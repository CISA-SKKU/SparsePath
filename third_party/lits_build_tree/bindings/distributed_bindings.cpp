// bindings/distributed_bindings.cpp — Thin pybind11 wrapper
// Target: ~1000 lines (type dispatch + py::dict conversion only)
//
// All logic lives in the danns:: modules:
//   core/   — VectorID, PriorityQueue, LatencyStats
//   rdma/   — RDMA transport
//   store/  — NodeStore (ColocationStore, Fetcher)
//   search/ — ColocationBeamSearch, PQFilter
//   update/ — Consolidation
//   cluster/— ClusterSetup, PartitionMapper
//   trace/  — SearchTraceWriter, ConsolidationTraceWriter

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <map>
#include <mutex>
#include <omp.h>
#include <atomic>
#include <optional>
#include <fstream>

// DANNS_SEARCH_WARMER_OFF=1: do not wire the consolidation-cache warmer into
// search (used when search co-runs with an active consolidation that is
// concurrently consuming/compacting the cache). Read per call, NOT latched:
// ConcurrentOp toggles the env at op boundaries, possibly after an earlier
// warm-search op already exercised this path.
static bool danns_search_warmer_off() {
    const char* v = std::getenv("DANNS_SEARCH_WARMER_OFF");
    return v && v[0] && v[0] != '0';
}

#include "core/vector_id.h"
#include "core/priority_queue.h"
#include "core/latency_stats.h"
#include "rdma/common.h"
#include "rdma/connection.h"
#include "store/node_store.h"
#include "store/consolidation_cache.h"
#include "store/redirect_alloc.h"
#include "search/beam_search.h"
#include "search/pq_filter.h"
#include "update/consolidation.h"
#include "update/insertion.h"
#include "update/load_balancer.h"
#include "cluster/cluster_setup.h"
#include "cluster/partition_mapper.h"
#include "trace/search_trace.h"
#include "trace/consolidation_trace.h"

// DiskANN distance (for search/consolidation)
#include "distance.h"

namespace py = pybind11;

// Multi-dtype support
enum class DataType { FLOAT32, INT8, UINT8 };

// AVX2-aligned dimension rounding
#ifndef ROUND_UP
#define ROUND_UP(X, Y) (((X) + (Y) - 1) / (Y) * (Y))
#endif

// =============================================================================
// DANNSEngine — Python-visible orchestrator combining all modules
// =============================================================================
class DANNSEngine {
public:
    DANNSEngine(uint32_t node_id, uint32_t total_nodes, uint32_t vector_dim,
              const std::string& rdma_device = "", int gid_index = 1)
        : node_id_(node_id), total_nodes_(total_nodes), vector_dim_(vector_dim),
          cluster_(node_id, total_nodes, rdma_device, gid_index) {
        aligned_dim_ = ROUND_UP(vector_dim, 8);
    }

    // === RDMA Setup (cluster/) ===
    void init_rdma() { cluster_.init(); }

    void connect_to_peer(uint32_t peer_id, const std::string& host, uint16_t port) {
        cluster_.connect_to_peer(peer_id, host, port);
        peer_hosts_[peer_id] = host;
    }

    void cleanup() {
        // Tear down per-thread QP/MR/CQ state while the endpoints are still
        // alive — node_store_.cleanup()/cluster_.cleanup() destroy them and
        // teardown_multi_qp would then skip everything as orphaned.
        teardown_multi_qp();
        redirect_alloc_.stop();
        redirect_endpoints_.clear();
        node_store_.cleanup();
        cluster_.cleanup();
    }

    // === Partition Mapping (cluster/) ===
    void set_total_vectors(uint32_t total) { mapper_.set_total_vectors(total); }

    void set_partition_assignments(py::array_t<uint32_t> ids) {
        auto buf = ids.request();
        auto* ptr = static_cast<uint32_t*>(buf.ptr);
        mapper_.set_partition_assignments(std::vector<uint32_t>(ptr, ptr + buf.size));
    }

    void set_partition_node_mapping(py::list sorted_ids) {
        std::vector<uint32_t> v;
        for (auto& item : sorted_ids) v.push_back(item.cast<uint32_t>());
        mapper_.set_partition_node_mapping(v);
    }

    // Per-partition slot bound for search-path VID sanity checks
    // (resolve_vid). 0 disables the check.
    void set_partition_max_local_locs(size_t max_local_locs) {
        mapper_.set_max_local_locs(max_local_locs);
    }

    void load_tags(const std::string& tags_file) { mapper_.load_tags(tags_file); }

    void update_tag_to_local_entry(uint32_t partition, uint32_t tag, uint32_t local_loc) {
        auto& t2l = node_store_.tag_to_local_mut();
        auto& vec = t2l[partition];
        if (tag >= vec.size()) {
            vec.resize(tag + 1, UINT32_MAX);
        }
        vec[tag] = local_loc;
    }

    void update_local_to_tag_entry(uint32_t partition, uint32_t local_loc, uint32_t tag) {
        auto& l2t = node_store_.local_to_tag_mut();
        if (partition >= l2t.size()) {
            l2t.resize(partition + 1);
        }
        auto& vec = l2t[partition];
        if (local_loc >= vec.size()) {
            vec.resize(local_loc + 1, 0);
        }
        vec[local_loc] = tag;
    }

    // =========================================================================
    // Phase 1: Distance init + entry points + utilities
    // =========================================================================

    void init_distance_only(uint32_t dim, const std::string& data_type = "float32") {
        if (data_type == "int8") data_type_ = DataType::INT8;
        else if (data_type == "uint8") data_type_ = DataType::UINT8;
        else data_type_ = DataType::FLOAT32;

        aligned_dim_ = ROUND_UP(dim, 8);

        switch (data_type_) {
            case DataType::INT8:
                dist_i8_ = std::make_unique<diskann::DistanceL2Int8>();
                break;
            case DataType::UINT8:
                dist_u8_ = std::make_unique<diskann::DistanceL2UInt8>();
                break;
            default:
                dist_f32_ = std::make_unique<diskann::DistanceL2Float>();
                break;
        }
    }


    void set_centroid_local_locs(py::dict locs, uint32_t max_points) {
        std::unordered_map<uint32_t, uint32_t> m;
        for (auto& [key, val] : locs) {
            m[key.cast<uint32_t>()] = val.cast<uint32_t>();
        }
        mapper_.set_centroid_local_locs(m, max_points);
    }

    void set_frozen_local_loc(uint32_t loc) { mapper_.set_frozen_local_loc(loc); }
    void set_frozen_canonical_vid(uint64_t vid) { mapper_.set_frozen_canonical_vid(vid); }

    uint32_t get_max_points() const { return mapper_.max_points(); }


    uint32_t get_location_count() const {
        auto* store = node_store_.local_store();
        return store ? store->num_entries() : 0;
    }

    // === Store (store/) ===
    void load_colocation(
        const std::string& coloc_file,
        const std::string& global_to_vectorid_file,
        py::dict tag_to_local_files_py,
        py::dict local_to_tag_files_py,
        uint32_t partition_id, uint32_t dim, uint32_t R, uint32_t slack_R,
        uint32_t entry_size, uint32_t alignment, uint32_t num_entries,
        uint32_t total_vectors, uint32_t element_size = 4) {

        // Convert py::dict to C++ maps
        std::unordered_map<uint32_t, std::string> t2l_files, l2t_files;
        for (auto item : tag_to_local_files_py)
            t2l_files[item.first.cast<uint32_t>()] = item.second.cast<std::string>();
        for (auto item : local_to_tag_files_py)
            l2t_files[item.first.cast<uint32_t>()] = item.second.cast<std::string>();

        rdma::ColocationConfig config;
        config.dim = dim;
        config.R = R;
        config.slack_R = slack_R;
        config.entry_size = entry_size;
        config.alignment = alignment;
        config.num_entries = num_entries;
        config.element_size = element_size;
        config.compute_offsets();

        node_store_.load(coloc_file, global_to_vectorid_file,
                         t2l_files, l2t_files, partition_id, config, total_vectors);
        // Pin node_id eagerly so single-node runs (no peer MR exchange) still
        // see a correct my_node value during consolidation. Multi-node runs
        // still call set_node_id again from exchange_mr_with_peer; same value.
        node_store_.set_node_id(node_id_);
    }

    void exchange_coloc_mr_with_peer(uint32_t peer_id, bool exchange_counter = false) {
        auto ep = cluster_.endpoint(peer_id);
        if (!ep) throw std::runtime_error("Not connected to peer " + std::to_string(peer_id));
        node_store_.exchange_mr_with_peer(peer_id, node_id_, ep, fetch_mode_,
                                          exchange_counter);
    }

    // === LoadBalancer counter MR (Stage B) ===
    // Python pushes slot_manager.active_count here at insert-batch
    // boundaries; peers read it via one-sided RDMA READ.
    void set_local_vector_count(uint64_t n) {
        node_store_.set_local_vector_count(n);
    }
    uint64_t local_vector_count() const {
        return node_store_.local_vector_count();
    }
    // Debug/diagnostic: read a peer's live counter over RDMA.
    uint64_t read_peer_vector_count(uint32_t peer_id) {
        return node_store_.read_peer_vector_count(peer_id);
    }

    // === Live redirect alloc channel ===
    // Two dedicated endpoints per pair, role-split so each QP/CQ has one
    // owner thread: on the LOW-port channel the lower node id serves; on
    // the HIGH-port channel the higher id serves. Bootstrap mirrors
    // ClusterSetup (lower id listens on both).
    void connect_redirect_peer(uint32_t peer_id, const std::string& host,
                               uint16_t port_low, uint16_t port_high,
                               const std::string& rdma_device = "",
                               int gid_index = 1) {
        auto ep_low = std::make_shared<rdma::RDMAEndpoint>(rdma_device, gid_index);
        if (node_id_ < peer_id) ep_low->listen_and_connect(port_low);
        else                    ep_low->connect_to(host, port_low);

        auto ep_high = std::make_shared<rdma::RDMAEndpoint>(rdma_device, gid_index);
        if (node_id_ < peer_id) ep_high->listen_and_connect(port_high);
        else                    ep_high->connect_to(host, port_high);

        auto server_ep = (node_id_ < peer_id) ? ep_low : ep_high;
        auto client_ep = (node_id_ < peer_id) ? ep_high : ep_low;
        redirect_alloc_.add_peer(peer_id, server_ep, client_ep);
        redirect_endpoints_.push_back(ep_low);    // keep alive
        redirect_endpoints_.push_back(ep_high);
    }
    void start_redirect_server() { redirect_alloc_.start(); }
    void open_redirect_window(py::array_t<uint32_t> reserve_arr,
                              uint32_t max_tag, uint32_t num_partitions) {
        auto buf = reserve_arr.request();
        const uint32_t* data = static_cast<const uint32_t*>(buf.ptr);
        std::vector<uint32_t> reserve(data, data + buf.shape[0]);
        redirect_alloc_.open_window(std::move(reserve), max_tag, num_partitions);
    }
    py::dict close_redirect_window() {
        auto ledger = redirect_alloc_.close_window();
        py::dict d;
        py::list granted;
        for (const auto& [tag, floc] : ledger.granted)
            granted.append(py::make_tuple(tag, floc));
        py::list unused;
        for (uint32_t loc : ledger.reserve_unused)
            unused.append(loc);
        d["granted"] = granted;
        d["reserve_unused"] = unused;
        return d;
    }

    void set_fetch_mode(const std::string& mode) {
        if (mode == "colocation") fetch_mode_ = rdma::FetchMode::COLOCATION;
        else if (mode == "separated") fetch_mode_ = rdma::FetchMode::SEPARATED;
        else if (mode == "local_mmap") fetch_mode_ = rdma::FetchMode::LOCAL_MMAP;
        else throw std::runtime_error("Unknown fetch mode: " + mode);
        node_store_.set_fetch_mode(fetch_mode_);
    }

    // Register a peer's partition data (mmap'd locally) for LOCAL_MMAP fetch mode.
    // Caller passes the address as uintptr_t (Python side does ctypes mmap).
    // The Python side owns the mmap's lifetime — keep the mmap object alive
    // for the duration of the engine.
    void register_local_partition_data(uint32_t peer_id, uintptr_t addr) {
        const char* ptr = reinterpret_cast<const char*>(addr);
        node_store_.set_local_partition_data(peer_id, ptr);
    }

    // === Local diagnostic access ===
    // Entry identity vid (= encode(partition, tag)) at a local slot — used
    // by redirect settlement to verify a granted slot was actually claimed
    // (entry written) before adopting/publishing its mapping.
    uint64_t local_get_entry_vid(uint32_t local_loc) {
        auto* store = node_store_.local_store();
        if (!store || local_loc >= store->num_entries()) return 0;
        uint64_t vid;
        std::memcpy(&vid, store->get_entry(local_loc) +
                    node_store_.config().vector_id_offset, sizeof(uint64_t));
        return vid;
    }
    py::array_t<uint64_t> local_get_neighbors(uint32_t local_loc) {
        auto* store = node_store_.local_store();
        if (!store) throw std::runtime_error("Colocation not loaded");
        uint32_t deg = store->get_degree(local_loc);
        const uint64_t* nbrs = store->get_neighbors(local_loc);
        py::array_t<uint64_t> result(deg);
        auto* ptr = static_cast<uint64_t*>(result.request().ptr);
        std::memcpy(ptr, nbrs, deg * sizeof(uint64_t));
        return result;
    }

    // Dump the local colocation buffer to a file, byte-identical to the
    // .coloc on-disk format (ColocationStore::load is the exact inverse).
    // Callers must be quiescent; CheckpointOp runs between op barriers,
    // which guarantees no peer is mid-RDMA-write into this buffer.
    void dump_colocation(const std::string& path) {
        auto* store = node_store_.local_store();
        if (!store) throw std::runtime_error("Colocation not loaded");
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("dump_colocation: cannot open " + path);
        }
        out.write(store->get_entry(0), store->total_size());
        if (!out) {
            throw std::runtime_error("dump_colocation: write failed: " + path);
        }
        // NFS defers write-back errors to close(); failbit catches the
        // final flush so a failed dump can never report success.
        out.close();
        if (out.fail()) {
            throw std::runtime_error("dump_colocation: close/flush failed: " + path);
        }
    }


    // =========================================================================
    // Phase 2: Single-query search — search_colocation_pure
    // =========================================================================

private:
    template <typename T>
    std::tuple<py::array_t<uint32_t>, py::array_t<float>, py::dict>
    search_colocation_pure_typed(
        py::array_t<T> query,
        uint32_t K, uint32_t L,
        uint32_t entry_mode, uint32_t query_partition,
        bool verbose, uint32_t trace_query_id,
        uint32_t batch_size, uint32_t conv_window, float conv_threshold_pct,
        bool return_pool, bool lits_filter, uint32_t lits_my_partition
    ) {
        auto query_buf = query.request();
        if (query_buf.size != static_cast<ssize_t>(vector_dim_))
            throw std::runtime_error("Query dimension mismatch");
        if (L < K) L = K;

        T* query_ptr = static_cast<T*>(query_buf.ptr);

        // Distance function
        diskann::Distance<T>* dist_fn = nullptr;
        if constexpr (std::is_same_v<T, float>) {
            if (dist_f32_) dist_fn = dist_f32_.get();
        } else if constexpr (std::is_same_v<T, int8_t>) {
            if (dist_i8_) dist_fn = dist_i8_.get();
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            if (dist_u8_) dist_fn = dist_u8_.get();
        }
        if (!dist_fn)
            throw std::runtime_error("No distance function. Call init_distance_only() first.");

        const size_t cmp_dim = vector_dim_;

        // Allocate 32-byte aligned query buffer for AVX2
        size_t aq_bytes = ((aligned_dim_ * sizeof(T) + 31) / 32) * 32;
        void* aq_mem = std::aligned_alloc(32, aq_bytes);
        if (!aq_mem) throw std::runtime_error("aligned_alloc failed");
        T* aligned_query = static_cast<T*>(aq_mem);
        std::memcpy(aligned_query, query_ptr, vector_dim_ * sizeof(T));
        if (aligned_dim_ > vector_dim_)
            std::memset(aligned_query + vector_dim_, 0,
                        (aligned_dim_ - vector_dim_) * sizeof(T));


        using hrc = std::chrono::high_resolution_clock;
        auto t_start = hrc::now();

        consolidation_cache_.configure(cmp_dim * sizeof(T), node_store_.config().slack_R);

        // Build search config
        danns::SearchConfig config;
        config.K = K;
        config.L = L;
        config.entry_mode = entry_mode;
        config.query_partition = query_partition;
        config.batch_size = batch_size;
        config.conv_window = conv_window;
        config.conv_threshold_pct = conv_threshold_pct;
        config.return_pool = return_pool;
        config.lits_filter = lits_filter;
        config.lits_my_partition = lits_my_partition;
        config.trace_query_id = trace_query_id;
        config.trace_level = search_vfct_enabled_
            ? std::max<uint16_t>(trace_writer_.level(), 1)
            : trace_writer_.level();
        config.consolidation_cache_warmer =
            danns_search_warmer_off() ? nullptr : &consolidation_cache_;

        auto sr = beam_search_.search<T>(
            aligned_query, aligned_dim_, cmp_dim, dist_fn, config,
            &node_store_, &mapper_, &pq_filter_, nullptr, 1, 0);

        auto t_end = hrc::now();
        double total_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_end - t_start).count();

        // Convert results to numpy
        auto indices = py::array_t<uint32_t>(K);
        auto distances = py::array_t<float>(K);
        uint32_t* idx_ptr = static_cast<uint32_t*>(indices.request().ptr);
        float* dist_ptr = static_cast<float*>(distances.request().ptr);

        size_t copy_k = std::min(static_cast<size_t>(K), sr.indices.size());
        std::memcpy(idx_ptr, sr.indices.data(), copy_k * sizeof(uint32_t));
        std::memcpy(dist_ptr, sr.distances.data(), copy_k * sizeof(float));

        // Write trace if active
        if (trace_writer_.level() >= 1) {
            danns::QueryHeader qh{};
            qh.query_id = trace_query_id;
            qh.K = static_cast<uint16_t>(K);
            qh.L = static_cast<uint16_t>(L);
            qh.n_hops = static_cast<uint16_t>(sr.trace_hops.size());
            qh.n_results = sr.result_count;
            trace_writer_.write_query(qh, sr.trace_results.data(),
                                      sr.result_count,
                                      sr.trace_hops, sr.trace_fetched);
        }
        // Single-query VFCT accumulate. mode 0 (deleted/option-4): only
        // deleted Di hits; mode 1 (neighbors/option-2): every visited vid.
        if (search_vfct_enabled_) {
            for (auto const& hop : sr.trace_hops) {
                const uint64_t vid = hop.expanded_vid;
                if (search_vfct_mode_ == 0 &&
                    !mapper_.is_vid_deleted(vid)) continue;
                uint32_t part = danns::decode_partition(vid);
                uint32_t loc = static_cast<uint32_t>(
                    danns::decode_local_loc(vid));
                if (part < search_vec_fetch_count_.size() &&
                    loc < search_vec_fetch_count_[part].size()) {
                    search_vec_fetch_count_[part][loc].fetch_add(
                        1, std::memory_order_relaxed);
                }
            }
        }

        // Build stats dict
        py::dict stats_dict;
        stats_dict["total_us"] = total_us;
        stats_dict["n_hops"] = sr.n_hops;
        stats_dict["n_cmps"] = sr.n_cmps;
        stats_dict["local_vector_accesses"] = sr.local_vector_accesses;
        stats_dict["remote_vector_accesses"] = sr.remote_vector_accesses;
        stats_dict["local_neighbor_accesses"] = sr.local_neighbor_accesses;
        stats_dict["remote_neighbor_accesses"] = sr.remote_neighbor_accesses;
        stats_dict["resolve_fail"] = sr.resolve_fail;
        stats_dict["rdma_vector_us"] = sr.rdma_vector_ns / 1000.0;
        stats_dict["rdma_neighbor_us"] = sr.rdma_neighbor_ns / 1000.0;
        stats_dict["rdma_entry_calls"] = sr.rdma_entry_calls;
        stats_dict["rdma_entry_items"] = sr.rdma_entry_items;
        stats_dict["rdma_vec_fetch_calls"] = sr.rdma_vec_fetch_calls;
        stats_dict["rdma_nbr_expand_calls"] = sr.rdma_nbr_expand_calls;
        stats_dict["rdma_full_entry_fetches"] = sr.rdma_full_entry_fetches;
        stats_dict["rdma_vec_only_fetches"] = sr.rdma_vec_only_fetches;
        stats_dict["visited_count"] = sr.visited_count;
        stats_dict["pq_filtered"] = sr.pq_filtered;
        stats_dict["unresolved_vids"] = sr.unresolved_vids;
        stats_dict["candidate_fetch_phase_us"] = sr.candidate_fetch_phase_ns / 1000.0;
        stats_dict["pass1_us"] = sr.pass1_ns / 1000.0;
        stats_dict["pass2_us"] = sr.pass2_ns / 1000.0;
        stats_dict["pass3_us"] = sr.pass3_ns / 1000.0;
        stats_dict["pass3_poll_us"] = sr.pass3_poll_ns / 1000.0;
        stats_dict["pass3_compute_us"] = sr.pass3_compute_ns / 1000.0;
        stats_dict["batch_nbr_poll_us"] = sr.batch_nbr_poll_ns / 1000.0;
        stats_dict["batch_vec_poll_us"] = sr.batch_vec_poll_ns / 1000.0;

        if (batch_size > 0) {
            stats_dict["batch_size"] = batch_size;
            stats_dict["convergence_hop"] = sr.convergence_hop;
            stats_dict["batch_rounds"] = sr.n_batch_rounds;
            stats_dict["batch_candidates_total"] = sr.n_batch_candidates_total;
            stats_dict["fallback_count"] = sr.fallback_count;
        }



        // Return full candidate pool if requested
        if (return_pool && !sr.pool.empty()) {
            auto pool_vids = py::array_t<uint64_t>(sr.pool.size());
            auto pool_dists = py::array_t<float>(sr.pool.size());
            auto* pv = static_cast<uint64_t*>(pool_vids.mutable_data());
            auto* pd = static_cast<float*>(pool_dists.mutable_data());
            for (size_t i = 0; i < sr.pool.size(); i++) {
                pv[i] = sr.pool[i].first;
                pd[i] = sr.pool[i].second;
            }
            stats_dict["pool_vids"] = pool_vids;
            stats_dict["pool_dists"] = pool_dists;
        }

        std::free(aq_mem);
        return std::make_tuple(indices, distances, stats_dict);
    }

public:
    // Type dispatch for single search
    std::tuple<py::array_t<uint32_t>, py::array_t<float>, py::dict>
    search_colocation_pure(
        py::object query_obj,
        uint32_t K, uint32_t L,
        uint32_t entry_mode = 0,
        uint32_t query_partition = 0,
        bool verbose = false,
        uint32_t trace_query_id = 0,
        uint32_t batch_size = 0,
        uint32_t conv_window = 3,
        float conv_threshold_pct = 1.0f,
        bool return_pool = false,
        bool lits_filter = false,
        uint32_t lits_my_partition = 0
    ) {
        if (!node_store_.local_store())
            throw std::runtime_error("Co-location not loaded. Call load_colocation() first.");
        switch (data_type_) {
            case DataType::INT8: {
                auto q = query_obj.cast<py::array_t<int8_t>>();
                return search_colocation_pure_typed<int8_t>(
                    q, K, L, entry_mode, query_partition, verbose, trace_query_id,
                    batch_size, conv_window, conv_threshold_pct,
                    return_pool,
                    lits_filter, lits_my_partition);
            }
            case DataType::UINT8: {
                auto q = query_obj.cast<py::array_t<uint8_t>>();
                return search_colocation_pure_typed<uint8_t>(
                    q, K, L, entry_mode, query_partition, verbose, trace_query_id,
                    batch_size, conv_window, conv_threshold_pct,
                    return_pool,
                    lits_filter, lits_my_partition);
            }
            default: {
                auto q = query_obj.cast<py::array_t<float>>();
                return search_colocation_pure_typed<float>(
                    q, K, L, entry_mode, query_partition, verbose, trace_query_id,
                    batch_size, conv_window, conv_threshold_pct,
                    return_pool,
                    lits_filter, lits_my_partition);
            }
        }
    }

    // =========================================================================
    // Phase 3: Batch search — search_colocation_pure_batch
    // =========================================================================

private:
    template <typename T>
    py::dict search_colocation_pure_batch_typed(
        py::array_t<T> queries, py::array_t<uint32_t> qids,
        uint32_t K, uint32_t L, uint32_t query_partition,
        int num_threads, uint32_t batch_size, uint32_t conv_window,
        float conv_threshold_pct, uint32_t entry_mode, bool skip_timing,
        bool compute_skip, uint32_t interleave_factor,
        int qp_tid_offset = 0
    ) {
        auto q_buf = queries.request();
        if (q_buf.ndim != 2 || q_buf.shape[1] != static_cast<ssize_t>(vector_dim_))
            throw std::runtime_error("queries must be (N, dim) array");
        const ssize_t N = q_buf.shape[0];
        const T* q_data = static_cast<const T*>(q_buf.ptr);

        auto qid_buf = qids.request();
        if (qid_buf.ndim != 1 || qid_buf.shape[0] != N)
            throw std::runtime_error("qids must be (N,) array matching queries");
        const uint32_t* qid_data = static_cast<const uint32_t*>(qid_buf.ptr);

        if (L < K) L = K;

        // Distance function
        diskann::Distance<T>* dist_fn = nullptr;
        if constexpr (std::is_same_v<T, float>) {
            if (dist_f32_) dist_fn = dist_f32_.get();
        } else if constexpr (std::is_same_v<T, int8_t>) {
            if (dist_i8_) dist_fn = dist_i8_.get();
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            if (dist_u8_) dist_fn = dist_u8_.get();
        }
        if (!dist_fn)
            throw std::runtime_error("No distance function. Call init_distance_only() first.");
        const size_t cmp_dim = vector_dim_;

        // Pre-allocate output arrays
        auto out_indices = py::array_t<uint32_t>({N, static_cast<ssize_t>(K)});
        auto out_distances = py::array_t<float>({N, static_cast<ssize_t>(K)});
        uint32_t* out_idx_ptr = static_cast<uint32_t*>(out_indices.request().ptr);
        float* out_dist_ptr = static_cast<float*>(out_distances.request().ptr);

        // Create fetcher mutexes if multi-threaded
        std::unique_ptr<std::map<uint32_t, std::mutex>> fetcher_mutexes;
        if (num_threads > 1) {
            fetcher_mutexes = std::make_unique<std::map<uint32_t, std::mutex>>();
            // Initialize mutexes for all peers with fetchers
            for (uint32_t p = 0; p < total_nodes_; p++) {
                if (p != node_id_ && node_store_.fetcher(p))
                    (*fetcher_mutexes)[p];
            }
        }


        // Trace buffers
        // Effective trace level: at least 1 if VFCT is enabled (need HopRecord
        // for expanded_vid lookup; FetchedEntry not required for option-4
        // "expanded deleted Di" metric). otherwise whatever the trace_writer wants.
        const uint16_t writer_tl = trace_writer_.level();
        const uint16_t tl = search_vfct_enabled_
            ? std::max<uint16_t>(writer_tl, 1)
            : writer_tl;
        struct TraceEntry {
            ssize_t query_idx;
            uint32_t query_id;
            uint16_t K_val, L_val;
            danns::SearchResult<T> sr;
        };
        std::vector<std::vector<TraceEntry>> thread_traces;
        if (tl >= 1) thread_traces.resize(num_threads > 1 ? num_threads : 1);

        // Reduction variables
        uint64_t total_n_hops = 0, total_n_cmps = 0;
        uint64_t total_local_vec = 0, total_remote_vec = 0;
        uint64_t total_local_nbr = 0, total_remote_nbr = 0;
        uint64_t total_resolve_fail = 0, total_visited = 0;
        long long total_rdma_vec_ns = 0, total_rdma_nbr_ns = 0;
        uint64_t total_rdma_entry_calls = 0, total_rdma_entry_items = 0;
        uint64_t total_rdma_vec_fetch_calls = 0, total_rdma_nbr_expand_calls = 0;
        uint64_t total_rdma_full_entry_fetches = 0, total_rdma_vec_only_fetches = 0;
        uint64_t total_convergence_hop = 0, total_batch_rounds = 0;
        uint64_t total_batch_cands = 0, total_fallback = 0;
        uint64_t total_pq_filtered = 0, total_unresolved_vids = 0;
        long long total_cpu_visited_ns = 0, total_cpu_resolve_ns = 0;
        long long total_cpu_distance_ns = 0, total_cpu_queue_ns = 0;
        long long total_cpu_nbr_loop_ns = 0, total_cpu_map_ns = 0, total_cpu_pq_ns = 0;
        long long total_lut_build_ns = 0;
        long long total_candidate_fetch_phase_ns = 0;
        long long total_pass1_ns = 0, total_pass2_ns = 0, total_pass3_ns = 0;
        long long total_pass3_poll_ns = 0, total_pass3_compute_ns = 0;
        long long total_batch_nbr_poll_ns = 0, total_batch_vec_poll_ns = 0;

        std::vector<double> per_query_latency_us(N, 0.0);

        consolidation_cache_.configure(cmp_dim * sizeof(T), node_store_.config().slack_R);

        using hrc = std::chrono::high_resolution_clock;
        auto t_wall_start = hrc::now();

        // Decide whether to use the hop-interleaved multi-query path. Constant
        // across threads; gated on multi-QP + SEPARATED + all optimizations off.
        danns::SearchConfig probe_cfg;
        probe_cfg.interleave_factor = interleave_factor;
        probe_cfg.batch_size = batch_size;
        probe_cfg.trace_level = tl;
        probe_cfg.return_pool = false;
        const bool use_interleave = danns::ColocationBeamSearch::interleave_feasible(
            probe_cfg, &node_store_, &mapper_, &pq_filter_, num_threads);
        if (use_interleave) {
            rdma::Logger::info("search_batch: hop-interleaved path (Q=%u, threads=%d)",
                              interleave_factor, num_threads);
        } else if (interleave_factor > 1) {
            rdma::Logger::info("search_batch: interleave_factor=%u requested but path "
                              "infeasible (need multi-QP + separated + no PQ/batch/trace); "
                              "using per-query path", interleave_factor);
        }

        // C2: capture exceptions escaping the OMP region. An RDMAException that
        // unwinds out of a parallel region calls std::terminate -> SIGABRT (no
        // Python exception, no FAILED marking). First error wins; it is re-thrown
        // serially after the region so pybind converts it to a Python exception.
        std::atomic<bool> err_flag{false};
        std::string err_msg;
        std::mutex err_mtx;
        {
            py::gil_scoped_release release;
            if (num_threads > 1) omp_set_num_threads(num_threads);

            #pragma omp parallel if(num_threads > 1)
            {
                size_t aq_bytes = ((aligned_dim_ * sizeof(T) + 31) / 32) * 32;
                void* aq_mem = std::aligned_alloc(32, aq_bytes);
                T* aq = aq_mem ? static_cast<T*>(aq_mem) : nullptr;

                int tid = 0;
                if (num_threads > 1) {
                    tid = omp_get_thread_num();
                }

                danns::SearchConfig cfg;
                cfg.K = K; cfg.L = L;
                cfg.entry_mode = entry_mode;
                cfg.query_partition = query_partition;
                cfg.batch_size = batch_size;
                cfg.conv_window = conv_window;
                cfg.conv_threshold_pct = conv_threshold_pct;
                cfg.skip_timing = skip_timing;
                cfg.compute_skip = compute_skip;
                cfg.interleave_factor = interleave_factor;
                cfg.consolidation_cache_warmer =
                    danns_search_warmer_off() ? nullptr : &consolidation_cache_;

              if (!use_interleave) {
                #pragma omp for schedule(dynamic, 16) \
                    reduction(+:total_n_hops, total_n_cmps, total_local_vec, total_remote_vec, \
                              total_local_nbr, total_remote_nbr, total_resolve_fail, total_visited, \
                              total_rdma_vec_ns, total_rdma_nbr_ns, \
                              total_rdma_entry_calls, total_rdma_entry_items, \
                              total_rdma_vec_fetch_calls, total_rdma_nbr_expand_calls, \
                              total_rdma_full_entry_fetches, total_rdma_vec_only_fetches, \
                              total_convergence_hop, total_batch_rounds, total_batch_cands, \
                              total_fallback, total_pq_filtered, \
                              total_unresolved_vids, \
                              total_cpu_visited_ns, total_cpu_resolve_ns, \
                              total_cpu_distance_ns, total_cpu_queue_ns, \
                              total_cpu_nbr_loop_ns, total_cpu_map_ns, total_cpu_pq_ns, \
                              total_lut_build_ns, \
                              total_candidate_fetch_phase_ns, \
                              total_pass1_ns, total_pass2_ns, total_pass3_ns, \
                              total_pass3_poll_ns, total_pass3_compute_ns, \
                              total_batch_nbr_poll_ns, total_batch_vec_poll_ns)
                for (ssize_t qi = 0; qi < N; qi++) {
                    if (!aq) continue;
                    if (err_flag.load(std::memory_order_relaxed)) continue;
                    try {

                    std::memcpy(aq, q_data + qi * vector_dim_, vector_dim_ * sizeof(T));
                    if (aligned_dim_ > vector_dim_)
                        std::memset(aq + vector_dim_, 0, (aligned_dim_ - vector_dim_) * sizeof(T));


                    cfg.trace_query_id = qid_data[qi];
                    cfg.trace_level = tl;

                    auto t_q_start = hrc::now();
                    // qp_tid_offset: shift this region's tids into a disjoint
                    // slice of the shared multi-QP pool (ConcurrentOp gives
                    // consolidation tids [0, Cn) and search [Cn, Cn+Sn)).
                    auto sr = beam_search_.search<T>(
                        aq, aligned_dim_, cmp_dim, dist_fn, cfg,
                        &node_store_, &mapper_, &pq_filter_,
                        fetcher_mutexes.get(), num_threads,
                        tid + qp_tid_offset);
                    auto t_q_end = hrc::now();
                    per_query_latency_us[qi] = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t_q_end - t_q_start).count() / 1000.0;

                    size_t copy_k = std::min(static_cast<size_t>(K), sr.indices.size());
                    std::memcpy(out_idx_ptr + qi * K, sr.indices.data(), copy_k * sizeof(uint32_t));
                    std::memcpy(out_dist_ptr + qi * K, sr.distances.data(), copy_k * sizeof(float));

                    // Accumulate stats
                    total_n_hops += sr.n_hops;
                    total_n_cmps += sr.n_cmps;
                    total_local_vec += sr.local_vector_accesses;
                    total_remote_vec += sr.remote_vector_accesses;
                    total_local_nbr += sr.local_neighbor_accesses;
                    total_remote_nbr += sr.remote_neighbor_accesses;
                    total_resolve_fail += sr.resolve_fail;
                    total_visited += sr.visited_count;
                    total_rdma_vec_ns += sr.rdma_vector_ns;
                    total_rdma_nbr_ns += sr.rdma_neighbor_ns;
                    total_rdma_entry_calls += sr.rdma_entry_calls;
                    total_rdma_entry_items += sr.rdma_entry_items;
                    total_rdma_vec_fetch_calls += sr.rdma_vec_fetch_calls;
                    total_rdma_nbr_expand_calls += sr.rdma_nbr_expand_calls;
                    total_rdma_full_entry_fetches += sr.rdma_full_entry_fetches;
                    total_rdma_vec_only_fetches += sr.rdma_vec_only_fetches;
                    total_convergence_hop += sr.convergence_hop;
                    total_batch_rounds += sr.n_batch_rounds;
                    total_batch_cands += sr.n_batch_candidates_total;
                    total_fallback += sr.fallback_count;
                    total_pq_filtered += sr.pq_filtered;
                    total_unresolved_vids += sr.unresolved_vids;
                    total_cpu_visited_ns += sr.cpu_visited_ns;
                    total_cpu_resolve_ns += sr.cpu_resolve_ns;
                    total_cpu_distance_ns += sr.cpu_distance_ns;
                    total_cpu_queue_ns += sr.cpu_queue_ns;
                    total_cpu_nbr_loop_ns += sr.cpu_nbr_loop_ns;
                    total_cpu_map_ns += sr.cpu_map_ns;
                    total_cpu_pq_ns += sr.cpu_pq_ns;
                    total_lut_build_ns += sr.lut_build_ns;
                    total_candidate_fetch_phase_ns += sr.candidate_fetch_phase_ns;
                    total_pass1_ns += sr.pass1_ns;
                    total_pass2_ns += sr.pass2_ns;
                    total_pass3_ns += sr.pass3_ns;
                    total_pass3_poll_ns += sr.pass3_poll_ns;
                    total_pass3_compute_ns += sr.pass3_compute_ns;
                    total_batch_nbr_poll_ns += sr.batch_nbr_poll_ns;
                    total_batch_vec_poll_ns += sr.batch_vec_poll_ns;

                    // VFCT accumulate. mode 0 (deleted/option-4): "expanded
                    // deleted Di" — count every deleted node search expanded.
                    // mode 1 (neighbors/option-2): count every visited vid
                    // (deleted or not), used by recompute_coverage_v2.
                    if (search_vfct_enabled_) {
                        for (auto const& hop : sr.trace_hops) {
                            const uint64_t vid = hop.expanded_vid;
                            if (search_vfct_mode_ == 0 &&
                                !mapper_.is_vid_deleted(vid)) continue;
                            uint32_t part = danns::decode_partition(vid);
                            uint32_t loc = static_cast<uint32_t>(
                                danns::decode_local_loc(vid));
                            if (part < search_vec_fetch_count_.size() &&
                                loc < search_vec_fetch_count_[part].size()) {
                                search_vec_fetch_count_[part][loc].fetch_add(
                                    1, std::memory_order_relaxed);
                            }
                        }
                    }
                    // Disk-trace path still buffers per-query for sorted writes.
                    // Skip when only VFCT is on (writer level == 0) to save RAM.
                    if (writer_tl >= 1) {
                        int trace_tid = (num_threads > 1) ? tid : 0;
                        TraceEntry te;
                        te.query_idx = qi;
                        te.query_id = qid_data[qi];
                        te.K_val = static_cast<uint16_t>(K);
                        te.L_val = static_cast<uint16_t>(L);
                        te.sr = std::move(sr);
                        thread_traces[trace_tid].push_back(std::move(te));
                    }
                    } catch (const std::exception& e) {
                        bool expected = false;
                        if (err_flag.compare_exchange_strong(expected, true)) {
                            std::lock_guard<std::mutex> lk(err_mtx);
                            err_msg = e.what();
                        }
                    }
                }
              } else {
                // ---- Hop-interleaved multi-query path ----
                // Each thread drives Q queries per group, interleaving their
                // per-hop RDMA. Results are bit-identical to the per-query path.
                const uint32_t Q = interleave_factor;
                const ssize_t n_groups = (N + Q - 1) / Q;
                size_t aq_bytes_g = ((aligned_dim_ * sizeof(T) + 31) / 32) * 32;
                size_t aq_stride = aq_bytes_g / sizeof(T);
                void* aqg_mem = std::aligned_alloc(32, aq_bytes_g * Q);
                T* aqg = aqg_mem ? static_cast<T*>(aqg_mem) : nullptr;
                std::vector<const T*> qptrs(Q, nullptr);

                #pragma omp for schedule(dynamic, 1) \
                    reduction(+:total_n_hops, total_n_cmps, total_local_vec, total_remote_vec, \
                              total_local_nbr, total_remote_nbr, total_resolve_fail, total_visited, \
                              total_rdma_vec_ns, total_rdma_nbr_ns, \
                              total_rdma_entry_calls, total_rdma_entry_items, \
                              total_rdma_vec_fetch_calls, total_rdma_nbr_expand_calls, \
                              total_rdma_full_entry_fetches, total_rdma_vec_only_fetches, \
                              total_convergence_hop, total_batch_rounds, total_batch_cands, \
                              total_fallback, total_pq_filtered, \
                              total_unresolved_vids, \
                              total_cpu_visited_ns, total_cpu_resolve_ns, \
                              total_cpu_distance_ns, total_cpu_queue_ns, \
                              total_cpu_nbr_loop_ns, total_cpu_map_ns, total_cpu_pq_ns, \
                              total_lut_build_ns, \
                              total_candidate_fetch_phase_ns, \
                              total_pass1_ns, total_pass2_ns, total_pass3_ns, \
                              total_pass3_poll_ns, total_pass3_compute_ns, \
                              total_batch_nbr_poll_ns, total_batch_vec_poll_ns)
                for (ssize_t g = 0; g < n_groups; g++) {
                    if (!aqg) continue;
                    if (err_flag.load(std::memory_order_relaxed)) continue;
                    try {
                    ssize_t base = g * static_cast<ssize_t>(Q);
                    uint32_t gsize = static_cast<uint32_t>(std::min<ssize_t>(Q, N - base));
                    for (uint32_t b = 0; b < gsize; b++) {
                        T* slot = aqg + static_cast<size_t>(b) * aq_stride;
                        std::memcpy(slot, q_data + (base + b) * vector_dim_,
                                    vector_dim_ * sizeof(T));
                        if (aligned_dim_ > vector_dim_)
                            std::memset(slot + vector_dim_, 0,
                                        (aligned_dim_ - vector_dim_) * sizeof(T));
                        qptrs[b] = slot;
                    }

                    auto t_g_start = hrc::now();
                    auto srs = beam_search_.search_interleaved<T>(
                        qptrs.data(), qid_data + base, gsize,
                        aligned_dim_, cmp_dim, dist_fn, cfg,
                        &node_store_, &mapper_, num_threads, tid);
                    auto t_g_end = hrc::now();
                    double per_us = gsize
                        ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                              t_g_end - t_g_start).count() / 1000.0 / gsize
                        : 0.0;

                    for (uint32_t b = 0; b < gsize; b++) {
                        ssize_t qi = base + b;
                        auto& sr = srs[b];
                        per_query_latency_us[qi] = per_us;
                        size_t copy_k = std::min(static_cast<size_t>(K), sr.indices.size());
                        std::memcpy(out_idx_ptr + qi * K, sr.indices.data(), copy_k * sizeof(uint32_t));
                        std::memcpy(out_dist_ptr + qi * K, sr.distances.data(), copy_k * sizeof(float));

                        total_n_hops += sr.n_hops;
                        total_n_cmps += sr.n_cmps;
                        total_local_vec += sr.local_vector_accesses;
                        total_remote_vec += sr.remote_vector_accesses;
                        total_local_nbr += sr.local_neighbor_accesses;
                        total_remote_nbr += sr.remote_neighbor_accesses;
                        total_resolve_fail += sr.resolve_fail;
                        total_visited += sr.visited_count;
                        total_rdma_vec_ns += sr.rdma_vector_ns;
                        total_rdma_nbr_ns += sr.rdma_neighbor_ns;
                        total_rdma_entry_calls += sr.rdma_entry_calls;
                        total_rdma_entry_items += sr.rdma_entry_items;
                        total_rdma_vec_fetch_calls += sr.rdma_vec_fetch_calls;
                        total_rdma_nbr_expand_calls += sr.rdma_nbr_expand_calls;
                        total_rdma_full_entry_fetches += sr.rdma_full_entry_fetches;
                        total_rdma_vec_only_fetches += sr.rdma_vec_only_fetches;
                        total_convergence_hop += sr.convergence_hop;
                        total_batch_rounds += sr.n_batch_rounds;
                        total_batch_cands += sr.n_batch_candidates_total;
                        total_fallback += sr.fallback_count;
                        total_pq_filtered += sr.pq_filtered;
                        total_unresolved_vids += sr.unresolved_vids;
                        total_cpu_visited_ns += sr.cpu_visited_ns;
                        total_cpu_resolve_ns += sr.cpu_resolve_ns;
                        total_cpu_distance_ns += sr.cpu_distance_ns;
                        total_cpu_queue_ns += sr.cpu_queue_ns;
                        total_cpu_nbr_loop_ns += sr.cpu_nbr_loop_ns;
                        total_cpu_map_ns += sr.cpu_map_ns;
                        total_cpu_pq_ns += sr.cpu_pq_ns;
                        total_lut_build_ns += sr.lut_build_ns;
                        total_candidate_fetch_phase_ns += sr.candidate_fetch_phase_ns;
                        total_pass1_ns += sr.pass1_ns;
                        total_pass2_ns += sr.pass2_ns;
                        total_pass3_ns += sr.pass3_ns;
                        total_pass3_poll_ns += sr.pass3_poll_ns;
                        total_pass3_compute_ns += sr.pass3_compute_ns;
                        total_batch_nbr_poll_ns += sr.batch_nbr_poll_ns;
                        total_batch_vec_poll_ns += sr.batch_vec_poll_ns;
                    }
                    } catch (const std::exception& e) {
                        bool expected = false;
                        if (err_flag.compare_exchange_strong(expected, true)) {
                            std::lock_guard<std::mutex> lk(err_mtx);
                            err_msg = e.what();
                        }
                    }
                }
                if (aqg_mem) std::free(aqg_mem);
              }
                std::free(aq_mem);
            }
        } // GIL re-acquired

        // C2: re-raise any worker exception serially (GIL held -> Python exception).
        if (err_flag.load()) {
            throw std::runtime_error(err_msg);
        }

        auto t_wall_end = hrc::now();
        double wall_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_wall_end - t_wall_start).count();

        // Disk-trace merge + write (only when trace_writer is active). VFCT was
        // already accumulated inside the parallel loop above.
        if (writer_tl >= 1) {
            std::vector<TraceEntry*> all_traces;
            for (auto& tt : thread_traces)
                for (auto& te : tt) all_traces.push_back(&te);
            std::sort(all_traces.begin(), all_traces.end(),
                [](const TraceEntry* a, const TraceEntry* b) {
                    return a->query_idx < b->query_idx;
                });
            for (auto* te : all_traces) {
                danns::QueryHeader qh{};
                qh.query_id = te->query_id;
                qh.K = te->K_val;
                qh.L = te->L_val;
                qh.n_hops = static_cast<uint16_t>(te->sr.trace_hops.size());
                qh.n_results = te->sr.result_count;
                trace_writer_.write_query(qh, te->sr.trace_results.data(),
                                          te->sr.result_count,
                                          te->sr.trace_hops, te->sr.trace_fetched);
            }
        }
        // Build per-query latency array
        py::array_t<double> py_query_latency(N);
        std::memcpy(py_query_latency.mutable_data(), per_query_latency_us.data(), N * sizeof(double));

        // Build result dict
        py::dict result;
        result["indices"] = out_indices;
        result["distances"] = out_distances;
        result["per_query_latency_us"] = py_query_latency;
        result["wall_us"] = wall_us;
        result["num_queries"] = N;
        result["num_threads"] = num_threads;

        py::dict stats;
        stats["total_n_hops"] = total_n_hops;
        stats["total_n_cmps"] = total_n_cmps;
        stats["total_local_vector_accesses"] = total_local_vec;
        stats["total_remote_vector_accesses"] = total_remote_vec;
        stats["total_local_neighbor_accesses"] = total_local_nbr;
        stats["total_remote_neighbor_accesses"] = total_remote_nbr;
        stats["total_resolve_fail"] = total_resolve_fail;
        stats["total_visited_count"] = total_visited;
        stats["total_rdma_vector_us"] = total_rdma_vec_ns / 1000.0;
        stats["total_rdma_neighbor_us"] = total_rdma_nbr_ns / 1000.0;
        stats["total_rdma_entry_calls"] = total_rdma_entry_calls;
        stats["total_rdma_entry_items"] = total_rdma_entry_items;
        stats["total_rdma_vec_fetch_calls"] = total_rdma_vec_fetch_calls;
        stats["total_rdma_nbr_expand_calls"] = total_rdma_nbr_expand_calls;
        stats["total_rdma_full_entry_fetches"] = total_rdma_full_entry_fetches;
        stats["total_rdma_vec_only_fetches"] = total_rdma_vec_only_fetches;
        if (N > 0) {
            stats["avg_n_hops"] = static_cast<double>(total_n_hops) / N;
            stats["avg_n_cmps"] = static_cast<double>(total_n_cmps) / N;
            stats["avg_visited"] = static_cast<double>(total_visited) / N;
            stats["avg_us"] = wall_us / N;
        }
        if (batch_size > 0) {
            stats["total_convergence_hop"] = total_convergence_hop;
            stats["total_batch_rounds"] = total_batch_rounds;
            stats["total_batch_candidates"] = total_batch_cands;
            stats["total_fallback_count"] = total_fallback;
        }
        if (pq_filter_.enabled()) {
            stats["total_pq_filtered"] = total_pq_filtered;
        }
        stats["total_unresolved_vids"] = total_unresolved_vids;
        stats["total_cpu_visited_us"] = total_cpu_visited_ns / 1000.0;
        stats["total_cpu_resolve_us"] = total_cpu_resolve_ns / 1000.0;
        stats["total_cpu_distance_us"] = total_cpu_distance_ns / 1000.0;
        stats["total_cpu_queue_us"] = total_cpu_queue_ns / 1000.0;
        stats["total_cpu_nbr_loop_us"] = total_cpu_nbr_loop_ns / 1000.0;
        stats["total_cpu_map_us"] = total_cpu_map_ns / 1000.0;
        stats["total_cpu_pq_us"] = total_cpu_pq_ns / 1000.0;
        stats["total_lut_build_us"] = total_lut_build_ns / 1000.0;
        stats["total_candidate_fetch_phase_us"] = total_candidate_fetch_phase_ns / 1000.0;
        stats["total_pass1_us"] = total_pass1_ns / 1000.0;
        stats["total_pass2_us"] = total_pass2_ns / 1000.0;
        stats["total_pass3_us"] = total_pass3_ns / 1000.0;
        stats["total_pass3_poll_us"] = total_pass3_poll_ns / 1000.0;
        stats["total_pass3_compute_us"] = total_pass3_compute_ns / 1000.0;
        stats["total_batch_nbr_poll_us"] = total_batch_nbr_poll_ns / 1000.0;
        stats["total_batch_vec_poll_us"] = total_batch_vec_poll_ns / 1000.0;

        // Cache stats


        result["stats"] = stats;
        return result;
    }

public:
    // Type dispatch for batch search
    py::dict search_colocation_pure_batch(
        py::object queries_obj, py::array_t<uint32_t> qids,
        uint32_t K, uint32_t L, uint32_t query_partition,
        int num_threads = 1, uint32_t batch_size = 0,
        uint32_t conv_window = 3, float conv_threshold_pct = 1.0f,
        uint32_t entry_mode = 0, bool skip_timing = false,
        bool compute_skip = false, uint32_t interleave_factor = 1,
        int qp_tid_offset = 0
    ) {
        if (!node_store_.local_store())
            throw std::runtime_error("Co-location not loaded. Call load_colocation() first.");
        switch (data_type_) {
            case DataType::INT8: {
                auto q = queries_obj.cast<py::array_t<int8_t>>();
                return search_colocation_pure_batch_typed<int8_t>(
                    q, qids, K, L, query_partition, num_threads, batch_size,
                    conv_window, conv_threshold_pct, entry_mode, skip_timing,
                    compute_skip, interleave_factor, qp_tid_offset);
            }
            case DataType::UINT8: {
                auto q = queries_obj.cast<py::array_t<uint8_t>>();
                return search_colocation_pure_batch_typed<uint8_t>(
                    q, qids, K, L, query_partition, num_threads, batch_size,
                    conv_window, conv_threshold_pct, entry_mode, skip_timing,
                    compute_skip, interleave_factor, qp_tid_offset);
            }
            default: {
                auto q = queries_obj.cast<py::array_t<float>>();
                return search_colocation_pure_batch_typed<float>(
                    q, qids, K, L, query_partition, num_threads, batch_size,
                    conv_window, conv_threshold_pct, entry_mode, skip_timing,
                    compute_skip, interleave_factor, qp_tid_offset);
            }
        }
    }

    // =========================================================================
    // Phase 4: Consolidation — consolidate_local
    // =========================================================================

    py::dict consolidate_local(
        py::array_t<uint32_t> active_local_locs,
        py::array_t<uint32_t> active_tags,
        py::array_t<uint64_t> deleted_vids_arr,
        py::array_t<uint32_t> delete_tags_arr,
        uint32_t R, float alpha, uint32_t maxc,
        const std::string& trace_path = "",
        bool enable_vec_cache = false,
        int num_threads = 1,
        bool enable_lits_filter = false,
        int pass2_num_threads = 0,
        const std::string& vec_fetch_count_dump_path = "",
        size_t cache_neighbor_records_per_thread = 0,
        size_t cache_global_vector_bytes = 0,
        bool lits_tau_include_deleted = true
    ) {
        if (!node_store_.local_store())
            throw std::runtime_error("consolidate_local: ColocationStore not initialized");

        // Build deleted VID set from numpy array
        auto del_buf = deleted_vids_arr.request();
        const uint64_t* del_ptr = static_cast<const uint64_t*>(del_buf.ptr);
        std::unordered_set<uint64_t> deleted_set(del_ptr, del_ptr + del_buf.shape[0]);

        // Build config
        danns::ConsolidationConfig config;
        config.R = R;
        config.alpha = alpha;
        config.maxc = maxc;
        config.num_threads = static_cast<uint32_t>(num_threads);
        config.trace_path = trace_path;
        config.lits_filter = enable_lits_filter;
        config.lits_my_partition = node_store_.partition_id();
        config.lits_tau_include_deleted = lits_tau_include_deleted;
        config.enable_vec_cache = enable_vec_cache;
        config.consolidation_cache = &consolidation_cache_;
        config.pass2_num_threads = static_cast<uint32_t>(std::max(0, pass2_num_threads));
        config.vec_fetch_count_dump_path = vec_fetch_count_dump_path;

        // Get distance function and dispatch by type. GIL is released for the
        // long-running consolidation body so a concurrent Python thread (e.g.
        // ConcurrentOp's search loop) can enter its own binding meanwhile.
        danns::ConsolidationStats cstats;
        {
            py::gil_scoped_release release;
            switch (data_type_) {
                case DataType::INT8: {
                    if (!dist_i8_) throw std::runtime_error("No int8 distance function");
                    cstats = consolidation_.run<int8_t>(
                        config, &node_store_, &mapper_, dist_i8_.get(),
                        deleted_set, vector_dim_, aligned_dim_);
                    break;
                }
                case DataType::UINT8: {
                    if (!dist_u8_) throw std::runtime_error("No uint8 distance function");
                    cstats = consolidation_.run<uint8_t>(
                        config, &node_store_, &mapper_, dist_u8_.get(),
                        deleted_set, vector_dim_, aligned_dim_);
                    break;
                }
                default: {
                    if (!dist_f32_) throw std::runtime_error("No float32 distance function");
                    cstats = consolidation_.run<float>(
                        config, &node_store_, &mapper_, dist_f32_.get(),
                        deleted_set, vector_dim_, aligned_dim_);
                    break;
                }
            }
        }

        // Convert stats to py::dict
        py::dict result;
        result["entries_processed"] = cstats.entries_processed;
        result["entries_modified"] = cstats.entries_modified;
        result["entries_pruned"] = cstats.entries_pruned;
        result["total_deleted_nbrs"] = cstats.total_deleted_nbrs;
        result["total_expanded"] = cstats.total_expanded;
        result["local_vec_accesses"] = cstats.local_vec_accesses;
        result["remote_vec_accesses"] = cstats.remote_vec_accesses;
        result["local_nbr_accesses"] = cstats.local_nbr_accesses;
        result["remote_nbr_accesses"] = cstats.remote_nbr_accesses;
        result["cache_hits"] = cstats.cache_hits;
        result["cache_misses"] = cstats.cache_misses;
        result["cache_neighbor_hits"] = cstats.cache_neighbor_hits;
        result["cache_neighbor_misses"] = cstats.cache_neighbor_misses;
        result["cache_vector_hits"] = cstats.cache_vector_hits;
        result["cache_vector_misses"] = cstats.cache_vector_misses;
        result["cache_writebacks"] = cstats.cache_writebacks;
        result["cache_nbr_writebacks"] = cstats.cache_nbr_writebacks;
        result["cache_entries"] = cstats.cache_entries;
        result["cache_warm_expansion_events"] = cstats.cache_warm_expansion_events;
        result["cache_warm_neighbor_records"] = cstats.cache_warm_neighbor_records;
        result["cache_warm_vector_records"] = cstats.cache_warm_vector_records;
        result["cache_warm_bytes"] = cstats.cache_warm_bytes;
        result["cache_buffer_neighbor_bytes"] = cstats.cache_buffer_neighbor_bytes;
        result["cache_buffer_vector_bytes"] = cstats.cache_buffer_vector_bytes;
        result["cache_compacted_neighbor_bytes"] = cstats.cache_compacted_neighbor_bytes;
        result["cache_compacted_vector_bytes"] = cstats.cache_compacted_vector_bytes;
        result["cache_compacted_index_bytes"] = cstats.cache_compacted_index_bytes;
        result["cache_dropped_neighbor_records"] = cstats.cache_dropped_neighbor_records;
        result["cache_dropped_vector_records"] = cstats.cache_dropped_vector_records;
        result["unique_vec_fetches"] = cstats.unique_vec_fetches;
        result["vec_bytes_fetched"] = cstats.vec_bytes_fetched;
        result["vec_bytes_saved_by_cache"] = cstats.vec_bytes_saved_by_cache;
        result["vec_buffer_peak_bytes"] = cstats.vec_buffer_peak_bytes;
        result["twohop_lock_wait_ns"] = cstats.twohop_lock_wait_ns;
        result["vecfetch_lock_wait_ns"] = cstats.vecfetch_lock_wait_ns;
        result["twohop_lock_contentions"] = cstats.twohop_lock_contentions;
        result["vecfetch_lock_contentions"] = cstats.vecfetch_lock_contentions;
        result["use_multi_qp"] = cstats.use_multi_qp;
        result["elapsed_sec"] = cstats.elapsed_sec;
        result["loop_only_sec"] = cstats.loop_only_sec;
        // Phase offsets (seconds, relative to consolidate start) for NIC BW
        // timeline slicing. ON path populates all five; OFF collapses scan
        // and populate to t_loop_start so the loop window covers body wall.
        result["scan_end_offset_sec"] = cstats.scan_end_offset_sec;
        result["pop_start_offset_sec"] = cstats.pop_start_offset_sec;
        result["pop_end_offset_sec"] = cstats.pop_end_offset_sec;
        result["pass2_loop_start_offset_sec"] = cstats.pass2_loop_start_offset_sec;
        result["pass2_loop_end_offset_sec"] = cstats.pass2_loop_end_offset_sec;

        if (!cstats.degree_histogram.empty()) {
            py::array_t<uint32_t> hist(cstats.degree_histogram.size());
            std::memcpy(hist.mutable_data(), cstats.degree_histogram.data(),
                        cstats.degree_histogram.size() * sizeof(uint32_t));
            result["degree_histogram"] = hist;
        }

        return result;
    }

    // =========================================================================
    // Phase 4b: Insertion — insert_batch
    // =========================================================================

    py::dict insert_batch(
        py::object vectors_obj,               // (N, vector_dim), dtype MUST match data_type_
        py::array_t<uint32_t> tags_arr,       // (N,) 1-based tags
        py::array_t<uint32_t> local_locs_arr, // (N,) pre-allocated slots
        uint32_t my_partition,
        uint32_t R, uint32_t L, float alpha, uint32_t maxc,
        uint32_t slack_R,
        py::list special_vids_list,           // frozen/centroid VIDs to exclude
        int num_threads = 1,
        uint32_t entry_mode = 1,              // FROZEN_ONLY by default
        const std::string& trace_path = "",
        bool lits_filter = false,             // LITS (long∧inter) filter on forward pool
        uint32_t lits_my_partition = 0,       // Partition for the inter judgment
        bool load_balance = false,            // LoadBalancer (Stage B, measure-only)
        double lb_threshold = 0.7,            // candidate cut: v_i >= θ * v_max
        bool redirect = false,                // live redirect (in-batch move)
        uint32_t redirect_timeout_ms = 500    // alloc request timeout
    ) {
        if (!node_store_.local_store())
            throw std::runtime_error("insert_batch: ColocationStore not initialized");

        // Parse tag/loc arrays (uint32, dtype-safe). The vectors array is cast
        // per data_type_ in the dispatch below — taking it as a typed
        // py::array_t<float> here would forcecast int8/uint8 input to float32
        // and then reinterpret it as int8 = garbage (the C4 bug).
        auto tags_buf = tags_arr.request();
        auto locs_buf = local_locs_arr.request();
        uint32_t count = static_cast<uint32_t>(tags_buf.shape[0]);

        if (count == 0) {
            py::dict result;
            result["inserted"] = 0;
            result["skipped"] = 0;
            result["failed"] = 0;
            return result;
        }

        // Build special_vids set
        std::unordered_set<uint64_t> special_vids;
        for (auto item : special_vids_list) {
            special_vids.insert(item.cast<uint64_t>());
        }

        // Build config
        danns::InsertionConfig config;
        config.R = R;
        config.L = L;
        config.alpha = alpha;
        config.maxc = maxc;
        config.slack_R = slack_R;
        config.entry_mode = entry_mode;
        config.num_threads = static_cast<uint32_t>(num_threads);
        config.trace_path = trace_path;
        config.lits_filter = lits_filter;
        config.lits_my_partition = lits_my_partition;
        config.load_balance = load_balance;
        config.lb_threshold = lb_threshold;
        config.redirect = redirect;
        config.redirect_timeout_ms = redirect_timeout_ms;

        // Type dispatch: cast the vectors array to the engine's native dtype
        // (mirrors search_colocation_pure). The cast happens under the GIL; the
        // typed array `v` stays alive for the case scope so vbuf.ptr is valid
        // while the GIL is released for the heavy insert. Shape-guard (count,
        // vector_dim) since pybind value-converts (forcecast) on dtype mismatch.
        danns::InsertionStats istats;

        {
            switch (data_type_) {
                case DataType::INT8: {
                    if (!dist_i8_) throw std::runtime_error("No int8 distance function");
                    auto v = vectors_obj.cast<py::array_t<int8_t>>();
                    auto vbuf = v.request();
                    if (vbuf.ndim != 2 ||
                        static_cast<uint32_t>(vbuf.shape[0]) != count ||
                        vbuf.shape[1] != static_cast<ssize_t>(vector_dim_))
                        throw std::runtime_error("insert_batch: vectors must be (count, vector_dim) int8");
                    py::gil_scoped_release release;
                    istats = insertion_.insert_batch<int8_t>(
                        config, &node_store_, &mapper_, &beam_search_,
                        &pq_filter_, dist_i8_.get(),
                        vector_dim_, aligned_dim_,
                        static_cast<const int8_t*>(vbuf.ptr),
                        static_cast<const uint32_t*>(tags_buf.ptr),
                        static_cast<const uint32_t*>(locs_buf.ptr),
                        count, my_partition, special_vids,
                        redirect ? &redirect_alloc_ : nullptr);
                    break;
                }
                case DataType::UINT8: {
                    if (!dist_u8_) throw std::runtime_error("No uint8 distance function");
                    auto v = vectors_obj.cast<py::array_t<uint8_t>>();
                    auto vbuf = v.request();
                    if (vbuf.ndim != 2 ||
                        static_cast<uint32_t>(vbuf.shape[0]) != count ||
                        vbuf.shape[1] != static_cast<ssize_t>(vector_dim_))
                        throw std::runtime_error("insert_batch: vectors must be (count, vector_dim) uint8");
                    py::gil_scoped_release release;
                    istats = insertion_.insert_batch<uint8_t>(
                        config, &node_store_, &mapper_, &beam_search_,
                        &pq_filter_, dist_u8_.get(),
                        vector_dim_, aligned_dim_,
                        static_cast<const uint8_t*>(vbuf.ptr),
                        static_cast<const uint32_t*>(tags_buf.ptr),
                        static_cast<const uint32_t*>(locs_buf.ptr),
                        count, my_partition, special_vids,
                        redirect ? &redirect_alloc_ : nullptr);
                    break;
                }
                default: {
                    if (!dist_f32_) throw std::runtime_error("No float32 distance function");
                    auto v = vectors_obj.cast<py::array_t<float>>();
                    auto vbuf = v.request();
                    if (vbuf.ndim != 2 ||
                        static_cast<uint32_t>(vbuf.shape[0]) != count ||
                        vbuf.shape[1] != static_cast<ssize_t>(vector_dim_))
                        throw std::runtime_error("insert_batch: vectors must be (count, vector_dim) float32");
                    py::gil_scoped_release release;
                    istats = insertion_.insert_batch<float>(
                        config, &node_store_, &mapper_, &beam_search_,
                        &pq_filter_, dist_f32_.get(),
                        vector_dim_, aligned_dim_,
                        static_cast<const float*>(vbuf.ptr),
                        static_cast<const uint32_t*>(tags_buf.ptr),
                        static_cast<const uint32_t*>(locs_buf.ptr),
                        count, my_partition, special_vids,
                        redirect ? &redirect_alloc_ : nullptr);
                    break;
                }
            }
        }

        // Convert stats to py::dict
        py::dict result;
        result["inserted"] = istats.inserted;
        result["skipped"] = istats.skipped;
        result["failed"] = istats.failed;
        result["elapsed_sec"] = istats.elapsed_sec;
        result["search_sec"] = istats.search_sec;
        result["prune_sec"] = istats.prune_sec;
        result["write_sec"] = istats.write_sec;
        result["reverse_sec"] = istats.reverse_sec;
        result["local_vec_reads"] = istats.local_vec_reads;
        result["remote_vec_reads"] = istats.remote_vec_reads;
        result["local_nbr_reads"] = istats.local_nbr_reads;
        result["remote_nbr_reads"] = istats.remote_nbr_reads;
        result["local_nbr_writes"] = istats.local_nbr_writes;
        result["remote_nbr_writes"] = istats.remote_nbr_writes;
        result["lits_checked"] = istats.lits_checked;
        result["lits_filtered"] = istats.lits_filtered;
        result["redirect_considered"] = istats.redirect_considered;
        result["redirect_would_happen"] = istats.redirect_would_happen;
        result["redirect_fallback"] = istats.redirect_fallback;
        if (!istats.redirected.empty()) {
            py::list redirected_out;
            for (const auto& r : istats.redirected)
                redirected_out.append(py::make_tuple(r.tag, r.old_loc,
                                                     r.fpart, r.floc));
            result["redirected_out"] = redirected_out;
        }
        if (!istats.lb_counter_snapshot.empty()) {
            py::dict counters;
            for (const auto& [nid, cnt] : istats.lb_counter_snapshot)
                counters[py::int_(nid)] = cnt;
            result["lb_counters"] = counters;
        }

        // Per-vector timing (for detailed analysis)
        if (!istats.per_vector.empty()) {
            py::list per_vec;
            for (const auto& pv : istats.per_vector) {
                py::dict d;
                d["tag"] = pv.tag;
                d["local_loc"] = pv.local_loc;
                d["forward_degree"] = pv.forward_degree;
                d["reverse_appends"] = pv.reverse_appends;
                d["reverse_prunes"] = pv.reverse_prunes;
                d["search_us"] = pv.search_ns / 1000.0;
                d["prune_us"] = pv.prune_ns / 1000.0;
                d["write_us"] = pv.write_ns / 1000.0;
                d["reverse_edge_us"] = pv.reverse_edge_ns / 1000.0;
                per_vec.append(d);
            }
            result["per_vector"] = per_vec;
        }

        return result;
    }

    // =========================================================================
    // Phase 5: PQ filter
    // =========================================================================

    void load_pq_data(const std::string& codebook_path,
                      const std::string& all_codes_path,
                      float alpha,
                      size_t max_capacity = 0) {
        uint32_t dim = vector_dim_;
        pq_filter_.load(codebook_path, all_codes_path, 0 /* M read from file */,
                        dim, max_capacity);
        pq_filter_.set_alpha(alpha);
    }

    void encode_and_update_pq(py::array_t<uint32_t> global_ids,
                              py::array_t<float> vectors) {
        if (!pq_filter_.enabled()) return;
        auto gids = global_ids.request();
        auto vecs = vectors.request();
        pq_filter_.encode_and_update(
            static_cast<const uint32_t*>(gids.ptr),
            static_cast<const float*>(vecs.ptr),
            static_cast<uint32_t>(gids.shape[0]),
            vector_dim_);
    }

    void update_pq_codes(py::array_t<uint32_t> global_ids,
                         py::array_t<uint8_t> codes) {
        if (!pq_filter_.enabled()) return;
        auto gids_buf = global_ids.unchecked<1>();
        auto codes_buf = codes.unchecked<2>();
        size_t n = gids_buf.shape(0);
        for (size_t i = 0; i < n; i++) {
            uint8_t buf[64];
            uint32_t M = static_cast<uint32_t>(codes_buf.shape(1));
            for (uint32_t m = 0; m < M && m < 64; m++) buf[m] = codes_buf(i, m);
            pq_filter_.set_code(gids_buf(i), buf);
        }
    }

    void set_pq_alpha(float alpha) { pq_filter_.set_alpha(alpha); }


    // =========================================================================
    // Multi-QP: per-thread CQ/Connection/MR allocation + TCP exchange
    // Ported from the legacy baseline tree distributed_bindings.cpp:6148-6290
    // =========================================================================

    void setup_multi_qp(int num_threads_in, uint16_t base_port, bool same_peer_mode = false) {
        teardown_multi_qp();

        uint32_t num_threads = static_cast<uint32_t>(num_threads_in);
        rdma::Logger::info("Setting up %u per-thread QPs (same_peer_mode=%s)...",
                          num_threads, same_peer_mode ? "true" : "false");

        const auto& endpoints = cluster_.endpoints();
        if (endpoints.empty()) {
            throw std::runtime_error("No peer connections. Call connect_to_peer first.");
        }

        // Determine target peers. Iterate in globally-consistent SORTED order so
        // every node walks node pairs identically — unordered_map iteration is
        // per-node-arbitrary and was the root cause of the setup race (a server
        // opened a pair the client reached last, so the accept window expired).
        std::vector<uint32_t> target_peers;
        for (const auto& [peer_id, _] : endpoints) {
            target_peers.push_back(peer_id);
        }
        std::sort(target_peers.begin(), target_peers.end());
        if (same_peer_mode) {
            // Deterministic single peer = smallest id (was endpoints.begin()->first,
            // order-dependent: two nodes could pick different peers and never meet).
            target_peers.resize(1);
            rdma::Logger::info("same_peer_mode: creating %u QPs to peer %u only",
                              num_threads, target_peers[0]);
        }

        for (uint32_t peer_id : target_peers) {
            setup_qp_for_peer(num_threads, base_port, peer_id);
        }

        node_store_.set_num_thread_qps(num_threads);
        rdma::Logger::info("Multi-QP setup complete: %u threads", num_threads);
    }

    // -------------------------------------------------------------------------
    // Per-pair multi-QP setup, driven from Python so a cluster-wide sync.barrier
    // can fence each pair: the server is listening before the client connects by
    // CONSTRUCTION, not by timing margin. Mirrors node_worker.py connect_to_peer.
    //   begin()    — reset prior state (idempotent).
    //   pair()     — handshake ONE peer; does NOT publish num_thread_qps.
    //   finalize() — publish num_thread_qps only after the LAST pair, so a
    //                half-built setup reads num_thread_qps()==0 and search /
    //                consolidation fall back cleanly instead of using dead QPs.
    // -------------------------------------------------------------------------
    void setup_multi_qp_begin() {
        teardown_multi_qp();
    }

    void setup_multi_qp_pair(int num_threads_in, uint16_t base_port, uint32_t peer_id) {
        setup_qp_for_peer(static_cast<uint32_t>(num_threads_in), base_port, peer_id);
    }

    void setup_multi_qp_finalize(int num_threads_in) {
        node_store_.set_num_thread_qps(static_cast<uint32_t>(num_threads_in));
        rdma::Logger::info("Multi-QP setup complete: %u threads (per-pair)",
                          static_cast<uint32_t>(num_threads_in));
    }

    // Create CQ/QP/MR for all threads of ONE peer, then the blocking TCP
    // rendezvous + connect. Body extracted verbatim from the old setup_multi_qp
    // loop so single-call and per-pair paths share identical logic.
    void setup_qp_for_peer(uint32_t num_threads, uint16_t base_port, uint32_t peer_id) {
        const auto& endpoints = cluster_.endpoints();
        auto ep_it = endpoints.find(peer_id);
        if (ep_it == endpoints.end()) {
            throw std::runtime_error("setup_qp_for_peer: no endpoint for peer " +
                                     std::to_string(peer_id));
        }
        auto ctx = ep_it->second->get_context();

        auto& thread_states = node_store_.thread_states_mut();
        const auto& coloc_cfg = node_store_.config();

        std::vector<danns::NodeStore::ThreadRDMAState>& states = thread_states[peer_id];
        states.resize(num_threads);

            std::vector<rdma::ConnectionInfo> local_infos;
            local_infos.reserve(num_threads);

            for (uint32_t t = 0; t < num_threads; t++) {
                auto [send_cq, recv_cq] = ctx->create_cq();
                states[t].send_cq = send_cq;
                states[t].recv_cq = recv_cq;

                states[t].conn = std::make_shared<rdma::RDMAConnection>(ctx, send_cq, recv_cq);
                local_infos.push_back(states[t].conn->get_local_info());

                // Per-thread fetch buffer (enough for a batch of vectors).
                // Consolidation/insertion address vectors with entry_size stride
                // (sgl[i].local_addr = buf + i*entry_size + vector_offset) even
                // under SEPARATED fetch_mode, so the buffer must be at least
                // entry_size * batch_max bytes regardless of colocation_mode.
                constexpr size_t FETCH_BUF_ENTRIES = 1024;
                size_t buf_size = static_cast<size_t>(aligned_dim_) * FETCH_BUF_ENTRIES;
                if (coloc_cfg.entry_size > 0) {
                    size_t coloc_buf_size = static_cast<size_t>(coloc_cfg.entry_size) * FETCH_BUF_ENTRIES;
                    size_t coloc_floats = coloc_buf_size / sizeof(float) + 1;
                    buf_size = std::max(buf_size, coloc_floats);
                }
                states[t].fetch_buffer.resize(buf_size);
                states[t].fetch_buffer_mr = ctx->register_memory(
                    states[t].fetch_buffer.data(),
                    buf_size * sizeof(float),
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);

                // Per-thread neighbor fetch buffer. 2-hop neighbor reads in
                // consolidation issue post_read of (degree_u32 + slack_R*vid)
                // bytes; the 256B default was sized for an old non-coloc path.
                // Without slack_R-based sizing, the buffer is too small for
                // SEPARATED mode and post_read trips LOCAL_PROTECTION_ERROR.
                uint32_t nbr_buf_size = 256;
                if (coloc_cfg.slack_R > 0) {
                    uint32_t coloc_nbr_size = static_cast<uint32_t>(
                        sizeof(uint32_t) + coloc_cfg.slack_R * sizeof(uint64_t));
                    nbr_buf_size = std::max(nbr_buf_size, coloc_nbr_size);
                }
                states[t].neighbor_fetch_buffer.resize(nbr_buf_size);
                states[t].neighbor_fetch_mr = ctx->register_memory(
                    states[t].neighbor_fetch_buffer.data(), nbr_buf_size,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
            }

            // TCP exchange — unique port per peer pair
            uint16_t exchange_port = base_port + 100 +
                static_cast<uint16_t>(std::min(node_id_, peer_id)) * 10 +
                static_cast<uint16_t>(std::max(node_id_, peer_id));

            auto host_it = peer_hosts_.find(peer_id);
            if (host_it == peer_hosts_.end()) {
                throw std::runtime_error("No host info for peer " + std::to_string(peer_id));
            }
            const std::string& peer_host = host_it->second;

            std::vector<rdma::ConnectionInfo> remote_infos;
            if (node_id_ < peer_id) {
                remote_infos = rdma::ConnectionExchanger::exchange_multi_as_server(
                    exchange_port, local_infos, 60);
            } else {
                // max_retries 60 → 200 (30s → 100s ceiling): q50M setup_multi_qp
                // race observed when 128-QP server start exceeds 30s under load.
                remote_infos = rdma::ConnectionExchanger::exchange_multi_as_client(
                    peer_host, exchange_port, local_infos, 30, 200, 500);
            }

            for (uint32_t t = 0; t < num_threads; t++) {
                states[t].conn->connect(remote_infos[t]);
            }

            rdma::Logger::info("Multi-QP: %u QPs connected to peer %u (port=%u)",
                              num_threads, peer_id, exchange_port);
    }

    void teardown_multi_qp() {
        auto& thread_states = node_store_.thread_states_mut();
        if (thread_states.empty()) return;

        const auto& endpoints = cluster_.endpoints();

        for (auto& [peer_id, states] : thread_states) {
            auto ep_it = endpoints.find(peer_id);
            // ctx is null if the endpoint was already destroyed (out-of-order
            // teardown). MR dereg / CQ destroy need the context, but conn reset
            // and buffer release must still happen — don't skip the whole peer.
            std::shared_ptr<rdma::RDMAContext> ctx =
                (ep_it != endpoints.end()) ? ep_it->second->get_context() : nullptr;

            for (auto& state : states) {
                if (ctx && state.fetch_buffer_mr) {
                    ctx->deregister_memory(state.fetch_buffer_mr);
                }
                state.fetch_buffer_mr = nullptr;
                if (ctx && state.neighbor_fetch_mr) {
                    ctx->deregister_memory(state.neighbor_fetch_mr);
                }
                state.neighbor_fetch_mr = nullptr;

                state.conn.reset();

                if (ctx && state.send_cq) ctx->destroy_cq(state.send_cq);
                if (ctx && state.recv_cq) ctx->destroy_cq(state.recv_cq);
                state.send_cq = state.recv_cq = nullptr;
            }
        }

        thread_states.clear();
        node_store_.set_num_thread_qps(0);
        rdma::Logger::info("Multi-QP teardown complete");
    }

    // === Trace ===
    void start_search_trace(const std::string& path, uint16_t level) {
        trace_writer_.open(path, level, node_id_, vector_dim_);
    }
    void stop_search_trace() { trace_writer_.close(); }

    // === Search VFCT (per-vector fetch counter, lightweight in-memory) ======
    // Allocates [partition][local_loc] atomic counter sized from local_to_tag().
    // Increment happens at end of each batch search call by walking
    // sr.trace_fetched (forces trace_level >= 2 internally).
    // Snapshot via dump_search_vec_fetch_count(path) — VFCT binary format.
    void start_search_vec_fetch_count(int mode = 0) {
        // Idempotent: if already enabled with a sized counter array, leave it
        // alone so cumulative counts survive across multiple search ops.
        // mode is captured on first activation; subsequent calls do not change
        // it (mixing counts under different modes would corrupt the snapshot).
        if (search_vfct_enabled_ && !search_vec_fetch_count_.empty()) return;
        search_vfct_mode_ = mode;
        const auto& l2t = node_store_.local_to_tag();
        search_vec_fetch_count_.clear();
        search_vec_fetch_count_.resize(l2t.size());
        for (size_t p = 0; p < l2t.size(); p++) {
            search_vec_fetch_count_[p] =
                std::vector<std::atomic<uint32_t>>(l2t[p].size());
        }
        search_vfct_enabled_ = true;
    }
    void clear_search_vec_fetch_count() {
        for (auto& part : search_vec_fetch_count_) {
            for (auto& c : part) c.store(0, std::memory_order_relaxed);
        }
    }
    void stop_search_vec_fetch_count() {
        search_vfct_enabled_ = false;
        search_vfct_mode_ = 0;
        search_vec_fetch_count_.clear();
    }
    py::dict dump_search_vec_fetch_count(const std::string& path) {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) {
            throw std::runtime_error("[SEARCH VFCT] failed to open dump path: " + path);
        }
        const char magic[4] = {'V', 'F', 'C', 'T'};
        uint32_t version = 1;
        uint32_t num_parts = static_cast<uint32_t>(search_vec_fetch_count_.size());
        f.write(magic, 4);
        f.write(reinterpret_cast<const char*>(&version), sizeof(version));
        f.write(reinterpret_cast<const char*>(&num_parts), sizeof(num_parts));
        uint64_t total_fetches = 0;
        uint64_t unique_fetched = 0;
        uint32_t max_fetches = 0;
        for (uint32_t p = 0; p < num_parts; p++) {
            uint32_t n_locs = static_cast<uint32_t>(search_vec_fetch_count_[p].size());
            f.write(reinterpret_cast<const char*>(&p), sizeof(p));
            f.write(reinterpret_cast<const char*>(&n_locs), sizeof(n_locs));
            std::vector<uint32_t> snapshot(n_locs);
            for (uint32_t i = 0; i < n_locs; i++) {
                uint32_t c = search_vec_fetch_count_[p][i].load(
                    std::memory_order_relaxed);
                snapshot[i] = c;
                total_fetches += c;
                if (c > 0) unique_fetched++;
                if (c > max_fetches) max_fetches = c;
            }
            f.write(reinterpret_cast<const char*>(snapshot.data()),
                    n_locs * sizeof(uint32_t));
        }
        f.close();
        py::dict info;
        info["path"] = path;
        info["num_partitions"] = num_parts;
        info["total_fetches"] = total_fetches;
        info["unique_fetched"] = unique_fetched;
        info["max_fetches"] = max_fetches;
        return info;
    }

    // === Update (update/) ===
    void mark_vid_deleted(uint64_t vid) { mapper_.mark_vid_deleted(vid); }
    void unmark_vid_deleted(uint64_t vid) { mapper_.unmark_vid_deleted(vid); }
    void clear_deleted_vids() { mapper_.clear_deleted_vids(); consolidation_cache_.clear(); }
    bool is_vid_deleted(uint64_t vid) const { return mapper_.is_vid_deleted(vid); }
    uint64_t deleted_vids_count() const { return mapper_.deleted_vids_count(); }

    void mark_tag_deleted(uint32_t tag) { mapper_.mark_tag_deleted(tag); }
    void unmark_tag_deleted(uint32_t tag) { mapper_.unmark_tag_deleted(tag); }
    void clear_deleted_tags() { mapper_.clear_deleted_tags(); }
    size_t deleted_tags_count() const { return mapper_.deleted_tags_count(); }

    void clear_consolidation_cache() { consolidation_cache_.clear(); }

    void configure_consolidation_cache(size_t neighbor_records_per_thread = 0,
                                       size_t global_vector_bytes = 0,
                                       size_t max_local_locs = 0,
                                       size_t num_partitions = 1,
                                       size_t vector_bytes = 0) {
        consolidation_cache_.configure_limits(neighbor_records_per_thread,
                                             global_vector_bytes,
                                             max_local_locs,
                                             num_partitions,
                                             vector_bytes);
        const int numa_mode = consolidation_cache_.arena_numa_mode();
        if (numa_mode >= 0) {
            rdma::Logger::info("[CONSOL CACHE] arenas bound to NUMA node %d (mbind)",
                               numa_mode);
        } else if (numa_mode == -2) {
            rdma::Logger::info("[CONSOL CACHE] arena mbind REFUSED — "
                               "placement falls back to first-touch");
        }
    }

    py::dict get_consolidation_cache_stats() const {
        auto s = consolidation_cache_.stats();
        py::dict d;
        d["expansion_events"] = s.expansion_events;
        d["neighbor_records"] = s.neighbor_records;
        d["vector_records"] = s.vector_records;
        d["entries"] = s.entries;
        d["bytes"] = s.bytes;
        d["buffer_neighbor_bytes"] = s.buffer_neighbor_bytes;
        d["buffer_vector_bytes"] = s.buffer_vector_bytes;
        d["compacted_neighbor_bytes"] = s.compacted_neighbor_bytes;
        d["compacted_vector_bytes"] = s.compacted_vector_bytes;
        d["compacted_index_bytes"] = s.compacted_index_bytes;
        d["dropped_neighbor_records"] = s.dropped_neighbor_records;
        d["dropped_vector_records"] = s.dropped_vector_records;
        return d;
    }

    // === Stats ===
    void enable_detailed_latency(bool enable) {
        latency_stats_.detailed_sampling = enable;
    }

    py::dict get_latency_stats() {
        auto s = latency_stats_.get_stats();
        py::dict d;
        d["local_count"] = s.local_count;
        d["remote_count"] = s.remote_count;
        d["total_count"] = s.local_count + s.remote_count;
        d["avg_local_us"] = s.avg_local_us;
        d["avg_remote_us"] = s.avg_remote_us;
        d["total_local_ms"] = s.total_local_ms;
        d["total_remote_ms"] = s.total_remote_ms;
        d["bytes_transferred"] = s.bytes_transferred;
        d["bandwidth_mbps"] = s.bandwidth_mbps;
        if (s.local_samples > 0) {
            d["local_p50_us"] = s.local_p50_us;
            d["local_p90_us"] = s.local_p90_us;
            d["local_p99_us"] = s.local_p99_us;
        }
        if (s.remote_samples > 0) {
            d["remote_p50_us"] = s.remote_p50_us;
            d["remote_p90_us"] = s.remote_p90_us;
            d["remote_p99_us"] = s.remote_p99_us;
        }
        if (s.remote_overhead_ratio > 0)
            d["remote_overhead_ratio"] = s.remote_overhead_ratio;
        return d;
    }

    // === Properties ===
    uint32_t node_id() const { return node_id_; }
    uint32_t total_nodes() const { return total_nodes_; }
    uint32_t vector_dim() const { return vector_dim_; }

private:
    uint32_t node_id_;
    uint32_t total_nodes_;
    uint32_t vector_dim_;
    uint32_t aligned_dim_;

    // Data type for multi-dtype dispatch
    DataType data_type_ = DataType::FLOAT32;

    // Standalone distance functions (no DiskANN index needed)
    std::unique_ptr<diskann::DistanceL2Float> dist_f32_;
    std::unique_ptr<diskann::DistanceL2Int8> dist_i8_;
    std::unique_ptr<diskann::DistanceL2UInt8> dist_u8_;


    // Peer host addresses (for multi-QP TCP exchange)
    std::unordered_map<uint32_t, std::string> peer_hosts_;

    danns::ClusterSetup cluster_;
    danns::PartitionMapper mapper_;
    danns::NodeStore node_store_;
    danns::RedirectAlloc redirect_alloc_{&node_store_};
    std::vector<std::shared_ptr<rdma::RDMAEndpoint>> redirect_endpoints_;
    danns::ColocationBeamSearch beam_search_;
    danns::PQFilter pq_filter_;
    danns::ConsolidationCache consolidation_cache_;
    danns::Consolidation consolidation_;
    danns::Insertion insertion_;
    danns::SearchTraceWriter trace_writer_;
    danns::LatencyStats latency_stats_;

    // Search VFCT (per-vector fetch counter, [partition][local_loc])
    // mode: 0=deleted (option-4, count only deleted Di hits — default, matches
    // pre-toggle behavior), 1=neighbors (option-2, count every visited vid).
    bool search_vfct_enabled_ = false;
    int search_vfct_mode_ = 0;
    std::vector<std::vector<std::atomic<uint32_t>>> search_vec_fetch_count_;

    rdma::FetchMode fetch_mode_ = rdma::FetchMode::COLOCATION;
};

// =============================================================================
// pybind11 module definition
// =============================================================================
PYBIND11_MODULE(distributed_bindings, m) {
    m.doc() = "DANNS — Distributed Approximate Nearest Neighbor Search";

    // LoadBalancer selection rule, exposed standalone for oracle-parity
    // testing (runs the host-side test vectors against the exact C++
    // implementation used by insert_batch). Not used by the engine runtime.
    m.def("lb_compute",
          [](const std::vector<uint32_t>& pool_parts,
             uint32_t my_partition,
             double threshold,
             const std::map<uint32_t, uint32_t>& part_to_node,
             const std::map<uint32_t, uint64_t>& node_counters) {
              std::vector<std::pair<uint64_t, float>> pool;
              pool.reserve(pool_parts.size());
              for (uint32_t p : pool_parts)
                  pool.emplace_back(danns::encode_vector_id(p, 0), 0.0f);
              auto d = danns::compute_load_balance_assignment(
                  pool, my_partition, threshold,
                  [&part_to_node](uint32_t p) { return part_to_node.at(p); },
                  [&node_counters](uint32_t n) -> std::optional<uint64_t> {
                      auto it = node_counters.find(n);
                      if (it == node_counters.end()) return std::nullopt;
                      return it->second;
                  });
              return py::make_tuple(d.m_assign, d.considered, d.would_redirect);
          },
          py::arg("pool_parts"), py::arg("my_partition"), py::arg("threshold"),
          py::arg("part_to_node"), py::arg("node_counters"));

    // Consolidation LITS filter, exposed standalone for oracle-parity
    // testing of the deleted-inclusive tau rule. Mirrors
    // filter_lits_consolidation_candidates (same source compiled here).
    m.def("lits_consol_compute",
          [](std::vector<std::pair<uint64_t, float>> candidates,
             const std::vector<uint64_t>& orig_vids,
             uint32_t my_partition,
             const std::vector<double>& deleted_dists,
             bool tau_include_deleted) {
              std::unordered_set<uint64_t> orig(orig_vids.begin(),
                                                orig_vids.end());
              double extra_sum = 0.0;
              uint32_t extra_cnt = 0;
              if (tau_include_deleted) {
                  for (double d : deleted_dists) { extra_sum += d; extra_cnt++; }
              }
              auto [checked, filtered] =
                  danns::filter_lits_consolidation_candidates_for_test(
                      candidates, orig, my_partition, extra_sum, extra_cnt);
              py::list kept;
              for (const auto& [vid, dist] : candidates)
                  kept.append(py::make_tuple(vid, dist));
              return py::make_tuple(kept, checked, filtered);
          },
          py::arg("candidates"), py::arg("orig_vids"), py::arg("my_partition"),
          py::arg("deleted_dists"), py::arg("tau_include_deleted") = true);

    py::class_<DANNSEngine>(m, "MultiPeerGraphStore")
        // Constructor (same signature as legacy)
        .def(py::init<uint32_t, uint32_t, uint32_t, std::string, int>(),
             py::arg("node_id"), py::arg("total_nodes"), py::arg("vector_dim"),
             py::arg("rdma_device") = "", py::arg("gid_index") = 1)

        // RDMA setup
        .def("init_rdma", &DANNSEngine::init_rdma)
        .def("connect_to_peer", &DANNSEngine::connect_to_peer)
        .def("cleanup", &DANNSEngine::cleanup)

        // Partition mapping
        .def("set_total_vectors", &DANNSEngine::set_total_vectors)
        .def("set_partition_assignments", &DANNSEngine::set_partition_assignments)
        .def("set_partition_node_mapping", &DANNSEngine::set_partition_node_mapping)
        .def("set_partition_max_local_locs", &DANNSEngine::set_partition_max_local_locs)
        .def("load_tags", &DANNSEngine::load_tags)
        .def("update_tag_to_local_entry", &DANNSEngine::update_tag_to_local_entry)
        .def("update_local_to_tag_entry", &DANNSEngine::update_local_to_tag_entry)

        // Phase 1: Distance + entry points + utilities
        .def("init_distance_only", &DANNSEngine::init_distance_only,
             py::arg("dim"), py::arg("data_type") = "float32")
        .def("set_centroid_local_locs", &DANNSEngine::set_centroid_local_locs,
             py::arg("locs"), py::arg("max_points"))
        .def("get_max_points", &DANNSEngine::get_max_points)
        .def("set_frozen_local_loc", &DANNSEngine::set_frozen_local_loc,
             py::arg("loc"))
        .def("set_frozen_canonical_vid", &DANNSEngine::set_frozen_canonical_vid,
             py::arg("vid"))
        .def("get_location_count", &DANNSEngine::get_location_count)

        // Store
        .def("load_colocation", &DANNSEngine::load_colocation,
             py::arg("coloc_file"), py::arg("global_to_vectorid_file"),
             py::arg("tag_to_local_files"), py::arg("local_to_tag_files"),
             py::arg("partition_id"), py::arg("dim"), py::arg("R"), py::arg("slack_R"),
             py::arg("entry_size"), py::arg("alignment"), py::arg("num_entries"),
             py::arg("total_vectors"), py::arg("element_size") = 4)
        .def("exchange_coloc_mr_with_peer", &DANNSEngine::exchange_coloc_mr_with_peer,
             py::arg("peer_id"), py::arg("exchange_counter") = false)
        .def("set_local_vector_count", &DANNSEngine::set_local_vector_count,
             py::arg("count"))
        .def("local_vector_count", &DANNSEngine::local_vector_count)
        .def("read_peer_vector_count", &DANNSEngine::read_peer_vector_count,
             py::arg("peer_id"))
        .def("connect_redirect_peer", &DANNSEngine::connect_redirect_peer,
             py::arg("peer_id"), py::arg("host"),
             py::arg("port_low"), py::arg("port_high"),
             py::arg("rdma_device") = "", py::arg("gid_index") = 1)
        .def("start_redirect_server", &DANNSEngine::start_redirect_server)
        .def("open_redirect_window", &DANNSEngine::open_redirect_window,
             py::arg("reserve"), py::arg("max_tag"), py::arg("num_partitions"))
        .def("close_redirect_window", &DANNSEngine::close_redirect_window)
        .def("local_get_entry_vid", &DANNSEngine::local_get_entry_vid,
             py::arg("local_loc"))
        .def("set_fetch_mode", &DANNSEngine::set_fetch_mode)
        .def("register_local_partition_data", &DANNSEngine::register_local_partition_data,
             py::arg("peer_id"), py::arg("addr"))

        // Local diagnostic access
        .def("local_get_neighbors", &DANNSEngine::local_get_neighbors)
        .def("dump_colocation", &DANNSEngine::dump_colocation,
             py::arg("path"))

        // Phase 2: Single-query search
        .def("search_colocation_pure", &DANNSEngine::search_colocation_pure,
             py::arg("query"), py::arg("K"), py::arg("L"),
             py::arg("entry_mode") = 0,
             py::arg("query_partition") = 0,
             py::arg("verbose") = false,
             py::arg("trace_query_id") = 0,
             py::arg("batch_size") = 0,
             py::arg("conv_window") = 3,
             py::arg("conv_threshold_pct") = 1.0f,
             py::arg("return_pool") = false,
             py::arg("lits_filter") = false,
             py::arg("lits_my_partition") = 0)

        // Phase 3: Batch search
        .def("search_colocation_pure_batch", &DANNSEngine::search_colocation_pure_batch,
             py::arg("queries"), py::arg("qids"),
             py::arg("K"), py::arg("L"),
             py::arg("query_partition"),
             py::arg("num_threads") = 1,
             py::arg("batch_size") = 0,
             py::arg("conv_window") = 3,
             py::arg("conv_threshold_pct") = 1.0f,
             py::arg("entry_mode") = 0,
             py::arg("skip_timing") = false,
             py::arg("compute_skip") = false,
             py::arg("interleave_factor") = 1,
             py::arg("qp_tid_offset") = 0)

        // Phase 6: Multi-QP
        .def("setup_multi_qp", &DANNSEngine::setup_multi_qp,
             py::arg("num_threads"), py::arg("base_port"),
             py::arg("same_peer_mode") = false)
        .def("setup_multi_qp_begin", &DANNSEngine::setup_multi_qp_begin)
        .def("setup_multi_qp_pair", &DANNSEngine::setup_multi_qp_pair,
             py::arg("num_threads"), py::arg("base_port"), py::arg("peer_id"))
        .def("setup_multi_qp_finalize", &DANNSEngine::setup_multi_qp_finalize,
             py::arg("num_threads"))
        .def("teardown_multi_qp", &DANNSEngine::teardown_multi_qp)

        // Trace
        .def("start_search_trace", &DANNSEngine::start_search_trace)
        .def("stop_search_trace", &DANNSEngine::stop_search_trace)

        // Search VFCT (lightweight per-vector fetch counter, dump format = VFCT)
        .def("start_search_vec_fetch_count",
             &DANNSEngine::start_search_vec_fetch_count,
             py::arg("mode") = 0)
        .def("clear_search_vec_fetch_count",
             &DANNSEngine::clear_search_vec_fetch_count)
        .def("stop_search_vec_fetch_count",
             &DANNSEngine::stop_search_vec_fetch_count)
        .def("dump_search_vec_fetch_count",
             &DANNSEngine::dump_search_vec_fetch_count, py::arg("path"))

        // Update
        .def("mark_vid_deleted", &DANNSEngine::mark_vid_deleted)
        .def("unmark_vid_deleted", &DANNSEngine::unmark_vid_deleted)
        .def("clear_deleted_vids", &DANNSEngine::clear_deleted_vids)
        .def("is_vid_deleted", &DANNSEngine::is_vid_deleted)
        .def("deleted_vids_count", &DANNSEngine::deleted_vids_count)
        .def("mark_tag_deleted", &DANNSEngine::mark_tag_deleted)
        .def("unmark_tag_deleted", &DANNSEngine::unmark_tag_deleted)
        .def("clear_deleted_tags", &DANNSEngine::clear_deleted_tags)
        .def("deleted_tags_count", &DANNSEngine::deleted_tags_count)
        .def("clear_consolidation_cache", &DANNSEngine::clear_consolidation_cache)
        .def("configure_consolidation_cache", &DANNSEngine::configure_consolidation_cache,
             py::arg("neighbor_records_per_thread") = 0,
             py::arg("global_vector_bytes") = 0,
             py::arg("max_local_locs") = 0,
             py::arg("num_partitions") = 1,
             py::arg("vector_bytes") = 0)
        .def("get_consolidation_cache_stats", &DANNSEngine::get_consolidation_cache_stats)

        // Phase 4: Consolidation
        .def("consolidate_local", &DANNSEngine::consolidate_local,
             py::arg("active_local_locs"), py::arg("active_tags"),
             py::arg("deleted_vids"), py::arg("delete_tags"),
             py::arg("R"), py::arg("alpha"), py::arg("maxc"),
             py::arg("trace_path") = "",
             py::arg("enable_vec_cache") = false,
             py::arg("num_threads") = 1,
             py::arg("enable_lits_filter") = false,
             py::arg("pass2_num_threads") = 0,
             py::arg("vec_fetch_count_dump_path") = "",
             py::arg("cache_neighbor_records_per_thread") = 0,
             py::arg("cache_global_vector_bytes") = 0,
             py::arg("lits_tau_include_deleted") = true)

        // Phase 4b: Insertion
        .def("insert_batch", &DANNSEngine::insert_batch,
             py::arg("vectors"), py::arg("tags"), py::arg("local_locs"),
             py::arg("my_partition"),
             py::arg("R"), py::arg("L"), py::arg("alpha"), py::arg("maxc"),
             py::arg("slack_R"),
             py::arg("special_vids_list"),
             py::arg("num_threads") = 1,
             py::arg("entry_mode") = 1,
             py::arg("trace_path") = "",
             py::arg("lits_filter") = false,
             py::arg("lits_my_partition") = 0,
             py::arg("load_balance") = false,
             py::arg("lb_threshold") = 0.7,
             py::arg("redirect") = false,
             py::arg("redirect_timeout_ms") = 500)

        // Phase 5: PQ filter
        .def("load_pq_data", &DANNSEngine::load_pq_data,
             py::arg("codebook_path"), py::arg("all_codes_path"), py::arg("alpha"),
             py::arg("max_capacity") = 0)
        .def("update_pq_codes", &DANNSEngine::update_pq_codes,
             py::arg("global_ids"), py::arg("codes"))
        .def("encode_and_update_pq", &DANNSEngine::encode_and_update_pq,
             py::arg("global_ids"), py::arg("vectors"))
        .def("set_pq_alpha", &DANNSEngine::set_pq_alpha, py::arg("alpha"))

        // Stats
        .def("enable_detailed_latency", &DANNSEngine::enable_detailed_latency)
        .def("get_latency_stats", &DANNSEngine::get_latency_stats)

        // Properties
        .def_property_readonly("node_id", &DANNSEngine::node_id)
        .def_property_readonly("total_nodes", &DANNSEngine::total_nodes)
        .def_property_readonly("vector_dim", &DANNSEngine::vector_dim)
    ;
}
