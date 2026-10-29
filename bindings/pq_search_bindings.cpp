// pq_search_bindings.cpp — single-node search engine (search half of `single_engine`)
// ============================================================================
// Originally the M11 PQ false-filtering measurement harness; now the search side
// of the unified `single_engine` module (build/update half = diskann_bindings.cpp).
// Also still built as the legacy `pq_search_bindings` module for existing drivers.
// Purpose: single-node search experiments without RDMA/colocation — PQ filter
// false-filtering (M11), quantized-search recall (Q1/M08: search_dist modes),
// rerank variants, and local/remote fetch accounting via partition labels.
//
// Why single-node is faithful for THIS question: the PQ-filter DECISION
// (pq_dist > d_worst*(1+alpha) -> drop) and its consequence (candidate never gets
// an exact distance -> can't enter the beam) are layout-independent. RDMA/coloc
// only change WHAT fetch is skipped (performance), not WHICH candidates are
// dropped (recall). Here every vector is local, so we get the PQ-approx AND the
// exact distance for every candidate for free -> false filtering is directly
// classifiable.
//
// Faithfulness contract (mirrors search/src/beam_search.cpp @ ColocationBeamSearch):
//   * distances are SQUARED L2 (DiskANN DistanceL2*, no sqrt); LUT stores ||q-c||^2.
//   * filter fires only when  converged && beam_full(size>=L)  (converge-only).
//   * per hop: worst = queue.back().distance (L-th/worst); threshold = worst*(1+alpha);
//     computed ONCE at hop entry (before this hop's insertions), applied to all nbrs.
//   * convergence: latched per query; conv_detector.update(queue[0].distance) each hop,
//     window/threshold_pct copied verbatim from beam_search.cpp ConvergenceDetector.
//   * PQ gid = tag - 1 (base-vector index); code lookup via danns::PQFilter.
//
// filter-all mode: this harness applies the filter to EVERY candidate (the M->inf /
// all-remote conservative upper bound). In the real M-machine system only remote-owned
// candidates (owner != my_node) are filtered; local ones are computed exact. If false
// filtering is negligible when EVERYONE is filtered, it is a fortiori negligible at M=4.
// (A partition-aware mode can be layered on later by loading partition_ids.)
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <thread>
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <fstream>
#include <algorithm>
#include <type_traits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <random>
#include <omp.h>

#include "index.h"
#include "utils.h"
#include "parameters.h"
#include "search/pq_filter.h"   // danns::PQFilter

namespace py = pybind11;

static constexpr uint32_t KSUB = 256;   // centroids per subquantizer (PQFilter enforces ==256)

template <typename T>
class PQSearchHarness {
public:
    PQSearchHarness(size_t dim, size_t max_points, uint32_t R = 64, uint32_t L_build = 100,
                    float alpha_build = 1.2f, uint32_t num_threads = 0)
        : _dim(dim), _max_points(max_points), _L_build(L_build) {
        if (num_threads == 0) num_threads = std::thread::hardware_concurrency();
        _num_threads = num_threads;

        auto index_params = std::make_shared<diskann::IndexWriteParameters>(
            diskann::IndexWriteParametersBuilder(L_build, R)
                .with_alpha(alpha_build)
                .with_saturate_graph(false)
                .with_num_threads(num_threads)
                .build());
        auto search_params = std::make_shared<diskann::IndexSearchParams>(
            diskann::IndexSearchParams(L_build, num_threads));

        // Dynamic index with 1 frozen point — matches the pipeline build stage.
        _index = std::make_unique<diskann::Index<T, uint32_t, uint32_t>>(
            diskann::Metric::L2, dim, max_points, index_params, search_params,
            /*num_frozen_pts*/ 1, /*dynamic*/ true, /*enable_tags*/ true,
            false, false, 0, false, false);
    }

    // Load the Vamana graph + vectors (from base_file, since pipeline defaults
    // save_data_file=false) + tags. Mirrors diskann_bindings::load_graph_only,
    // which also restores the frozen start vector from the <graph>.frozen sidecar.
    void load_index(const std::string& graph_file, const std::string& base_file,
                    const std::string& tags_file, uint32_t search_L = 0) {
        if (search_L == 0) search_L = _L_build;
        _index->load_graph_only(graph_file.c_str(), base_file, tags_file, _num_threads, search_L);
        _num_points = _index->get_num_points();

        // Restore frozen point vector from <graph>.frozen (data-less checkpoint).
        {
            std::string fp_path = graph_file + ".frozen";
            std::ifstream in(fp_path, std::ios::binary);
            if (in) {
                uint32_t header[2] = {0u, 0u};
                in.read(reinterpret_cast<char*>(header), sizeof(header));
                if (in && header[0] == 1u && header[1] == static_cast<uint32_t>(_dim)) {
                    std::vector<T> fp_vec(_dim);
                    in.read(reinterpret_cast<char*>(fp_vec.data()),
                            static_cast<std::streamsize>(_dim * sizeof(T)));
                    if (in) _index->get_data_store().set_vector(
                                static_cast<uint32_t>(_max_points), fp_vec.data());
                }
            }
        }

        // Load tags: file layout [u32 npts][u32 dim=1][u32 tags...], tags[loc]=tag.
        // gid = tag - 1 (base-vector index). Frozen loc's tag == 0 (sentinel).
        {
            std::ifstream tin(tags_file, std::ios::binary);
            if (!tin) throw std::runtime_error("Failed to open tags file: " + tags_file);
            uint32_t thdr[2] = {0u, 0u};
            tin.read(reinterpret_cast<char*>(thdr), sizeof(thdr));
            if (!tin) throw std::runtime_error("Short read on tags header: " + tags_file);
            _tag_of_loc.assign(thdr[0], 0u);
            tin.read(reinterpret_cast<char*>(_tag_of_loc.data()),
                     static_cast<std::streamsize>(thdr[0] * sizeof(uint32_t)));
            if (!tin) throw std::runtime_error("Short read on tags body: " + tags_file);
        }

        // Entry point: dynamic index's frozen start is at location _max_points.
        _start = static_cast<uint32_t>(_max_points);
        _index_loaded = true;
    }

    // Load ONLY the graph adjacency + tags (no base vectors, no PQ) — for
    // graph_stats. This sidesteps the base-size problem on post-update
    // checkpoints (iter150 tags are NON-sequential with gids beyond the 100M
    // base, so load_graph_only's per-loc base read would go out of range) and is
    // fast (no 100M+ vector load). Search is NOT usable after this (no vectors);
    // use load_index for recall.
    void load_topology(const std::string& graph_file, const std::string& tags_file) {
        // tags: [u32 npts][u32 dim=1][u32 tags...], tags[loc] = tag (0 = frozen).
        std::ifstream tin(tags_file, std::ios::binary);
        if (!tin) throw std::runtime_error("Failed to open tags file: " + tags_file);
        uint32_t thdr[2] = {0u, 0u};
        tin.read(reinterpret_cast<char*>(thdr), sizeof(thdr));
        if (!tin) throw std::runtime_error("Short read on tags header: " + tags_file);
        _tag_of_loc.assign(thdr[0], 0u);
        tin.read(reinterpret_cast<char*>(_tag_of_loc.data()),
                 static_cast<std::streamsize>(thdr[0]) * sizeof(uint32_t));
        if (!tin) throw std::runtime_error("Short read on tags body: " + tags_file);

        // Graph adjacency only. AbstractGraphStore::load reads exactly
        // expected_num_points nodes, so pass the tags' node count (the checkpoint
        // pairs graph and tags with the same node count). Returns (nodes_read,
        // start, frozen_pts).
        auto* gs = _index->get_graph_store();
        auto res = gs->load(graph_file, static_cast<size_t>(thdr[0]));
        _num_points = static_cast<size_t>(std::get<0>(res));
        if (_num_points == 0) _num_points = thdr[0];
        _index_loaded = true;
    }

    // Load PQ codebook + all codes into danns::PQFilter (reused verbatim).
    void load_pq(const std::string& codebook_path, const std::string& codes_path,
                 size_t max_capacity, float alpha) {
        _pq.load(codebook_path, codes_path, /*M_hint*/ 0, /*dim_hint*/ static_cast<uint32_t>(_dim),
                 max_capacity);
        _pq.set_alpha(alpha);
        _M = _pq.M();
        if (!_pq.enabled()) throw std::runtime_error("PQFilter failed to load (enabled=false)");
        if (_M == 0) throw std::runtime_error("PQFilter M == 0 after load");
    }

    void set_alpha(float alpha) { _pq.set_alpha(alpha); }
    float alpha() const { return _pq.alpha(); }
    uint32_t M() const { return _M; }
    size_t num_points() const { return _num_points; }

    // ---- Core: search a batch of queries -------------------------------------
    // Returns a dict:
    //   result_tags : (Nq, K) uint32  — top-K tags (UINT32_MAX = padding)
    //   result_dists: (Nq, K) float   — corresponding squared-L2 distances
    //   aggregate stats: n_filter_eligible, n_pq_filtered, n_false_local,
    //                    n_pq_filtered_gt-less (see below), sum_hops
    //   drop sample  : drop_qidx, drop_gid, drop_pqdist, drop_exact, drop_worst
    //                  (up to max_drops rows; for the PQ-error-distribution plot)
    //
    // filter_on=false  -> oracle run (no candidate ever dropped) for recall(OFF).
    // filter_on=true   -> real filter; every dropped candidate is classified with
    //                     its exact distance (benign vs false-local: exact <= worst).
    // partition_of (gid -> partition, len >= max gid) and coordinator (per-query
    // owner partition, len Nq) enable PARTITION-AWARE filtering: a candidate is
    // filtered only if it is REMOTE (partition_of[gid] != coordinator[qi]); local
    // candidates are always computed exact (never filtered), mirroring the real
    // M-machine system where only remote-owned nodes go through the PQ gate.
    // Both None -> filter-all mode (every candidate treated as remote / M->inf).
    py::dict search_batch(py::array_t<T, py::array::c_style | py::array::forcecast> queries,
                          uint32_t K, uint32_t L_search,
                          uint32_t conv_window, float conv_threshold_pct,
                          bool filter_on, bool record_drops, size_t max_drops,
                          py::object partition_of = py::none(),
                          py::object coordinator = py::none(),
                          const std::string& search_dist = std::string("exact"),
                          const std::string& rerank_scope = std::string("full"),
                          size_t trace_max_queries = 0,
                          float rerank_gate_alpha = -1.0f,
                          uint32_t batch_size = 0,
                          py::object opq_rotation = py::none()) {
        if (!_index_loaded) throw std::runtime_error("index not loaded");
        if (filter_on && !_pq.enabled()) throw std::runtime_error("filter_on but PQ not loaded");
        // Q1 (M08): rank the beam by PQ-estimated distance = "SparsePath under
        // quantized vectors". Default "exact" keeps the M11 behaviour bit-for-bit.
        //   "pq"        : traverse AND return top-K by PQ distance (no rerank; min memory).
        //   "pq_rerank" : traverse by PQ (cheap), then recompute EXACT distance on the
        //                 final beam and re-sort -> top-K by exact (mirrors quantized
        //                 ANNS that rerank with full vectors; rerank needs the raw vector
        //                 for every beam candidate = a remote fetch in the distributed
        //                 setting, i.e. the bandwidth SparsePath's placement reduces).
        const bool use_pq_dist = (search_dist == "pq" || search_dist == "pq_rerank");
        const bool use_sq8     = (search_dist == "sq8" || search_dist == "sq8_rerank");
        const bool use_rabitq  = (search_dist == "rabitq" || search_dist == "rabitq_rerank");
        // LVQ (Aguerrebere et al., VLDB'23 — LVQ-B / LVQ-B1xB2): quantize the
        // MEAN-CENTERED vector with B bits using that vector's own [min,max]
        // (per-vector scale+offset); the two-level variant adds a second B2-bit
        // level on the primary's residual and reranks with primary+residual
        // (NOT with the raw vector). Simulated on the fly here (recall-only).
        //   lvq8   : 8-bit primary only            -> D + 8 B/vec
        //   lvq4   : 4-bit primary only            -> D/2 + 8 B/vec  (max SQ compression)
        //   lvq4x8 : 4-bit primary (traversal) + 8-bit residual (terminal rerank)
        //                                          -> D/2 + D + 16 B/vec
        const bool use_lvq8    = (search_dist == "lvq8");
        const bool use_lvq4    = (search_dist == "lvq4");
        const bool use_lvq4x8  = (search_dist == "lvq4x8");
        const bool use_lvq     = use_lvq8 || use_lvq4 || use_lvq4x8;
        const bool do_rerank   = (search_dist == "pq_rerank" || search_dist == "sq8_rerank"
                                  || search_dist == "rabitq_rerank" || use_lvq4x8);
        // rerank_scope: "full" = DiskANN full_retset (every expanded node exact-rescored;
        // disk-resident semantics). "beam" = exact-rescore only the FINAL beam (fetch
        // count bounded by L — the distributed-reproducible rerank).
        if (rerank_scope != "full" && rerank_scope != "beam")
            throw std::runtime_error("rerank_scope must be 'full' or 'beam'");
        const bool rerank_beam_only = (rerank_scope == "beam");
        if (use_pq_dist && !_pq.enabled())
            throw std::runtime_error("search_dist='pq'/'pq_rerank' but PQ not loaded");
        if (use_rabitq) ensure_rabitq();   // one-time rotation + encode of all active vectors
        if (use_lvq)    ensure_lvq_mean(); // one-time global mean (LVQ centers the data)

        // OPQ: the PQ codes were built in a ROTATED frame (codes encode x @ R^T,
        // i.e. row i of R dotted with x). Rotate the query the same way and use
        // R*q for the PQ LUT / PQ distances ONLY — every exact distance (raw
        // traversal + terminal rerank) keeps the ORIGINAL query and the ORIGINAL
        // base, which is exact because R is orthogonal: ||Rq - Rx|| == ||q - x||.
        // Result is mathematically identical to running the whole harness on a
        // pre-rotated base + pre-rotated queries (the 2026-08-06 protocol),
        // without materialising a rotated copy of the base on disk.
        std::vector<float> Rmat;   // row-major D x D
        if (!opq_rotation.is_none()) {
            py::object robj = opq_rotation;
            if (py::isinstance<py::str>(opq_rotation))
                robj = py::module_::import("numpy").attr("load")(opq_rotation);
            auto Ra = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(robj);
            if (Ra.ndim() != 2 || static_cast<size_t>(Ra.shape(0)) != _dim
                               || static_cast<size_t>(Ra.shape(1)) != _dim)
                throw std::runtime_error("opq_rotation must be a (dim x dim) matrix");
            Rmat.assign(Ra.data(), Ra.data() + _dim * _dim);
        }
        const bool use_opq = !Rmat.empty();

        auto qbuf = queries.request();
        if (qbuf.ndim != 2) throw std::runtime_error("queries must be 2D [Nq, dim]");
        const size_t Nq  = static_cast<size_t>(qbuf.shape[0]);
        const size_t qd  = static_cast<size_t>(qbuf.shape[1]);
        if (qd != _dim) throw std::runtime_error("query dim != index dim");
        const T* qptr = static_cast<const T*>(qbuf.ptr);

        // partition-aware setup (optional). partition_of/coordinator each accept a
        // single uint32 array OR a parallel LIST of arrays — multi-label accounting:
        // ONE traversal, one counter set per placement labeling (recall is
        // label-independent, so N labelings cost N array lookups per candidate,
        // not N searches). Label set 0 drives the PQ filter's remote gate.
        using U32Arr = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
        std::vector<U32Arr> part_arrs, coord_arrs;   // keep numpy buffers alive
        std::vector<const uint32_t*> part_ptrs, coord_ptrs;
        std::vector<size_t> part_lens;
        if (!partition_of.is_none() && !coordinator.is_none()) {
            auto push_pair = [&](py::handle po, py::handle co) {
                part_arrs.push_back(py::cast<U32Arr>(py::reinterpret_borrow<py::object>(po)));
                coord_arrs.push_back(py::cast<U32Arr>(py::reinterpret_borrow<py::object>(co)));
                if (static_cast<size_t>(coord_arrs.back().size()) != Nq)
                    throw std::runtime_error("coordinator length must equal number of queries");
                part_ptrs.push_back(part_arrs.back().data());
                part_lens.push_back(static_cast<size_t>(part_arrs.back().size()));
                coord_ptrs.push_back(coord_arrs.back().data());
            };
            const bool is_list = py::isinstance<py::list>(partition_of)
                              || py::isinstance<py::tuple>(partition_of);
            if (is_list) {
                auto pl = py::cast<py::sequence>(partition_of);
                auto cl = py::cast<py::sequence>(coordinator);
                if (py::len(pl) != py::len(cl))
                    throw std::runtime_error("partition_of/coordinator list lengths differ");
                for (size_t i = 0; i < py::len(pl); i++) push_pair(pl[i], cl[i]);
            } else {
                push_pair(partition_of, coordinator);
            }
        }
        const size_t n_labels = part_ptrs.size();
        const bool has_part = n_labels > 0;

        const float alpha = _pq.alpha();
        auto* dist_fn = _index->get_distance_fn();
        auto& data_store = _index->get_data_store();
        auto* graph_store = _index->get_graph_store();

        // outputs
        std::vector<uint32_t> out_tags(Nq * K, std::numeric_limits<uint32_t>::max());
        std::vector<float>    out_dists(Nq * K, std::numeric_limits<float>::infinity());

        uint64_t n_filter_eligible = 0;   // candidates seen while converged && beam_full
        uint64_t n_pq_filtered     = 0;    // dropped by the filter
        uint64_t n_false_local     = 0;    // dropped AND exact <= worst (would-be-useful)
        uint64_t sum_hops          = 0;
        uint64_t n_rerank_gated    = 0;    // W4: beam-rerank candidates skipped by the α-margin gate
        // ---- M10 CAP (converge-phase batched fetching) telemetry ----
        uint64_t n_batch_rounds    = 0;    // batch rounds across all queries
        uint64_t n_batch_cands     = 0;    // candidates expanded via the batch path
        uint64_t n_fallbacks       = 0;    // fallback (un-converge) events
        uint64_t sum_fetch_rounds  = 0;    // per-query fetch rounds summed (ALL rounds)
        uint64_t sum_remote_fetch_rounds = 0; // rounds issuing >=1 REMOTE fetch (RDMA round-trip)
        uint64_t sum_conv_hop      = 0;    // sum over queries of the hop at first convergence
        uint64_t n_converged       = 0;    // queries that reached convergence

        // W1 (convergence-under-quantization): per-query latch telemetry. The
        // detector observes the RANKING distance (quantized in pq/sq8/rabitq
        // modes), so these arrays measure whether/when the production
        // ConvergenceDetector latches under estimated distances. 0 = never.
        std::vector<uint32_t> conv_hop_q(Nq, 0);   // hop (1-based) at which converged latched
        std::vector<uint32_t> hops_q(Nq, 0);       // total hops per query

        // W2 (trace-based convergence profile): per-hop stream for the first
        // trace_max_queries queries. Slots indexed by qi -> no cross-thread
        // contention. topk = beam's current top-K tags (ranking order, tag!=0),
        // padded with UINT32_MAX — enables offline oracle/recall@hop/detector
        // comparison without re-searching.
        struct HopRec { float best, worst; uint32_t beam_size, n_inserted, expand_tag; };
        const size_t Nt = std::min(trace_max_queries, Nq);
        std::vector<std::vector<HopRec>> trace_hops(Nt);
        std::vector<std::vector<uint32_t>> trace_topk(Nt);   // K entries per hop
        // Full-beam snapshot per hop (L_search entries, UINT32_MAX padding) —
        // enables offline final-list overlap / full-beam turnover (2026-08-11).
        std::vector<std::vector<uint32_t>> trace_beam(Nt);

        // Fetch accounting (single-node proxy for distributed traffic):
        //   scored — candidates whose ranking distance was computed (exact mode = raw
        //            vector accesses; quantized modes = code accesses, no raw fetch).
        //   expand — expanded frontier nodes (= neighbor-list accesses).
        //   rerank — exact rescoring events (= raw-vector fetches in quantized modes).
        // local/remote classified via partition_of/coordinator, per label set; without
        // partitions everything counts as remote (filter-all convention, M -> inf).
        // Layout per set: [scored_l, scored_r, expand_l, expand_r, rerank_l, rerank_r].
        const size_t n_ctr_sets = has_part ? n_labels : 1;
        std::vector<std::array<uint64_t, 6>> label_cnt(
            n_ctr_sets, std::array<uint64_t, 6>{0, 0, 0, 0, 0, 0});

        std::vector<uint32_t> drop_qidx;
        std::vector<uint32_t> drop_gid;
        std::vector<float>    drop_pqdist, drop_exact, drop_worst;

        const auto& centroids        = _pq.centroids();
        const auto& chunk_offsets    = _pq.chunk_offsets();
        const auto& centroid_offsets = _pq.centroid_offsets();

        // Parallelize over queries (independent). Counters via reduction; drop
        // recording via a critical section. Per-thread scratch declared inside.
        const int nthreads = (_num_threads > 0) ? static_cast<int>(_num_threads) : omp_get_max_threads();
        #pragma omp parallel for schedule(dynamic, 8) num_threads(nthreads) \
            reduction(+ : n_filter_eligible, n_pq_filtered, n_false_local, sum_hops, n_rerank_gated, \
                          n_batch_rounds, n_batch_cands, n_fallbacks, sum_fetch_rounds, \
                          sum_conv_hop, n_converged, sum_remote_fetch_rounds)
        for (long long qi_s = 0; qi_s < static_cast<long long>(Nq); qi_s++) {
            const size_t qi = static_cast<size_t>(qi_s);
            const bool do_trace = (qi < Nt);
            std::vector<float> pq_lut(static_cast<size_t>(_M) * KSUB);   // per-thread
            std::vector<float> qf(_dim);
            std::vector<T>     nbuf(_dim), qnat(_dim);
            const T* q = qptr + qi * _dim;
            for (size_t d = 0; d < _dim; d++) {
                qf[d]   = static_cast<float>(q[d]);
                qnat[d] = q[d];
            }

            // remote iff partition_of[tag-1] != this query's coordinator partition,
            // per label set l. No partitions -> everything remote (filter-all).
            // tag==0 (frozen) -> local.
            auto is_remote_tag_l = [&](uint32_t tag_, size_t l) -> bool {
                if (!has_part) return true;
                if (tag_ == 0u) return false;
                const uint32_t gid_ = tag_ - 1u;
                const uint32_t p = (gid_ < part_lens[l]) ? part_ptrs[l][gid_] : 0u;
                return p != coord_ptrs[l][qi];
            };
            auto is_remote_tag = [&](uint32_t tag_) -> bool {   // label 0 = filter gate
                return is_remote_tag_l(tag_, 0);
            };
            // per-query local counters, merged once per query under a critical section
            std::vector<std::array<uint64_t, 6>> lc(
                n_ctr_sets, std::array<uint64_t, 6>{0, 0, 0, 0, 0, 0});
            auto count_event = [&](uint32_t tag_, int base) {  // base: 0 scored, 2 expand, 4 rerank
                if (tag_ == 0u) return;
                for (size_t l = 0; l < n_ctr_sets; l++)
                    lc[l][base + (is_remote_tag_l(tag_, l) ? 1 : 0)]++;
            };

            // RaBitQ query prep: rotate the query once and take ||q||^2 (per-thread).
            std::vector<float> qr;
            float qnorm2 = 0.0f;
            if (use_rabitq) {
                qr.assign(_dim, 0.0f);
                for (size_t i = 0; i < _dim; i++) {
                    const float* Ri = &_rbq_R[i * _dim];
                    float acc = 0.0f;
                    for (size_t k = 0; k < _dim; k++) acc += Ri[k] * (qf[k] - _rbq_c[k]);
                    qr[i] = acc;
                }
                for (size_t k = 0; k < _dim; k++) { float v = qf[k] - _rbq_c[k]; qnorm2 += v * v; }
            }

            // Build PQ query LUT (squared L2), identical to beam_search.cpp:401-434.
            // Under OPQ the LUT — and therefore every PQ-estimated distance and
            // every PQ-filter decision — is built from R*q instead of q.
            if (filter_on || use_pq_dist) {
                const float* qlut = qf.data();
                std::vector<float> qrot;
                if (use_opq) {
                    qrot.assign(_dim, 0.0f);
                    for (size_t i = 0; i < _dim; i++) {
                        const float* Ri = &Rmat[i * _dim];
                        float acc = 0.0f;
                        for (size_t k = 0; k < _dim; k++) acc += Ri[k] * qf[k];
                        qrot[i] = acc;
                    }
                    qlut = qrot.data();
                }
                build_lut(qlut, centroids, chunk_offsets, centroid_offsets, pq_lut.data());
            }

            // ---- beam search over the Vamana graph ----
            std::vector<Cand> cl;                 // bounded ascending candidate list, cap L_search
            cl.reserve(L_search + 8);
            tsl_like_visited visited;

            ConvergenceDetector conv(conv_window, conv_threshold_pct);
            bool converged = false;
            uint64_t hops = 0;              // candidates expanded (best-first + batch)
            uint64_t fetch_rounds = 0;      // 1 per best-first hop, 1 per batch round
            uint64_t conv_hop = 0;          // hop at which this query first converged (0 = never)
            uint64_t remote_fetch_rounds = 0; // rounds with >=1 remote fetch (partition-aware)
            bool     round_remote = false;  // set by process_neighbor on a remote fetch

            // DiskANN-faithful rerank pool: every EXPANDED (frontier) node is scored by
            // EXACT distance and accumulated here (mirrors pq_flash_index.cpp full_retset).
            // Final results = top-K by exact over this set (NOT just the final PQ-beam),
            // so a true neighbor expanded early but PQ-evicted from the beam still counts.
            std::vector<Cand> full_retset;
            if (do_rerank) full_retset.reserve(L_search * 4);

            // seed the frozen start point
            {
                data_store.get_vector(_start, nbuf.data());
                float d0 = dist_fn->compare(qnat.data(), nbuf.data(), static_cast<uint32_t>(_dim));
                cl.push_back(Cand{d0, _start, false});
                visited.insert(_start);
            }

            // Neighbour processing, shared by the best-first and the CAP batch path.
            // worst / beam_full / threshold are snapshotted from the queue state BEFORE
            // this hop's (or round's) inserts — hop-fixed in best-first, round-fixed in a
            // batch round — mirroring ColocationBeamSearch. `converged` is read live
            // (captured by reference), so the PQ gate reflects the current phase.
            auto process_neighbor = [&](uint32_t nbr, float worst, bool beam_full,
                                        float threshold, uint32_t& ins_ctr) {
                if (nbr == std::numeric_limits<uint32_t>::max()) return;
                if (visited.count(nbr)) return;
                visited.insert(nbr);

                    // gid for PQ code (skip frozen/sentinel tag==0). Needed when
                    // filtering OR when ranking the beam by PQ distance (search_dist="pq").
                    uint32_t tag = (nbr < _tag_of_loc.size()) ? _tag_of_loc[nbr] : 0u;
                    const bool pq_valid = (filter_on || use_pq_dist) && (tag != 0u);
                    uint32_t gid = (tag != 0u) ? (tag - 1u) : 0u;

                    const bool is_remote = is_remote_tag(tag);

                    if (filter_on && pq_valid && is_remote && converged && beam_full) {
                        n_filter_eligible++;
                        float pq_dist = _pq.compute_distance(pq_lut.data(), gid);
                        // pq_dist < 0 means gid out of range -> treat as not filterable
                        if (pq_dist >= 0.0f && pq_dist > threshold) {
                            // FILTERED. Classify with the exact distance (free: vector is local).
                            data_store.get_vector(nbr, nbuf.data());
                            float exact = dist_fn->compare(qnat.data(), nbuf.data(),
                                                           static_cast<uint32_t>(_dim));
                            n_pq_filtered++;
                            if (exact <= worst) n_false_local++;   // would-be-useful
                            if (record_drops) {
                                #pragma omp critical
                                {
                                    if (drop_qidx.size() < max_drops) {
                                        drop_qidx.push_back(static_cast<uint32_t>(qi));
                                        drop_gid.push_back(gid);
                                        drop_pqdist.push_back(pq_dist);
                                        drop_exact.push_back(exact);
                                        drop_worst.push_back(worst);
                                    }
                                }
                            }
                            return;   // do NOT insert -> faithful recall(ON)
                        }
                    }

                    // ranking distance for the beam:
                    //  * search_dist="exact" (default): true squared-L2 from the local vector.
                    //  * search_dist="pq": PQ-estimated squared-L2 = SparsePath "under quantized
                    //    vectors" (the whole beam is ranked on the compressed distance -> recall
                    //    degrades; this is the honest quantized-search Pareto point).
                    float dist;
                    if (use_pq_dist && pq_valid) {
                        dist = _pq.compute_distance(pq_lut.data(), gid);
                        if (dist < 0.0f) {   // gid out of code range -> fall back to exact
                            data_store.get_vector(nbr, nbuf.data());
                            dist = dist_fn->compare(qnat.data(), nbuf.data(),
                                                    static_cast<uint32_t>(_dim));
                        }
                    } else if (use_sq8) {
                        // 8-bit per-vector scalar quantization (LVQ-style): quantize the
                        // data vector to int8 on the fly, exact query (asymmetric). Models
                        // "vectors stored at SQ8 precision" (1 byte/dim).
                        data_store.get_vector(nbr, nbuf.data());
                        dist = sq8_dist(qnat.data(), nbuf.data(), _dim);
                    } else if (use_lvq) {
                        // LVQ PRIMARY level drives the traversal (lvq8 -> 8-bit,
                        // lvq4/lvq4x8 -> 4-bit). Query stays exact (asymmetric).
                        data_store.get_vector(nbr, nbuf.data());
                        dist = lvq_dist(qf.data(), nbuf.data(), use_lvq8 ? 8 : 4);
                    } else if (use_rabitq) {
                        // 1 bit/dim RaBitQ estimate from pre-encoded codes (no raw fetch).
                        dist = rabitq_dist(nbr, qr.data(), qnorm2);
                    } else {
                        data_store.get_vector(nbr, nbuf.data());
                        dist = dist_fn->compare(qnat.data(), nbuf.data(),
                                                static_cast<uint32_t>(_dim));
                    }
                    count_event(tag, 0);
                    if (insert_bounded(cl, Cand{dist, nbr, false}, L_search)) ins_ctr++;
            };

            while (true) {
                uint32_t ins_this_hop = 0;   // beam insertions this hop/round (turnover signal)
                uint32_t cur_tag = 0;        // W2 trace: tag of the (last) expanded node

                if (!converged || batch_size == 0) {
                    // ---- best-first (approach phase, or CAP disabled: batch_size == 0) ----
                    // Expand ONE closest-unexpanded candidate, then re-select. This is the
                    // standard best-first visit order M10 contrasts batching against.
                    int k = closest_unexpanded(cl);
                    if (k < 0) break;
                    cl[static_cast<size_t>(k)].expanded = true;
                    const uint32_t cur = cl[static_cast<size_t>(k)].loc;
                    hops++;
                    fetch_rounds++;
                    round_remote = false;

                    cur_tag = (cur < _tag_of_loc.size()) ? _tag_of_loc[cur] : 0u;
                    count_event(cur_tag, 2);

                    // DiskANN-faithful: score this expanded (frontier) node by EXACT and add
                    // it to the rerank pool (mirrors full_retset). Frozen/sentinel skipped.
                    // "beam" scope skips this and reranks only the final beam after the loop.
                    if (do_rerank && !rerank_beam_only && cur_tag != 0u) {
                        data_store.get_vector(cur, nbuf.data());
                        // lvq4x8 reranks with the two-level RECONSTRUCTION (primary
                        // 4-bit + 8-bit residual), never with the raw vector.
                        float cur_exact = use_lvq4x8
                            ? lvq2_dist(qf.data(), nbuf.data())
                            : dist_fn->compare(qnat.data(), nbuf.data(),
                                               static_cast<uint32_t>(_dim));
                        full_retset.push_back(Cand{cur_exact, cur, true});
                        count_event(cur_tag, 4);
                    }

                    // hop-fixed filter parameters (queue state at hop entry)
                    const bool  beam_full = (cl.size() >= L_search);
                    const float worst     = cl.back().dist;   // cl ascending -> back = worst
                    const float threshold = worst * (1.0f + alpha);

                    const std::vector<uint32_t>& nbrs = graph_store->get_neighbours(cur);
                    for (uint32_t nbr : nbrs)
                        process_neighbor(nbr, worst, beam_full, threshold, ins_this_hop);
                    if (round_remote) remote_fetch_rounds++;

                    // convergence update (once per hop, on current best), latched
                    if (!cl.empty()) {
                        bool just = conv.update(cl[0].dist);
                        if (just && !converged) {
                            converged = true;
                            conv_hop_q[qi] = static_cast<uint32_t>(hops);   // W1: latch hop
                            if (conv_hop == 0) conv_hop = hops;
                        }
                    }
                } else {
                    // ---- M10 CAP converge-phase batch expansion (batch_size = B) ----
                    // Pop up to B closest-unexpanded candidates AS A BATCH (all marked
                    // expanded before any of their neighbours is processed), then fetch and
                    // insert all their neighbours in ONE round under a single filter
                    // snapshot. This is exactly the visit-order change M10 asks about;
                    // recall(B>0) vs recall(B=0) measures whether it hurts recall.
                    // Mirrors beam_search.cpp (batch expansion + fallback).
                    const float pre_best =
                        cl.empty() ? std::numeric_limits<float>::infinity() : cl[0].dist;

                    std::vector<uint32_t> batch_locs;
                    batch_locs.reserve(batch_size);
                    while (batch_locs.size() < batch_size) {
                        int k = closest_unexpanded(cl);
                        if (k < 0) break;
                        cl[static_cast<size_t>(k)].expanded = true;
                        batch_locs.push_back(cl[static_cast<size_t>(k)].loc);
                    }
                    if (batch_locs.empty()) break;
                    hops += batch_locs.size();
                    fetch_rounds++;                 // one batched fetch round for the whole batch
                    round_remote = false;
                    n_batch_rounds++;
                    n_batch_cands += batch_locs.size();

                    // filter parameters snapshot ONCE per round, before this round's inserts.
                    const bool  beam_full = (cl.size() >= L_search);
                    const float worst     = cl.back().dist;
                    const float threshold = worst * (1.0f + alpha);

                    for (uint32_t cloc : batch_locs) {
                        const uint32_t ctag = (cloc < _tag_of_loc.size()) ? _tag_of_loc[cloc] : 0u;
                        count_event(ctag, 2);
                        if (do_rerank && !rerank_beam_only && ctag != 0u) {
                            data_store.get_vector(cloc, nbuf.data());
                            // lvq4x8 reranks with the two-level RECONSTRUCTION.
                            float cur_exact = use_lvq4x8
                                ? lvq2_dist(qf.data(), nbuf.data())
                                : dist_fn->compare(qnat.data(), nbuf.data(),
                                                   static_cast<uint32_t>(_dim));
                            full_retset.push_back(Cand{cur_exact, cloc, true});
                            count_event(ctag, 4);
                        }
                        const std::vector<uint32_t>& nbrs = graph_store->get_neighbours(cloc);
                        for (uint32_t nbr : nbrs)
                            process_neighbor(nbr, worst, beam_full, threshold, ins_this_hop);
                        cur_tag = ctag;
                    }
                    if (round_remote) remote_fetch_rounds++;

                    // Fallback: if this round improved the best by > 5%, un-converge and
                    // resume best-first (mirrors beam_search.cpp).
                    const float post_best =
                        cl.empty() ? std::numeric_limits<float>::infinity() : cl[0].dist;
                    if (pre_best > 0 && pre_best < std::numeric_limits<float>::infinity()) {
                        float imp = (pre_best - post_best) / pre_best * 100.0f;
                        if (imp > 5.0f) { converged = false; conv.reset(); n_fallbacks++; }
                    }
                }

                // W2: hop-end snapshot (beam state after this hop's/round's insertions).
                if (do_trace) {
                    HopRec hr;
                    hr.best  = cl.empty() ? std::numeric_limits<float>::infinity() : cl[0].dist;
                    hr.worst = cl.empty() ? std::numeric_limits<float>::infinity() : cl.back().dist;
                    hr.beam_size  = static_cast<uint32_t>(cl.size());
                    hr.n_inserted = ins_this_hop;
                    hr.expand_tag = cur_tag;
                    trace_hops[qi].push_back(hr);
                    uint32_t fk = 0;
                    for (size_t ci = 0; ci < cl.size() && fk < K; ci++) {
                        const uint32_t t2 = (cl[ci].loc < _tag_of_loc.size()) ? _tag_of_loc[cl[ci].loc] : 0u;
                        if (t2 == 0u) continue;
                        trace_topk[qi].push_back(t2);
                        fk++;
                    }
                    for (; fk < K; fk++) trace_topk[qi].push_back(std::numeric_limits<uint32_t>::max());
                    uint32_t fb = 0;
                    for (size_t ci = 0; ci < cl.size() && fb < static_cast<uint32_t>(L_search); ci++) {
                        const uint32_t t2 = (cl[ci].loc < _tag_of_loc.size()) ? _tag_of_loc[cl[ci].loc] : 0u;
                        if (t2 == 0u) continue;
                        trace_beam[qi].push_back(t2);
                        fb++;
                    }
                    for (; fb < static_cast<uint32_t>(L_search); fb++)
                        trace_beam[qi].push_back(std::numeric_limits<uint32_t>::max());
                }
            }

            sum_hops += hops;
            hops_q[qi] = static_cast<uint32_t>(hops);
            sum_fetch_rounds += fetch_rounds;
            sum_remote_fetch_rounds += remote_fetch_rounds;
            if (conv_hop > 0) { sum_conv_hop += conv_hop; n_converged++; }

            // "beam" rerank scope: exact-rescore only the FINAL beam (fetch count <= L
            // per query — the rerank a distributed system can afford). Recall parity
            // with "full" verified on deep100M (0.7346 vs 0.7353, navigation-bound).
            // W4 (rerank-pool gate): with rerank_gate_alpha >= 0, only candidates
            // within the estimated-distance margin  d̂ <= d̂_K * (1 + α)  are raw-
            // fetched (M11 α-margin math transplanted to the terminal rerank —
            // the SP-q replacement for the traversal-time PQ filter).
            if (do_rerank && rerank_beam_only) {
                full_retset.clear();
                float gate_thr = std::numeric_limits<float>::infinity();
                if (rerank_gate_alpha >= 0.0f) {
                    uint32_t seen = 0;
                    for (const Cand& c : cl) {
                        const uint32_t ctag = (c.loc < _tag_of_loc.size()) ? _tag_of_loc[c.loc] : 0u;
                        if (ctag == 0u) continue;
                        if (++seen == K) { gate_thr = c.dist * (1.0f + rerank_gate_alpha); break; }
                    }   // fewer than K valid candidates -> gate stays open (inf)
                }
                for (const Cand& c : cl) {
                    uint32_t ctag = (c.loc < _tag_of_loc.size()) ? _tag_of_loc[c.loc] : 0u;
                    if (ctag == 0u) continue;
                    if (c.dist > gate_thr) { n_rerank_gated++; continue; }   // gated: no raw fetch
                    data_store.get_vector(c.loc, nbuf.data());
                    // lvq4x8: rescore from the two-level LVQ reconstruction (the
                    // "fetch" is the D-byte residual code, not the raw vector).
                    float ex = use_lvq4x8
                        ? lvq2_dist(qf.data(), nbuf.data())
                        : dist_fn->compare(qnat.data(), nbuf.data(),
                                           static_cast<uint32_t>(_dim));
                    full_retset.push_back(Cand{ex, c.loc, true});
                    count_event(ctag, 4);
                }
            }

            // Result source: for pq_rerank, the DiskANN-faithful full_retset (every
            // expanded frontier node scored by EXACT), sorted by exact; otherwise the
            // PQ/exact beam itself. This matches pq_flash_index.cpp: navigate by PQ,
            // rank the results by exact over all expanded nodes (not just the final beam).
            if (do_rerank) {
                std::sort(full_retset.begin(), full_retset.end(),
                          [](const Cand& a, const Cand& b) { return a.dist < b.dist; });
            }
            const std::vector<Cand>& result_list = do_rerank ? full_retset : cl;

            // collect top-K (skip frozen start / tag==0)
            uint32_t filled = 0;
            for (size_t i = 0; i < result_list.size() && filled < K; i++) {
                uint32_t loc = result_list[i].loc;
                uint32_t tag = (loc < _tag_of_loc.size()) ? _tag_of_loc[loc] : 0u;
                if (tag == 0u) continue;   // frozen / sentinel — not a real result
                out_tags[qi * K + filled]  = tag;
                out_dists[qi * K + filled] = result_list[i].dist;
                filled++;
            }

            // merge this query's label counters (once per query — negligible contention)
            #pragma omp critical
            {
                for (size_t l = 0; l < n_ctr_sets; l++)
                    for (int j = 0; j < 6; j++) label_cnt[l][j] += lc[l][j];
            }
        }

        // ---- pack results ----
        py::dict out;
        out["result_tags"]  = to_numpy_2d<uint32_t>(out_tags, Nq, K);
        out["result_dists"] = to_numpy_2d<float>(out_dists, Nq, K);
        out["n_filter_eligible"] = n_filter_eligible;
        out["n_pq_filtered"]     = n_pq_filtered;
        out["n_false_local"]     = n_false_local;
        out["sum_hops"]          = sum_hops;
        out["n_queries"]         = static_cast<uint64_t>(Nq);
        out["alpha"]             = alpha;
        // M10 CAP telemetry
        out["n_batch_rounds"]     = n_batch_rounds;
        out["n_batch_candidates"] = n_batch_cands;
        out["n_fallbacks"]        = n_fallbacks;
        out["sum_fetch_rounds"]   = sum_fetch_rounds;
        out["sum_remote_fetch_rounds"] = sum_remote_fetch_rounds;
        out["sum_conv_hop"]       = sum_conv_hop;
        out["n_converged"]        = n_converged;
        out["batch_size"]         = static_cast<uint64_t>(batch_size);
        out["n_scored_local"]    = label_cnt[0][0];
        out["n_scored_remote"]   = label_cnt[0][1];
        out["n_expand_local"]    = label_cnt[0][2];
        out["n_expand_remote"]   = label_cnt[0][3];
        out["n_rerank_local"]    = label_cnt[0][4];
        out["n_rerank_remote"]   = label_cnt[0][5];
        out["has_partitions"]    = has_part;
        out["n_label_sets"]      = static_cast<uint64_t>(n_ctr_sets);
        out["rerank_scope"]      = rerank_scope;
        {
            py::list lcs;
            for (size_t l = 0; l < n_ctr_sets; l++) {
                py::dict d;
                d["scored_local"]  = label_cnt[l][0];
                d["scored_remote"] = label_cnt[l][1];
                d["expand_local"]  = label_cnt[l][2];
                d["expand_remote"] = label_cnt[l][3];
                d["rerank_local"]  = label_cnt[l][4];
                d["rerank_remote"] = label_cnt[l][5];
                lcs.append(d);
            }
            out["label_counters"] = lcs;
        }
        out["drop_qidx"]   = to_numpy_1d<uint32_t>(drop_qidx);
        out["drop_gid"]    = to_numpy_1d<uint32_t>(drop_gid);
        out["drop_pqdist"] = to_numpy_1d<float>(drop_pqdist);
        out["drop_exact"]  = to_numpy_1d<float>(drop_exact);
        out["drop_worst"]  = to_numpy_1d<float>(drop_worst);

        // W1: per-query convergence latch telemetry (0 = never latched).
        out["conv_hop_per_query"] = to_numpy_1d<uint32_t>(conv_hop_q);
        out["hops_per_query"]     = to_numpy_1d<uint32_t>(hops_q);
        // W4: rerank-pool α-margin gate accounting.
        out["n_rerank_gated"]     = n_rerank_gated;
        out["rerank_gate_alpha"]  = rerank_gate_alpha;

        // W2: flatten the ragged per-hop trace into offset-indexed arrays.
        // Query i's hops = [trace_offsets[i], trace_offsets[i+1]); trace_topk row
        // j = beam top-K tags at that hop (UINT32_MAX padding).
        out["trace_n_queries"] = static_cast<uint64_t>(Nt);
        if (Nt > 0) {
            std::vector<uint64_t> offs(Nt + 1, 0);
            for (size_t i = 0; i < Nt; i++) offs[i + 1] = offs[i] + trace_hops[i].size();
            const size_t tot = static_cast<size_t>(offs[Nt]);
            std::vector<float>    tb(tot), tw(tot);
            std::vector<uint32_t> tbs(tot), tni(tot), tet(tot);
            std::vector<uint32_t> ttk(tot * K);
            const size_t Lcap = static_cast<size_t>(L_search);
            std::vector<uint32_t> tbm(tot * Lcap);
            for (size_t i = 0; i < Nt; i++) {
                const size_t base = static_cast<size_t>(offs[i]);
                for (size_t hi = 0; hi < trace_hops[i].size(); hi++) {
                    const HopRec& hr = trace_hops[i][hi];
                    tb[base + hi]  = hr.best;
                    tw[base + hi]  = hr.worst;
                    tbs[base + hi] = hr.beam_size;
                    tni[base + hi] = hr.n_inserted;
                    tet[base + hi] = hr.expand_tag;
                }
                std::memcpy(ttk.data() + base * K, trace_topk[i].data(),
                            trace_topk[i].size() * sizeof(uint32_t));
                std::memcpy(tbm.data() + base * Lcap, trace_beam[i].data(),
                            trace_beam[i].size() * sizeof(uint32_t));
            }
            out["trace_offsets"]    = to_numpy_1d<uint64_t>(offs);
            out["trace_best"]       = to_numpy_1d<float>(tb);
            out["trace_worst"]      = to_numpy_1d<float>(tw);
            out["trace_beam_size"]  = to_numpy_1d<uint32_t>(tbs);
            out["trace_n_inserted"] = to_numpy_1d<uint32_t>(tni);
            out["trace_expand_tag"] = to_numpy_1d<uint32_t>(tet);
            out["trace_topk"]       = to_numpy_2d<uint32_t>(ttk, tot, K);
            out["trace_beam"]       = to_numpy_2d<uint32_t>(tbm, tot, Lcap);
        }
        return out;
    }

    // ---- Graph structure stats (LITS: inter-partition / "remote" edge %) --------
    // Walks the loaded graph's adjacency and classifies every active out-edge as
    // intra- or inter-partition, reproducing the paper's remote-edge headline
    // (e.g. deep100M lits 7.56% vs vamana+ncb 24.8%) single-node, no RDMA. This is
    // a MEASUREMENT of an already-built graph (lits or vamana) + its partition
    // labels — nothing is rebuilt.
    //
    // part: partition id per node. by_tag=true -> indexed by TAG (lits `.labels`,
    // which is tag-indexed and pool-sized; tags are NON-sequential after update
    // cycles, so partition MUST be looked up by tag). by_tag=false -> indexed by
    // GID = tag-1 (vamana partition_ids.npy). Nodes whose partition index is out of
    // range are skipped (n_skipped). Edges to the frozen start (tag==0) count toward
    // degree/total but never as inter-partition.
    py::dict graph_stats(py::array_t<uint32_t, py::array::c_style | py::array::forcecast> part,
                         bool by_tag = true) {
        if (!_index_loaded) throw std::runtime_error("index not loaded");
        auto* graph_store = _index->get_graph_store();
        const uint32_t* pp = part.data();
        const size_t part_len = static_cast<size_t>(part.size());

        auto part_of = [&](uint32_t tag) -> long long {
            size_t idx = by_tag ? static_cast<size_t>(tag)
                                : static_cast<size_t>(tag - 1u);
            if (idx >= part_len) return -1;
            return static_cast<long long>(pp[idx]);
        };

        // max partition id (over active nodes) to size the per-partition histograms
        uint32_t max_part = 0;
        for (uint32_t loc = 0; loc < _num_points; loc++) {
            uint32_t tag = (loc < _tag_of_loc.size()) ? _tag_of_loc[loc] : 0u;
            if (tag == 0u) continue;
            long long sp = part_of(tag);
            if (sp >= 0 && static_cast<uint32_t>(sp) > max_part) max_part = static_cast<uint32_t>(sp);
        }

        std::vector<uint64_t> part_nodes(static_cast<size_t>(max_part) + 1, 0);   // active nodes / partition
        std::vector<uint64_t> part_inter(static_cast<size_t>(max_part) + 1, 0);   // inter out-edges / partition
        uint64_t n_active = 0, n_skipped = 0;
        uint64_t total_edges = 0, inter_edges = 0, sum_deg = 0, zero_deg = 0;

        for (uint32_t loc = 0; loc < _num_points; loc++) {
            uint32_t tag = (loc < _tag_of_loc.size()) ? _tag_of_loc[loc] : 0u;
            if (tag == 0u) continue;                 // frozen / sentinel
            long long sp = part_of(tag);
            if (sp < 0) { n_skipped++; continue; }   // partition index out of range
            n_active++;
            part_nodes[static_cast<size_t>(sp)]++;

            const std::vector<uint32_t>& nbrs = graph_store->get_neighbours(loc);
            uint32_t deg = 0;
            for (uint32_t nbr : nbrs) {
                if (nbr == std::numeric_limits<uint32_t>::max()) continue;
                deg++;
                total_edges++;
                uint32_t ntag = (nbr < _tag_of_loc.size()) ? _tag_of_loc[nbr] : 0u;
                if (ntag == 0u) continue;            // edge to frozen: counts as edge, not inter
                long long np = part_of(ntag);
                if (np >= 0 && np != sp) { inter_edges++; part_inter[static_cast<size_t>(sp)]++; }
            }
            sum_deg += deg;
            if (deg == 0) zero_deg++;
        }

        py::dict out;
        out["n_active"]          = n_active;
        out["n_skipped"]         = n_skipped;
        out["total_edges"]       = total_edges;
        out["inter_edges"]       = inter_edges;
        out["inter_pct"]         = total_edges ? (100.0 * static_cast<double>(inter_edges)
                                                        / static_cast<double>(total_edges)) : 0.0;
        out["avg_degree"]        = n_active ? (static_cast<double>(sum_deg)
                                                      / static_cast<double>(n_active)) : 0.0;
        out["zero_degree_nodes"] = zero_deg;
        out["num_partitions"]    = static_cast<uint64_t>(max_part) + 1;
        out["partition_node_counts"] = to_numpy_1d<uint64_t>(part_nodes);
        out["partition_inter_edges"] = to_numpy_1d<uint64_t>(part_inter);
        return out;
    }

private:
    // ---- small helpers -------------------------------------------------------
    struct Cand { float dist; uint32_t loc; bool expanded; };

    // Verbatim copy of ColocationBeamSearch's convergence detector.
    struct ConvergenceDetector {
        float prev_best = std::numeric_limits<float>::infinity();
        uint32_t stable_count = 0;
        uint32_t window;
        float threshold_pct;
        ConvergenceDetector(uint32_t w, float t) : window(w), threshold_pct(t) {}
        bool update(float best_distance) {
            if (prev_best > 0 && prev_best < std::numeric_limits<float>::infinity()) {
                float delta_pct = std::abs(best_distance - prev_best) / prev_best * 100.0f;
                if (delta_pct < threshold_pct) stable_count++;
                else stable_count = 0;
            }
            prev_best = best_distance;
            return stable_count >= window;
        }
        void reset() { stable_count = 0; }   // fallback un-converge: forget the stable streak
    };

    // A tiny open-addressing visited set (avoids pulling tsl here).
    struct tsl_like_visited {
        std::vector<uint64_t> keys;
        std::vector<uint8_t>  used;
        size_t mask = 0, count_ = 0;
        tsl_like_visited() { rehash(1024); }
        void rehash(size_t cap) {
            size_t n = 1; while (n < cap * 2) n <<= 1;
            std::vector<uint64_t> nk(n); std::vector<uint8_t> nu(n, 0);
            for (size_t i = 0; i < keys.size(); i++) if (used[i]) {
                size_t j = keys[i] & (n - 1);
                while (nu[j]) j = (j + 1) & (n - 1);
                nk[j] = keys[i]; nu[j] = 1;
            }
            keys.swap(nk); used.swap(nu); mask = n - 1;
        }
        bool count(uint32_t v) const {
            size_t j = static_cast<uint64_t>(v) & mask;
            while (used[j]) { if (keys[j] == v) return true; j = (j + 1) & mask; }
            return false;
        }
        void insert(uint32_t v) {
            if ((count_ + 1) * 2 > keys.size()) rehash(keys.size());
            size_t j = static_cast<uint64_t>(v) & mask;
            while (used[j]) { if (keys[j] == v) return; j = (j + 1) & mask; }
            keys[j] = v; used[j] = 1; count_++;
        }
    };

    static int closest_unexpanded(const std::vector<Cand>& cl) {
        for (size_t i = 0; i < cl.size(); i++) if (!cl[i].expanded) return static_cast<int>(i);
        return -1;
    }

    // Insert keeping ascending order, capped at L (mirrors NeighborPriorityQueue).
    // Returns true iff the candidate actually entered the beam (turnover signal).
    static bool insert_bounded(std::vector<Cand>& cl, Cand c, uint32_t L) {
        if (cl.size() >= L && c.dist >= cl.back().dist) return false;   // worse than worst -> drop
        // binary search for insertion point
        size_t lo = 0, hi = cl.size();
        while (lo < hi) { size_t mid = (lo + hi) / 2; if (cl[mid].dist < c.dist) lo = mid + 1; else hi = mid; }
        // dedup by loc is handled by the visited set upstream
        cl.insert(cl.begin() + static_cast<std::ptrdiff_t>(lo), c);
        if (cl.size() > L) cl.pop_back();
        return true;
    }

    // 8-bit per-vector scalar quant (LVQ-style): quantize x to int8 with its own
    // [min,max], decode, and return squared-L2 to the (exact) query. Asymmetric.
    static float sq8_dist(const T* q, const T* x, size_t dim) {
        float lo = std::numeric_limits<float>::max();
        float hi = -std::numeric_limits<float>::max();
        for (size_t d = 0; d < dim; d++) {
            float v = static_cast<float>(x[d]);
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        const float scale = (hi > lo) ? (hi - lo) / 255.0f : 1.0f;
        float acc = 0.0f;
        for (size_t d = 0; d < dim; d++) {
            int code = static_cast<int>(std::lround((static_cast<float>(x[d]) - lo) / scale));
            if (code < 0) code = 0; else if (code > 255) code = 255;
            const float xq = lo + static_cast<float>(code) * scale;
            const float diff = static_cast<float>(q[d]) - xq;
            acc += diff * diff;
        }
        return acc;
    }

    // ---- LVQ (Aguerrebere et al., VLDB'23: "Similarity search in the blink of
    // an eye with compressed indices" — LVQ-B / two-level LVQ-B1xB2) ------------
    // LVQ-B: r = x - mu (global mean); quantize r with B bits using THIS vector's
    // own [min,max] -> per-vector (scale, offset) stored alongside the codes.
    // Query stays uncompressed (asymmetric distance), same as the paper.
    float lvq_dist(const float* q, const T* x, int bits) const {
        const size_t dim = _dim;
        const int levels = (1 << bits) - 1;
        float lo = std::numeric_limits<float>::max();
        float hi = -std::numeric_limits<float>::max();
        for (size_t d = 0; d < dim; d++) {
            const float r = static_cast<float>(x[d]) - _lvq_mean[d];
            if (r < lo) lo = r;
            if (r > hi) hi = r;
        }
        const float scale = (hi > lo) ? (hi - lo) / static_cast<float>(levels) : 1.0f;
        float acc = 0.0f;
        for (size_t d = 0; d < dim; d++) {
            const float r = static_cast<float>(x[d]) - _lvq_mean[d];
            int code = static_cast<int>(std::lround((r - lo) / scale));
            if (code < 0) code = 0; else if (code > levels) code = levels;
            const float xq = _lvq_mean[d] + lo + static_cast<float>(code) * scale;
            const float diff = q[d] - xq;
            acc += diff * diff;
        }
        return acc;
    }

    // Two-level LVQ-4x8: primary 4-bit on r = x - mu, then an 8-bit level on the
    // primary's residual (its own [min,max]). Reconstruction = mu + level1 + level2.
    float lvq2_dist(const float* q, const T* x) const {
        const size_t dim = _dim;
        const int L1 = 15, L2 = 255;
        float lo = std::numeric_limits<float>::max();
        float hi = -std::numeric_limits<float>::max();
        for (size_t d = 0; d < dim; d++) {
            const float r = static_cast<float>(x[d]) - _lvq_mean[d];
            if (r < lo) lo = r;
            if (r > hi) hi = r;
        }
        const float s1 = (hi > lo) ? (hi - lo) / static_cast<float>(L1) : 1.0f;
        // pass 1: primary residual range
        float rlo = std::numeric_limits<float>::max();
        float rhi = -std::numeric_limits<float>::max();
        for (size_t d = 0; d < dim; d++) {
            const float r = static_cast<float>(x[d]) - _lvq_mean[d];
            int c1 = static_cast<int>(std::lround((r - lo) / s1));
            if (c1 < 0) c1 = 0; else if (c1 > L1) c1 = L1;
            const float res = r - (lo + static_cast<float>(c1) * s1);
            if (res < rlo) rlo = res;
            if (res > rhi) rhi = res;
        }
        const float s2 = (rhi > rlo) ? (rhi - rlo) / static_cast<float>(L2) : 1.0f;
        // pass 2: reconstruct with both levels and score against the exact query
        float acc = 0.0f;
        for (size_t d = 0; d < dim; d++) {
            const float r = static_cast<float>(x[d]) - _lvq_mean[d];
            int c1 = static_cast<int>(std::lround((r - lo) / s1));
            if (c1 < 0) c1 = 0; else if (c1 > L1) c1 = L1;
            const float lvl1 = lo + static_cast<float>(c1) * s1;
            int c2 = static_cast<int>(std::lround((r - lvl1 - rlo) / s2));
            if (c2 < 0) c2 = 0; else if (c2 > L2) c2 = L2;
            const float xq = _lvq_mean[d] + lvl1 + rlo + static_cast<float>(c2) * s2;
            const float diff = q[d] - xq;
            acc += diff * diff;
        }
        return acc;
    }

    // Global mean over the active vectors (strided sample for big N), computed
    // once per harness instance. LVQ centers the dataset before quantizing.
    void ensure_lvq_mean() {
        if (_lvq_ready) return;
        const size_t d = _dim, N = _num_points;
        auto& data_store = _index->get_data_store();
        std::vector<double> csum(d, 0.0);
        const size_t stride = (N > 2000000) ? 101 : 1;
        size_t cnt = 0;
        std::vector<T> xb(d);
        for (size_t loc = 0; loc < N; loc += stride) {
            data_store.get_vector(static_cast<uint32_t>(loc), xb.data());
            for (size_t k = 0; k < d; k++) csum[k] += static_cast<double>(xb[k]);
            cnt++;
        }
        if (cnt == 0) cnt = 1;
        _lvq_mean.assign(d, 0.0f);
        for (size_t k = 0; k < d; k++)
            _lvq_mean[k] = static_cast<float>(csum[k] / static_cast<double>(cnt));
        _lvq_ready = true;
    }

    // Build PQ query LUT: pq_lut[m*256+k] = || q_chunk_m - centroid(m,k) ||^2.
    void build_lut(const float* qf,
                   const std::vector<float>& centroids,
                   const std::vector<uint32_t>& chunk_offsets,
                   const std::vector<uint32_t>& centroid_offsets,
                   float* pq_lut) const {
        for (uint32_t m = 0; m < _M; m++) {
            const uint32_t chunk_start = chunk_offsets[m];
            const uint32_t chunk_size  = chunk_offsets[m + 1] - chunk_offsets[m];
            const float* cent_m = centroids.data() + centroid_offsets[m];
            for (uint32_t k = 0; k < KSUB; k++) {
                float d = 0.0f;
                const float* c = cent_m + static_cast<size_t>(k) * chunk_size;
                for (uint32_t di = 0; di < chunk_size; di++) {
                    float diff = qf[chunk_start + di] - c[di];
                    d += diff * diff;
                }
                pq_lut[static_cast<size_t>(m) * KSUB + k] = d;
            }
        }
    }

    template <typename U>
    static py::array_t<U> to_numpy_2d(const std::vector<U>& v, size_t r, size_t c) {
        py::array_t<U> a(std::vector<py::ssize_t>{static_cast<py::ssize_t>(r),
                                                  static_cast<py::ssize_t>(c)});
        if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(U));
        return a;
    }
    template <typename U>
    static py::array_t<U> to_numpy_1d(const std::vector<U>& v) {
        py::array_t<U> a(std::vector<py::ssize_t>{static_cast<py::ssize_t>(v.size())});
        if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(U));
        return a;
    }

    std::unique_ptr<diskann::Index<T, uint32_t, uint32_t>> _index;
    danns::PQFilter _pq;
    size_t   _dim = 0, _max_points = 0, _num_points = 0;
    uint32_t _L_build = 100, _num_threads = 1, _M = 0, _start = 0;
    std::vector<uint32_t> _tag_of_loc;
    bool _index_loaded = false;

    // LVQ: global mean (data centering), computed lazily by ensure_lvq_mean().
    std::vector<float> _lvq_mean;
    bool _lvq_ready = false;

    // ---- RaBitQ (1 bit/dim, random rotation + sign; unbiased L2 estimator) ----
    // Lazy pre-encode: rotate every active vector once, store packed sign bits +
    // per-vector ||x|| and <o_bar, x_bar> factor. Indexed by loc (== search candidate).
    std::vector<float>   _rbq_R;      // d*d rotation (row-major, orthonormal)
    std::vector<uint8_t> _rbq_code;   // N * ceil(d/8) packed sign bits
    std::vector<float>   _rbq_norm;   // ||x-c|| per loc
    std::vector<float>   _rbq_fac;    // <o_bar, x_bar> per loc (unbias factor)
    std::vector<float>   _rbq_c;      // data centroid (RaBitQ centers before rotation)
    size_t _rbq_bytes = 0;            // ceil(d/8)
    bool _rbq_ready = false;

public:
    // Build the rotation + encode all active vectors (one-time, parallel).
    void ensure_rabitq() {
        if (_rbq_ready) return;
        const size_t d = _dim, nb = (d + 7) / 8, N = _num_points;
        _rbq_bytes = nb;
        // 1) random orthonormal rotation via modified Gram-Schmidt on a seeded Gaussian.
        _rbq_R.assign(d * d, 0.0f);
        {
            std::mt19937 rng(1234567u);
            std::normal_distribution<double> g(0.0, 1.0);
            std::vector<double> Mx(d * d);
            for (auto& v : Mx) v = g(rng);
            for (size_t i = 0; i < d; i++) {
                double* ri = &Mx[i * d];
                for (size_t j = 0; j < i; j++) {
                    const double* rj = &Mx[j * d];
                    double dot = 0; for (size_t k = 0; k < d; k++) dot += ri[k] * rj[k];
                    for (size_t k = 0; k < d; k++) ri[k] -= dot * rj[k];
                }
                double nrm = 0; for (size_t k = 0; k < d; k++) nrm += ri[k] * ri[k];
                nrm = std::sqrt(nrm); if (nrm < 1e-12) nrm = 1.0;
                for (size_t k = 0; k < d; k++) ri[k] /= nrm;
            }
            for (size_t i = 0; i < d * d; i++) _rbq_R[i] = static_cast<float>(Mx[i]);
        }
        auto& data_store = _index->get_data_store();
        const int nth = (_num_threads > 0) ? static_cast<int>(_num_threads) : omp_get_max_threads();
        // centroid c = mean of active vectors (strided sample; RaBitQ centers the data).
        _rbq_c.assign(d, 0.0f);
        {
            std::vector<double> csum(d, 0.0);
            const size_t stride = (N > 2000000) ? 101 : 1;
            size_t cnt = 0;
            std::vector<T> xb(d);
            for (size_t loc = 0; loc < N; loc += stride) {
                data_store.get_vector(static_cast<uint32_t>(loc), xb.data());
                for (size_t k = 0; k < d; k++) csum[k] += static_cast<double>(xb[k]);
                cnt++;
            }
            for (size_t k = 0; k < d; k++) _rbq_c[k] = static_cast<float>(csum[k] / static_cast<double>(cnt));
        }
        // 2) encode every active vector (index by loc): rotate (x - c), sign bits.
        _rbq_code.assign(N * nb, 0);
        _rbq_norm.assign(N, 0.0f);
        _rbq_fac.assign(N, 1.0f);
        const float invsqd = 1.0f / std::sqrt(static_cast<float>(d));
        #pragma omp parallel num_threads(nth)
        {
            std::vector<T> xbuf(d);
            std::vector<float> xc(d);
            #pragma omp for schedule(static)
            for (long long loc = 0; loc < static_cast<long long>(N); loc++) {
                data_store.get_vector(static_cast<uint32_t>(loc), xbuf.data());
                for (size_t k = 0; k < d; k++) xc[k] = static_cast<float>(xbuf[k]) - _rbq_c[k];
                float norm2 = 0.0f, absum = 0.0f;
                uint8_t* code = &_rbq_code[static_cast<size_t>(loc) * nb];
                for (size_t i = 0; i < d; i++) {
                    const float* Ri = &_rbq_R[i * d];
                    float acc = 0.0f;
                    for (size_t k = 0; k < d; k++) acc += Ri[k] * xc[k];
                    norm2 += acc * acc; absum += std::fabs(acc);
                    if (acc >= 0.0f) code[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
                }
                float norm = std::sqrt(norm2);
                _rbq_norm[static_cast<size_t>(loc)] = norm;
                _rbq_fac[static_cast<size_t>(loc)] = (norm > 1e-12f) ? (absum * invsqd / norm) : 1.0f;
            }
        }
        _rbq_ready = true;
    }

private:
    // RaBitQ estimated squared-L2 for vector at loc, given rotated query qr and ||q||^2.
    inline float rabitq_dist(uint32_t loc, const float* qr, float qnorm2) const {
        const size_t d = _dim, nb = _rbq_bytes;
        const uint8_t* code = &_rbq_code[static_cast<size_t>(loc) * nb];
        float qx = 0.0f;
        for (size_t i = 0; i < d; i++) {
            float s = ((code[i >> 3] >> (i & 7)) & 1u) ? 1.0f : -1.0f;
            qx += qr[i] * s;
        }
        qx *= 1.0f / std::sqrt(static_cast<float>(d));      // <q_r, x_bar>
        const float norm = _rbq_norm[static_cast<size_t>(loc)];
        const float fac = _rbq_fac[static_cast<size_t>(loc)];
        const float ip = (fac > 1e-12f) ? (norm * qx / fac) : 0.0f;   // <q,x> estimate
        return qnorm2 + norm * norm - 2.0f * ip;
    }
};

template <typename T>
void bind_harness(py::module_& m, const char* name) {
    // module_local: the same classes are registered in both `pq_search_bindings`
    // (legacy) and `single_engine` — keeps double registration legal per process.
    py::class_<PQSearchHarness<T>>(m, name, py::module_local())
        .def(py::init<size_t, size_t, uint32_t, uint32_t, float, uint32_t>(),
             py::arg("dim"), py::arg("max_points"), py::arg("R") = 64,
             py::arg("L_build") = 100, py::arg("alpha_build") = 1.2f, py::arg("num_threads") = 0)
        .def("load_index", &PQSearchHarness<T>::load_index,
             py::arg("graph_file"), py::arg("base_file"), py::arg("tags_file"),
             py::arg("search_L") = 0)
        .def("load_topology", &PQSearchHarness<T>::load_topology,
             py::arg("graph_file"), py::arg("tags_file"))
        .def("load_pq", &PQSearchHarness<T>::load_pq,
             py::arg("codebook_path"), py::arg("codes_path"), py::arg("max_capacity"),
             py::arg("alpha"))
        .def("set_alpha", &PQSearchHarness<T>::set_alpha, py::arg("alpha"))
        .def("alpha", &PQSearchHarness<T>::alpha)
        .def("M", &PQSearchHarness<T>::M)
        .def("num_points", &PQSearchHarness<T>::num_points)
        .def("search_batch", &PQSearchHarness<T>::search_batch,
             py::arg("queries"), py::arg("K") = 10, py::arg("L_search") = 100,
             py::arg("conv_window") = 3, py::arg("conv_threshold_pct") = 1.0f,
             py::arg("filter_on") = true, py::arg("record_drops") = false,
             py::arg("max_drops") = 2000000,
             py::arg("partition_of") = py::none(), py::arg("coordinator") = py::none(),
             py::arg("search_dist") = "exact", py::arg("rerank_scope") = "full",
             py::arg("trace_max_queries") = 0, py::arg("rerank_gate_alpha") = -1.0f,
             py::arg("batch_size") = 0, py::arg("opq_rotation") = py::none())
        .def("graph_stats", &PQSearchHarness<T>::graph_stats,
             py::arg("part"), py::arg("by_tag") = true);
}

// Registration split so these classes can be exposed by both modules
// (legacy `pq_search_bindings` + unified `single_engine`).
void register_single_search(py::module_& m) {
    bind_harness<float>(m,   "PQSearchHarnessFloat");
    bind_harness<uint8_t>(m, "PQSearchHarnessUInt8");
    bind_harness<int8_t>(m,  "PQSearchHarnessInt8");
    // single-engine identity aliases (same classes, clearer names going forward)
    m.attr("SingleNodeSearchFloat") = m.attr("PQSearchHarnessFloat");
    m.attr("SingleNodeSearchUInt8") = m.attr("PQSearchHarnessUInt8");
    m.attr("SingleNodeSearchInt8")  = m.attr("PQSearchHarnessInt8");
}

// Legacy module — kept so existing drivers (run_m11/run_tierB, deployed run_q1)
// keep working. New code should import `single_engine`.
PYBIND11_MODULE(pq_search_bindings, m) {
    m.doc() = "Single-node Vamana + PQ false-filtering measurement harness (M11).";
    register_single_search(m);
}
