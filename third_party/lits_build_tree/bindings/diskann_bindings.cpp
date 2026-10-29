// DiskANN Python bindings using pybind11
// Build & Update + Search functionality

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <thread>
#include <atomic>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <type_traits>

#include "index.h"
#include "utils.h"
#include "parameters.h"
#include "percentile_stats.h"

namespace py = pybind11;

// Wrapper class for DiskANN Index with simplified Python interface
template <typename T>
class DiskANNIndex {
public:
    DiskANNIndex(
        size_t dim,
        size_t max_points,
        uint32_t R = 64,           // graph degree
        uint32_t L = 100,          // search list size during build
        float alpha = 1.2f,        // distance threshold for pruning
        uint32_t num_threads = 0,  // 0 = auto
        bool dynamic = true        // enable dynamic updates
    ) : _dim(dim), _max_points(max_points), _R(R), _L(L), _alpha(alpha) {

        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }
        _num_threads = num_threads;

        // Create index parameters
        auto index_params = std::make_shared<diskann::IndexWriteParameters>(
            diskann::IndexWriteParametersBuilder(L, R)
                .with_alpha(alpha)
                .with_saturate_graph(false)
                .with_num_threads(num_threads)
                .build()
        );

        auto search_params = std::make_shared<diskann::IndexSearchParams>(
            diskann::IndexSearchParams(L, num_threads)
        );

        // Create the index
        // For dynamic index, we need 1 frozen point
        size_t num_frozen_pts = dynamic ? 1 : 0;
        _index = std::make_unique<diskann::Index<T, uint32_t, uint32_t>>(
            diskann::Metric::L2,
            dim,
            max_points,
            index_params,
            search_params,
            num_frozen_pts,
            dynamic,    // dynamic_index
            true,       // enable_tags
            false,      // concurrent_consolidate
            false,      // pq_dist_build
            0,          // num_pq_chunks
            false,      // use_opq
            false       // filtered_index
        );

        _dynamic = dynamic;
        _built = false;
        _next_tag = 0;
    }

    // Compute per-dim mean of the given data and install it as the dynamic
    // index's frozen start point. Replaces the previous random-unit-norm init
    // so search entry reflects the initial active distribution.
    void set_mean_frozen_point(const T* data_ptr, size_t num_points) {
        if (!_dynamic || num_points == 0) return;
        std::vector<double> acc(_dim, 0.0);
        for (size_t i = 0; i < num_points; i++) {
            const T* row = data_ptr + i * _dim;
            for (size_t d = 0; d < _dim; d++) acc[d] += static_cast<double>(row[d]);
        }
        std::vector<T> mean_vec(_dim);
        const double inv_n = 1.0 / static_cast<double>(num_points);
        for (size_t d = 0; d < _dim; d++) {
            double m = acc[d] * inv_n;
            if constexpr (std::is_integral_v<T>) {
                m = std::round(m);
                if (m > static_cast<double>(std::numeric_limits<T>::max())) m = std::numeric_limits<T>::max();
                if (m < static_cast<double>(std::numeric_limits<T>::min())) m = std::numeric_limits<T>::min();
            }
            mean_vec[d] = static_cast<T>(m);
        }
        _index->set_start_points(mean_vec.data(), _dim);
    }

    // Build index from numpy array
    void build(py::array_t<T> data) {
        auto buf = data.request();

        if (buf.ndim != 2) {
            throw std::runtime_error("Data must be 2D array");
        }

        size_t num_points = buf.shape[0];
        size_t dim = buf.shape[1];

        if (dim != _dim) {
            throw std::runtime_error("Dimension mismatch: expected " +
                                     std::to_string(_dim) + ", got " + std::to_string(dim));
        }

        if (num_points > _max_points) {
            throw std::runtime_error("Number of points exceeds max_points");
        }

        // Generate tags
        std::vector<uint32_t> tags(num_points);
        for (size_t i = 0; i < num_points; i++) {
            tags[i] = static_cast<uint32_t>(i + 1);  // Tags start from 1
        }
        _next_tag = static_cast<uint32_t>(num_points + 1);

        // Build the index
        T* data_ptr = static_cast<T*>(buf.ptr);

        if (_dynamic) {
            // Frozen point = mean of initial active vectors
            set_mean_frozen_point(data_ptr, num_points);
        }

        _index->build(data_ptr, num_points, tags);
        _built = true;
        _num_points = num_points;
    }

    // Build index from file
    void build_from_file(const std::string& filename, size_t num_points = 0) {
        if (num_points == 0) {
            // Read from file header
            size_t npts, ndim;
            diskann::get_bin_metadata(filename, npts, ndim);
            num_points = npts;
        }

        // Generate tags
        std::vector<uint32_t> tags(num_points);
        for (size_t i = 0; i < num_points; i++) {
            tags[i] = static_cast<uint32_t>(i + 1);
        }
        _next_tag = static_cast<uint32_t>(num_points + 1);

        if (_dynamic) {
            // Frozen point = mean of initial active vectors.
            // build_from_file: briefly read the first num_points vectors to compute mean.
            std::ifstream in(filename, std::ios::binary);
            if (!in) throw std::runtime_error("Failed to open " + filename);
            in.seekg(8);  // skip [npts:uint32, dim:uint32] header
            std::vector<T> first_block(num_points * _dim);
            in.read(reinterpret_cast<char*>(first_block.data()),
                    static_cast<std::streamsize>(num_points * _dim * sizeof(T)));
            if (!in) throw std::runtime_error("Short read while computing mean frozen point");
            set_mean_frozen_point(first_block.data(), num_points);
        }

        _index->build(filename.c_str(), num_points, tags);
        _built = true;
        _num_points = num_points;
    }

    // Build index with custom tags (for sharded graphs where tag = global_id + 1)
    void build_with_tags(py::array_t<T> data, py::array_t<uint32_t> custom_tags) {
        auto data_buf = data.request();
        auto tags_buf = custom_tags.request();

        if (data_buf.ndim != 2) {
            throw std::runtime_error("Data must be 2D array");
        }

        size_t num_points = data_buf.shape[0];
        size_t dim = data_buf.shape[1];

        if (dim != _dim) {
            throw std::runtime_error("Dimension mismatch: expected " +
                                     std::to_string(_dim) + ", got " + std::to_string(dim));
        }

        if (num_points > _max_points) {
            throw std::runtime_error("Number of points exceeds max_points");
        }

        if (tags_buf.size != static_cast<ssize_t>(num_points)) {
            throw std::runtime_error("Tags array size must match number of points");
        }

        // Copy custom tags
        uint32_t* tags_ptr = static_cast<uint32_t*>(tags_buf.ptr);
        std::vector<uint32_t> tags(tags_ptr, tags_ptr + num_points);

        // Find max tag for _next_tag
        uint32_t max_tag = 0;
        for (auto t : tags) {
            if (t == 0) {
                throw std::runtime_error("Tag 0 is reserved, tags must be >= 1");
            }
            if (t > max_tag) max_tag = t;
        }
        _next_tag = max_tag + 1;

        // Build the index
        T* data_ptr = static_cast<T*>(data_buf.ptr);

        if (_dynamic) {
            // Frozen point = mean of initial active vectors
            set_mean_frozen_point(data_ptr, num_points);
        }

        _index->build(data_ptr, num_points, tags);
        _built = true;
        _num_points = num_points;
    }

    // Insert single point, returns tag
    uint32_t insert(py::array_t<T> point) {
        if (!_dynamic) {
            throw std::runtime_error("Index is not dynamic, cannot insert");
        }

        auto buf = point.request();
        if (buf.size != static_cast<ssize_t>(_dim)) {
            throw std::runtime_error("Point dimension mismatch");
        }

        T* ptr = static_cast<T*>(buf.ptr);
        uint32_t tag = _next_tag++;

        if (_lits_balanced) {
            _index->precharge_lits_balanced(ptr, 1, _dim);
        }

        int result = _index->insert_point(ptr, tag);
        if (result != 0) {
            throw std::runtime_error("Insert failed with code: " + std::to_string(result));
        }

        _num_points++;
        return tag;
    }

    // Insert batch of points, returns list of tags
    std::vector<uint32_t> insert_batch(py::array_t<T> data) {
        if (!_dynamic) {
            throw std::runtime_error("Index is not dynamic, cannot insert");
        }

        auto buf = data.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Data must be 2D array");
        }

        size_t num_points = buf.shape[0];
        size_t dim = buf.shape[1];

        if (dim != _dim) {
            throw std::runtime_error("Dimension mismatch");
        }

        T* data_ptr = static_cast<T*>(buf.ptr);

        // LITS balanced alloc: pre-charge this batch's nearest-centroid homes
        // into the per-cluster counters (B-style accounting; the per-point
        // placement then only moves counters on redirects).
        if (_lits_balanced) {
            _index->precharge_lits_balanced(data_ptr, num_points, dim);
        }

        // Pre-allocate tags so the parallel loop has no atomic on _next_tag.
        uint32_t start_tag = _next_tag;
        _next_tag += static_cast<uint32_t>(num_points);
        std::vector<uint32_t> tags(num_points);
        for (size_t i = 0; i < num_points; i++) tags[i] = start_tag + static_cast<uint32_t>(i);

        std::atomic<int> first_err(0);
        std::atomic<size_t> err_index(0);
        #pragma omp parallel for num_threads(static_cast<int>(_num_threads)) schedule(dynamic, 64)
        for (size_t i = 0; i < num_points; i++) {
            if (first_err.load(std::memory_order_relaxed) != 0) continue;
            int r = _index->insert_point(data_ptr + i * dim, tags[i]);
            if (r != 0) {
                int expected = 0;
                if (first_err.compare_exchange_strong(expected, r)) {
                    err_index.store(i);
                }
            }
        }
        if (first_err.load() != 0) {
            throw std::runtime_error("Insert failed at index " + std::to_string(err_index.load())
                + " (code " + std::to_string(first_err.load()) + ")");
        }
        _num_points += num_points;
        return tags;
    }

    // Insert single point with specific tag (for sharded graphs)
    void insert_with_tag(py::array_t<T> point, uint32_t tag) {
        if (!_dynamic) {
            throw std::runtime_error("Index is not dynamic, cannot insert");
        }

        if (tag == 0) {
            throw std::runtime_error("Tag 0 is reserved, tags must be >= 1");
        }

        auto buf = point.request();
        if (buf.size != static_cast<ssize_t>(_dim)) {
            throw std::runtime_error("Point dimension mismatch");
        }

        T* ptr = static_cast<T*>(buf.ptr);
        int result = _index->insert_point(ptr, tag);
        if (result != 0) {
            throw std::runtime_error("Insert failed with code: " + std::to_string(result));
        }

        _num_points++;
        if (tag >= _next_tag) {
            _next_tag = tag + 1;
        }
    }

    // Insert batch of points with specific tags (for sharded graphs)
    void insert_batch_with_tags(py::array_t<T> data, py::array_t<uint32_t> tags) {
        if (!_dynamic) {
            throw std::runtime_error("Index is not dynamic, cannot insert");
        }

        auto data_buf = data.request();
        auto tags_buf = tags.request();

        if (data_buf.ndim != 2) {
            throw std::runtime_error("Data must be 2D array");
        }

        size_t num_points = data_buf.shape[0];
        size_t dim = data_buf.shape[1];

        if (dim != _dim) {
            throw std::runtime_error("Dimension mismatch");
        }

        if (tags_buf.size != static_cast<ssize_t>(num_points)) {
            throw std::runtime_error("Tags array size must match number of points");
        }

        // Custom-tag inserts bypass the sequential-tag accounting the LITS
        // balanced allocator assumes (precharge would drift for tags inside
        // the preloaded window). Not a supported combination.
        if (_lits_balanced) {
            throw std::runtime_error(
                "insert_batch_with_tags is not supported with LITS balanced alloc; "
                "use insert_batch (sequential tags)");
        }

        T* data_ptr = static_cast<T*>(data_buf.ptr);
        uint32_t* tags_ptr = static_cast<uint32_t*>(tags_buf.ptr);

        // Validate and compute next_tag serially (cheap vs insert_point cost)
        uint32_t max_tag = 0;
        for (size_t i = 0; i < num_points; i++) {
            if (tags_ptr[i] == 0) {
                throw std::runtime_error("Tag 0 is reserved at index " + std::to_string(i));
            }
            if (tags_ptr[i] > max_tag) max_tag = tags_ptr[i];
        }

        std::atomic<int> first_err(0);
        std::atomic<size_t> err_index(0);
        #pragma omp parallel for num_threads(static_cast<int>(_num_threads)) schedule(dynamic, 64)
        for (size_t i = 0; i < num_points; i++) {
            if (first_err.load(std::memory_order_relaxed) != 0) continue;
            int r = _index->insert_point(data_ptr + i * dim, tags_ptr[i]);
            if (r != 0) {
                int expected = 0;
                if (first_err.compare_exchange_strong(expected, r)) {
                    err_index.store(i);
                }
            }
        }
        if (first_err.load() != 0) {
            throw std::runtime_error("Insert failed at index " + std::to_string(err_index.load())
                + " with tag " + std::to_string(tags_ptr[err_index.load()])
                + " (code " + std::to_string(first_err.load()) + ")");
        }
        _num_points += num_points;
        if (max_tag >= _next_tag) _next_tag = max_tag + 1;
    }

    // Delete point by tag
    void delete_point(uint32_t tag) {
        if (!_dynamic) {
            throw std::runtime_error("Index is not dynamic, cannot delete");
        }

        _index->enable_delete();
        int result = _index->lazy_delete(tag);
        if (result != 0) {
            throw std::runtime_error("Delete failed, tag not found: " + std::to_string(tag));
        }
        _pending_deletes++;
        if (_lits_balanced) {
            std::vector<uint32_t> one{tag};
            _index->on_tags_deleted_for_balanced_alloc(one);
        }
    }

    // Delete batch of points
    void delete_batch(const std::vector<uint32_t>& tags) {
        if (!_dynamic) {
            throw std::runtime_error("Index is not dynamic, cannot delete");
        }

        _index->enable_delete();
        std::vector<uint32_t> failed;
        _index->lazy_delete(tags, failed);

        if (!failed.empty()) {
            throw std::runtime_error("Delete failed for " + std::to_string(failed.size()) + " tags");
        }
        _pending_deletes += tags.size();
        // LITS balanced alloc: release the deleted tags' cluster counts (lazy
        // delete time, matching the diskann-lits batch loop accounting).
        if (_lits_balanced) {
            _index->on_tags_deleted_for_balanced_alloc(tags);
        }
    }

    // Consolidate pending deletes
    size_t consolidate() {
        if (_pending_deletes == 0) {
            return _num_points;
        }

        auto params = diskann::IndexWriteParametersBuilder(_L, _R)
            .with_alpha(_alpha)
            .with_num_threads(_num_threads)
            .build();

        auto report = _index->consolidate_deletes(params);
        _num_points = report._active_points;
        _pending_deletes = 0;

        return _num_points;
    }

    // Search for K nearest neighbors
    std::tuple<py::array_t<uint32_t>, py::array_t<float>> search(
        py::array_t<T> query,
        size_t K,
        uint32_t L = 0
    ) {
        if (!_built) {
            throw std::runtime_error("Index not built yet");
        }

        if (L == 0) L = _L;
        if (L < K) L = static_cast<uint32_t>(K);

        auto buf = query.request();
        size_t num_queries = 1;
        size_t dim = _dim;

        if (buf.ndim == 1) {
            if (buf.size != static_cast<ssize_t>(_dim)) {
                throw std::runtime_error("Query dimension mismatch");
            }
        } else if (buf.ndim == 2) {
            num_queries = buf.shape[0];
            dim = buf.shape[1];
            if (dim != _dim) {
                throw std::runtime_error("Query dimension mismatch");
            }
        } else {
            throw std::runtime_error("Query must be 1D or 2D array");
        }

        // Allocate output arrays
        std::vector<ssize_t> shape = {static_cast<ssize_t>(num_queries), static_cast<ssize_t>(K)};
        auto indices = py::array_t<uint32_t>(shape);
        auto distances = py::array_t<float>(shape);

        T* query_ptr = static_cast<T*>(buf.ptr);
        uint32_t* indices_ptr = static_cast<uint32_t*>(indices.request().ptr);
        float* distances_ptr = static_cast<float*>(distances.request().ptr);

        // Perform search (returns locations, not tags)
        #pragma omp parallel for num_threads(_num_threads)
        for (size_t i = 0; i < num_queries; i++) {
            _index->search(
                query_ptr + i * _dim,
                K,
                L,
                indices_ptr + i * K,
                distances_ptr + i * K
            );
            // Convert locations to tags
            for (size_t j = 0; j < K; j++) {
                indices_ptr[i * K + j] = static_cast<uint32_t>(
                    _index->get_tag_by_location(indices_ptr[i * K + j]));
            }
        }

        return std::make_tuple(indices, distances);
    }

    // Save index to file. When save_data_file=false, the <filename>.data blob
    // (raw vectors) is skipped — combine with load_graph_only() + base file to
    // restore. Useful for checkpointing at scale without duplicating vectors.
    // When save_data_file=false, the frozen point vector is persisted to a small
    // auxiliary file <filename>.frozen so load_graph_only() can restore it (the
    // frozen point is not part of the base file and would otherwise be lost).
    void save(const std::string& filename, bool compact_before_save = true,
              bool save_data_file = true) {
        _index->save(filename.c_str(), compact_before_save, save_data_file);
        // LITS balanced alloc: dump the tag-indexed label array next to the
        // checkpoint (same format as the input label file; consumed for
        // partitioning and for --resume).
        if (_lits_balanced) {
            _index->save_lits_cluster_labels(filename + ".labels");
        }
        if (!save_data_file && _dynamic) {
            // Persist frozen point vector (single D-dim row)
            uint32_t fp_loc = static_cast<uint32_t>(_max_points);
            std::vector<T> fp_vec(_dim);
            _index->get_data_store().get_vector(fp_loc, fp_vec.data());
            std::string fp_path = filename + ".frozen";
            std::ofstream out(fp_path, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("Failed to write " + fp_path);
            uint32_t header[2] = { 1u, static_cast<uint32_t>(_dim) };
            out.write(reinterpret_cast<const char*>(header), sizeof(header));
            out.write(reinterpret_cast<const char*>(fp_vec.data()),
                      static_cast<std::streamsize>(_dim * sizeof(T)));
            if (!out) throw std::runtime_error("Short write to " + fp_path);
        }
    }

    // Load index from file
    void load(const std::string& filename, uint32_t search_L = 0) {
        if (search_L == 0) search_L = _L;
        _index->load(filename.c_str(), _num_threads, search_L);
        _built = true;
        _num_points = _index->get_num_points();
        _next_tag = static_cast<uint32_t>(_num_points + 1);
    }

    // Load index from graph + tags + base file (without .data file)
    // Useful when .data was not saved to save disk space. If a <graph_file>.frozen
    // sidecar exists, the frozen point vector is restored from it (required for
    // dynamic indices that skipped .data).
    void load_graph_only(const std::string& graph_file, const std::string& base_file,
                         const std::string& tags_file, uint32_t search_L = 0) {
        if (search_L == 0) search_L = _L;
        _index->load_graph_only(graph_file.c_str(), base_file, tags_file, _num_threads, search_L);
        _built = true;
        _num_points = _index->get_num_points();

        // Recover _next_tag = max_existing_tag + 1 by scanning tags file.
        // load_graph_only preserves tag values (not sequential), so using
        // num_points + 1 would collide with tags from past inserts.
        {
            std::ifstream tin(tags_file, std::ios::binary);
            if (!tin) throw std::runtime_error("Failed to open tags file: " + tags_file);
            uint32_t thdr[2] = { 0u, 0u };
            tin.read(reinterpret_cast<char*>(thdr), sizeof(thdr));
            if (!tin) throw std::runtime_error("Short read on tags header: " + tags_file);
            std::vector<uint32_t> tag_buf(thdr[0]);
            tin.read(reinterpret_cast<char*>(tag_buf.data()),
                     static_cast<std::streamsize>(thdr[0] * sizeof(uint32_t)));
            if (!tin) throw std::runtime_error("Short read on tags body: " + tags_file);
            uint32_t max_tag = 0;
            for (auto t : tag_buf) if (t > max_tag) max_tag = t;
            _next_tag = max_tag + 1;
        }

        std::string fp_path = graph_file + ".frozen";
        std::ifstream in(fp_path, std::ios::binary);
        if (in) {
            uint32_t header[2] = { 0u, 0u };
            in.read(reinterpret_cast<char*>(header), sizeof(header));
            if (!in || header[0] != 1u || header[1] != static_cast<uint32_t>(_dim)) {
                throw std::runtime_error(".frozen file has unexpected header: " + fp_path);
            }
            std::vector<T> fp_vec(_dim);
            in.read(reinterpret_cast<char*>(fp_vec.data()),
                    static_cast<std::streamsize>(_dim * sizeof(T)));
            if (!in) throw std::runtime_error("Short read from " + fp_path);
            uint32_t fp_loc = static_cast<uint32_t>(_max_points);
            _index->get_data_store().set_vector(fp_loc, fp_vec.data());
        }
    }

    // Getters
    size_t get_num_points() const { return _num_points; }
    size_t get_max_points() const { return _max_points; }
    size_t get_dim() const { return _dim; }
    bool is_built() const { return _built; }
    size_t get_pending_deletes() const { return _pending_deletes; }

    // Get frozen point (start node) vector and neighbor list
    py::dict get_frozen_point_info() {
        py::dict result;
        uint32_t start = static_cast<uint32_t>(_max_points);
        result["location"] = start;

        // Vector: read from data store
        auto vec = py::array_t<T>(_dim);
        T* vec_ptr = static_cast<T*>(vec.mutable_data());
        _index->get_data_store().get_vector(start, vec_ptr);
        result["vector"] = vec;

        // Neighbors
        auto& nbrs = _index->get_graph_store()->get_neighbours(start);
        auto nbr_arr = py::array_t<uint32_t>(nbrs.size());
        auto nbr_buf = nbr_arr.mutable_unchecked<1>();
        for (size_t i = 0; i < nbrs.size(); i++) nbr_buf(i) = nbrs[i];
        result["neighbors"] = nbr_arr;
        result["degree"] = static_cast<uint32_t>(nbrs.size());

        return result;
    }

    // Print status
    void print_status() {
        _index->print_status();
    }

    // ===============================================
    // === Latency profiling methods with stats ===
    // ===============================================

    // Search with detailed latency breakdown
    std::tuple<py::array_t<uint32_t>, py::array_t<float>, py::dict> search_with_stats(
        py::array_t<T> query,
        size_t K,
        uint32_t L = 0
    ) {
        if (!_built) {
            throw std::runtime_error("Index not built yet");
        }

        if (L == 0) L = _L;
        if (L < K) L = static_cast<uint32_t>(K);

        auto buf = query.request();
        if (buf.size != static_cast<ssize_t>(_dim)) {
            throw std::runtime_error("Query dimension mismatch (single query only)");
        }

        // Allocate output arrays
        auto indices = py::array_t<uint32_t>(K);
        auto distances = py::array_t<float>(K);

        T* query_ptr = static_cast<T*>(buf.ptr);
        uint32_t* indices_ptr = static_cast<uint32_t*>(indices.request().ptr);
        float* distances_ptr = static_cast<float*>(distances.request().ptr);

        // Perform search with stats (returns locations, not tags)
        diskann::QueryStats stats;
        _index->search_with_stats(query_ptr, K, L, indices_ptr, distances_ptr, stats);

        // Convert locations to tags for user-facing API
        for (size_t i = 0; i < K; i++) {
            indices_ptr[i] = static_cast<uint32_t>(_index->get_tag_by_location(indices_ptr[i]));
        }

        // Convert stats to dict
        py::dict stats_dict;
        stats_dict["total_us"] = stats.total_us;
        stats_dict["dist_us"] = stats.dist_us;
        stats_dict["fetch_us"] = stats.fetch_us;
        stats_dict["queue_us"] = stats.queue_us;
        stats_dict["visited_us"] = stats.visited_us;
        stats_dict["n_hops"] = stats.n_hops;
        stats_dict["n_cmps"] = stats.n_cmps;

        return std::make_tuple(indices, distances, stats_dict);
    }

    // Search with partition-aware latency breakdown (for distributed latency profiling)
    std::tuple<py::array_t<uint32_t>, py::array_t<float>, py::dict> search_with_partition_stats(
        py::array_t<T> query,
        size_t K,
        uint32_t L,
        py::array_t<uint32_t> partition_ids,
        uint32_t query_partition
    ) {
        if (!_built) {
            throw std::runtime_error("Index not built yet");
        }

        if (L == 0) L = _L;
        if (L < K) L = static_cast<uint32_t>(K);

        auto buf = query.request();
        if (buf.size != static_cast<ssize_t>(_dim)) {
            throw std::runtime_error("Query dimension mismatch (single query only)");
        }

        auto partition_buf = partition_ids.request();
        if (partition_buf.ndim != 1) {
            throw std::runtime_error("partition_ids must be 1D array");
        }

        // Allocate output arrays
        auto indices = py::array_t<uint32_t>(K);
        auto distances = py::array_t<float>(K);

        T* query_ptr = static_cast<T*>(buf.ptr);
        uint32_t* indices_ptr = static_cast<uint32_t*>(indices.request().ptr);
        float* distances_ptr = static_cast<float*>(distances.request().ptr);
        const uint32_t* partition_ids_ptr = static_cast<const uint32_t*>(partition_buf.ptr);

        // Perform search with partition stats
        diskann::QueryStats stats;
        _index->search_with_partition_stats(query_ptr, K, L, indices_ptr, distances_ptr,
                                            partition_ids_ptr, query_partition, stats);

        // Convert stats to dict
        py::dict stats_dict;
        stats_dict["total_us"] = stats.total_us;
        stats_dict["dist_us"] = stats.dist_us;
        stats_dict["fetch_us"] = stats.fetch_us;
        stats_dict["queue_us"] = stats.queue_us;
        stats_dict["visited_us"] = stats.visited_us;
        stats_dict["n_hops"] = stats.n_hops;
        stats_dict["n_cmps"] = stats.n_cmps;
        // Partition-aware stats
        stats_dict["dist_local_us"] = stats.dist_local_us;
        stats_dict["dist_remote_us"] = stats.dist_remote_us;
        stats_dict["fetch_local_us"] = stats.fetch_local_us;
        stats_dict["fetch_remote_us"] = stats.fetch_remote_us;
        stats_dict["local_vector_accesses"] = stats.local_vector_accesses;
        stats_dict["remote_vector_accesses"] = stats.remote_vector_accesses;
        stats_dict["local_neighbor_accesses"] = stats.local_neighbor_accesses;
        stats_dict["remote_neighbor_accesses"] = stats.remote_neighbor_accesses;
        stats_dict["rdma_round_trips"] = stats.rdma_round_trips;

        return std::make_tuple(indices, distances, stats_dict);
    }

    // Multi-threaded throughput benchmark (local-only, no RDMA)
    py::dict search_throughput(
        py::array_t<T> queries_array,
        uint32_t K,
        uint32_t L,
        uint32_t num_threads,
        double duration_sec,
        double warmup_sec
    ) {
        if (!_built) {
            throw std::runtime_error("Index not built yet");
        }

        auto queries_buf = queries_array.request();
        if (queries_buf.ndim != 2 || queries_buf.shape[1] != static_cast<ssize_t>(_dim)) {
            throw std::runtime_error("queries must be 2D array with matching dim");
        }
        const T* all_queries = static_cast<const T*>(queries_buf.ptr);
        uint32_t num_queries = static_cast<uint32_t>(queries_buf.shape[0]);
        uint32_t query_dim = static_cast<uint32_t>(queries_buf.shape[1]);

        if (L < K) L = K;

        // Shared stop flag and counters
        std::atomic<bool> warmup_done{false};
        std::atomic<bool> stop{false};
        std::atomic<uint64_t> total_queries_done{0};
        std::atomic<uint64_t> warmup_queries_done{0};

        // Per-thread results
        struct ThreadResult {
            uint64_t queries_done = 0;
            std::vector<double> latencies_us;
        };
        std::vector<ThreadResult> thread_results(num_threads);

        // Release GIL for C++ threading
        py::gil_scoped_release release;

        // Launch worker threads
        std::vector<std::thread> workers;
        for (uint32_t t = 0; t < num_threads; t++) {
            workers.emplace_back([&, t]() {
                auto& result = thread_results[t];

                // Per-thread output buffers
                std::vector<uint32_t> indices(K);
                std::vector<float> distances(K);

                uint32_t query_idx = t;  // Start at different offsets

                while (!stop.load(std::memory_order_relaxed)) {
                    const T* query_ptr = all_queries +
                        static_cast<size_t>(query_idx % num_queries) * query_dim;

                    auto t0 = std::chrono::high_resolution_clock::now();

                    diskann::QueryStats stats;
                    _index->search_with_stats(
                        query_ptr, K, L, indices.data(), distances.data(), stats);

                    auto t1 = std::chrono::high_resolution_clock::now();
                    double latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        t1 - t0).count();

                    if (warmup_done.load(std::memory_order_relaxed)) {
                        result.queries_done++;
                        result.latencies_us.push_back(latency_us);
                        total_queries_done.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        warmup_queries_done.fetch_add(1, std::memory_order_relaxed);
                    }

                    query_idx += num_threads;
                }
            });
        }

        // Warmup phase
        std::this_thread::sleep_for(std::chrono::duration<double>(warmup_sec));
        warmup_done.store(true, std::memory_order_release);

        // Measurement phase
        auto measure_start = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::duration<double>(duration_sec));
        stop.store(true, std::memory_order_release);

        // Join all threads
        for (auto& w : workers) w.join();

        auto measure_end = std::chrono::steady_clock::now();
        double actual_duration = std::chrono::duration<double>(
            measure_end - measure_start).count();

        // Aggregate results
        py::gil_scoped_acquire acquire;

        uint64_t total_q = total_queries_done.load();
        double qps = total_q / actual_duration;

        // Collect all latencies
        std::vector<double> all_latencies;
        for (auto& tr : thread_results) {
            all_latencies.insert(all_latencies.end(),
                tr.latencies_us.begin(), tr.latencies_us.end());
        }
        std::sort(all_latencies.begin(), all_latencies.end());

        auto pctile = [&](double p) -> double {
            if (all_latencies.empty()) return 0;
            size_t idx = static_cast<size_t>(p * (all_latencies.size() - 1));
            return all_latencies[idx];
        };

        double mean_lat = 0;
        if (!all_latencies.empty()) {
            mean_lat = std::accumulate(all_latencies.begin(), all_latencies.end(), 0.0)
                       / all_latencies.size();
        }

        py::dict result;
        result["qps"] = qps;
        result["total_queries"] = total_q;
        result["warmup_queries"] = warmup_queries_done.load();
        result["duration_sec"] = actual_duration;
        result["latency_mean_us"] = mean_lat;
        result["latency_p50_us"] = pctile(0.50);
        result["latency_p90_us"] = pctile(0.90);
        result["latency_p99_us"] = pctile(0.99);
        result["latency_p999_us"] = pctile(0.999);
        result["latency_min_us"] = all_latencies.empty() ? 0.0 : all_latencies.front();
        result["latency_max_us"] = all_latencies.empty() ? 0.0 : all_latencies.back();

        // Per-thread breakdown
        py::list per_thread;
        for (uint32_t t = 0; t < num_threads; t++) {
            py::dict td;
            td["thread_id"] = t;
            td["queries_done"] = thread_results[t].queries_done;
            td["qps"] = thread_results[t].queries_done / actual_duration;
            per_thread.append(td);
        }
        result["per_thread"] = per_thread;

        return result;
    }

    // Insert single point with detailed latency breakdown
    std::tuple<uint32_t, py::dict> insert_with_stats(py::array_t<T> point) {
        if (!_dynamic) {
            throw std::runtime_error("Index is not dynamic, cannot insert");
        }

        auto buf = point.request();
        if (buf.size != static_cast<ssize_t>(_dim)) {
            throw std::runtime_error("Point dimension mismatch");
        }

        T* ptr = static_cast<T*>(buf.ptr);
        uint32_t tag = _next_tag++;

        diskann::InsertStats stats;
        int result = _index->insert_point_with_stats(ptr, tag, stats);
        if (result != 0) {
            throw std::runtime_error("Insert failed with code: " + std::to_string(result));
        }

        _num_points++;

        // Convert stats to dict
        py::dict stats_dict;
        stats_dict["total_us"] = stats.total_us;
        stats_dict["lock_us"] = stats.lock_us;
        stats_dict["reserve_us"] = stats.reserve_us;
        stats_dict["data_store_us"] = stats.data_store_us;
        stats_dict["search_prune_us"] = stats.search_prune_us;
        stats_dict["link_us"] = stats.link_us;

        return std::make_tuple(tag, stats_dict);
    }

    // Delete single point with detailed latency breakdown
    py::dict delete_with_stats(uint32_t tag) {
        if (!_dynamic) {
            throw std::runtime_error("Index is not dynamic, cannot delete");
        }

        _index->enable_delete();

        diskann::DeleteStats stats;
        int result = _index->lazy_delete_with_stats(tag, stats);
        if (result != 0) {
            throw std::runtime_error("Delete failed, tag not found: " + std::to_string(tag));
        }
        _pending_deletes++;

        // Convert stats to dict
        py::dict stats_dict;
        stats_dict["total_us"] = stats.total_us;
        stats_dict["lock_us"] = stats.lock_us;
        stats_dict["tag_lookup_us"] = stats.tag_lookup_us;
        stats_dict["mark_us"] = stats.mark_us;

        return stats_dict;
    }

    // Consolidate with detailed latency breakdown
    py::dict consolidate_with_stats() {
        auto params = diskann::IndexWriteParametersBuilder(_L, _R)
            .with_alpha(_alpha)
            .with_num_threads(_num_threads)
            .build();

        diskann::ConsolidateStats stats;
        auto report = _index->consolidate_deletes_with_stats(params, stats);
        _num_points = report._active_points;
        _pending_deletes = 0;

        // Convert stats to dict
        py::dict stats_dict;
        stats_dict["total_us"] = stats.total_us;
        stats_dict["compaction_us"] = stats.compaction_us;
        stats_dict["relink_us"] = stats.relink_us;
        stats_dict["points_removed"] = stats.points_removed;
        stats_dict["points_remaining"] = stats.points_remaining;

        return stats_dict;
    }

    // Pruning ablation: runtime strategy switch
    void set_pruning_strategy(const std::string &strategy) {
        if (strategy == "robust_prune")
            _index->set_pruning_strategy(diskann::PruningStrategy::ROBUST_PRUNE);
        else if (strategy == "median_evict")
            _index->set_pruning_strategy(diskann::PruningStrategy::MEDIAN_EVICT);
        else if (strategy == "greedy_nearest")
            _index->set_pruning_strategy(diskann::PruningStrategy::GREEDY_NEAREST);
        else if (strategy == "random_prune")
            _index->set_pruning_strategy(diskann::PruningStrategy::RANDOM_PRUNE);
        else if (strategy == "partition_aware")
            _index->set_pruning_strategy(diskann::PruningStrategy::PARTITION_AWARE_PRUNE);
        else
            throw std::runtime_error("Unknown pruning strategy: " + strategy);
    }

    std::string get_pruning_strategy() const {
        switch (_index->get_pruning_strategy()) {
            case diskann::PruningStrategy::ROBUST_PRUNE: return "robust_prune";
            case diskann::PruningStrategy::MEDIAN_EVICT: return "median_evict";
            case diskann::PruningStrategy::GREEDY_NEAREST: return "greedy_nearest";
            case diskann::PruningStrategy::RANDOM_PRUNE: return "random_prune";
            case diskann::PruningStrategy::PARTITION_AWARE_PRUNE: return "partition_aware";
            default: return "unknown";
        }
    }

    void set_skip_consolidation_pruning(bool skip) {
        _index->set_skip_consolidation_pruning(skip);
    }

    bool get_skip_consolidation_pruning() const {
        return _index->get_skip_consolidation_pruning();
    }

    // ----- LITS (ported from the diskann-lits fork) -----
    // Call AFTER constructing the index and BEFORE build()/insert_batch(): the
    // initial build then already applies the LITS edge filter, matching the
    // diskann-lits streaming scenario (configure_lits runs before the initial
    // build). Labels are tag-indexed ({num_entries u32, K u32} + u32[]); the
    // frozen start point is tag 0 and must be entry 0 of the label file.
    //
    // num_clusters > 0 enables capacity-balanced placement (requires
    // centroids_file; labels_file must cover the initial window with
    // active_window+1 entries, or pass a checkpoint .labels + resume_tags_file
    // to resume). num_clusters == 0 with a labels_file enables the edge filter
    // over a static label set only.
    void enable_lits(const std::string& labels_file,
                     const std::string& centroids_file,
                     uint32_t num_clusters,
                     float threshold,
                     uint64_t active_window,
                     uint64_t max_points_to_insert,
                     const std::string& filter_mode,
                     float tau_scale,
                     bool tau_median,
                     const std::string& resume_tags_file) {
        if (num_clusters > 0) {
            _index->enable_lits_balanced_alloc(num_clusters, threshold, active_window,
                                               max_points_to_insert, labels_file,
                                               resume_tags_file);
            if (centroids_file.empty()) {
                throw std::runtime_error(
                    "enable_lits: balanced alloc requires centroids_file "
                    "(precharge+redirect accounting)");
            }
            _index->set_lits_balanced_centroids(centroids_file);
            _lits_balanced = true;
        } else if (!labels_file.empty()) {
            _index->load_lits_build_cluster_labels(labels_file);
        } else {
            throw std::runtime_error("enable_lits: need labels_file (filter) and/or "
                                     "num_clusters > 0 (balanced alloc)");
        }

        if (filter_mode == "full") {
            _index->set_lits_build_filter(true);
        } else if (filter_mode == "notostart") {
            _index->set_lits_build_filter(true);
            _index->set_lits_build_notostart_filter(true);
        } else if (filter_mode == "long_inter_only") {
            _index->set_lits_build_filter(true);
            _index->set_lits_build_long_inter_only(true);
        } else if (filter_mode != "off") {
            throw std::runtime_error("enable_lits: unknown filter_mode '" + filter_mode +
                                     "' (off|full|notostart|long_inter_only)");
        }
        _index->set_lits_tau_scale(tau_scale);
        _index->set_lits_tau_median(tau_median);
        _lits_enabled = true;
    }

    // LITS counters. reset_redirect=true additionally returns and resets the
    // per-interval redirected/placed counters (diskann-lits per-batch log).
    py::dict lits_stats(bool reset_redirect = false) {
        py::dict d;
        d["enabled"] = _lits_enabled;
        d["balanced"] = _lits_balanced;
        d["filtered_total"] = _index->get_lits_build_filtered_total();
        if (_lits_balanced) {
            d["counts_sum"] = _index->get_lits_cluster_counts_sum();
            if (reset_redirect) {
                auto rd = _index->get_reset_lits_balanced_redirect_stats();
                d["redirected"] = rd.first;
                d["placed"] = rd.second;
            }
        }
        return d;
    }

    // Manual label dump (save() already dumps <filename>.labels when balanced
    // alloc is on).
    void save_lits_labels(const std::string& path) {
        _index->save_lits_cluster_labels(path);
    }

    // Graph metrics: degree distribution + BFS connectivity
    py::dict get_graph_metrics() {
        auto metrics = _index->compute_graph_metrics();
        py::dict d;
        d["mean"] = metrics.mean_degree;
        d["std"] = metrics.std_degree;
        d["median"] = metrics.median_degree;
        d["p95"] = metrics.p95_degree;
        d["p99"] = metrics.p99_degree;
        d["max"] = metrics.max_degree;
        d["min"] = metrics.min_degree;
        d["num_nodes"] = metrics.num_active_nodes;
        d["reachable_count"] = metrics.reachable_count;
        d["unreachable_count"] = metrics.unreachable_count;
        d["unreachable_fraction"] = metrics.unreachable_fraction;

        // Convert histogram vectors to numpy arrays
        py::list hist_counts;
        for (auto c : metrics.histogram_counts)
            hist_counts.append(c);
        d["histogram_counts"] = hist_counts;

        py::list hist_edges;
        for (auto e : metrics.histogram_edges)
            hist_edges.append(e);
        d["histogram_edges"] = hist_edges;

        return d;
    }

private:
    std::unique_ptr<diskann::Index<T, uint32_t, uint32_t>> _index;
    size_t _dim;
    size_t _max_points;
    size_t _num_points = 0;
    uint32_t _R;
    uint32_t _L;
    float _alpha;
    uint32_t _num_threads;
    bool _dynamic;
    bool _built;
    uint32_t _next_tag;
    size_t _pending_deletes = 0;
    bool _lits_enabled = false;   // any LITS feature on (filter and/or balanced)
    bool _lits_balanced = false;  // balanced placement on (labels evolve, counters live)
};


PYBIND11_MODULE(diskann_bindings, m) {
    m.doc() = "DiskANN Python bindings for build, update, and search";

    // Float32 index
    py::class_<DiskANNIndex<float>>(m, "DiskANNIndex")
        .def(py::init<size_t, size_t, uint32_t, uint32_t, float, uint32_t, bool>(),
             py::arg("dim"),
             py::arg("max_points"),
             py::arg("R") = 64,
             py::arg("L") = 100,
             py::arg("alpha") = 1.2f,
             py::arg("num_threads") = 0,
             py::arg("dynamic") = true,
             "Create a DiskANN index\n\n"
             "Args:\n"
             "    dim: Vector dimension\n"
             "    max_points: Maximum number of points\n"
             "    R: Graph degree (default: 64)\n"
             "    L: Search list size during build (default: 100)\n"
             "    alpha: Distance threshold for pruning (default: 1.2)\n"
             "    num_threads: Number of threads (0 = auto)\n"
             "    dynamic: Enable dynamic updates (default: True)")

        // Build
        .def("build", &DiskANNIndex<float>::build,
             py::arg("data"),
             "Build index from numpy array (N x dim)")
        .def("build_from_file", &DiskANNIndex<float>::build_from_file,
             py::arg("filename"),
             py::arg("num_points") = 0,
             "Build index from binary file")
        .def("build_with_tags", &DiskANNIndex<float>::build_with_tags,
             py::arg("data"),
             py::arg("tags"),
             "Build index with custom tags (tag = global_id + 1 for sharded graphs)")

        // Update
        .def("insert", &DiskANNIndex<float>::insert,
             py::arg("point"),
             "Insert single point, returns tag")
        .def("insert_batch", &DiskANNIndex<float>::insert_batch,
             py::arg("data"),
             "Insert batch of points, returns list of tags")
        .def("insert_with_tag", &DiskANNIndex<float>::insert_with_tag,
             py::arg("point"),
             py::arg("tag"),
             "Insert single point with specific tag")
        .def("insert_batch_with_tags", &DiskANNIndex<float>::insert_batch_with_tags,
             py::arg("data"),
             py::arg("tags"),
             "Insert batch of points with specific tags")
        .def("delete_point", &DiskANNIndex<float>::delete_point,
             py::arg("tag"),
             "Delete point by tag (lazy)")
        .def("delete_batch", &DiskANNIndex<float>::delete_batch,
             py::arg("tags"),
             "Delete batch of points by tags (lazy)")
        .def("consolidate", &DiskANNIndex<float>::consolidate,
             "Consolidate pending deletes, returns remaining point count")

        // Search
        .def("search", &DiskANNIndex<float>::search,
             py::arg("query"),
             py::arg("K"),
             py::arg("L") = 0,
             "Search for K nearest neighbors\n\n"
             "Args:\n"
             "    query: Query vector(s) - 1D (single) or 2D (batch)\n"
             "    K: Number of neighbors to return\n"
             "    L: Search list size (default: use build-time L)\n\n"
             "Returns:\n"
             "    (indices, distances) tuple of numpy arrays")

        // Save/Load
        .def("save", &DiskANNIndex<float>::save,
             py::arg("filename"),
             py::arg("compact_before_save") = true,
             py::arg("save_data_file") = true,
             "Save index to file (compacts data before save by default). "
             "Set save_data_file=False to skip the .data blob.")
        .def("load", &DiskANNIndex<float>::load,
             py::arg("filename"),
             py::arg("search_L") = 0,
             "Load index from file")
        .def("enable_lits", &DiskANNIndex<float>::enable_lits,
             py::arg("labels_file"),
             py::arg("centroids_file") = "",
             py::arg("num_clusters") = 0,
             py::arg("threshold") = 0.7f,
             py::arg("active_window") = 0,
             py::arg("max_points_to_insert") = 0,
             py::arg("filter_mode") = "long_inter_only",
             py::arg("tau_scale") = 1.0f,
             py::arg("tau_median") = false,
             py::arg("resume_tags_file") = "",
             "Enable LITS (diskann-lits port). Call before build().\n\n"
             "num_clusters > 0 enables capacity-balanced placement (requires\n"
             "centroids_file + active_window + max_points_to_insert; labels_file\n"
             "covers the initial window with active_window+1 entries). With\n"
             "num_clusters == 0, labels_file enables the edge filter only.\n"
             "filter_mode: off | full | notostart | long_inter_only")
        .def("lits_stats", &DiskANNIndex<float>::lits_stats,
             py::arg("reset_redirect") = false,
             "LITS counters: filtered_total, counts_sum (invariant: active+1),\n"
             "and (reset_redirect=True) per-interval redirected/placed")
        .def("save_lits_labels", &DiskANNIndex<float>::save_lits_labels,
             py::arg("path"),
             "Dump the tag-indexed cluster label array ({n u32, K u32} + u32[])")

        .def("load_graph_only", &DiskANNIndex<float>::load_graph_only,
             py::arg("graph_file"),
             py::arg("base_file"),
             py::arg("tags_file"),
             py::arg("search_L") = 0,
             "Load index from graph + tags + base file (without .data file)")

        // Properties
        .def_property_readonly("num_points", &DiskANNIndex<float>::get_num_points)
        .def_property_readonly("max_points", &DiskANNIndex<float>::get_max_points)
        .def_property_readonly("dim", &DiskANNIndex<float>::get_dim)
        .def_property_readonly("is_built", &DiskANNIndex<float>::is_built)
        .def_property_readonly("pending_deletes", &DiskANNIndex<float>::get_pending_deletes)

        // Debug
        .def("print_status", &DiskANNIndex<float>::print_status)

        // Latency profiling with stats
        .def("search_with_stats", &DiskANNIndex<float>::search_with_stats,
             py::arg("query"),
             py::arg("K"),
             py::arg("L") = 0,
             "Search with detailed latency breakdown (single query)\n\n"
             "Returns:\n"
             "    (indices, distances, stats) where stats is dict with:\n"
             "    total_us, dist_us, fetch_us, queue_us, visited_us, n_hops, n_cmps")
        .def("search_with_partition_stats", &DiskANNIndex<float>::search_with_partition_stats,
             py::arg("query"),
             py::arg("K"),
             py::arg("L"),
             py::arg("partition_ids"),
             py::arg("query_partition"),
             "Search with partition-aware latency breakdown (for distributed profiling)\n\n"
             "Args:\n"
             "    partition_ids: np.array of partition assignments for each point\n"
             "    query_partition: partition ID for this query\n\n"
             "Returns:\n"
             "    (indices, distances, stats) where stats includes:\n"
             "    dist_local_us, dist_remote_us, fetch_local_us, fetch_remote_us,\n"
             "    local_vector_accesses, remote_vector_accesses, rdma_round_trips")
        .def("insert_with_stats", &DiskANNIndex<float>::insert_with_stats,
             py::arg("point"),
             "Insert with detailed latency breakdown\n\n"
             "Returns:\n"
             "    (tag, stats) where stats is dict with:\n"
             "    total_us, lock_us, reserve_us, data_store_us, search_prune_us, link_us")
        .def("delete_with_stats", &DiskANNIndex<float>::delete_with_stats,
             py::arg("tag"),
             "Delete with detailed latency breakdown\n\n"
             "Returns:\n"
             "    stats dict with: total_us, lock_us, tag_lookup_us, mark_us")
        .def("consolidate_with_stats", &DiskANNIndex<float>::consolidate_with_stats,
             "Consolidate with detailed latency breakdown\n\n"
             "Returns:\n"
             "    stats dict with: total_us, compaction_us, relink_us, points_removed, points_remaining")
        .def("search_throughput", &DiskANNIndex<float>::search_throughput,
             py::arg("queries"),
             py::arg("K"),
             py::arg("L"),
             py::arg("num_threads"),
             py::arg("duration_sec"),
             py::arg("warmup_sec"),
             "Multi-threaded throughput benchmark (local-only)\n\n"
             "Returns:\n"
             "    dict with: qps, total_queries, latency_mean_us, latency_p50_us,\n"
             "    latency_p99_us, latency_p999_us, per_thread")

        // Pruning ablation
        .def("set_pruning_strategy", &DiskANNIndex<float>::set_pruning_strategy,
             py::arg("strategy"),
             "Set pruning strategy at runtime\n\n"
             "Strategies: 'robust_prune', 'median_evict', 'greedy_nearest', 'random_prune', 'partition_aware'")
        .def("get_pruning_strategy", &DiskANNIndex<float>::get_pruning_strategy,
             "Get current pruning strategy name")
        .def("set_skip_consolidation_pruning", &DiskANNIndex<float>::set_skip_consolidation_pruning,
             py::arg("skip"),
             "Skip expansion and pruning during consolidate_deletes (just remove deleted edges)")
        .def("get_skip_consolidation_pruning", &DiskANNIndex<float>::get_skip_consolidation_pruning,
             "Get current skip_consolidation_pruning flag")

        // Frozen point
        .def("get_frozen_point_info", &DiskANNIndex<float>::get_frozen_point_info,
             "Get frozen point vector, neighbors, and location")

        // Graph metrics
        .def("get_graph_metrics", &DiskANNIndex<float>::get_graph_metrics,
             "Compute graph metrics: degree distribution + BFS connectivity\n\n"
             "Returns:\n"
             "    dict with: mean, std, median, p95, p99, max, min, num_nodes,\n"
             "    reachable_count, unreachable_count, unreachable_fraction,\n"
             "    histogram_counts, histogram_edges");

    // UInt8 index (for SIFT1B, BigANN datasets)
    py::class_<DiskANNIndex<uint8_t>>(m, "DiskANNIndexUInt8")
        .def(py::init<size_t, size_t, uint32_t, uint32_t, float, uint32_t, bool>(),
             py::arg("dim"),
             py::arg("max_points"),
             py::arg("R") = 64,
             py::arg("L") = 100,
             py::arg("alpha") = 1.2f,
             py::arg("num_threads") = 0,
             py::arg("dynamic") = true,
             "Create a DiskANN index for uint8 data (SIFT1B, BigANN)")

        // Build
        .def("build", &DiskANNIndex<uint8_t>::build,
             py::arg("data"),
             "Build index from numpy array (N x dim)")
        .def("build_from_file", &DiskANNIndex<uint8_t>::build_from_file,
             py::arg("filename"),
             py::arg("num_points") = 0,
             "Build index from binary file")
        .def("build_with_tags", &DiskANNIndex<uint8_t>::build_with_tags,
             py::arg("data"),
             py::arg("tags"),
             "Build index with custom tags")

        // Update
        .def("insert", &DiskANNIndex<uint8_t>::insert,
             py::arg("point"),
             "Insert single point, returns tag")
        .def("insert_batch", &DiskANNIndex<uint8_t>::insert_batch,
             py::arg("data"),
             "Insert batch of points, returns list of tags")
        .def("insert_with_tag", &DiskANNIndex<uint8_t>::insert_with_tag,
             py::arg("point"),
             py::arg("tag"),
             "Insert single point with specific tag")
        .def("insert_batch_with_tags", &DiskANNIndex<uint8_t>::insert_batch_with_tags,
             py::arg("data"),
             py::arg("tags"),
             "Insert batch of points with specific tags")
        .def("delete_point", &DiskANNIndex<uint8_t>::delete_point,
             py::arg("tag"),
             "Delete point by tag (lazy)")
        .def("delete_batch", &DiskANNIndex<uint8_t>::delete_batch,
             py::arg("tags"),
             "Delete batch of points by tags (lazy)")
        .def("consolidate", &DiskANNIndex<uint8_t>::consolidate,
             "Consolidate pending deletes, returns remaining point count")

        // Search
        .def("search", &DiskANNIndex<uint8_t>::search,
             py::arg("query"),
             py::arg("K"),
             py::arg("L") = 0,
             "Search for K nearest neighbors")

        // Save/Load
        .def("save", &DiskANNIndex<uint8_t>::save,
             py::arg("filename"),
             py::arg("compact_before_save") = true,
             py::arg("save_data_file") = true,
             "Save index to file. Set save_data_file=False to skip .data blob.")
        .def("load", &DiskANNIndex<uint8_t>::load,
             py::arg("filename"),
             py::arg("search_L") = 0,
             "Load index from file")
        .def("enable_lits", &DiskANNIndex<uint8_t>::enable_lits,
             py::arg("labels_file"),
             py::arg("centroids_file") = "",
             py::arg("num_clusters") = 0,
             py::arg("threshold") = 0.7f,
             py::arg("active_window") = 0,
             py::arg("max_points_to_insert") = 0,
             py::arg("filter_mode") = "long_inter_only",
             py::arg("tau_scale") = 1.0f,
             py::arg("tau_median") = false,
             py::arg("resume_tags_file") = "",
             "Enable LITS (diskann-lits port). Call before build().\n\n"
             "num_clusters > 0 enables capacity-balanced placement (requires\n"
             "centroids_file + active_window + max_points_to_insert; labels_file\n"
             "covers the initial window with active_window+1 entries). With\n"
             "num_clusters == 0, labels_file enables the edge filter only.\n"
             "filter_mode: off | full | notostart | long_inter_only")
        .def("lits_stats", &DiskANNIndex<uint8_t>::lits_stats,
             py::arg("reset_redirect") = false,
             "LITS counters: filtered_total, counts_sum (invariant: active+1),\n"
             "and (reset_redirect=True) per-interval redirected/placed")
        .def("save_lits_labels", &DiskANNIndex<uint8_t>::save_lits_labels,
             py::arg("path"),
             "Dump the tag-indexed cluster label array ({n u32, K u32} + u32[])")

        .def("load_graph_only", &DiskANNIndex<uint8_t>::load_graph_only,
             py::arg("graph_file"),
             py::arg("base_file"),
             py::arg("tags_file"),
             py::arg("search_L") = 0,
             "Load index from graph + tags + base file (without .data file)")

        // Properties
        .def_property_readonly("num_points", &DiskANNIndex<uint8_t>::get_num_points)
        .def_property_readonly("max_points", &DiskANNIndex<uint8_t>::get_max_points)
        .def_property_readonly("dim", &DiskANNIndex<uint8_t>::get_dim)
        .def_property_readonly("is_built", &DiskANNIndex<uint8_t>::is_built)
        .def_property_readonly("pending_deletes", &DiskANNIndex<uint8_t>::get_pending_deletes)

        // Debug
        .def("print_status", &DiskANNIndex<uint8_t>::print_status)

        // Latency profiling with stats
        .def("search_with_stats", &DiskANNIndex<uint8_t>::search_with_stats,
             py::arg("query"),
             py::arg("K"),
             py::arg("L") = 0,
             "Search with detailed latency breakdown")
        .def("search_with_partition_stats", &DiskANNIndex<uint8_t>::search_with_partition_stats,
             py::arg("query"),
             py::arg("K"),
             py::arg("L"),
             py::arg("partition_ids"),
             py::arg("query_partition"),
             "Search with partition-aware latency breakdown")
        .def("insert_with_stats", &DiskANNIndex<uint8_t>::insert_with_stats,
             py::arg("point"),
             "Insert with detailed latency breakdown")
        .def("delete_with_stats", &DiskANNIndex<uint8_t>::delete_with_stats,
             py::arg("tag"),
             "Delete with detailed latency breakdown")
        .def("consolidate_with_stats", &DiskANNIndex<uint8_t>::consolidate_with_stats,
             "Consolidate with detailed latency breakdown")
        .def("search_throughput", &DiskANNIndex<uint8_t>::search_throughput,
             py::arg("queries"), py::arg("K"), py::arg("L"),
             py::arg("num_threads"), py::arg("duration_sec"), py::arg("warmup_sec"),
             "Multi-threaded throughput benchmark (local-only)");

    // Int8 index (for SpaceV datasets)
    py::class_<DiskANNIndex<int8_t>>(m, "DiskANNIndexInt8")
        .def(py::init<size_t, size_t, uint32_t, uint32_t, float, uint32_t, bool>(),
             py::arg("dim"),
             py::arg("max_points"),
             py::arg("R") = 64,
             py::arg("L") = 100,
             py::arg("alpha") = 1.2f,
             py::arg("num_threads") = 0,
             py::arg("dynamic") = true,
             "Create a DiskANN index for int8 data (SpaceV)")

        // Build
        .def("build", &DiskANNIndex<int8_t>::build,
             py::arg("data"),
             "Build index from numpy array (N x dim)")
        .def("build_from_file", &DiskANNIndex<int8_t>::build_from_file,
             py::arg("filename"),
             py::arg("num_points") = 0,
             "Build index from binary file")
        .def("build_with_tags", &DiskANNIndex<int8_t>::build_with_tags,
             py::arg("data"),
             py::arg("tags"),
             "Build index with custom tags")

        // Update
        .def("insert", &DiskANNIndex<int8_t>::insert,
             py::arg("point"),
             "Insert single point, returns tag")
        .def("insert_batch", &DiskANNIndex<int8_t>::insert_batch,
             py::arg("data"),
             "Insert batch of points, returns list of tags")
        .def("insert_with_tag", &DiskANNIndex<int8_t>::insert_with_tag,
             py::arg("point"),
             py::arg("tag"),
             "Insert single point with specific tag")
        .def("insert_batch_with_tags", &DiskANNIndex<int8_t>::insert_batch_with_tags,
             py::arg("data"),
             py::arg("tags"),
             "Insert batch of points with specific tags")
        .def("delete_point", &DiskANNIndex<int8_t>::delete_point,
             py::arg("tag"),
             "Delete point by tag (lazy)")
        .def("delete_batch", &DiskANNIndex<int8_t>::delete_batch,
             py::arg("tags"),
             "Delete batch of points by tags (lazy)")
        .def("consolidate", &DiskANNIndex<int8_t>::consolidate,
             "Consolidate pending deletes, returns remaining point count")

        // Search
        .def("search", &DiskANNIndex<int8_t>::search,
             py::arg("query"),
             py::arg("K"),
             py::arg("L") = 0,
             "Search for K nearest neighbors")

        // Save/Load
        .def("save", &DiskANNIndex<int8_t>::save,
             py::arg("filename"),
             py::arg("compact_before_save") = true,
             py::arg("save_data_file") = true,
             "Save index to file. Set save_data_file=False to skip .data blob.")
        .def("load", &DiskANNIndex<int8_t>::load,
             py::arg("filename"),
             py::arg("search_L") = 0,
             "Load index from file")
        .def("enable_lits", &DiskANNIndex<int8_t>::enable_lits,
             py::arg("labels_file"),
             py::arg("centroids_file") = "",
             py::arg("num_clusters") = 0,
             py::arg("threshold") = 0.7f,
             py::arg("active_window") = 0,
             py::arg("max_points_to_insert") = 0,
             py::arg("filter_mode") = "long_inter_only",
             py::arg("tau_scale") = 1.0f,
             py::arg("tau_median") = false,
             py::arg("resume_tags_file") = "",
             "Enable LITS (diskann-lits port). Call before build().\n\n"
             "num_clusters > 0 enables capacity-balanced placement (requires\n"
             "centroids_file + active_window + max_points_to_insert; labels_file\n"
             "covers the initial window with active_window+1 entries). With\n"
             "num_clusters == 0, labels_file enables the edge filter only.\n"
             "filter_mode: off | full | notostart | long_inter_only")
        .def("lits_stats", &DiskANNIndex<int8_t>::lits_stats,
             py::arg("reset_redirect") = false,
             "LITS counters: filtered_total, counts_sum (invariant: active+1),\n"
             "and (reset_redirect=True) per-interval redirected/placed")
        .def("save_lits_labels", &DiskANNIndex<int8_t>::save_lits_labels,
             py::arg("path"),
             "Dump the tag-indexed cluster label array ({n u32, K u32} + u32[])")

        .def("load_graph_only", &DiskANNIndex<int8_t>::load_graph_only,
             py::arg("graph_file"),
             py::arg("base_file"),
             py::arg("tags_file"),
             py::arg("search_L") = 0,
             "Load index from graph + tags + base file (without .data file)")

        // Properties
        .def_property_readonly("num_points", &DiskANNIndex<int8_t>::get_num_points)
        .def_property_readonly("max_points", &DiskANNIndex<int8_t>::get_max_points)
        .def_property_readonly("dim", &DiskANNIndex<int8_t>::get_dim)
        .def_property_readonly("is_built", &DiskANNIndex<int8_t>::is_built)
        .def_property_readonly("pending_deletes", &DiskANNIndex<int8_t>::get_pending_deletes)

        // Debug
        .def("print_status", &DiskANNIndex<int8_t>::print_status)

        // Latency profiling with stats
        .def("search_with_stats", &DiskANNIndex<int8_t>::search_with_stats,
             py::arg("query"),
             py::arg("K"),
             py::arg("L") = 0,
             "Search with detailed latency breakdown")
        .def("search_with_partition_stats", &DiskANNIndex<int8_t>::search_with_partition_stats,
             py::arg("query"),
             py::arg("K"),
             py::arg("L"),
             py::arg("partition_ids"),
             py::arg("query_partition"),
             "Search with partition-aware latency breakdown")
        .def("insert_with_stats", &DiskANNIndex<int8_t>::insert_with_stats,
             py::arg("point"),
             "Insert with detailed latency breakdown")
        .def("delete_with_stats", &DiskANNIndex<int8_t>::delete_with_stats,
             py::arg("tag"),
             "Delete with detailed latency breakdown")
        .def("consolidate_with_stats", &DiskANNIndex<int8_t>::consolidate_with_stats,
             "Consolidate with detailed latency breakdown")
        .def("search_throughput", &DiskANNIndex<int8_t>::search_throughput,
             py::arg("queries"), py::arg("K"), py::arg("L"),
             py::arg("num_threads"), py::arg("duration_sec"), py::arg("warmup_sec"),
             "Multi-threaded throughput benchmark (local-only)")

        // Pruning ablation
        .def("set_pruning_strategy", &DiskANNIndex<int8_t>::set_pruning_strategy,
             py::arg("strategy"),
             "Set pruning strategy at runtime\n\n"
             "Strategies: 'robust_prune', 'median_evict', 'greedy_nearest', 'random_prune', 'partition_aware'")
        .def("get_pruning_strategy", &DiskANNIndex<int8_t>::get_pruning_strategy,
             "Get current pruning strategy name")
        .def("set_skip_consolidation_pruning", &DiskANNIndex<int8_t>::set_skip_consolidation_pruning,
             py::arg("skip"),
             "Skip expansion and pruning during consolidate_deletes (just remove deleted edges)")
        .def("get_skip_consolidation_pruning", &DiskANNIndex<int8_t>::get_skip_consolidation_pruning,
             "Get current skip_consolidation_pruning flag")

        // Graph metrics
        .def("get_graph_metrics", &DiskANNIndex<int8_t>::get_graph_metrics,
             "Compute graph metrics: degree distribution + BFS connectivity\n\n"
             "Returns:\n"
             "    dict with: mean, std, median, p95, p99, max, min, num_nodes,\n"
             "    reachable_count, unreachable_count, unreachable_fraction,\n"
             "    histogram_counts, histogram_edges");

    // Utility functions
    m.def("read_bin_metadata", [](const std::string& filename) {
        size_t npts, ndim;
        diskann::get_bin_metadata(filename, npts, ndim);
        return std::make_tuple(npts, ndim);
    }, py::arg("filename"), "Read binary file metadata (num_points, dim)");

    m.def("load_bin_f32", [](const std::string& filename) {
        size_t npts, ndim;
        float* data = nullptr;
        diskann::load_bin<float>(filename, data, npts, ndim);

        // Create numpy array that owns the data
        auto capsule = py::capsule(data, [](void* p) {
            diskann::aligned_free(p);
        });

        std::vector<ssize_t> arr_shape = {static_cast<ssize_t>(npts), static_cast<ssize_t>(ndim)};
        std::vector<ssize_t> arr_strides = {static_cast<ssize_t>(ndim * sizeof(float)), static_cast<ssize_t>(sizeof(float))};
        return py::array_t<float>(arr_shape, arr_strides, data, capsule);
    }, py::arg("filename"), "Load float32 binary file as numpy array");
}
