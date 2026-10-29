// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// Modified version: compute_groundtruth_with_tags
// Only loads active vectors specified in tags file for efficient GT computation.
// Tags file format: [npts (uint32), dim=1 (uint32), tag1, tag2, ...] where tag = base_index + 1

#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <algorithm>
#include <cstddef>
#include <random>
#include <limits>
#include <cstring>
#include <queue>
#include <chrono>
#include <omp.h>
#include <mkl.h>
#include <boost/program_options.hpp>

#ifdef _WINDOWS
#include <malloc.h>
#else
#include <stdlib.h>
#endif
#include "utils.h"

#define ALIGNMENT 512

namespace po = boost::program_options;

template <class T> T div_round_up(const T numerator, const T denominator)
{
    return (numerator % denominator == 0) ? (numerator / denominator) : 1 + (numerator / denominator);
}

using pairIF = std::pair<size_t, float>;
struct cmpmaxstruct
{
    bool operator()(const pairIF &l, const pairIF &r)
    {
        return l.second < r.second;
    };
};

using maxPQIFCS = std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;

template <class T> T *aligned_malloc(const size_t n, const size_t alignment)
{
#ifdef _WINDOWS
    return (T *)_aligned_malloc(sizeof(T) * n, alignment);
#else
    return static_cast<T *>(aligned_alloc(alignment, sizeof(T) * n));
#endif
}

inline bool custom_dist(const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b)
{
    return a.second < b.second;
}

void compute_l2sq(float *const points_l2sq, const float *const matrix, const int64_t num_points, const uint64_t dim)
{
    assert(points_l2sq != NULL);
#pragma omp parallel for schedule(static, 65536)
    for (int64_t d = 0; d < num_points; ++d)
        points_l2sq[d] = cblas_sdot((int64_t)dim, matrix + (ptrdiff_t)d * (ptrdiff_t)dim, 1,
                                    matrix + (ptrdiff_t)d * (ptrdiff_t)dim, 1);
}

void distsq_to_points(const size_t dim,
                      float *dist_matrix,
                      size_t npoints, const float *const points,
                      const float *const points_l2sq,
                      size_t nqueries, const float *const queries,
                      const float *const queries_l2sq,
                      float *ones_vec = NULL)
{
    bool ones_vec_alloc = false;
    if (ones_vec == NULL)
    {
        ones_vec = new float[nqueries > npoints ? nqueries : npoints];
        std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float)1.0);
        ones_vec_alloc = true;
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float)-2.0, points, dim, queries, dim,
                (float)0.0, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, points_l2sq, npoints,
                ones_vec, nqueries, (float)1.0, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, ones_vec, npoints,
                queries_l2sq, nqueries, (float)1.0, dist_matrix, npoints);
    if (ones_vec_alloc)
        delete[] ones_vec;
}

void inner_prod_to_points(const size_t dim,
                          float *dist_matrix,
                          size_t npoints, const float *const points,
                          size_t nqueries, const float *const queries,
                          float *ones_vec = NULL)
{
    bool ones_vec_alloc = false;
    if (ones_vec == NULL)
    {
        ones_vec = new float[nqueries > npoints ? nqueries : npoints];
        std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float)1.0);
        ones_vec_alloc = true;
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float)-1.0, points, dim, queries, dim,
                (float)0.0, dist_matrix, npoints);

    if (ones_vec_alloc)
        delete[] ones_vec;
}

// Load tags file and return vector of tags (tag = base_index + 1)
std::vector<uint32_t> load_tags(const std::string &tags_file)
{
    std::ifstream reader(tags_file, std::ios::binary);
    if (!reader.is_open())
    {
        throw std::runtime_error("Cannot open tags file: " + tags_file);
    }

    uint32_t npts, ndims;
    reader.read((char *)&npts, sizeof(uint32_t));
    reader.read((char *)&ndims, sizeof(uint32_t));

    if (ndims != 1)
    {
        throw std::runtime_error("Tags file dimension must be 1, got " + std::to_string(ndims));
    }

    std::vector<uint32_t> tags(npts);
    reader.read((char *)tags.data(), npts * sizeof(uint32_t));
    reader.close();

    std::cout << "Loaded " << npts << " tags from " << tags_file << std::endl;
    return tags;
}

// Load active vectors from base file using sequential I/O
// Reads base file in one pass, extracts active indices, converts to float32.
// Peak memory: n_active * dim * sizeof(float) + read_buffer
template <typename T>
float *load_vectors_by_indices(const std::string &base_file, const std::vector<uint32_t> &indices,
                                size_t &dim)
{
    std::ifstream reader(base_file, std::ios::binary);
    if (!reader.is_open())
        throw std::runtime_error("Cannot open base file: " + base_file);

    int32_t npts_i32, ndims_i32;
    reader.read((char *)&npts_i32, sizeof(int32_t));
    reader.read((char *)&ndims_i32, sizeof(int32_t));
    dim = ndims_i32;
    size_t npts = npts_i32;

    size_t n_active = indices.size();
    std::cout << "Loading " << n_active << " active vectors (dim=" << dim
              << ") from " << npts << " total, sequential I/O..." << std::endl;

    // Build sorted index → output position mapping for O(1) lookup
    // Sort indices for sequential access pattern
    std::vector<std::pair<uint32_t, size_t>> sorted_idx(n_active);  // (base_idx, output_pos)
    for (size_t i = 0; i < n_active; i++)
        sorted_idx[i] = {indices[i], i};
    std::sort(sorted_idx.begin(), sorted_idx.end());

    // Allocate output
    float *data = aligned_malloc<float>(n_active * dim, ALIGNMENT);

    // Sequential read in chunks
    const size_t chunk_size = 10000000;  // 10M vectors per chunk
    size_t vector_bytes = dim * sizeof(T);
    size_t active_ptr = 0;  // pointer into sorted_idx
    size_t vectors_loaded = 0;

    auto t0 = std::chrono::steady_clock::now();

    for (size_t chunk_start = 0; chunk_start < npts && active_ptr < n_active; chunk_start += chunk_size)
    {
        size_t chunk_end = std::min(chunk_start + chunk_size, npts);
        size_t chunk_count = chunk_end - chunk_start;

        // How many active vectors fall in this chunk?
        size_t active_in_chunk = 0;
        size_t peek = active_ptr;
        while (peek < n_active && sorted_idx[peek].first < chunk_end)
        {
            active_in_chunk++;
            peek++;
        }

        if (active_in_chunk == 0)
            continue;

        // Read entire chunk — seek explicitly. The `continue` above skips
        // empty chunks without advancing the reader, so without seeking we
        // would load base_file[0..chunk_count) when the first non-empty
        // chunk is reached, silently mislabeling GT whenever the active
        // set doesn't start at gid 0 (i.e., any post-iteration-0 checkpoint).
        // header = 2 int32 fields (8 bytes).
        reader.seekg(8 + static_cast<std::streamoff>(chunk_start) * vector_bytes,
                     std::ios::beg);
        std::vector<T> chunk_data(chunk_count * dim);
        reader.read((char *)chunk_data.data(), chunk_count * vector_bytes);

        // Extract active vectors and convert to float32
        for (size_t i = 0; i < active_in_chunk; i++)
        {
            uint32_t base_idx = sorted_idx[active_ptr + i].first;
            size_t out_pos = sorted_idx[active_ptr + i].second;
            size_t local_idx = base_idx - chunk_start;

            const T *src = chunk_data.data() + local_idx * dim;
            float *dst = data + out_pos * dim;
            for (size_t j = 0; j < dim; j++)
                dst[j] = static_cast<float>(src[j]);
        }

        vectors_loaded += active_in_chunk;
        active_ptr += active_in_chunk;

        if (chunk_start % (chunk_size * 10) == 0 || active_ptr >= n_active)
        {
            auto t1 = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "  Loaded " << vectors_loaded << "/" << n_active
                      << " active vectors (" << sec << "s)" << std::endl;
        }
    }

    reader.close();
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Finished loading " << vectors_loaded << " active vectors in " << sec << "s" << std::endl;
    return data;
}

// Load query file as float
template <typename T>
float *load_queries(const std::string &query_file, size_t &nqueries, size_t &dim)
{
    std::ifstream reader(query_file, std::ios::binary);
    if (!reader.is_open())
    {
        throw std::runtime_error("Cannot open query file: " + query_file);
    }

    int32_t npts_i32, ndims_i32;
    reader.read((char *)&npts_i32, sizeof(int32_t));
    reader.read((char *)&ndims_i32, sizeof(int32_t));
    nqueries = npts_i32;
    dim = ndims_i32;

    std::cout << "Loading " << nqueries << " queries (dim=" << dim << ")..." << std::endl;

    std::vector<T> temp_data(nqueries * dim);
    reader.read((char *)temp_data.data(), nqueries * dim * sizeof(T));
    reader.close();

    float *data = aligned_malloc<float>(nqueries * dim, ALIGNMENT);
    for (size_t i = 0; i < nqueries * dim; i++)
    {
        data[i] = static_cast<float>(temp_data[i]);
    }

    return data;
}

void exact_knn(const size_t dim, const size_t k,
               size_t *const closest_points,
               float *const dist_closest_points,
               size_t npoints, float *points_in,
               size_t nqueries, float *queries_in,
               diskann::Metric metric = diskann::Metric::L2)
{
    float *points_l2sq = new float[npoints];
    float *queries_l2sq = new float[nqueries];
    compute_l2sq(points_l2sq, points_in, npoints, dim);
    compute_l2sq(queries_l2sq, queries_in, nqueries, dim);

    float *points = points_in;
    float *queries = queries_in;

    if (metric == diskann::Metric::COSINE)
    {
        points = new float[npoints * dim];
        queries = new float[nqueries * dim];
#pragma omp parallel for schedule(static, 4096)
        for (int64_t i = 0; i < (int64_t)npoints; i++)
        {
            float norm = std::sqrt(points_l2sq[i]);
            if (norm == 0)
            {
                norm = std::numeric_limits<float>::epsilon();
            }
            for (uint32_t j = 0; j < dim; j++)
            {
                points[i * dim + j] = points_in[i * dim + j] / norm;
            }
        }

#pragma omp parallel for schedule(static, 4096)
        for (int64_t i = 0; i < (int64_t)nqueries; i++)
        {
            float norm = std::sqrt(queries_l2sq[i]);
            if (norm == 0)
            {
                norm = std::numeric_limits<float>::epsilon();
            }
            for (uint32_t j = 0; j < dim; j++)
            {
                queries[i * dim + j] = queries_in[i * dim + j] / norm;
            }
        }
        compute_l2sq(points_l2sq, points, npoints, dim);
        compute_l2sq(queries_l2sq, queries, nqueries, dim);
    }

    std::cout << "Computing " << k << " NNs for " << nqueries << " queries over " << npoints << " active points in "
              << dim << " dimensions using ";
    if (metric == diskann::Metric::INNER_PRODUCT)
        std::cout << "MIPS";
    else if (metric == diskann::Metric::COSINE)
        std::cout << "Cosine";
    else
        std::cout << "L2";
    std::cout << " distance." << std::endl;

    // Adaptive query batch size to avoid OOM on large datasets.
    // dist_matrix needs q_batch_size * npoints * 4 bytes.
    // Target: keep dist_matrix under ~24 GB.
    size_t max_dist_matrix_bytes = (size_t)24 * 1024 * 1024 * 1024ULL;
    size_t q_batch_size = std::max((size_t)1, max_dist_matrix_bytes / (npoints * sizeof(float)));
    q_batch_size = std::min(q_batch_size, nqueries);
    std::cout << "Query batch size: " << q_batch_size
              << " (dist_matrix: " << (q_batch_size * npoints * sizeof(float)) / (1024ULL*1024*1024) << " GB)" << std::endl;
    float *dist_matrix = new float[(size_t)q_batch_size * (size_t)npoints];

    for (size_t b = 0; b < div_round_up(nqueries, q_batch_size); ++b)
    {
        int64_t q_b = b * q_batch_size;
        int64_t q_e = ((b + 1) * q_batch_size > nqueries) ? nqueries : (b + 1) * q_batch_size;

        if (metric == diskann::Metric::L2 || metric == diskann::Metric::COSINE)
        {
            distsq_to_points(dim, dist_matrix, npoints, points, points_l2sq, q_e - q_b,
                             queries + (ptrdiff_t)q_b * (ptrdiff_t)dim, queries_l2sq + q_b);
        }
        else
        {
            inner_prod_to_points(dim, dist_matrix, npoints, points, q_e - q_b,
                                 queries + (ptrdiff_t)q_b * (ptrdiff_t)dim);
        }
        std::cout << "Computed distances for queries: [" << q_b << "," << q_e << ")" << std::endl;

#pragma omp parallel for schedule(dynamic, 16)
        for (long long q = q_b; q < q_e; q++)
        {
            maxPQIFCS point_dist;
            for (size_t p = 0; p < k; p++)
                point_dist.emplace(p, dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints]);
            for (size_t p = k; p < npoints; p++)
            {
                if (point_dist.top().second > dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints])
                    point_dist.emplace(p, dist_matrix[(ptrdiff_t)p + (ptrdiff_t)(q - q_b) * (ptrdiff_t)npoints]);
                if (point_dist.size() > k)
                    point_dist.pop();
            }
            for (ptrdiff_t l = 0; l < (ptrdiff_t)k; ++l)
            {
                closest_points[(ptrdiff_t)(k - 1 - l) + (ptrdiff_t)q * (ptrdiff_t)k] = point_dist.top().first;
                dist_closest_points[(ptrdiff_t)(k - 1 - l) + (ptrdiff_t)q * (ptrdiff_t)k] = point_dist.top().second;
                point_dist.pop();
            }
        }
        std::cout << "Computed exact k-NN for queries: [" << q_b << "," << q_e << ")" << std::endl;
    }

    delete[] dist_matrix;
    delete[] points_l2sq;
    delete[] queries_l2sq;

    if (metric == diskann::Metric::COSINE)
    {
        delete[] points;
        delete[] queries;
    }
}

void save_groundtruth(const std::string &filename, int32_t *data, float *distances, size_t npts, size_t ndims)
{
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    int npts_i32 = (int)npts, ndims_i32 = (int)ndims;
    writer.write((char *)&npts_i32, sizeof(int));
    writer.write((char *)&ndims_i32, sizeof(int));
    std::cout << "Saving groundtruth: npts=" << npts << ", K=" << ndims
              << ", size=" << 2 * npts * ndims * sizeof(uint32_t) + 2 * sizeof(int) << "B" << std::endl;

    writer.write((char *)data, npts * ndims * sizeof(uint32_t));
    writer.write((char *)distances, npts * ndims * sizeof(float));
    writer.close();
    std::cout << "Finished writing groundtruth to " << filename << std::endl;
}

template <typename T>
int aux_main(const std::string &base_file, const std::string &query_file, const std::string &gt_file,
             const std::string &tags_file, size_t k, const diskann::Metric &metric)
{
    // 1. Load tags (active vectors)
    std::vector<uint32_t> tags = load_tags(tags_file);

    // 2. Convert tags to base indices (tag - 1) and filter out tag=0
    std::vector<uint32_t> active_indices;
    active_indices.reserve(tags.size());
    for (uint32_t tag : tags)
    {
        if (tag > 0)
        {
            active_indices.push_back(tag - 1);
        }
    }
    std::cout << "Active vectors: " << active_indices.size() << std::endl;

    // Keep original tags for result mapping (index in active array -> original tag)
    std::vector<uint32_t> idx_to_tag(active_indices.size());
    for (size_t i = 0; i < tags.size(); i++)
    {
        if (tags[i] > 0)
        {
            // Find position in active_indices
            // Since we're iterating in order, we can track position
        }
    }
    // Simpler: just rebuild
    idx_to_tag.clear();
    for (uint32_t tag : tags)
    {
        if (tag > 0)
        {
            idx_to_tag.push_back(tag);
        }
    }

    // 3. Load active vectors from base file
    size_t dim;
    float *base_data = load_vectors_by_indices<T>(base_file, active_indices, dim);
    size_t n_active = active_indices.size();

    // 4. Load queries
    size_t nqueries, query_dim;
    float *query_data = load_queries<T>(query_file, nqueries, query_dim);

    if (dim != query_dim)
    {
        throw std::runtime_error("Dimension mismatch: base=" + std::to_string(dim) +
                                 ", query=" + std::to_string(query_dim));
    }

    // 5. Compute exact k-NN
    size_t *closest_points = new size_t[nqueries * k];
    float *dist_closest_points = new float[nqueries * k];

    exact_knn(dim, k, closest_points, dist_closest_points, n_active, base_data, nqueries, query_data, metric);

    // 6. Convert results: local index -> tag
    int32_t *result_tags = new int32_t[nqueries * k];
    for (size_t i = 0; i < nqueries * k; i++)
    {
        size_t local_idx = closest_points[i];
        result_tags[i] = static_cast<int32_t>(idx_to_tag[local_idx]);
    }

    // Adjust distances for MIPS
    if (metric == diskann::Metric::INNER_PRODUCT)
    {
        for (size_t i = 0; i < nqueries * k; i++)
        {
            dist_closest_points[i] = -dist_closest_points[i];
        }
    }

    // 7. Save groundtruth
    save_groundtruth(gt_file, result_tags, dist_closest_points, nqueries, k);

    // Cleanup
    delete[] closest_points;
    delete[] dist_closest_points;
    delete[] result_tags;
    diskann::aligned_free(base_data);
    diskann::aligned_free(query_data);

    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, base_file, query_file, gt_file, tags_file;
    uint64_t K;

    try
    {
        po::options_description desc{"Arguments"};

        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "distance function <l2/mips/cosine>");
        desc.add_options()("base_file", po::value<std::string>(&base_file)->required(),
                           "File containing the base vectors in binary format");
        desc.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                           "File containing the query vectors in binary format");
        desc.add_options()("gt_file", po::value<std::string>(&gt_file)->required(),
                           "Output file for ground truth");
        desc.add_options()("tags_file", po::value<std::string>(&tags_file)->required(),
                           "File containing active tags (index.tags format)");
        desc.add_options()("K", po::value<uint64_t>(&K)->required(),
                           "Number of ground truth nearest neighbors to compute");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    if (data_type != std::string("float") && data_type != std::string("int8") && data_type != std::string("uint8"))
    {
        std::cout << "Unsupported type. float, int8 and uint8 types are supported." << std::endl;
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cerr << "Unsupported distance function. Use l2/mips/cosine." << std::endl;
        return -1;
    }

    try
    {
        if (data_type == std::string("float"))
            aux_main<float>(base_file, query_file, gt_file, tags_file, K, metric);
        else if (data_type == std::string("int8"))
            aux_main<int8_t>(base_file, query_file, gt_file, tags_file, K, metric);
        else if (data_type == std::string("uint8"))
            aux_main<uint8_t>(base_file, query_file, gt_file, tags_file, K, metric);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
