// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// compute_groundtruth_batched: Memory-efficient batched GT computation
//
// Loads base vectors in batches (default 10M) instead of all at once.
// Uses MKL cblas_sgemm for distance computation, OpenMP for parallelism.
// Peak memory: ~batch_size * dim * 4 + q_batch * batch_active * 4 bytes
//
// Tags file format: [npts (uint32), dim=1 (uint32), tag1, tag2, ...] where tag = base_index + 1

#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cstddef>
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

template <class T>
T div_round_up(const T numerator, const T denominator)
{
    return (numerator % denominator == 0) ? (numerator / denominator) : 1 + (numerator / denominator);
}

template <class T>
T *aligned_malloc(const size_t n, const size_t alignment)
{
#ifdef _WINDOWS
    return (T *)_aligned_malloc(sizeof(T) * n, alignment);
#else
    return static_cast<T *>(aligned_alloc(alignment, sizeof(T) * n));
#endif
}

void compute_l2sq(float *const points_l2sq, const float *const matrix,
                  const int64_t num_points, const uint64_t dim)
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
                      const float *const queries_l2sq)
{
    float *ones_vec = new float[nqueries > npoints ? nqueries : npoints];
    std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, 1.0f);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, -2.0f,
                points, dim, queries, dim, 0.0f, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, 1.0f,
                points_l2sq, npoints, ones_vec, nqueries, 1.0f, dist_matrix, npoints);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, 1.0f,
                ones_vec, npoints, queries_l2sq, nqueries, 1.0f, dist_matrix, npoints);
    delete[] ones_vec;
}

// Load tags file
std::vector<uint32_t> load_tags(const std::string &tags_file)
{
    std::ifstream reader(tags_file, std::ios::binary);
    if (!reader.is_open())
        throw std::runtime_error("Cannot open tags file: " + tags_file);

    uint32_t npts, ndims;
    reader.read((char *)&npts, sizeof(uint32_t));
    reader.read((char *)&ndims, sizeof(uint32_t));
    if (ndims != 1)
        throw std::runtime_error("Tags file dimension must be 1, got " + std::to_string(ndims));

    std::vector<uint32_t> tags(npts);
    reader.read((char *)tags.data(), npts * sizeof(uint32_t));
    reader.close();
    std::cout << "Loaded " << npts << " tags" << std::endl;
    return tags;
}

// Load query file as float
template <typename T>
float *load_queries(const std::string &query_file, size_t &nqueries, size_t &dim)
{
    std::ifstream reader(query_file, std::ios::binary);
    if (!reader.is_open())
        throw std::runtime_error("Cannot open query file: " + query_file);

    int32_t npts_i32, ndims_i32;
    reader.read((char *)&npts_i32, sizeof(int32_t));
    reader.read((char *)&ndims_i32, sizeof(int32_t));
    nqueries = npts_i32;
    dim = ndims_i32;

    std::vector<T> temp_data(nqueries * dim);
    reader.read((char *)temp_data.data(), nqueries * dim * sizeof(T));
    reader.close();

    float *data = aligned_malloc<float>(nqueries * dim, ALIGNMENT);
#pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (int64_t)(nqueries * dim); i++)
        data[i] = static_cast<float>(temp_data[i]);

    std::cout << "Loaded " << nqueries << " queries (dim=" << dim << ")" << std::endl;
    return data;
}

void save_groundtruth(const std::string &filename, int32_t *data, float *distances,
                      size_t npts, size_t ndims)
{
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    int npts_i32 = (int)npts, ndims_i32 = (int)ndims;
    writer.write((char *)&npts_i32, sizeof(int));
    writer.write((char *)&ndims_i32, sizeof(int));
    writer.write((char *)data, npts * ndims * sizeof(uint32_t));
    writer.write((char *)distances, npts * ndims * sizeof(float));
    writer.close();
    std::cout << "Saved groundtruth: " << npts << " queries × K=" << ndims << std::endl;
}

template <typename T>
int aux_main(const std::string &base_file, const std::string &query_file,
             const std::string &gt_file, const std::string &tags_file,
             size_t k, size_t base_batch_size, const diskann::Metric &metric)
{
    auto t_start = std::chrono::steady_clock::now();

    // 1. Load tags → build sorted active indices + tag mapping
    std::vector<uint32_t> tags = load_tags(tags_file);
    std::vector<uint32_t> active_indices;
    std::vector<uint32_t> active_tags;
    active_indices.reserve(tags.size());
    active_tags.reserve(tags.size());
    for (uint32_t tag : tags)
    {
        if (tag > 0)
        {
            active_indices.push_back(tag - 1);
            active_tags.push_back(tag);
        }
    }
    size_t n_active = active_indices.size();
    std::cout << "Active vectors: " << n_active << std::endl;

    // Sort by base index for sequential I/O
    std::vector<size_t> sort_order(n_active);
    std::iota(sort_order.begin(), sort_order.end(), 0);
    std::sort(sort_order.begin(), sort_order.end(),
              [&](size_t a, size_t b) { return active_indices[a] < active_indices[b]; });

    std::vector<uint32_t> sorted_indices(n_active);
    std::vector<uint32_t> sorted_tags(n_active);
    for (size_t i = 0; i < n_active; i++)
    {
        sorted_indices[i] = active_indices[sort_order[i]];
        sorted_tags[i] = active_tags[sort_order[i]];
    }
    // Free originals
    { std::vector<uint32_t>().swap(active_indices); }
    { std::vector<uint32_t>().swap(active_tags); }
    { std::vector<size_t>().swap(sort_order); }
    { std::vector<uint32_t>().swap(tags); }

    // 2. Load queries
    size_t nqueries, dim;
    float *query_data = load_queries<T>(query_file, nqueries, dim);

    // Precompute query L2 norms
    float *queries_l2sq = new float[nqueries];
    compute_l2sq(queries_l2sq, query_data, nqueries, dim);

    // 3. Read base file header
    std::ifstream base_reader(base_file, std::ios::binary);
    if (!base_reader.is_open())
        throw std::runtime_error("Cannot open base file: " + base_file);

    int32_t base_npts, base_dim;
    base_reader.read((char *)&base_npts, sizeof(int32_t));
    base_reader.read((char *)&base_dim, sizeof(int32_t));
    base_reader.close();

    if ((size_t)base_dim != dim)
        throw std::runtime_error("Dimension mismatch: base=" + std::to_string(base_dim) +
                                 ", query=" + std::to_string(dim));

    std::cout << "Base file: " << base_npts << " points, dim=" << base_dim << std::endl;

    // 4. Allocate per-query top-K results
    //    For each query, maintain a max-heap of (distance, tag) pairs
    std::vector<std::vector<std::pair<float, uint32_t>>> topk_results(nqueries);
    for (size_t q = 0; q < nqueries; q++)
        topk_results[q].reserve(k + 1);

    // 5. Process base in batches
    size_t header_bytes = 2 * sizeof(int32_t);
    size_t vector_bytes = dim * sizeof(T);
    uint32_t max_base_idx = sorted_indices.back();
    size_t total_batches = div_round_up((size_t)(max_base_idx + 1), base_batch_size);

    std::cout << "\nBatched GT: " << nqueries << " queries × " << n_active
              << " active, K=" << k << ", base_batch=" << base_batch_size << std::endl;

    for (size_t batch_start = 0; batch_start <= max_base_idx; batch_start += base_batch_size)
    {
        size_t batch_end = std::min(batch_start + base_batch_size, (size_t)base_npts);
        size_t batch_idx = batch_start / base_batch_size;

        // Find active points in this batch range
        auto it_lo = std::lower_bound(sorted_indices.begin(), sorted_indices.end(), (uint32_t)batch_start);
        auto it_hi = std::lower_bound(sorted_indices.begin(), sorted_indices.end(), (uint32_t)batch_end);
        size_t active_offset = it_lo - sorted_indices.begin();
        size_t n_batch_active = it_hi - it_lo;

        if (n_batch_active == 0)
            continue;

        auto t_batch = std::chrono::steady_clock::now();

        // Read batch from file
        size_t batch_count = batch_end - batch_start;
        std::vector<T> raw_data(batch_count * dim);
        {
            std::ifstream reader(base_file, std::ios::binary);
            reader.seekg(header_bytes + batch_start * vector_bytes);
            reader.read((char *)raw_data.data(), batch_count * vector_bytes);
        }

        // Extract active vectors in this batch → float32
        float *batch_points = aligned_malloc<float>(n_batch_active * dim, ALIGNMENT);
#pragma omp parallel for schedule(static, 1024)
        for (int64_t i = 0; i < (int64_t)n_batch_active; i++)
        {
            uint32_t base_idx = sorted_indices[active_offset + i];
            size_t local_idx = base_idx - batch_start;
            for (size_t j = 0; j < dim; j++)
                batch_points[i * dim + j] = static_cast<float>(raw_data[local_idx * dim + j]);
        }
        { std::vector<T>().swap(raw_data); }

        // Compute L2 norms for batch points
        float *batch_l2sq = new float[n_batch_active];
        compute_l2sq(batch_l2sq, batch_points, n_batch_active, dim);

        // Determine query batch size to keep dist_matrix under ~8 GB
        size_t max_dist_bytes = (size_t)8 * 1024 * 1024 * 1024ULL;
        size_t q_batch = std::max((size_t)1, max_dist_bytes / (n_batch_active * sizeof(float)));
        q_batch = std::min(q_batch, nqueries);

        float *dist_matrix = new float[q_batch * n_batch_active];

        for (size_t qb = 0; qb < nqueries; qb += q_batch)
        {
            size_t qe = std::min(qb + q_batch, nqueries);
            size_t n_q = qe - qb;

            // Compute distances: MKL sgemm
            distsq_to_points(dim, dist_matrix, n_batch_active, batch_points,
                             batch_l2sq, n_q,
                             query_data + qb * dim, queries_l2sq + qb);

            // Update top-K for each query
#pragma omp parallel for schedule(dynamic, 64)
            for (int64_t qi = 0; qi < (int64_t)n_q; qi++)
            {
                size_t q = qb + qi;
                auto &heap = topk_results[q];
                float worst = (heap.size() >= k) ? heap.front().first : std::numeric_limits<float>::max();

                for (size_t p = 0; p < n_batch_active; p++)
                {
                    float d = dist_matrix[p + qi * n_batch_active];
                    if (d < worst || heap.size() < k)
                    {
                        heap.push_back({d, sorted_tags[active_offset + p]});
                        std::push_heap(heap.begin(), heap.end());
                        if (heap.size() > k)
                        {
                            std::pop_heap(heap.begin(), heap.end());
                            heap.pop_back();
                        }
                        worst = heap.front().first;
                    }
                }
            }
        }

        delete[] dist_matrix;
        delete[] batch_l2sq;
        free(batch_points);

        auto t_batch_end = std::chrono::steady_clock::now();
        double batch_sec = std::chrono::duration<double>(t_batch_end - t_batch).count();
        double total_sec = std::chrono::duration<double>(t_batch_end - t_start).count();
        std::cout << "  Batch " << (batch_idx + 1) << "/" << total_batches
                  << ": " << n_batch_active << " active, "
                  << batch_sec << "s (total " << total_sec << "s)" << std::endl;
    }

    // 6. Convert results to output format
    int32_t *result_tags = new int32_t[nqueries * k];
    float *result_dists = new float[nqueries * k];

#pragma omp parallel for schedule(static, 256)
    for (int64_t q = 0; q < (int64_t)nqueries; q++)
    {
        auto &heap = topk_results[q];
        // Sort by distance (ascending)
        std::sort_heap(heap.begin(), heap.end());
        std::reverse(heap.begin(), heap.end());  // sort_heap gives descending after pop order; reverse

        // Actually, sort properly
        std::sort(heap.begin(), heap.end(),
                  [](const std::pair<float, uint32_t> &a, const std::pair<float, uint32_t> &b) {
                      return a.first < b.first;
                  });

        for (size_t i = 0; i < k && i < heap.size(); i++)
        {
            result_tags[q * k + i] = static_cast<int32_t>(heap[i].second);
            result_dists[q * k + i] = heap[i].first;
        }
        // Fill remaining with zeros if heap < k
        for (size_t i = heap.size(); i < k; i++)
        {
            result_tags[q * k + i] = 0;
            result_dists[q * k + i] = std::numeric_limits<float>::max();
        }
    }

    // 7. Save
    save_groundtruth(gt_file, result_tags, result_dists, nqueries, k);

    auto t_end = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Total time: " << total << "s" << std::endl;

    // Cleanup
    delete[] result_tags;
    delete[] result_dists;
    delete[] queries_l2sq;
    free(query_data);

    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, base_file, query_file, gt_file, tags_file;
    uint64_t K;
    uint64_t base_batch_size = 50000000;

    try
    {
        po::options_description desc{"Arguments"};
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                           "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "distance function <l2/mips/cosine>");
        desc.add_options()("base_file", po::value<std::string>(&base_file)->required(),
                           "Base vectors file");
        desc.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                           "Query vectors file");
        desc.add_options()("gt_file", po::value<std::string>(&gt_file)->required(),
                           "Output ground truth file");
        desc.add_options()("tags_file", po::value<std::string>(&tags_file)->required(),
                           "Active tags file (DiskANN format)");
        desc.add_options()("K", po::value<uint64_t>(&K)->required(),
                           "Number of nearest neighbors");
        desc.add_options()("base_batch_size", po::value<uint64_t>(&base_batch_size)->default_value(50000000),
                           "Base vectors batch size (default 10M)");

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

    if (data_type != "float" && data_type != "int8" && data_type != "uint8")
    {
        std::cerr << "Unsupported type. float, int8 and uint8 types are supported." << std::endl;
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == "l2")
        metric = diskann::Metric::L2;
    else if (dist_fn == "mips")
        metric = diskann::Metric::INNER_PRODUCT;
    else if (dist_fn == "cosine")
        metric = diskann::Metric::COSINE;
    else
    {
        std::cerr << "Unsupported distance function. Use l2/mips/cosine." << std::endl;
        return -1;
    }

    try
    {
        if (data_type == "float")
            aux_main<float>(base_file, query_file, gt_file, tags_file, K, base_batch_size, metric);
        else if (data_type == "int8")
            aux_main<int8_t>(base_file, query_file, gt_file, tags_file, K, base_batch_size, metric);
        else if (data_type == "uint8")
            aux_main<uint8_t>(base_file, query_file, gt_file, tags_file, K, base_batch_size, metric);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
