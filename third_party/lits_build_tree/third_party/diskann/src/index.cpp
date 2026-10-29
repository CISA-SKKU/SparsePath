// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>

#include <type_traits>
#include <random>
#include <algorithm>
#include <numeric>
#include <queue>
#include <set>

#include "boost/dynamic_bitset.hpp"
#include "index_factory.h"
#include "memory_mapper.h"
#include "timer.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "windows_customizations.h"
#include "tag_uint128.h"
#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#ifdef _WINDOWS
#include <xmmintrin.h>
#endif

#include "index.h"
#include "trace_writer.h"

#define MAX_POINTS_FOR_USING_BITSET 10000000

namespace diskann
{
// Initialize an index with metric m, load the data of type T with filename
// (bin), and initialize max_points
template <typename T, typename TagT, typename LabelT>
Index<T, TagT, LabelT>::Index(const IndexConfig &index_config, std::shared_ptr<AbstractDataStore<T>> data_store,
                              std::unique_ptr<AbstractGraphStore> graph_store,
                              std::shared_ptr<AbstractDataStore<T>> pq_data_store)
    : _dist_metric(index_config.metric), _dim(index_config.dimension), _max_points(index_config.max_points),
      _num_frozen_pts(index_config.num_frozen_pts), _dynamic_index(index_config.dynamic_index),
      _enable_tags(index_config.enable_tags), _indexingMaxC(DEFAULT_MAXC), _query_scratch(nullptr),
      _pq_dist(index_config.pq_dist_build), _use_opq(index_config.use_opq),
      _filtered_index(index_config.filtered_index), _num_pq_chunks(index_config.num_pq_chunks),
      _delete_set(new tsl::robin_set<uint32_t>), _conc_consolidate(index_config.concurrent_consolidate)
{
    if (_dynamic_index && !_enable_tags)
    {
        throw ANNException("ERROR: Dynamic Indexing must have tags enabled.", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (_pq_dist)
    {
        if (_dynamic_index)
            throw ANNException("ERROR: Dynamic Indexing not supported with PQ distance based "
                               "index construction",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        if (_dist_metric == diskann::Metric::INNER_PRODUCT)
            throw ANNException("ERROR: Inner product metrics not yet supported "
                               "with PQ distance "
                               "base index",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (_dynamic_index && _num_frozen_pts == 0)
    {
        _num_frozen_pts = 1;
    }
    // Sanity check. While logically it is correct, max_points = 0 causes
    // downstream problems.
    if (_max_points == 0)
    {
        _max_points = 1;
    }
    const size_t total_internal_points = _max_points + _num_frozen_pts;

    _start = (uint32_t)_max_points;

    _data_store = data_store;
    _pq_data_store = pq_data_store;
    _graph_store = std::move(graph_store);

    _locks = std::vector<non_recursive_mutex>(total_internal_points);
    if (_enable_tags)
    {
        _location_to_tag.reserve(total_internal_points);
        _tag_to_location.reserve(total_internal_points);
    }

    if (_dynamic_index)
    {
        this->enable_delete(); // enable delete by default for dynamic index
        if (_filtered_index)
        {
            _location_to_labels.resize(total_internal_points);
        }
    }

    if (index_config.index_write_params != nullptr)
    {
        _indexingQueueSize = index_config.index_write_params->search_list_size;
        _indexingRange = index_config.index_write_params->max_degree;
        _indexingMaxC = index_config.index_write_params->max_occlusion_size;
        _indexingAlpha = index_config.index_write_params->alpha;
        _filterIndexingQueueSize = index_config.index_write_params->filter_list_size;
        _indexingThreads = index_config.index_write_params->num_threads;
        _saturate_graph = index_config.index_write_params->saturate_graph;
        _pruningStrategy = index_config.index_write_params->pruning_strategy;
        _randomPruneRatio = index_config.index_write_params->random_prune_ratio;

        // Load partition file if specified (for PARTITION_AWARE_PRUNE strategy)
        if (!index_config.index_write_params->partition_file.empty())
        {
            load_partition_file(index_config.index_write_params->partition_file);
        }

        if (index_config.index_search_params != nullptr)
        {
            uint32_t num_scratch_spaces = index_config.index_search_params->num_search_threads + _indexingThreads;
            initialize_query_scratch(num_scratch_spaces, index_config.index_search_params->initial_search_list_size,
                                     _indexingQueueSize, _indexingRange, _indexingMaxC, _data_store->get_dims());
        }
    }

    // Initialize pruning trace buffers (one per thread)
    uint32_t num_threads = _indexingThreads > 0 ? _indexingThreads : omp_get_max_threads();
    _inter_insert_trace_buffers.resize(num_threads);
    _delete_consolidate_trace_buffers.resize(num_threads);
    _delete_detail_trace_buffers.resize(num_threads);
}

template <typename T, typename TagT, typename LabelT>
Index<T, TagT, LabelT>::Index(Metric m, const size_t dim, const size_t max_points,
                              const std::shared_ptr<IndexWriteParameters> index_parameters,
                              const std::shared_ptr<IndexSearchParams> index_search_params, const size_t num_frozen_pts,
                              const bool dynamic_index, const bool enable_tags, const bool concurrent_consolidate,
                              const bool pq_dist_build, const size_t num_pq_chunks, const bool use_opq,
                              const bool filtered_index)
    : Index(
          IndexConfigBuilder()
              .with_metric(m)
              .with_dimension(dim)
              .with_max_points(max_points)
              .with_index_write_params(index_parameters)
              .with_index_search_params(index_search_params)
              .with_num_frozen_pts(num_frozen_pts)
              .is_dynamic_index(dynamic_index)
              .is_enable_tags(enable_tags)
              .is_concurrent_consolidate(concurrent_consolidate)
              .is_pq_dist_build(pq_dist_build)
              .with_num_pq_chunks(num_pq_chunks)
              .is_use_opq(use_opq)
              .is_filtered(filtered_index)
              .with_data_type(diskann_type_to_name<T>())
              .build(),
          IndexFactory::construct_datastore<T>(DataStoreStrategy::MEMORY,
                                               (max_points == 0 ? (size_t)1 : max_points) +
                                                   (dynamic_index && num_frozen_pts == 0 ? (size_t)1 : num_frozen_pts),
                                               dim, m),
          IndexFactory::construct_graphstore(GraphStoreStrategy::MEMORY,
                                             (max_points == 0 ? (size_t)1 : max_points) +
                                                 (dynamic_index && num_frozen_pts == 0 ? (size_t)1 : num_frozen_pts),
                                             (size_t)((index_parameters == nullptr ? 0 : index_parameters->max_degree) *
                                                      defaults::GRAPH_SLACK_FACTOR * 1.05)))
{
    if (_pq_dist)
    {
        _pq_data_store = IndexFactory::construct_pq_datastore<T>(DataStoreStrategy::MEMORY, max_points + num_frozen_pts,
                                                                 dim, m, num_pq_chunks, use_opq);
    }
    else
    {
        _pq_data_store = _data_store;
    }
}

template <typename T, typename TagT, typename LabelT> Index<T, TagT, LabelT>::~Index()
{
    // Ensure that no other activity is happening before dtor()
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    for (auto &lock : _locks)
    {
        LockGuard lg(lock);
    }

    if (_opt_graph != nullptr)
    {
        delete[] _opt_graph;
    }

    if (!_query_scratch.empty())
    {
        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
        manager.destroy();
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::initialize_query_scratch(uint32_t num_threads, uint32_t search_l, uint32_t indexing_l,
                                                      uint32_t r, uint32_t maxc, size_t dim)
{
    for (uint32_t i = 0; i < num_threads; i++)
    {
        auto scratch = new InMemQueryScratch<T>(search_l, indexing_l, r, maxc, dim, _data_store->get_aligned_dim(),
                                                _data_store->get_alignment_factor(), _pq_dist);
        _query_scratch.push(scratch);
    }
}

template <typename T, typename TagT, typename LabelT> size_t Index<T, TagT, LabelT>::save_tags(std::string tags_file)
{
    if (!_enable_tags)
    {
        diskann::cout << "Not saving tags as they are not enabled." << std::endl;
        return 0;
    }

    size_t tag_bytes_written;
    TagT *tag_data = new TagT[_nd + _num_frozen_pts];
    for (uint32_t i = 0; i < _nd; i++)
    {
        TagT tag;
        if (_location_to_tag.try_get(i, tag))
        {
            tag_data[i] = tag;
        }
        else
        {
            // catering to future when tagT can be any type.
            std::memset((char *)&tag_data[i], 0, sizeof(TagT));
        }
    }
    if (_num_frozen_pts > 0)
    {
        // Frozen points are saved at tag_data[_nd], not tag_data[_start]
        std::memset((char *)&tag_data[_nd], 0, sizeof(TagT) * _num_frozen_pts);
    }
    try
    {
        tag_bytes_written = save_bin<TagT>(tags_file, tag_data, _nd + _num_frozen_pts, 1);
    }
    catch (std::system_error &e)
    {
        throw FileException(tags_file, e, __FUNCSIG__, __FILE__, __LINE__);
    }
    delete[] tag_data;
    return tag_bytes_written;
}

template <typename T, typename TagT, typename LabelT> size_t Index<T, TagT, LabelT>::save_data(std::string data_file)
{
    // Note: at this point, either _nd == _max_points or any frozen points have
    // been temporarily moved to _nd, so _nd + _num_frozen_pts is the valid
    // location limit.
    return _data_store->save(data_file, (location_t)(_nd + _num_frozen_pts));
}

// save the graph index on a file as an adjacency list. For each point,
// first store the number of neighbors, and then the neighbor list (each as
// 4 byte uint32_t)
template <typename T, typename TagT, typename LabelT> size_t Index<T, TagT, LabelT>::save_graph(std::string graph_file)
{
    return _graph_store->store(graph_file, _nd + _num_frozen_pts, _num_frozen_pts, _start);
}

template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::save_delete_list(const std::string &filename)
{
    if (_delete_set->size() == 0)
    {
        return 0;
    }
    std::unique_ptr<uint32_t[]> delete_list = std::make_unique<uint32_t[]>(_delete_set->size());
    uint32_t i = 0;
    for (auto &del : *_delete_set)
    {
        delete_list[i++] = del;
    }
    return save_bin<uint32_t>(filename, delete_list.get(), _delete_set->size(), 1);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::save(const char *filename, bool compact_before_save, bool save_data_file)
{
    diskann::Timer timer;

    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    if (compact_before_save)
    {
        compact_data();
        compact_frozen_point();
    }
    else
    {
        if (!_data_compacted)
        {
            throw ANNException("Index save for non-compacted index is not yet implemented", -1, __FUNCSIG__, __FILE__,
                               __LINE__);
        }
    }

    if (!_save_as_one_file)
    {
        if (_filtered_index)
        {
            if (_label_to_start_id.size() > 0)
            {
                std::ofstream medoid_writer(std::string(filename) + "_labels_to_medoids.txt");
                if (medoid_writer.fail())
                {
                    throw diskann::ANNException(std::string("Failed to open file ") + filename, -1);
                }
                for (auto iter : _label_to_start_id)
                {
                    medoid_writer << iter.first << ", " << iter.second << std::endl;
                }
                medoid_writer.close();
            }

            if (_use_universal_label)
            {
                std::ofstream universal_label_writer(std::string(filename) + "_universal_label.txt");
                assert(universal_label_writer.is_open());
                universal_label_writer << _universal_label << std::endl;
                universal_label_writer.close();
            }

            if (_location_to_labels.size() > 0)
            {
                std::ofstream label_writer(std::string(filename) + "_labels.txt");
                assert(label_writer.is_open());
                for (uint32_t i = 0; i < _nd + _num_frozen_pts; i++)
                {
                    for (uint32_t j = 0; j + 1 < _location_to_labels[i].size(); j++)
                    {
                        label_writer << _location_to_labels[i][j] << ",";
                    }
                    if (_location_to_labels[i].size() != 0)
                        label_writer << _location_to_labels[i][_location_to_labels[i].size() - 1];

                    label_writer << std::endl;
                }
                label_writer.close();

                // write compacted raw_labels if data hence _location_to_labels was also compacted
                if (compact_before_save && _dynamic_index)
                {
                    _label_map = load_label_map(std::string(filename) + "_labels_map.txt");
                    std::unordered_map<LabelT, std::string> mapped_to_raw_labels;
                    // invert label map
                    for (const auto &[key, value] : _label_map)
                    {
                        mapped_to_raw_labels.insert({value, key});
                    }

                    // write updated labels
                    std::ofstream raw_label_writer(std::string(filename) + "_raw_labels.txt");
                    assert(raw_label_writer.is_open());
                    for (uint32_t i = 0; i < _nd + _num_frozen_pts; i++)
                    {
                        for (uint32_t j = 0; j + 1 < _location_to_labels[i].size(); j++)
                        {
                            raw_label_writer << mapped_to_raw_labels[_location_to_labels[i][j]] << ",";
                        }
                        if (_location_to_labels[i].size() != 0)
                            raw_label_writer
                                << mapped_to_raw_labels[_location_to_labels[i][_location_to_labels[i].size() - 1]];

                        raw_label_writer << std::endl;
                    }
                    raw_label_writer.close();
                }
            }
        }

        std::string graph_file = std::string(filename);
        std::string tags_file = std::string(filename) + ".tags";
        std::string data_file = std::string(filename) + ".data";
        std::string delete_list_file = std::string(filename) + ".del";

        // Because the save_* functions use append mode, ensure that
        // the files are deleted before save. Ideally, we should check
        // the error code for delete_file, but will ignore now because
        // delete should succeed if save will succeed.
        delete_file(graph_file);
        save_graph(graph_file);
        if (save_data_file)
        {
            delete_file(data_file);
            save_data(data_file);
        }
        delete_file(tags_file);
        save_tags(tags_file);
        delete_file(delete_list_file);
        save_delete_list(delete_list_file);
    }
    else
    {
        diskann::cout << "Save index in a single file currently not supported. "
                         "Not saving the index."
                      << std::endl;
    }

    // If frozen points were temporarily compacted to _nd, move back to
    // _max_points.
    reposition_frozen_point_to_end();

    diskann::cout << "Time taken for save: " << timer.elapsed() / 1000000.0 << "s." << std::endl;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::load_tags(AlignedFileReader &reader)
{
#else
template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::load_tags(const std::string tag_filename)
{
    if (_enable_tags && !file_exists(tag_filename))
    {
        diskann::cerr << "Tag file " << tag_filename << " does not exist!" << std::endl;
        throw diskann::ANNException("Tag file " + tag_filename + " does not exist!", -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
    }
#endif
    if (!_enable_tags)
    {
        diskann::cout << "Tags not loaded as tags not enabled." << std::endl;
        return 0;
    }

    size_t file_dim, file_num_points;
    TagT *tag_data;
#ifdef EXEC_ENV_OLS
    load_bin<TagT>(reader, tag_data, file_num_points, file_dim);
#else
    load_bin<TagT>(std::string(tag_filename), tag_data, file_num_points, file_dim);
#endif

    if (file_dim != 1)
    {
        std::stringstream stream;
        stream << "ERROR: Found " << file_dim << " dimensions for tags,"
               << "but tag file must have 1 dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        delete[] tag_data;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    const size_t num_data_points = file_num_points - _num_frozen_pts;
    _location_to_tag.reserve(num_data_points);
    _tag_to_location.reserve(num_data_points);
    for (uint32_t i = 0; i < (uint32_t)num_data_points; i++)
    {
        TagT tag = *(tag_data + i);
        if (_delete_set->find(i) == _delete_set->end())
        {
            _location_to_tag.set(i, tag);
            _tag_to_location[tag] = i;
        }
    }
    diskann::cout << "Tags loaded." << std::endl;
    delete[] tag_data;
    return file_num_points;
}

template <typename T, typename TagT, typename LabelT>
#ifdef EXEC_ENV_OLS
size_t Index<T, TagT, LabelT>::load_data(AlignedFileReader &reader)
{
#else
size_t Index<T, TagT, LabelT>::load_data(std::string filename)
{
#endif
    size_t file_dim, file_num_points;
#ifdef EXEC_ENV_OLS
    diskann::get_bin_metadata(reader, file_num_points, file_dim);
#else
    if (!file_exists(filename))
    {
        std::stringstream stream;
        stream << "ERROR: data file " << filename << " does not exist." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    diskann::get_bin_metadata(filename, file_num_points, file_dim);
#endif

    // since we are loading a new dataset, _empty_slots must be cleared
    _empty_slots.clear();

    if (file_dim != _dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_num_points > _max_points + _num_frozen_pts)
    {
        // update and tag lock acquired in load() before calling load_data
        resize(file_num_points - _num_frozen_pts);
    }

#ifdef EXEC_ENV_OLS
    // REFACTOR TODO: Must figure out how to support aligned reader in a clean
    // manner.
    copy_aligned_data_from_file<T>(reader, _data, file_num_points, file_dim, _data_store->get_aligned_dim());
#else
    _data_store->load(filename); // offset == 0.
#endif
    return file_num_points;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::load_delete_set(AlignedFileReader &reader)
{
#else
template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::load_delete_set(const std::string &filename)
{
#endif
    std::unique_ptr<uint32_t[]> delete_list;
    size_t npts, ndim;

#ifdef EXEC_ENV_OLS
    diskann::load_bin<uint32_t>(reader, delete_list, npts, ndim);
#else
    diskann::load_bin<uint32_t>(filename, delete_list, npts, ndim);
#endif
    assert(ndim == 1);
    for (uint32_t i = 0; i < npts; i++)
    {
        _delete_set->insert(delete_list[i]);
    }
    return npts;
}

// load the index from file and update the max_degree, cur (navigating
// node loc), and _final_graph (adjacency list)
template <typename T, typename TagT, typename LabelT>
#ifdef EXEC_ENV_OLS
void Index<T, TagT, LabelT>::load(AlignedFileReader &reader, uint32_t num_threads, uint32_t search_l)
{
#else
void Index<T, TagT, LabelT>::load(const char *filename, uint32_t num_threads, uint32_t search_l)
{
#endif
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    _has_built = true;

    size_t tags_file_num_pts = 0, graph_num_pts = 0, data_file_num_pts = 0, label_num_pts = 0;

    std::string mem_index_file(filename);
    std::string labels_file = mem_index_file + "_labels.txt";
    std::string labels_to_medoids = mem_index_file + "_labels_to_medoids.txt";
    std::string labels_map_file = mem_index_file + "_labels_map.txt";

    if (!_save_as_one_file)
    {
        // For DLVS Store, we will not support saving the index in multiple
        // files.
#ifndef EXEC_ENV_OLS
        std::string data_file = std::string(filename) + ".data";
        std::string tags_file = std::string(filename) + ".tags";
        std::string delete_set_file = std::string(filename) + ".del";
        std::string graph_file = std::string(filename);
        data_file_num_pts = load_data(data_file);
        if (file_exists(delete_set_file))
        {
            load_delete_set(delete_set_file);
        }
        if (_enable_tags)
        {
            tags_file_num_pts = load_tags(tags_file);
        }
        graph_num_pts = load_graph(graph_file, data_file_num_pts);
#endif
    }
    else
    {
        diskann::cout << "Single index file saving/loading support not yet "
                         "enabled. Not loading the index."
                      << std::endl;
        return;
    }

    if (data_file_num_pts != graph_num_pts || (data_file_num_pts != tags_file_num_pts && _enable_tags))
    {
        std::stringstream stream;
        stream << "ERROR: When loading index, loaded " << data_file_num_pts << " points from datafile, "
               << graph_num_pts << " from graph, and " << tags_file_num_pts
               << " tags, with num_frozen_pts being set to " << _num_frozen_pts << " in constructor." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_exists(labels_file))
    {
        _label_map = load_label_map(labels_map_file);
        parse_label_file(labels_file, label_num_pts);
        assert(label_num_pts == data_file_num_pts - _num_frozen_pts);
        if (file_exists(labels_to_medoids))
        {
            std::ifstream medoid_stream(labels_to_medoids);
            std::string line, token;
            uint32_t line_cnt = 0;

            _label_to_start_id.clear();

            while (std::getline(medoid_stream, line))
            {
                std::istringstream iss(line);
                uint32_t cnt = 0;
                uint32_t medoid = 0;
                LabelT label;
                while (std::getline(iss, token, ','))
                {
                    token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
                    token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
                    LabelT token_as_num = (LabelT)std::stoul(token);
                    if (cnt == 0)
                        label = token_as_num;
                    else
                        medoid = token_as_num;
                    cnt++;
                }
                _label_to_start_id[label] = medoid;
                line_cnt++;
            }
        }

        std::string universal_label_file(filename);
        universal_label_file += "_universal_label.txt";
        if (file_exists(universal_label_file))
        {
            std::ifstream universal_label_reader(universal_label_file);
            universal_label_reader >> _universal_label;
            _use_universal_label = true;
            universal_label_reader.close();
        }
    }

    _nd = data_file_num_pts - _num_frozen_pts;
    _empty_slots.clear();
    _empty_slots.reserve(_max_points);
    for (auto i = _nd; i < _max_points; i++)
    {
        _empty_slots.insert((uint32_t)i);
    }

    reposition_frozen_point_to_end();
    diskann::cout << "Num frozen points:" << _num_frozen_pts << " _nd: " << _nd << " _start: " << _start
                  << " size(_location_to_tag): " << _location_to_tag.size()
                  << " size(_tag_to_location):" << _tag_to_location.size() << " Max points: " << _max_points
                  << std::endl;

    // For incremental index, _query_scratch is initialized in the constructor.
    // For the bulk index, the params required to initialize _query_scratch
    // are known only at load time, hence this check and the call to
    // initialize_q_s().
    if (_query_scratch.size() == 0)
    {
        initialize_query_scratch(num_threads, search_l, search_l, (uint32_t)_graph_store->get_max_range_of_graph(),
                                 _indexingMaxC, _dim);
    }
}

#ifndef EXEC_ENV_OLS
template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::get_graph_num_frozen_points(const std::string &graph_file)
{
    size_t expected_file_size;
    uint32_t max_observed_degree, start;
    size_t file_frozen_pts;

    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);

    in.open(graph_file, std::ios::binary);
    in.read((char *)&expected_file_size, sizeof(size_t));
    in.read((char *)&max_observed_degree, sizeof(uint32_t));
    in.read((char *)&start, sizeof(uint32_t));
    in.read((char *)&file_frozen_pts, sizeof(size_t));

    return file_frozen_pts;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::load_graph_only(const char *graph_file, const std::string &base_file,
                                              const std::string &tags_file, uint32_t num_threads, uint32_t search_l)
{
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    _has_built = true;

    // Step 1: Load tags file to get tag array
    // Tags file format: [npts (uint32), dim=1 (uint32), tags... (uint32)]
    std::ifstream tags_reader(tags_file, std::ios::binary);
    if (!tags_reader.is_open())
    {
        throw diskann::ANNException("Failed to open tags file: " + tags_file, -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    uint32_t tags_npts, tags_dim;
    tags_reader.read(reinterpret_cast<char *>(&tags_npts), sizeof(uint32_t));
    tags_reader.read(reinterpret_cast<char *>(&tags_dim), sizeof(uint32_t));

    if (tags_dim != 1)
    {
        tags_reader.close();
        throw diskann::ANNException("Tags file dimension must be 1, got " + std::to_string(tags_dim), -1, __FUNCSIG__,
                                    __FILE__, __LINE__);
    }

    std::vector<uint32_t> raw_tags(tags_npts);
    tags_reader.read(reinterpret_cast<char *>(raw_tags.data()), tags_npts * sizeof(uint32_t));
    tags_reader.close();

    diskann::cout << "Loaded " << tags_npts << " tags from " << tags_file << std::endl;

    // Step 2: Read base file metadata
    std::ifstream base_reader(base_file, std::ios::binary);
    if (!base_reader.is_open())
    {
        throw diskann::ANNException("Failed to open base file: " + base_file, -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    uint32_t base_npts, base_dim;
    base_reader.read(reinterpret_cast<char *>(&base_npts), sizeof(uint32_t));
    base_reader.read(reinterpret_cast<char *>(&base_dim), sizeof(uint32_t));

    if (base_dim != _dim)
    {
        base_reader.close();
        throw diskann::ANNException("Base file dimension " + std::to_string(base_dim) + " != index dimension " +
                                        std::to_string(_dim),
                                    -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::cout << "Base file: " << base_npts << " points, " << base_dim << " dims" << std::endl;

    // Step 3: Count active points (non-zero tags) - frozen point has tag 0
    size_t num_points = tags_npts;

    // Step 4: Load vectors from base file using tags
    // Tag is 1-indexed: tag = base_index + 1
    // So base_index = tag - 1
    // NOTE: temp_data must use _dim spacing (not aligned_dim) because
    // populate_data() expects input data with _dim spacing and converts to aligned_dim internally
    T *temp_data = nullptr;
    // Round up allocation size to multiple of 8 for alloc_aligned requirement
    size_t alloc_size = ROUND_UP(num_points * _dim * sizeof(T), 8 * sizeof(T));
    alloc_aligned((void **)&temp_data, alloc_size, 8 * sizeof(T));
    std::memset(temp_data, 0, alloc_size);

    const size_t header_size = 2 * sizeof(uint32_t);
    const size_t point_size = _dim * sizeof(T);

    T *temp_vec = new T[_dim];
    for (size_t i = 0; i < num_points; i++)
    {
        uint32_t tag = raw_tags[i];

        // Tag 0 is reserved for frozen/deleted points - fill with zeros
        if (tag == 0)
        {
            // Already zeroed by memset above
            continue;
        }

        // Tags are 1-indexed: tag = base_index + 1
        size_t base_index = static_cast<size_t>(tag) - 1;

        if (base_index >= base_npts)
        {
            delete[] temp_vec;
            aligned_free(temp_data);
            throw diskann::ANNException("Tag " + std::to_string(tag) + " (base_index=" + std::to_string(base_index) +
                                            ") exceeds base file size " + std::to_string(base_npts),
                                        -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        // Seek to vector position and read
        size_t offset = header_size + base_index * point_size;
        base_reader.seekg(offset, std::ios::beg);
        base_reader.read(reinterpret_cast<char *>(temp_vec), point_size);

        // Copy to buffer with _dim spacing (populate_data will convert to aligned_dim)
        memcpy(temp_data + i * _dim, temp_vec, _dim * sizeof(T));
    }
    delete[] temp_vec;
    base_reader.close();

    diskann::cout << "Loaded " << num_points << " vectors (dim=" << _dim << ")" << std::endl;

    // Step 5: Populate data store
    _data_store->populate_data(temp_data, static_cast<location_t>(num_points));
    aligned_free(temp_data);

    // Step 6: Load graph
    size_t graph_num_pts = load_graph(std::string(graph_file), num_points);
    if (graph_num_pts != num_points)
    {
        std::stringstream stream;
        stream << "ERROR: Graph has " << graph_num_pts << " points but tags file has " << num_points << " points";
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Step 7: Setup tag mappings
    _location_to_tag.reserve(num_points);
    _tag_to_location.reserve(num_points);
    for (size_t i = 0; i < num_points; i++)
    {
        uint32_t raw_tag = raw_tags[i];
        if (raw_tag != 0)
        { // Skip frozen point tag (tag 0)
            TagT tag;
            tag = static_cast<uint64_t>(raw_tag);  // Use assignment operator (works for tag_uint128)
            _tag_to_location[tag] = static_cast<uint32_t>(i);
            _location_to_tag.set(static_cast<uint32_t>(i), tag);
        }
    }

    // Step 8: Finalize
    _nd = num_points - _num_frozen_pts;
    _empty_slots.clear();
    _empty_slots.reserve(_max_points);
    for (auto i = _nd; i < _max_points; i++)
    {
        _empty_slots.insert((uint32_t)i);
    }

    reposition_frozen_point_to_end();

    diskann::cout << "load_graph_only complete: Num frozen points=" << _num_frozen_pts << " _nd=" << _nd
                  << " _start=" << _start << " size(_location_to_tag)=" << _location_to_tag.size()
                  << " size(_tag_to_location)=" << _tag_to_location.size() << " Max points=" << _max_points
                  << std::endl;

    // Initialize query scratch if needed
    if (_query_scratch.size() == 0)
    {
        initialize_query_scratch(num_threads, search_l, search_l, (uint32_t)_graph_store->get_max_range_of_graph(),
                                 _indexingMaxC, _dim);
    }
}
#endif

#ifdef EXEC_ENV_OLS
template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::load_graph(AlignedFileReader &reader, size_t expected_num_points)
{
#else

template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::load_graph(std::string filename, size_t expected_num_points)
{
#endif
    auto res = _graph_store->load(filename, expected_num_points);
    _start = std::get<1>(res);
    _num_frozen_pts = std::get<2>(res);
    return std::get<0>(res);
}

template <typename T, typename TagT, typename LabelT>
int Index<T, TagT, LabelT>::_get_vector_by_tag(TagType &tag, DataType &vec)
{
    try
    {
        TagT tag_val = std::any_cast<TagT>(tag);
        T *vec_val = std::any_cast<T *>(vec);
        return this->get_vector_by_tag(tag_val, vec_val);
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad any cast while performing _get_vector_by_tags() " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error: " + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT> int Index<T, TagT, LabelT>::get_vector_by_tag(TagT &tag, T *vec)
{
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    if (_tag_to_location.find(tag) == _tag_to_location.end())
    {
        diskann::cout << "Tag " << get_tag_string(tag) << " does not exist" << std::endl;
        return -1;
    }

    location_t location = _tag_to_location[tag];
    _data_store->get_vector(location, vec);

    return 0;
}

template <typename T, typename TagT, typename LabelT> uint32_t Index<T, TagT, LabelT>::calculate_entry_point()
{
    // REFACTOR TODO: This function does not support multi-threaded calculation of medoid.
    // Must revisit if perf is a concern.
    return _data_store->calculate_medoid();
}

template <typename T, typename TagT, typename LabelT> std::vector<uint32_t> Index<T, TagT, LabelT>::get_init_ids()
{
    std::vector<uint32_t> init_ids;
    init_ids.reserve(1 + _num_frozen_pts);

    init_ids.emplace_back(_start);

    for (uint32_t frozen = (uint32_t)_max_points; frozen < _max_points + _num_frozen_pts; frozen++)
    {
        if (frozen != _start)
        {
            init_ids.emplace_back(frozen);
        }
    }

    return init_ids;
}

// Find common filter between a node's labels and a given set of labels, while
// taking into account universal label.
// Note: incoming_labels should be sorted.
template <typename T, typename TagT, typename LabelT>
bool Index<T, TagT, LabelT>::detect_common_filters(uint32_t point_id, bool search_invocation,
                                                   const std::vector<LabelT> &incoming_labels)
{
    auto &curr_node_labels = _location_to_labels[point_id];
    // Check for intersection between incoming_labels and curr_node_labels
    // using two-pointer approach (both vectors are sorted)
    auto it_inc = incoming_labels.begin();
    auto it_curr = curr_node_labels.begin();

    while (it_inc != incoming_labels.end() && it_curr != curr_node_labels.end())
    {
        if (*it_inc < *it_curr)
        {
            ++it_inc;
        }
        else if (*it_curr < *it_inc)
        {
            ++it_curr;
        }
        else
        {
            // common label found
            return true;
        }
    }
    // intersection empty; proceed to check the universal label logic
    
    if (_use_universal_label)
    {
        if (!search_invocation)
        {
            if (std::find(incoming_labels.begin(), incoming_labels.end(), _universal_label) != incoming_labels.end() ||
                std::find(curr_node_labels.begin(), curr_node_labels.end(), _universal_label) != curr_node_labels.end())
                return true;
        }
        else
        {
            if (std::find(curr_node_labels.begin(), curr_node_labels.end(), _universal_label) != curr_node_labels.end())
                return true;
        }
    }
    return false;
}

template <typename T, typename TagT, typename LabelT>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::iterate_to_fixed_point(
    InMemQueryScratch<T> *scratch, const uint32_t Lsize, const std::vector<uint32_t> &init_ids, bool use_filter,
    const std::vector<LabelT> &filter_labels, bool search_invocation)
{
    std::vector<Neighbor> &expanded_nodes = scratch->pool();
    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    best_L_nodes.reserve(Lsize);
    tsl::robin_set<uint32_t> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();
    boost::dynamic_bitset<> &inserted_into_pool_bs = scratch->inserted_into_pool_bs();
    std::vector<uint32_t> &id_scratch = scratch->id_scratch();
    std::vector<float> &dist_scratch = scratch->dist_scratch();
    assert(id_scratch.size() == 0);

    T *aligned_query = scratch->aligned_query();

    float *pq_dists = nullptr;

    _pq_data_store->preprocess_query(aligned_query, scratch);

    if (expanded_nodes.size() > 0 || id_scratch.size() > 0)
    {
        throw ANNException("ERROR: Clear scratch space before passing.", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Decide whether to use bitset or robin set to mark visited nodes
    auto total_num_points = _max_points + _num_frozen_pts;
    bool fast_iterate = total_num_points <= MAX_POINTS_FOR_USING_BITSET;

    if (fast_iterate)
    {
        if (inserted_into_pool_bs.size() < total_num_points)
        {
            // hopefully using 2X will reduce the number of allocations.
            auto resize_size =
                2 * total_num_points > MAX_POINTS_FOR_USING_BITSET ? MAX_POINTS_FOR_USING_BITSET : 2 * total_num_points;
            inserted_into_pool_bs.resize(resize_size);
        }
    }

    // Lambda to determine if a node has been visited
    auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](const uint32_t id) {
        return fast_iterate ? inserted_into_pool_bs[id] == 0
                            : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
    };

    // Lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, scratch, pq_dists](const std::vector<uint32_t> &ids, std::vector<float> &dists_out) {
        _pq_data_store->get_distance(scratch->aligned_query(), ids, dists_out, scratch);
    };

    // Initialize the candidate pool with starting points
    for (auto id : init_ids)
    {
        if (id >= _max_points + _num_frozen_pts)
        {
            diskann::cerr << "Out of range loc found as an edge : " << id << std::endl;
            throw diskann::ANNException(std::string("Wrong loc") + std::to_string(id), -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
        }

        if (use_filter)
        {
            if (!detect_common_filters(id, search_invocation, filter_labels))
                continue;
        }

        if (is_not_visited(id))
        {
            if (fast_iterate)
            {
                inserted_into_pool_bs[id] = 1;
            }
            else
            {
                inserted_into_pool_rs.insert(id);
            }

            float distance;
            uint32_t ids[] = {id};
            float distances[] = {std::numeric_limits<float>::max()};
            _pq_data_store->get_distance(aligned_query, ids, 1, distances, scratch);
            distance = distances[0];

            Neighbor nn = Neighbor(id, distance);
            best_L_nodes.insert(nn);
        }
    }

    uint32_t hops = 0;
    uint32_t cmps = 0;

    while (best_L_nodes.has_unexpanded_node())
    {
        auto nbr = best_L_nodes.closest_unexpanded();
        auto n = nbr.id;

        // Add node to expanded nodes to create pool for prune later
        if (!search_invocation)
        {
            if (!use_filter)
            {
                expanded_nodes.emplace_back(nbr);
            }
            else
            { // in filter based indexing, the same point might invoke
                // multiple iterate_to_fixed_points, so need to be careful
                // not to add the same item to pool multiple times.
                if (std::find(expanded_nodes.begin(), expanded_nodes.end(), nbr) == expanded_nodes.end())
                {
                    expanded_nodes.emplace_back(nbr);
                }
            }
        }

        // Find which of the nodes in des have not been visited before
        id_scratch.clear();
        dist_scratch.clear();
        if (_dynamic_index)
        {
            LockGuard guard(_locks[n]);
            for (auto id : _graph_store->get_neighbours(n))
            {
                assert(id < _max_points + _num_frozen_pts);

                if (use_filter)
                {
                    // NOTE: NEED TO CHECK IF THIS CORRECT WITH NEW LOCKS.
                    if (!detect_common_filters(id, search_invocation, filter_labels))
                        continue;
                }

                if (is_not_visited(id))
                {
                    id_scratch.push_back(id);
                }
            }
        }
        else
        {
            _locks[n].lock();
            auto nbrs = _graph_store->get_neighbours(n);
            _locks[n].unlock();
            for (auto id : nbrs)
            {
                assert(id < _max_points + _num_frozen_pts);

                if (use_filter)
                {
                    // NOTE: NEED TO CHECK IF THIS CORRECT WITH NEW LOCKS.
                    if (!detect_common_filters(id, search_invocation, filter_labels))
                        continue;
                }

                if (is_not_visited(id))
                {
                    id_scratch.push_back(id);
                }
            }
        }

        // Mark nodes visited
        for (auto id : id_scratch)
        {
            if (fast_iterate)
            {
                inserted_into_pool_bs[id] = 1;
            }
            else
            {
                inserted_into_pool_rs.insert(id);
            }
        }

        assert(dist_scratch.capacity() >= id_scratch.size());
        compute_dists(id_scratch, dist_scratch);
        cmps += (uint32_t)id_scratch.size();

        // Insert <id, dist> pairs into the pool of candidates
        for (size_t m = 0; m < id_scratch.size(); ++m)
        {
            best_L_nodes.insert(Neighbor(id_scratch[m], dist_scratch[m]));
        }
    }
    return std::make_pair(hops, cmps);
}

// iterate_to_fixed_point with detailed timing statistics for latency profiling
template <typename T, typename TagT, typename LabelT>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::iterate_to_fixed_point_timed(
    InMemQueryScratch<T> *scratch, const uint32_t Lsize, const std::vector<uint32_t> &init_ids, bool use_filter,
    const std::vector<LabelT> &filter_labels, bool search_invocation, QueryStats &stats)
{
    Timer dist_timer, fetch_timer, queue_timer, visited_timer;
    long long dist_us = 0, fetch_us = 0, queue_us = 0, visited_us = 0;

    std::vector<Neighbor> &expanded_nodes = scratch->pool();
    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    best_L_nodes.reserve(Lsize);
    tsl::robin_set<uint32_t> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();
    boost::dynamic_bitset<> &inserted_into_pool_bs = scratch->inserted_into_pool_bs();
    std::vector<uint32_t> &id_scratch = scratch->id_scratch();
    std::vector<float> &dist_scratch = scratch->dist_scratch();
    assert(id_scratch.size() == 0);

    T *aligned_query = scratch->aligned_query();

    float *pq_dists = nullptr;

    _pq_data_store->preprocess_query(aligned_query, scratch);

    if (expanded_nodes.size() > 0 || id_scratch.size() > 0)
    {
        throw ANNException("ERROR: Clear scratch space before passing.", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Decide whether to use bitset or robin set to mark visited nodes
    auto total_num_points = _max_points + _num_frozen_pts;
    bool fast_iterate = total_num_points <= MAX_POINTS_FOR_USING_BITSET;

    if (fast_iterate)
    {
        if (inserted_into_pool_bs.size() < total_num_points)
        {
            auto resize_size =
                2 * total_num_points > MAX_POINTS_FOR_USING_BITSET ? MAX_POINTS_FOR_USING_BITSET : 2 * total_num_points;
            inserted_into_pool_bs.resize(resize_size);
        }
    }

    // Lambda to determine if a node has been visited
    auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](const uint32_t id) {
        return fast_iterate ? inserted_into_pool_bs[id] == 0
                            : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
    };

    // Lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, scratch, pq_dists](const std::vector<uint32_t> &ids, std::vector<float> &dists_out) {
        _pq_data_store->get_distance(scratch->aligned_query(), ids, dists_out, scratch);
    };

    // Initialize the candidate pool with starting points
    for (auto id : init_ids)
    {
        if (id >= _max_points + _num_frozen_pts)
        {
            diskann::cerr << "Out of range loc found as an edge : " << id << std::endl;
            throw diskann::ANNException(std::string("Wrong loc") + std::to_string(id), -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
        }

        if (use_filter)
        {
            if (!detect_common_filters(id, search_invocation, filter_labels))
                continue;
        }

        visited_timer.reset();
        if (is_not_visited(id))
        {
            if (fast_iterate)
            {
                inserted_into_pool_bs[id] = 1;
            }
            else
            {
                inserted_into_pool_rs.insert(id);
            }
            visited_us += visited_timer.elapsed();

            dist_timer.reset();
            float distance;
            uint32_t ids[] = {id};
            float distances[] = {std::numeric_limits<float>::max()};
            _pq_data_store->get_distance(aligned_query, ids, 1, distances, scratch);
            distance = distances[0];
            dist_us += dist_timer.elapsed();

            queue_timer.reset();
            Neighbor nn = Neighbor(id, distance);
            best_L_nodes.insert(nn);
            queue_us += queue_timer.elapsed();
        }
        else
        {
            visited_us += visited_timer.elapsed();
        }
    }

    uint32_t hops = 0;
    uint32_t cmps = 0;

    while (best_L_nodes.has_unexpanded_node())
    {
        // Queue pop
        queue_timer.reset();
        auto nbr = best_L_nodes.closest_unexpanded();
        queue_us += queue_timer.elapsed();
        auto n = nbr.id;

        // Add node to expanded nodes to create pool for prune later
        if (!search_invocation)
        {
            if (!use_filter)
            {
                expanded_nodes.emplace_back(nbr);
            }
            else
            {
                if (std::find(expanded_nodes.begin(), expanded_nodes.end(), nbr) == expanded_nodes.end())
                {
                    expanded_nodes.emplace_back(nbr);
                }
            }
        }

        // Find which of the nodes in des have not been visited before
        id_scratch.clear();
        dist_scratch.clear();

        // Neighbor fetch + visited check
        fetch_timer.reset();
        if (_dynamic_index)
        {
            LockGuard guard(_locks[n]);
            auto neighbors = _graph_store->get_neighbours(n);
            fetch_us += fetch_timer.elapsed();

            visited_timer.reset();
            for (auto id : neighbors)
            {
                assert(id < _max_points + _num_frozen_pts);

                if (use_filter)
                {
                    if (!detect_common_filters(id, search_invocation, filter_labels))
                        continue;
                }

                if (is_not_visited(id))
                {
                    id_scratch.push_back(id);
                }
            }
            visited_us += visited_timer.elapsed();
        }
        else
        {
            _locks[n].lock();
            auto nbrs = _graph_store->get_neighbours(n);
            _locks[n].unlock();
            fetch_us += fetch_timer.elapsed();

            visited_timer.reset();
            for (auto id : nbrs)
            {
                assert(id < _max_points + _num_frozen_pts);

                if (use_filter)
                {
                    if (!detect_common_filters(id, search_invocation, filter_labels))
                        continue;
                }

                if (is_not_visited(id))
                {
                    id_scratch.push_back(id);
                }
            }
            visited_us += visited_timer.elapsed();
        }

        // Mark nodes visited
        visited_timer.reset();
        for (auto id : id_scratch)
        {
            if (fast_iterate)
            {
                inserted_into_pool_bs[id] = 1;
            }
            else
            {
                inserted_into_pool_rs.insert(id);
            }
        }
        visited_us += visited_timer.elapsed();

        // Distance computation
        dist_timer.reset();
        assert(dist_scratch.capacity() >= id_scratch.size());
        compute_dists(id_scratch, dist_scratch);
        dist_us += dist_timer.elapsed();
        cmps += (uint32_t)id_scratch.size();

        // Queue insert
        queue_timer.reset();
        for (size_t m = 0; m < id_scratch.size(); ++m)
        {
            best_L_nodes.insert(Neighbor(id_scratch[m], dist_scratch[m]));
        }
        queue_us += queue_timer.elapsed();

        hops++;
    }

    // Record stats
    stats.dist_us = (float)dist_us;
    stats.fetch_us = (float)fetch_us;
    stats.queue_us = (float)queue_us;
    stats.visited_us = (float)visited_us;
    stats.n_cmps = cmps;
    stats.n_hops = hops;

    return std::make_pair(hops, cmps);
}

// iterate_to_fixed_point with partition-aware timing for distributed latency profiling
// Tracks local/remote accesses and counts RDMA round trips per-partition-per-hop
template <typename T, typename TagT, typename LabelT>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::iterate_to_fixed_point_timed_partitioned(
    InMemQueryScratch<T> *scratch, const uint32_t Lsize, const std::vector<uint32_t> &init_ids, bool use_filter,
    const std::vector<LabelT> &filter_labels, bool search_invocation,
    const uint32_t *partition_ids, uint32_t query_partition,
    QueryStats &stats,
    const RemoteDistanceFn<T> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn)
{
    Timer dist_timer, fetch_timer, queue_timer, visited_timer;
    long long dist_ns = 0, fetch_ns = 0, queue_ns = 0, visited_ns = 0;
    long long dist_local_ns = 0, dist_remote_ns = 0;
    long long fetch_local_ns = 0, fetch_remote_ns = 0;
    unsigned local_vector_accesses = 0, remote_vector_accesses = 0;
    unsigned local_neighbor_accesses = 0, remote_neighbor_accesses = 0;
    unsigned rdma_round_trips = 0;
    long long rdma_neighbor_fetch_ns = 0;

    std::vector<Neighbor> &expanded_nodes = scratch->pool();
    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    best_L_nodes.reserve(Lsize);
    tsl::robin_set<uint32_t> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();
    boost::dynamic_bitset<> &inserted_into_pool_bs = scratch->inserted_into_pool_bs();
    std::vector<uint32_t> &id_scratch = scratch->id_scratch();
    std::vector<float> &dist_scratch = scratch->dist_scratch();
    assert(id_scratch.size() == 0);

    T *aligned_query = scratch->aligned_query();

    float *pq_dists = nullptr;

    _pq_data_store->preprocess_query(aligned_query, scratch);

    if (expanded_nodes.size() > 0 || id_scratch.size() > 0)
    {
        throw ANNException("ERROR: Clear scratch space before passing.", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Decide whether to use bitset or robin set to mark visited nodes
    auto total_num_points = _max_points + _num_frozen_pts;
    bool fast_iterate = total_num_points <= MAX_POINTS_FOR_USING_BITSET;

    if (fast_iterate)
    {
        if (inserted_into_pool_bs.size() < total_num_points)
        {
            auto resize_size =
                2 * total_num_points > MAX_POINTS_FOR_USING_BITSET ? MAX_POINTS_FOR_USING_BITSET : 2 * total_num_points;
            inserted_into_pool_bs.resize(resize_size);
        }
    }

    // Lambda to determine if a node has been visited
    auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](const uint32_t id) {
        return fast_iterate ? inserted_into_pool_bs[id] == 0
                            : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
    };

    // Lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, scratch, pq_dists](const std::vector<uint32_t> &ids, std::vector<float> &dists_out) {
        _pq_data_store->get_distance(scratch->aligned_query(), ids, dists_out, scratch);
    };

    // Initialize the candidate pool with starting points
    for (auto id : init_ids)
    {
        if (id >= _max_points + _num_frozen_pts)
        {
            diskann::cerr << "Out of range loc found as an edge : " << id << std::endl;
            throw diskann::ANNException(std::string("Wrong loc") + std::to_string(id), -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
        }

        if (use_filter)
        {
            if (!detect_common_filters(id, search_invocation, filter_labels))
                continue;
        }

        visited_timer.reset();
        if (is_not_visited(id))
        {
            if (fast_iterate)
            {
                inserted_into_pool_bs[id] = 1;
            }
            else
            {
                inserted_into_pool_rs.insert(id);
            }
            visited_ns += visited_timer.elapsed_ns();

            // Check if local or remote for init point
            bool is_local = (partition_ids[id] == query_partition);

            dist_timer.reset();
            float distance;
            if (!is_local && remote_dist_fn)
            {
                // Use RDMA callback for remote init points
                std::vector<uint32_t> tmp_ids = {id};
                std::vector<float> tmp_dists;
                remote_dist_fn(aligned_query, tmp_ids, tmp_dists);
                distance = tmp_dists[0];
            }
            else
            {
                uint32_t ids[] = {id};
                float distances[] = {std::numeric_limits<float>::max()};
                _pq_data_store->get_distance(aligned_query, ids, 1, distances, scratch);
                distance = distances[0];
            }
            auto elapsed = dist_timer.elapsed_ns();
            dist_ns += elapsed;

            if (is_local) {
                dist_local_ns += elapsed;
                local_vector_accesses++;
            } else {
                dist_remote_ns += elapsed;
                remote_vector_accesses++;
                rdma_round_trips++;  // Init point from remote partition
            }

            queue_timer.reset();
            Neighbor nn = Neighbor(id, distance);
            best_L_nodes.insert(nn);
            queue_ns += queue_timer.elapsed_ns();
        }
        else
        {
            visited_ns += visited_timer.elapsed_ns();
        }
    }

    uint32_t hops = 0;
    uint32_t cmps = 0;

    // Temporary vectors for local/remote separation
    std::vector<uint32_t> local_ids, remote_ids;
    std::vector<float> local_dists, remote_dists;
    std::set<uint32_t> remote_partitions_this_hop;

    while (best_L_nodes.has_unexpanded_node())
    {
        // Queue pop
        queue_timer.reset();
        auto nbr = best_L_nodes.closest_unexpanded();
        queue_ns += queue_timer.elapsed_ns();
        auto n = nbr.id;

        // Add node to expanded nodes to create pool for prune later
        if (!search_invocation)
        {
            if (!use_filter)
            {
                expanded_nodes.emplace_back(nbr);
            }
            else
            {
                if (std::find(expanded_nodes.begin(), expanded_nodes.end(), nbr) == expanded_nodes.end())
                {
                    expanded_nodes.emplace_back(nbr);
                }
            }
        }

        // Find which of the nodes in des have not been visited before
        id_scratch.clear();
        dist_scratch.clear();
        local_ids.clear();
        remote_ids.clear();
        remote_partitions_this_hop.clear();

        // Check if current node's neighbor list is local or remote
        bool neighbor_list_is_local = (partition_ids[n] == query_partition);

        // Neighbor fetch + visited check
        fetch_timer.reset();
        if (graph_access_fn)
        {
            // Co-location mode: use callback for neighbor access (handles both local and remote)
            const auto &neighbors = graph_access_fn(n);
            auto fetch_elapsed = fetch_timer.elapsed_ns();
            fetch_ns += fetch_elapsed;

            if (neighbor_list_is_local) {
                fetch_local_ns += fetch_elapsed;
                local_neighbor_accesses++;
            } else {
                fetch_remote_ns += fetch_elapsed;
                remote_neighbor_accesses++;
            }

            visited_timer.reset();
            for (auto id : neighbors)
            {
                if (id >= _max_points + _num_frozen_pts)
                    continue;  // skip invalid/padding entries from co-location

                if (use_filter)
                {
                    if (!detect_common_filters(id, search_invocation, filter_labels))
                        continue;
                }

                if (is_not_visited(id))
                {
                    id_scratch.push_back(id);
                }
            }
            visited_ns += visited_timer.elapsed_ns();
        }
        else if (_dynamic_index)
        {
            LockGuard guard(_locks[n]);
            auto neighbors = _graph_store->get_neighbours(n);
            auto fetch_elapsed = fetch_timer.elapsed_ns();
            fetch_ns += fetch_elapsed;

            if (neighbor_list_is_local) {
                fetch_local_ns += fetch_elapsed;
                local_neighbor_accesses++;
            } else {
                fetch_remote_ns += fetch_elapsed;
                remote_neighbor_accesses++;
                if (remote_neighbor_fn) {
                    long long rdma_ns = remote_neighbor_fn(n);
                    rdma_neighbor_fetch_ns += rdma_ns;
                    fetch_remote_ns += rdma_ns;
                    fetch_ns += rdma_ns;
                }
            }

            visited_timer.reset();
            for (auto id : neighbors)
            {
                assert(id < _max_points + _num_frozen_pts);

                if (use_filter)
                {
                    if (!detect_common_filters(id, search_invocation, filter_labels))
                        continue;
                }

                if (is_not_visited(id))
                {
                    id_scratch.push_back(id);
                }
            }
            visited_ns += visited_timer.elapsed_ns();
        }
        else
        {
            _locks[n].lock();
            auto nbrs = _graph_store->get_neighbours(n);
            _locks[n].unlock();
            auto fetch_elapsed = fetch_timer.elapsed_ns();
            fetch_ns += fetch_elapsed;

            if (neighbor_list_is_local) {
                fetch_local_ns += fetch_elapsed;
                local_neighbor_accesses++;
            } else {
                fetch_remote_ns += fetch_elapsed;
                remote_neighbor_accesses++;
                if (remote_neighbor_fn) {
                    long long rdma_ns = remote_neighbor_fn(n);
                    rdma_neighbor_fetch_ns += rdma_ns;
                    fetch_remote_ns += rdma_ns;
                    fetch_ns += rdma_ns;
                }
            }

            visited_timer.reset();
            for (auto id : nbrs)
            {
                assert(id < _max_points + _num_frozen_pts);

                if (use_filter)
                {
                    if (!detect_common_filters(id, search_invocation, filter_labels))
                        continue;
                }

                if (is_not_visited(id))
                {
                    id_scratch.push_back(id);
                }
            }
            visited_ns += visited_timer.elapsed_ns();
        }

        // Mark nodes visited
        visited_timer.reset();
        for (auto id : id_scratch)
        {
            if (fast_iterate)
            {
                inserted_into_pool_bs[id] = 1;
            }
            else
            {
                inserted_into_pool_rs.insert(id);
            }
        }
        visited_ns += visited_timer.elapsed_ns();

        // Separate local and remote ids for distance computation
        for (auto id : id_scratch)
        {
            uint32_t p = partition_ids[id];
            if (p == query_partition) {
                local_ids.push_back(id);
            } else {
                remote_ids.push_back(id);
                remote_partitions_this_hop.insert(p);
            }
        }

        // Distance computation for local vectors
        if (!local_ids.empty()) {
            local_dists.clear();
            local_dists.reserve(local_ids.size());
            dist_timer.reset();
            compute_dists(local_ids, local_dists);
            auto elapsed = dist_timer.elapsed_ns();
            dist_ns += elapsed;
            dist_local_ns += elapsed;
            local_vector_accesses += local_ids.size();
        }

        // Distance computation for remote vectors
        if (!remote_ids.empty()) {
            remote_dists.clear();
            remote_dists.reserve(remote_ids.size());
            dist_timer.reset();
            if (remote_dist_fn)
            {
                remote_dist_fn(aligned_query, remote_ids, remote_dists);
            }
            else
            {
                compute_dists(remote_ids, remote_dists);
            }
            auto elapsed = dist_timer.elapsed_ns();
            dist_ns += elapsed;
            dist_remote_ns += elapsed;
            remote_vector_accesses += remote_ids.size();
        }

        // Count RDMA round trips for this hop (per-partition batching)
        // Each unique remote partition accessed = 1 RTT (scatter-gather)
        rdma_round_trips += remote_partitions_this_hop.size();

        cmps += (uint32_t)id_scratch.size();

        // Queue insert - merge local and remote results
        queue_timer.reset();
        size_t local_idx = 0, remote_idx = 0;
        for (auto id : id_scratch) {
            float dist;
            if (partition_ids[id] == query_partition) {
                dist = local_dists[local_idx++];
            } else {
                dist = remote_dists[remote_idx++];
            }
            best_L_nodes.insert(Neighbor(id, dist));
        }
        queue_ns += queue_timer.elapsed_ns();

        hops++;
    }

    // Record stats (convert nanoseconds to microseconds with double precision)
    stats.dist_us = (float)(dist_ns / 1000.0);
    stats.fetch_us = (float)(fetch_ns / 1000.0);
    stats.queue_us = (float)(queue_ns / 1000.0);
    stats.visited_us = (float)(visited_ns / 1000.0);
    stats.n_cmps = cmps;
    stats.n_hops = hops;

    // Partition-aware stats (double precision)
    stats.dist_local_us = dist_local_ns / 1000.0;
    stats.dist_remote_us = dist_remote_ns / 1000.0;
    stats.fetch_local_us = fetch_local_ns / 1000.0;
    stats.fetch_remote_us = fetch_remote_ns / 1000.0;
    stats.local_vector_accesses = local_vector_accesses;
    stats.remote_vector_accesses = remote_vector_accesses;
    stats.local_neighbor_accesses = local_neighbor_accesses;
    stats.remote_neighbor_accesses = remote_neighbor_accesses;
    stats.rdma_round_trips = rdma_round_trips;
    stats.rdma_neighbor_fetch_ns = rdma_neighbor_fetch_ns;

    return std::make_pair(hops, cmps);
}

// My Function
template <typename T, typename TagT, typename LabelT>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::iterate_to_fixed_point_with_trace(
    InMemQueryScratch<T> *scratch, const uint32_t Lsize, const std::vector<uint32_t> &init_ids, bool use_filter,
    const std::vector<LabelT> &filter_labels, bool search_invocation, const std::vector<uint32_t>& partition_labels,
    std::vector<TraceInfo>& trace_path)
{
    std::vector<Neighbor> &expanded_nodes = scratch->pool();
    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    best_L_nodes.reserve(Lsize);
    tsl::robin_set<uint32_t> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();
    boost::dynamic_bitset<> &inserted_into_pool_bs = scratch->inserted_into_pool_bs();
    std::vector<uint32_t> &id_scratch = scratch->id_scratch();
    std::vector<float> &dist_scratch = scratch->dist_scratch();
    assert(id_scratch.size() == 0);

    T *aligned_query = scratch->aligned_query();

    float *pq_dists = nullptr;

    _pq_data_store->preprocess_query(aligned_query, scratch);

    if (expanded_nodes.size() > 0 || id_scratch.size() > 0)
    {
        throw ANNException("ERROR: Clear scratch space before passing.", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Decide whether to use bitset or robin set to mark visited nodes
    auto total_num_points = _max_points + _num_frozen_pts;
    bool fast_iterate = total_num_points <= MAX_POINTS_FOR_USING_BITSET;

    if (fast_iterate)
    {
        if (inserted_into_pool_bs.size() < total_num_points)
        {
            // hopefully using 2X will reduce the number of allocations.
            auto resize_size =
                2 * total_num_points > MAX_POINTS_FOR_USING_BITSET ? MAX_POINTS_FOR_USING_BITSET : 2 * total_num_points;
            inserted_into_pool_bs.resize(resize_size);
        }
    }

    // Lambda to determine if a node has been visited
    auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](const uint32_t id) {
        return fast_iterate ? inserted_into_pool_bs[id] == 0
                            : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
    };

    // Lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, scratch, pq_dists](const std::vector<uint32_t> &ids, std::vector<float> &dists_out) {
        _pq_data_store->get_distance(scratch->aligned_query(), ids, dists_out, scratch);
    };

    // Initialize the candidate pool with starting points
    for (auto id : init_ids)
    {
        if (id >= _max_points + _num_frozen_pts)
        {
            diskann::cerr << "Out of range loc found as an edge : " << id << std::endl;
            throw diskann::ANNException(std::string("Wrong loc") + std::to_string(id), -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
        }

        if (use_filter)
        {
            if (!detect_common_filters(id, search_invocation, filter_labels))
                continue;
        }

        if (is_not_visited(id))
        {
            if (fast_iterate)
            {
                inserted_into_pool_bs[id] = 1;
            }
            else
            {
                inserted_into_pool_rs.insert(id);
            }

            float distance;
            uint32_t ids[] = {id};
            float distances[] = {std::numeric_limits<float>::max()};
            _pq_data_store->get_distance(aligned_query, ids, 1, distances, scratch);
            distance = distances[0];

            Neighbor nn = Neighbor(id, distance);
            best_L_nodes.insert(nn);
        }
    }

    uint32_t hops = 0;
    uint32_t cmps = 0;

    while (best_L_nodes.has_unexpanded_node())
    {
        auto nbr = best_L_nodes.closest_unexpanded();
        auto n = nbr.id;
        
        // --- START OF MODIFICATION ---
        uint32_t source_partition = partition_labels[n]; // 현재 노드의 파티션
        uint32_t current_hop_intra_comps = 0;
        uint32_t current_hop_inter_comps = 0;
        // --- END OF MODIFICATION ---


        // REMOVED: Duplicate initialization trace (causes 0,0,0 lines)
        // trace_path.push_back({n, partition_labels[n], hops, Lsize});

        // Add node to expanded nodes to create pool for prune later
        if (!search_invocation)
        {
            if (!use_filter)
            {
                expanded_nodes.emplace_back(nbr);
            }
            else
            { // in filter based indexing, the same point might invoke
                // multiple iterate_to_fixed_points, so need to be careful
                // not to add the same item to pool multiple times.
                if (std::find(expanded_nodes.begin(), expanded_nodes.end(), nbr) == expanded_nodes.end())
                {
                    expanded_nodes.emplace_back(nbr);
                }
            }
        }

        // Find which of the nodes in des have not been visited before
        id_scratch.clear();
        dist_scratch.clear();
        if (_dynamic_index)
        {
            LockGuard guard(_locks[n]);
            for (auto id : _graph_store->get_neighbours(n))
            {
                assert(id < _max_points + _num_frozen_pts);

                if (use_filter)
                {
                    // NOTE: NEED TO CHECK IF THIS CORRECT WITH NEW LOCKS.
                    if (!detect_common_filters(id, search_invocation, filter_labels))
                        continue;
                }

                if (is_not_visited(id))
                {
                    id_scratch.push_back(id);
                }
            }
        }
        else
        {
            _locks[n].lock();
            auto nbrs = _graph_store->get_neighbours(n);
            _locks[n].unlock();
            for (auto id : nbrs)
            {
                assert(id < _max_points + _num_frozen_pts);

                if (use_filter)
                {
                    // NOTE: NEED TO CHECK IF THIS CORRECT WITH NEW LOCKS.
                    if (!detect_common_filters(id, search_invocation, filter_labels))
                        continue;
                }

                if (is_not_visited(id))
                {
                    id_scratch.push_back(id);
                }
            }
        }

        // Mark nodes visited
        for (auto id : id_scratch)
        {
            if (fast_iterate)
            {
                inserted_into_pool_bs[id] = 1;
            }
            else
            {
                inserted_into_pool_rs.insert(id);
            }
        }

        // --- START OF MODIFICATION ---
        // 방문하지 않은 이웃(id_scratch)들을 대상으로 파티션 비교 및 카운트
        for (const auto& neighbor_id : id_scratch) {
            if (partition_labels[neighbor_id] == source_partition) {
                current_hop_intra_comps++;
            } else {
                current_hop_inter_comps++;
            }
        }
        cmps += id_scratch.size();
        // --- END OF MODIFICATION ---

        assert(dist_scratch.capacity() >= id_scratch.size());
        compute_dists(id_scratch, dist_scratch);
        cmps += (uint32_t)id_scratch.size();

        // Insert <id, dist> pairs into the pool of candidates
        for (size_t m = 0; m < id_scratch.size(); ++m)
        {
            best_L_nodes.insert(Neighbor(id_scratch[m], dist_scratch[m]));
        }

        // --- 최종 Trace 기록 (수정) ---
        // Extract candidate set from best_L_nodes (IDs only, distances computed at analysis time)
        std::vector<uint32_t> candidate_ids;
        for (size_t i = 0; i < best_L_nodes.size(); ++i) {
            candidate_ids.push_back(best_L_nodes[i].id);
        }

        trace_path.push_back(TraceInfo{n, source_partition, hops, Lsize,
                            current_hop_intra_comps, current_hop_inter_comps,
                            candidate_ids});
        hops++;
    }
    return std::make_pair(hops, cmps);
}


template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::search_for_point_and_prune(int location, uint32_t Lindex,
                                                        std::vector<uint32_t> &pruned_list,
                                                        InMemQueryScratch<T> *scratch, bool use_filter,
                                                        uint32_t filteredLindex)
{
    const std::vector<uint32_t> init_ids = get_init_ids();
    const std::vector<LabelT> unused_filter_label;

    if (!use_filter)
    {
        _data_store->get_vector(location, scratch->aligned_query());
        iterate_to_fixed_point(scratch, Lindex, init_ids, false, unused_filter_label, false);
    }
    else
    {
        std::shared_lock<std::shared_timed_mutex> tl(_tag_lock, std::defer_lock);
        if (_dynamic_index)
            tl.lock();
        std::vector<uint32_t> filter_specific_start_nodes;
        for (auto &x : _location_to_labels[location])
            filter_specific_start_nodes.emplace_back(_label_to_start_id[x]);

        if (_dynamic_index)
            tl.unlock();

        _data_store->get_vector(location, scratch->aligned_query());
        iterate_to_fixed_point(scratch, filteredLindex, filter_specific_start_nodes, true,
                               _location_to_labels[location], false);

        // combine candidate pools obtained with filter and unfiltered criteria.
        std::set<Neighbor> best_candidate_pool;
        for (auto filtered_neighbor : scratch->pool())
        {
            best_candidate_pool.insert(filtered_neighbor);
        }

        // clear scratch for finding unfiltered candidates
        scratch->clear();

        _data_store->get_vector(location, scratch->aligned_query());
        iterate_to_fixed_point(scratch, Lindex, init_ids, false, unused_filter_label, false);

        for (auto unfiltered_neighbour : scratch->pool())
        {
            // insert if this neighbour is not already in best_candidate_pool
            if (best_candidate_pool.find(unfiltered_neighbour) == best_candidate_pool.end())
            {
                best_candidate_pool.insert(unfiltered_neighbour);
            }
        }

        scratch->pool().clear();
        std::copy(best_candidate_pool.begin(), best_candidate_pool.end(), std::back_inserter(scratch->pool()));
    }

    auto &pool = scratch->pool();

    for (uint32_t i = 0; i < pool.size(); i++)
    {
        if (pool[i].id == (uint32_t)location)
        {
            pool.erase(pool.begin() + i);
            i--;
        }
    }

    if (pruned_list.size() > 0)
    {
        throw diskann::ANNException("ERROR: non-empty pruned_list passed", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Capacity-balanced placement: decide this point's cluster based on visit
    // distribution and current per-cluster counters. Must run before
    // filter_lits_from_pool because that function reads the point's cluster.
    if (_lits_balanced_alloc_enabled)
    {
        place_vector_balanced(location, pool);
    }

    if (_lits_build_filter_enabled && !_lits_build_cluster_labels.empty())
    {
        filter_lits_from_pool(location, pool);
    }

    prune_neighbors(location, pool, pruned_list, scratch);

    assert(!pruned_list.empty());
    assert(_graph_store->get_total_points() == _max_points + _num_frozen_pts);
}

// ----- LITS build support (ported from the diskann-lits fork) -----

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::filter_lits_from_pool(int location, std::vector<Neighbor> &pool)
{
    if (pool.size() < 2)
        return;

    const auto &cluster_labels = _lits_build_cluster_labels;

    auto tag_to_uint32 = [](const TagT &tag) -> uint32_t {
        if constexpr (std::is_same_v<TagT, tag_uint128>)
            return static_cast<uint32_t>(tag._data1);
        else
            return static_cast<uint32_t>(tag);
    };

    // Get cluster of the inserted point via location -> tag -> cluster.
    // Frozen start is exempt: it is the global search entry point and must keep
    // cross-cluster fan-out (LITS-localizing it caused a c=0.5 recall runaway in
    // the diskann-lits fork).
    TagT loc_tag;
    if ((uint32_t)location == _start)
        return;
    if (!_location_to_tag.try_get((uint32_t)location, loc_tag))
        return; // tag not found, skip filtering
    uint32_t loc_tag_u32 = tag_to_uint32(loc_tag);
    if (loc_tag_u32 >= cluster_labels.size())
        throw ANNException("LITS filter: loc_tag " + std::to_string(loc_tag_u32) +
                               " out of bounds (cluster_labels size=" + std::to_string(cluster_labels.size()) + ")",
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    uint32_t loc_cluster = cluster_labels[loc_tag_u32];

    // Compute the long-threshold base over pool distances: mean, or median
    // (upper median for even sizes) when _lits_tau_use_median is set.
    double sum_dist = 0.0;
    for (const auto &nbr : pool)
        sum_dist += nbr.distance;
    float mean_dist = static_cast<float>(sum_dist / pool.size());
    if (_lits_tau_use_median)
    {
        std::vector<float> dists;
        dists.reserve(pool.size());
        for (const auto &nbr : pool)
            dists.push_back(nbr.distance);
        auto mid = dists.begin() + dists.size() / 2;
        std::nth_element(dists.begin(), mid, dists.end());
        mean_dist = *mid;
    }

    // Get vectors for dot product computation
    size_t dim = _data_store->get_dims();
    std::vector<T> loc_vec(dim), start_vec(dim), nbr_vec(dim);
    _data_store->get_vector((location_t)location, loc_vec.data());
    _data_store->get_vector((location_t)_start, start_vec.data());

    // Identify candidates for removal
    std::vector<bool> is_lits(pool.size(), false);
    size_t lits_count = 0;

    for (size_t i = 0; i < pool.size(); i++)
    {
        // 1. Long: distance > c * mean (c = _lits_tau_scale)
        if (pool[i].distance <= _lits_tau_scale * mean_dist)
            continue;

        // 2. Inter: different cluster
        uint32_t nbr_id = pool[i].id;
        TagT nbr_tag;
        if (nbr_id == _start)
        {
            nbr_tag = 0;
        }
        else
        {
            if (!_location_to_tag.try_get(nbr_id, nbr_tag))
                continue; // tag not found, not LITS
        }
        uint32_t nbr_tag_u32 = tag_to_uint32(nbr_tag);
        if (nbr_tag_u32 >= cluster_labels.size())
            throw ANNException("LITS filter: nbr_tag " + std::to_string(nbr_tag_u32) +
                                   " out of bounds (cluster_labels size=" + std::to_string(cluster_labels.size()) +
                                   ")",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        uint32_t nbr_cluster = cluster_labels[nbr_tag_u32];
        if (nbr_cluster == loc_cluster)
            continue;

        // 3. To-start check (skipped in long_inter_only mode)
        if (!_lits_build_long_inter_only)
        {
            _data_store->get_vector((location_t)nbr_id, nbr_vec.data());
            double dot = 0.0;
            for (size_t d = 0; d < dim; d++)
            {
                double to_start = (double)start_vec[d] - (double)loc_vec[d];
                double to_nbr = (double)nbr_vec[d] - (double)loc_vec[d];
                dot += to_start * to_nbr;
            }

            if (_lits_build_notostart_mode)
            {
                if (dot >= 0.0)
                    continue; // to-start -> keep
            }
            else
            {
                if (dot < 0.0)
                    continue; // not-to-start -> keep (original)
            }
        }

        // Marked for removal (LITS in original mode, NOT-to-start in notostart
        // mode, or long+inter in long_inter_only mode)
        is_lits[i] = true;
        lits_count++;
    }

    // Safety: don't empty the pool
    if (lits_count == 0 || lits_count >= pool.size())
        return;

    _lits_build_filtered_total.fetch_add(lits_count, std::memory_order_relaxed);

    // Remove marked entries
    size_t write_idx = 0;
    for (size_t i = 0; i < pool.size(); i++)
    {
        if (!is_lits[i])
        {
            pool[write_idx++] = pool[i];
        }
    }
    pool.resize(write_idx);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::compute_pool_cluster_histogram(const std::vector<Neighbor> &pool,
                                                            std::vector<uint32_t> &v)
{
    auto tag_to_uint32_fn = [](const TagT &tag) -> uint32_t {
        if constexpr (std::is_same_v<TagT, tag_uint128>)
            return static_cast<uint32_t>(tag._data1);
        else
            return static_cast<uint32_t>(tag);
    };
    const uint32_t K = _lits_balanced_num_clusters;
    const auto &labels = _lits_build_cluster_labels;
    for (const auto &nbr : pool)
    {
        TagT nbr_tag;
        if (nbr.id == _start)
            nbr_tag = 0;
        else if (!_location_to_tag.try_get(nbr.id, nbr_tag))
            continue;
        uint32_t nt = tag_to_uint32_fn(nbr_tag);
        if ((size_t)nt >= labels.size())
            continue;
        uint32_t c = labels[nt];
        if (c < K)
            v[c]++;
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::place_vector_balanced(int location, const std::vector<Neighbor> &pool)
{
    auto tag_to_uint32_fn = [](const TagT &tag) -> uint32_t {
        if constexpr (std::is_same_v<TagT, tag_uint128>)
            return static_cast<uint32_t>(tag._data1);
        else
            return static_cast<uint32_t>(tag);
    };

    // location -> tag
    TagT loc_tag;
    if ((uint32_t)location == _start)
        return; // start point: label[0] fixed by initial load
    if (!_location_to_tag.try_get((uint32_t)location, loc_tag))
        return;
    uint32_t loc_tag_u32 = tag_to_uint32_fn(loc_tag);

    // Initial active_window region was populated from the label file; don't overwrite.
    if ((uint64_t)loc_tag_u32 <= _lits_balanced_active_window)
        return;
    if ((size_t)loc_tag_u32 >= _lits_build_cluster_labels.size())
        return;

    const uint32_t K = _lits_balanced_num_clusters;

    // Visit distribution over clusters from the greedy-search pool.
    std::vector<uint32_t> v(K, 0);
    compute_pool_cluster_histogram(pool, v);

    uint32_t v_max = 0;
    for (uint32_t i = 0; i < K; i++)
        if (v[i] > v_max)
            v_max = v[i];
    if (v_max == 0)
        v_max = 1; // empty pool edge case

    // B-style accounting: this point's nearest-centroid home was pre-charged
    // into the counters before the batch (precharge_lits_balanced); move it
    // (home--, target++) only when >= 2 clusters are eligible and one has a
    // strictly smaller counter. Ties keep home (then lowest id). Mirrors the
    // distributed LoadBalancer (update/include/update/load_balancer.h).
    std::vector<T> vec(_dim);
    _data_store->get_vector((location_t)location, vec.data());
    const uint32_t home = nearest_balanced_centroid(vec.data());

    uint32_t n_eligible = 0;
    for (uint32_t i = 0; i < K; i++)
        if ((double)v[i] / (double)v_max >= (double)_lits_balanced_threshold)
            n_eligible++;

    uint32_t best = home;
    if (n_eligible >= 2)
    {
        int cand = -1;
        uint64_t min_c = UINT64_MAX;
        for (uint32_t i = 0; i < K; i++)
        {
            if ((double)v[i] / (double)v_max < (double)_lits_balanced_threshold)
                continue;
            uint64_t ci = _lits_cluster_counts[i].value.load(std::memory_order_relaxed);
            if (cand < 0 || ci < min_c || (ci == min_c && i == home))
            {
                min_c = ci;
                cand = (int)i;
            }
        }
        best = (uint32_t)cand;
    }
    if (best != home)
    {
        _lits_cluster_counts[home].value.fetch_sub(1, std::memory_order_relaxed);
        _lits_cluster_counts[best].value.fetch_add(1, std::memory_order_relaxed);
        _lits_balanced_redirected.fetch_add(1, std::memory_order_relaxed);
    }
    _lits_balanced_placed.fetch_add(1, std::memory_order_relaxed);
    _lits_build_cluster_labels[loc_tag_u32] = best;
}

template <typename T, typename TagT, typename LabelT>
uint32_t Index<T, TagT, LabelT>::nearest_balanced_centroid(const T *vec) const
{
    const uint32_t K = _lits_balanced_num_clusters;
    uint32_t best = 0;
    float best_d = std::numeric_limits<float>::max();
    for (uint32_t k = 0; k < K; k++)
    {
        const float *c = _lits_balanced_centroids.data() + (size_t)k * _dim;
        float d = 0;
        for (size_t j = 0; j < _dim; j++)
        {
            float diff = (float)vec[j] - c[j];
            d += diff * diff;
        }
        if (d < best_d)
        {
            best_d = d;
            best = k;
        }
    }
    return best;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::load_lits_build_cluster_labels(const std::string &file_path)
{
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open())
    {
        throw ANNException("Could not open LITS cluster label file: " + file_path, -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    }

    uint32_t num_entries, n_clusters;
    in.read(reinterpret_cast<char *>(&num_entries), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&n_clusters), sizeof(uint32_t));

    _lits_build_cluster_labels.resize(num_entries);
    in.read(reinterpret_cast<char *>(_lits_build_cluster_labels.data()), num_entries * sizeof(uint32_t));
    in.close();

    diskann::cout << "Loaded LITS build cluster labels: " << num_entries << " entries, " << n_clusters << " clusters"
                  << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_lits_build_filter(bool enabled)
{
    _lits_build_filter_enabled = enabled;
    diskann::cout << "[lits] build filter " << (enabled ? "ENABLED" : "disabled") << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_lits_build_notostart_filter(bool enabled)
{
    _lits_build_notostart_mode = enabled;
    if (enabled)
        diskann::cout << "[lits] filter mode: notostart (ablation)" << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_lits_build_long_inter_only(bool enabled)
{
    _lits_build_long_inter_only = enabled;
    if (enabled)
        diskann::cout << "[lits] filter mode: long_inter_only" << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_lits_tau_scale(float scale)
{
    _lits_tau_scale = scale;
    diskann::cout << "[lits] tau scale c = " << scale << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_lits_tau_median(bool use_median)
{
    _lits_tau_use_median = use_median;
    if (use_median)
        diskann::cout << "[lits] tau base = median" << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::enable_lits_balanced_alloc(uint32_t K, float threshold, uint64_t active_window,
                                                        uint64_t max_points_to_insert,
                                                        const std::string &label_file_path,
                                                        const std::string &resume_tags_file)
{
    if (K == 0)
        throw ANNException("enable_lits_balanced_alloc: K must be > 0", -1, __FUNCSIG__, __FILE__, __LINE__);

    const bool resume = !resume_tags_file.empty();

    // Step 1: pre-resize label vector to full stream size (append-only writes from insert threads)
    _lits_build_cluster_labels.assign(1 + max_points_to_insert, 0);

    // Step 2: read + validate header. Fresh start: label_file_path covers the initial
    // window (active_window + 1 entries). Resume: label_file_path is a checkpoint
    // .labels dump covering the full stream (1 + max_points_to_insert entries).
    std::ifstream in(label_file_path, std::ios::binary);
    if (!in.is_open())
        throw ANNException("enable_lits_balanced_alloc: could not open " + label_file_path, -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    uint32_t file_num_entries = 0, file_n_clusters = 0;
    in.read(reinterpret_cast<char *>(&file_num_entries), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&file_n_clusters), sizeof(uint32_t));
    const uint64_t expected_entries = resume ? (1 + max_points_to_insert) : (active_window + 1);
    if ((uint64_t)file_num_entries != expected_entries)
        throw ANNException("enable_lits_balanced_alloc: label num_entries=" + std::to_string(file_num_entries) +
                               " expected " + (resume ? "1+max_points_to_insert=" : "active_window+1=") +
                               std::to_string(expected_entries),
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    if (file_n_clusters != K)
        throw ANNException("enable_lits_balanced_alloc: label n_clusters=" + std::to_string(file_n_clusters) +
                               " expected K=" + std::to_string(K),
                           -1, __FUNCSIG__, __FILE__, __LINE__);

    // Step 3: load entries into positions [0..expected_entries)
    in.read(reinterpret_cast<char *>(_lits_build_cluster_labels.data()), expected_entries * sizeof(uint32_t));
    in.close();
    for (uint64_t t = 0; t < expected_entries; t++)
    {
        if (_lits_build_cluster_labels[t] >= K)
            throw ANNException("enable_lits_balanced_alloc: label entry " +
                                   std::to_string(_lits_build_cluster_labels[t]) +
                                   " out of range (K=" + std::to_string(K) + ")",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Step 4: initial per-cluster counts (single-threaded init -> plain uint64_t).
    _lits_cluster_counts = std::make_unique<PaddedAtomicU64[]>(K);
    std::vector<uint64_t> counts(K, 0);
    if (!resume)
    {
        // Fresh start: the initial window [0..active_window] is exactly what will be live.
        for (uint64_t t = 0; t <= active_window; t++)
            counts[_lits_build_cluster_labels[t]]++;
    }
    else
    {
        // Resume: the active set has slid past the initial window, so rebuild counts
        // from the checkpoint's .tags file: histogram labels[tag] over active tags (>0),
        // plus labels[0] once for the frozen start point.
        auto tag_to_u64_fn = [](const TagT &tag) -> uint64_t {
            if constexpr (std::is_same_v<TagT, tag_uint128>)
                return tag._data1;
            else
                return static_cast<uint64_t>(tag);
        };
        if (!file_exists(resume_tags_file))
            throw ANNException("enable_lits_balanced_alloc: resume tags file does not exist: " + resume_tags_file, -1,
                               __FUNCSIG__, __FILE__, __LINE__);
        TagT *tag_data = nullptr;
        size_t tag_pts = 0, tag_dim = 0;
        load_bin<TagT>(resume_tags_file, tag_data, tag_pts, tag_dim);
        if (tag_dim != 1)
        {
            delete[] tag_data;
            throw ANNException("enable_lits_balanced_alloc: resume tags file must have dim 1", -1, __FUNCSIG__,
                               __FILE__, __LINE__);
        }
        for (size_t i = 0; i < tag_pts; i++)
        {
            uint64_t tv = tag_to_u64_fn(tag_data[i]);
            if (tv == 0)
                continue; // frozen slot in the tags file; accounted for below
            if (tv >= (uint64_t)_lits_build_cluster_labels.size())
            {
                delete[] tag_data;
                throw ANNException("enable_lits_balanced_alloc: resume tag " + std::to_string(tv) +
                                       " out of label range",
                                   -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            counts[_lits_build_cluster_labels[tv]]++;
        }
        delete[] tag_data;
        counts[_lits_build_cluster_labels[0]]++; // frozen start point
        uint64_t sum = 0;
        for (uint32_t k = 0; k < K; k++)
            sum += counts[k];
        if (sum != active_window + 1)
            throw ANNException("enable_lits_balanced_alloc: resume counter sum=" + std::to_string(sum) +
                                   " expected active_window+1=" + std::to_string(active_window + 1) +
                                   " (checkpoint .tags inconsistent with a full active window)",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    for (uint32_t k = 0; k < K; k++)
        _lits_cluster_counts[k].value.store(counts[k], std::memory_order_relaxed);

    // Step 5: capacity_limit = ceil(active_window / K) * 1.1 (informational)
    _lits_balanced_capacity_limit = static_cast<uint64_t>(std::ceil((double)active_window / (double)K * 1.1));

    // Step 6: flags
    _lits_balanced_num_clusters = K;
    _lits_balanced_threshold = threshold;
    _lits_balanced_active_window = active_window;
    _lits_balanced_alloc_enabled = true;

    diskann::cout << "[balanced-alloc] K=" << K << " threshold=" << threshold << " active_window=" << active_window
                  << " capacity_limit=" << _lits_balanced_capacity_limit
                  << (resume ? " (counters restored from " + resume_tags_file + ")" : "") << std::endl;
    diskann::cout << "[balanced-alloc] initial per-cluster counts:";
    for (uint32_t k = 0; k < K; k++)
        diskann::cout << " c[" << k << "]=" << counts[k];
    diskann::cout << " sum=" << (active_window + 1) << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_lits_balanced_centroids(const std::string &centroids_file)
{
    float *data = nullptr;
    size_t npts = 0, dim = 0;
    diskann::load_bin<float>(centroids_file, data, npts, dim);
    if (_lits_balanced_num_clusters == 0 || npts != _lits_balanced_num_clusters || dim != _dim)
    {
        delete[] data;
        throw ANNException("set_lits_balanced_centroids: expected " + std::to_string(_lits_balanced_num_clusters) +
                               " x " + std::to_string(_dim) + ", got " + std::to_string(npts) + " x " +
                               std::to_string(dim) + " (call after enable_lits_balanced_alloc)",
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    _lits_balanced_centroids.assign(data, data + npts * dim);
    delete[] data;
    diskann::cout << "[balanced-alloc] centroids loaded: " << npts << " x " << dim << " from " << centroids_file
                  << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::precharge_lits_balanced(const void *block, size_t npts, size_t stride)
{
    if (!_lits_balanced_alloc_enabled)
        return;
    if (_lits_balanced_centroids.empty())
        throw ANNException("precharge_lits_balanced: centroids not set (see set_lits_balanced_centroids)", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    const T *data = static_cast<const T *>(block);
    const uint32_t K = _lits_balanced_num_clusters;
    std::vector<uint64_t> deltas(K, 0);
#pragma omp parallel
    {
        std::vector<uint64_t> local(K, 0);
#pragma omp for nowait
        for (int64_t i = 0; i < (int64_t)npts; i++)
            local[nearest_balanced_centroid(data + i * stride)]++;
#pragma omp critical
        for (uint32_t k = 0; k < K; k++)
            deltas[k] += local[k];
    }
    for (uint32_t k = 0; k < K; k++)
        _lits_cluster_counts[k].value.fetch_add(deltas[k], std::memory_order_relaxed);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::on_points_deleted_for_balanced_alloc(uint64_t start_del, uint64_t end_del)
{
    if (!_lits_balanced_alloc_enabled)
        return;
    const uint32_t K = _lits_balanced_num_clusters;
    std::vector<uint64_t> deltas(K, 0);
    for (uint64_t tag = 1 + start_del; tag < 1 + end_del; tag++)
    {
        if (tag >= _lits_build_cluster_labels.size())
            continue;
        uint32_t c = _lits_build_cluster_labels[tag];
        if (c < K)
            deltas[c]++;
    }
    for (uint32_t k = 0; k < K; k++)
    {
        uint64_t cur = _lits_cluster_counts[k].value.load(std::memory_order_relaxed);
        _lits_cluster_counts[k].value.store(cur - deltas[k], std::memory_order_relaxed);
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::on_tags_deleted_for_balanced_alloc(const std::vector<uint32_t> &tags)
{
    if (!_lits_balanced_alloc_enabled)
        return;
    const uint32_t K = _lits_balanced_num_clusters;
    std::vector<uint64_t> deltas(K, 0);
    for (uint32_t tag : tags)
    {
        if ((size_t)tag >= _lits_build_cluster_labels.size())
            continue;
        uint32_t c = _lits_build_cluster_labels[tag];
        if (c < K)
            deltas[c]++;
    }
    for (uint32_t k = 0; k < K; k++)
    {
        uint64_t cur = _lits_cluster_counts[k].value.load(std::memory_order_relaxed);
        _lits_cluster_counts[k].value.store(cur - deltas[k], std::memory_order_relaxed);
    }
}

template <typename T, typename TagT, typename LabelT>
uint64_t Index<T, TagT, LabelT>::get_lits_cluster_counts_sum() const
{
    uint64_t sum = 0;
    for (uint32_t k = 0; k < _lits_balanced_num_clusters; k++)
        sum += _lits_cluster_counts[k].value.load(std::memory_order_relaxed);
    return sum;
}

template <typename T, typename TagT, typename LabelT>
uint64_t Index<T, TagT, LabelT>::get_lits_build_filtered_total() const
{
    return _lits_build_filtered_total.load(std::memory_order_relaxed);
}

template <typename T, typename TagT, typename LabelT>
std::pair<uint64_t, uint64_t> Index<T, TagT, LabelT>::get_reset_lits_balanced_redirect_stats()
{
    return {_lits_balanced_redirected.exchange(0, std::memory_order_relaxed),
            _lits_balanced_placed.exchange(0, std::memory_order_relaxed)};
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::save_lits_cluster_labels(const std::string &path) const
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open())
        throw ANNException("save_lits_cluster_labels: could not open " + path, -1, __FUNCSIG__, __FILE__, __LINE__);
    uint32_t num_entries = static_cast<uint32_t>(_lits_build_cluster_labels.size());
    uint32_t n_clusters = _lits_balanced_num_clusters;
    out.write(reinterpret_cast<const char *>(&num_entries), sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(&n_clusters), sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(_lits_build_cluster_labels.data()),
              static_cast<std::streamsize>(num_entries) * sizeof(uint32_t));
    out.close();
    diskann::cout << "[balanced-alloc] saved cluster labels: " << num_entries << " entries, " << n_clusters
                  << " clusters -> " << path << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::occlude_list(const uint32_t location, std::vector<Neighbor> &pool, const float alpha,
                                          const uint32_t degree, const uint32_t maxc, std::vector<uint32_t> &result,
                                          InMemQueryScratch<T> *scratch,
                                          const tsl::robin_set<uint32_t> *const delete_set_ptr)
{
    if (pool.size() == 0)
        return;

    // Truncate pool at maxc and initialize scratch spaces
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(result.size() == 0);
    if (pool.size() > maxc)
        pool.resize(maxc);
    std::vector<float> &occlude_factor = scratch->occlude_factor();
    // occlude_list can be called with the same scratch more than once by
    // search_for_point_and_add_link through inter_insert.
    occlude_factor.clear();
    // Initialize occlude_factor to pool.size() many 0.0f values for correctness
    occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree)
    {
        // used for MIPS, where we store a value of eps in cur_alpha to
        // denote pruned out entries which we can skip in later rounds.
        float eps = cur_alpha + 0.01f;

        for (auto iter = pool.begin(); result.size() < degree && iter != pool.end(); ++iter)
        {
            if (occlude_factor[iter - pool.begin()] > cur_alpha)
            {
                continue;
            }
            // Set the entry to float::max so that is not considered again
            occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
            // Add the entry to the result if its not been deleted, and doesn't
            // add a self loop
            if (delete_set_ptr == nullptr || delete_set_ptr->find(iter->id) == delete_set_ptr->end())
            {
                if (iter->id != location)
                {
                    result.push_back(iter->id);
                }
            }

            // Update occlude factor for points from iter+1 to pool.end()
            for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++)
            {
                auto t = iter2 - pool.begin();
                if (occlude_factor[t] > alpha)
                    continue;

                bool prune_allowed = true;
                if (_filtered_index)
                {
                    uint32_t a = iter->id;
                    uint32_t b = iter2->id;
                    if (_location_to_labels.size() < b || _location_to_labels.size() < a)
                        continue;
                    for (auto &x : _location_to_labels[b])
                    {
                        if (std::find(_location_to_labels[a].begin(), _location_to_labels[a].end(), x) ==
                            _location_to_labels[a].end())
                        {
                            prune_allowed = false;
                        }
                        if (!prune_allowed)
                            break;
                    }
                }
                if (!prune_allowed)
                    continue;

                float djk = _data_store->get_distance(iter2->id, iter->id);
                if (_dist_metric == diskann::Metric::L2 || _dist_metric == diskann::Metric::COSINE)
                {
                    occlude_factor[t] = (djk == 0) ? std::numeric_limits<float>::max()
                                                   : std::max(occlude_factor[t], iter2->distance / djk);
                }
                else if (_dist_metric == diskann::Metric::INNER_PRODUCT)
                {
                    // Improvization for flipping max and min dist for MIPS
                    float x = -iter2->distance;
                    float y = -djk;
                    if (y > cur_alpha * x)
                    {
                        occlude_factor[t] = std::max(occlude_factor[t], eps);
                    }
                }
            }
        }
        cur_alpha *= 1.2f;
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::partition_aware_occlude_list(const uint32_t location, std::vector<Neighbor> &pool,
                                                           const float alpha, const uint32_t degree, const uint32_t maxc,
                                                           std::vector<uint32_t> &result, InMemQueryScratch<T> *scratch,
                                                           const tsl::robin_set<uint32_t> *const delete_set_ptr)
{
    if (pool.size() == 0)
        return;

    // Truncate pool at maxc and initialize scratch spaces
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(result.size() == 0);
    if (pool.size() > maxc)
        pool.resize(maxc);

    std::vector<float> &occlude_factor = scratch->occlude_factor();
    occlude_factor.clear();
    occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

    // Get source partition (partition of the node we're building neighbors for)
    uint32_t source_partition = _partition_loaded ? get_partition(location) : 0;
    const uint32_t INVALID_PARTITION = std::numeric_limits<uint32_t>::max();

    float cur_alpha = 1;

    while (cur_alpha <= alpha && result.size() < degree)
    {
        float eps = cur_alpha + 0.01f;

        for (auto iter = pool.begin(); result.size() < degree && iter != pool.end(); ++iter)
        {
            size_t iter_idx = iter - pool.begin();

            if (occlude_factor[iter_idx] > cur_alpha)
            {
                continue;
            }

            // Skip deleted nodes and self-loops
            if (delete_set_ptr != nullptr && delete_set_ptr->find(iter->id) != delete_set_ptr->end())
            {
                occlude_factor[iter_idx] = std::numeric_limits<float>::max();
                continue;
            }
            if (iter->id == location)
            {
                occlude_factor[iter_idx] = std::numeric_limits<float>::max();
                continue;
            }

            // Collect candidates that will be pruned by iter (same occlude group)
            // These are candidates where: occlude_factor will exceed cur_alpha after update
            std::vector<size_t> occlude_group;
            occlude_group.push_back(iter_idx);  // iter itself is part of the group

            // Check which candidates will be pruned by iter AND update occlude_factor
            for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++)
            {
                auto t = iter2 - pool.begin();
                if (occlude_factor[t] > alpha)
                    continue;

                // Check if pruning is allowed (filtered index compatibility)
                bool prune_allowed = true;
                if (_filtered_index)
                {
                    uint32_t a = iter->id;
                    uint32_t b = iter2->id;
                    if (_location_to_labels.size() < b || _location_to_labels.size() < a)
                        continue;
                    for (auto &x : _location_to_labels[b])
                    {
                        if (std::find(_location_to_labels[a].begin(), _location_to_labels[a].end(), x) ==
                            _location_to_labels[a].end())
                        {
                            prune_allowed = false;
                        }
                        if (!prune_allowed)
                            break;
                    }
                }
                if (!prune_allowed)
                    continue;

                // Calculate and UPDATE occlude factor (same as original occlude_list)
                float djk = _data_store->get_distance(iter2->id, iter->id);

                if (_dist_metric == diskann::Metric::L2 || _dist_metric == diskann::Metric::COSINE)
                {
                    occlude_factor[t] = (djk == 0) ? std::numeric_limits<float>::max()
                                                   : std::max(occlude_factor[t], iter2->distance / djk);
                }
                else if (_dist_metric == diskann::Metric::INNER_PRODUCT)
                {
                    float x = -iter2->distance;
                    float y = -djk;
                    if (y > cur_alpha * x)
                    {
                        occlude_factor[t] = std::max(occlude_factor[t], eps);
                    }
                }

                // If this candidate is now pruned by iter, add to occlude group
                // (only consider valid candidates - not deleted, not self-loop)
                if (occlude_factor[t] > cur_alpha)
                {
                    if (delete_set_ptr != nullptr && delete_set_ptr->find(iter2->id) != delete_set_ptr->end())
                        continue;
                    if (iter2->id == location)
                        continue;
                    occlude_group.push_back(t);
                }
            }

            // Partition-aware selection from occlude group:
            // Find the best candidate (prefer intra-partition)
            size_t selected_idx = iter_idx;  // default to iter

            if (_partition_loaded)
            {
                uint32_t iter_partition = get_partition(iter->id);

                // If iter is inter-partition, look for intra-partition in the group
                if (iter_partition != source_partition)
                {
                    for (size_t i = 1; i < occlude_group.size(); ++i)  // start from 1, skip iter
                    {
                        size_t candidate_idx = occlude_group[i];
                        uint32_t candidate_partition = get_partition(pool[candidate_idx].id);

                        if (candidate_partition == source_partition)
                        {
                            // Found intra-partition candidate in the same occlude group
                            selected_idx = candidate_idx;
                            break;
                        }
                    }
                }
                // If iter is intra-partition, just use iter (selected_idx = iter_idx)
            }

            // Mark iter as processed (selected or not, it's been evaluated)
            occlude_factor[iter_idx] = std::numeric_limits<float>::max();

            // Add selected to result
            result.push_back(pool[selected_idx].id);
        }
        cur_alpha *= 1.2f;
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::greedy_nearest_prune(const uint32_t location, std::vector<Neighbor> &pool,
                                                   const uint32_t range, const uint32_t maxc,
                                                   std::vector<uint32_t> &result,
                                                   const tsl::robin_set<uint32_t> *const delete_set_ptr)
{
    if (pool.size() == 0)
        return;

    // Greedy strategy: simply select the K nearest neighbors
    // No occlusion calculation - just take top K by distance
    assert(std::is_sorted(pool.begin(), pool.end()));  // Pool is sorted by distance
    assert(result.size() == 0);

    // Truncate pool at maxc
    if (pool.size() > maxc)
        pool.resize(maxc);

    // Select up to 'range' nearest neighbors
    size_t target_count = std::min((size_t)range, pool.size());

    for (size_t i = 0; i < pool.size() && result.size() < target_count; ++i)
    {
        const auto &neighbor = pool[i];
        // Skip deleted nodes and self-loops
        if (delete_set_ptr == nullptr || delete_set_ptr->find(neighbor.id) == delete_set_ptr->end())
        {
            if (neighbor.id != location)
            {
                result.push_back(neighbor.id);
            }
        }
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::median_evict_prune(const uint32_t location, std::vector<Neighbor> &pool,
                                                 const uint32_t range, std::vector<uint32_t> &result)
{
    // Pool is sorted by distance (ascending).
    // Keep short-range (front) + long-range (back), evict middle-distance edges.
    if (pool.size() == 0)
        return;

    // Filter out self-loops first
    std::vector<Neighbor> filtered;
    filtered.reserve(pool.size());
    for (auto &n : pool)
    {
        if (n.id != location)
            filtered.push_back(n);
    }

    if (filtered.size() <= range)
    {
        for (auto &n : filtered)
            result.push_back(n.id);
        return;
    }

    // Keep first keep_short from front (nearest), last keep_long from back (farthest)
    size_t keep_short = range / 2;
    size_t keep_long = range - keep_short;

    for (size_t i = 0; i < keep_short; i++)
        result.push_back(filtered[i].id);

    size_t long_start = filtered.size() - keep_long;
    for (size_t i = long_start; i < filtered.size(); i++)
        result.push_back(filtered[i].id);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::random_prune(const uint32_t location, std::vector<Neighbor> &pool,
                                          const uint32_t target_degree, const uint32_t maxc,
                                          std::vector<uint32_t> &result, InMemQueryScratch<T> *scratch,
                                          const tsl::robin_set<uint32_t> *const delete_set_ptr)
{
    if (pool.size() == 0)
        return;

    // Truncate pool at maxc
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(result.size() == 0);
    if (pool.size() > maxc)
        pool.resize(maxc);

    // Stratified random sampling strategy:
    // - Sample more from closer neighbors (top half)
    // - Sample less from farther neighbors (bottom half)
    // This maintains locality while reducing degree

    size_t pool_size = pool.size();
    size_t actual_target = std::min((size_t)target_degree, pool_size);

    // Create index array for shuffling
    std::vector<size_t> indices(pool_size);
    std::iota(indices.begin(), indices.end(), 0);

    // Use thread-local random generator for thread safety
    static thread_local std::mt19937 gen(std::random_device{}());

    // Stratified sampling: 70% from near half, 30% from far half
    size_t half = pool_size / 2;
    size_t from_near = std::min((size_t)(actual_target * 0.7), half);
    size_t from_far = actual_target - from_near;

    // Shuffle near half and select
    std::shuffle(indices.begin(), indices.begin() + half, gen);

    // Shuffle far half and select
    if (pool_size > half)
        std::shuffle(indices.begin() + half, indices.end(), gen);

    // Collect selected indices
    for (size_t i = 0; i < from_near; ++i)
    {
        const auto &neighbor = pool[indices[i]];
        // Skip deleted nodes and self-loops
        if (delete_set_ptr == nullptr || delete_set_ptr->find(neighbor.id) == delete_set_ptr->end())
        {
            if (neighbor.id != location)
            {
                result.push_back(neighbor.id);
            }
        }
    }

    for (size_t i = 0; i < from_far && (half + i) < pool_size; ++i)
    {
        const auto &neighbor = pool[indices[half + i]];
        // Skip deleted nodes and self-loops
        if (delete_set_ptr == nullptr || delete_set_ptr->find(neighbor.id) == delete_set_ptr->end())
        {
            if (neighbor.id != location)
            {
                result.push_back(neighbor.id);
            }
        }
    }

    // If we couldn't get enough valid nodes, fill from the remaining candidates
    if (result.size() < actual_target)
    {
        for (size_t i = 0; i < pool_size && result.size() < actual_target; ++i)
        {
            bool already_added = false;
            for (auto id : result)
            {
                if (id == pool[i].id)
                {
                    already_added = true;
                    break;
                }
            }
            if (!already_added && pool[i].id != location &&
                (delete_set_ptr == nullptr || delete_set_ptr->find(pool[i].id) == delete_set_ptr->end()))
            {
                result.push_back(pool[i].id);
            }
        }
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::apply_random_reduction(std::vector<uint32_t> &pruned_list, float ratio)
{
    // Apply random reduction to achieve target degree
    // This is used by RANDOM_PRUNE strategy to reduce degree after robust pruning
    if (pruned_list.size() == 0 || ratio >= 1.0f)
        return;

    size_t target_size = static_cast<size_t>(pruned_list.size() * ratio);
    if (target_size < pruned_list.size())
    {
        // Randomly shuffle and keep only target_size elements
        static thread_local std::mt19937 gen(std::random_device{}());
        std::shuffle(pruned_list.begin(), pruned_list.end(), gen);
        pruned_list.resize(target_size);
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool,
                                             std::vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch)
{
    prune_neighbors(location, pool, _indexingRange, _indexingMaxC, _indexingAlpha, pruned_list, scratch);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool, const uint32_t range,
                                             const uint32_t max_candidate_size, const float alpha,
                                             std::vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch)
{
    if (pool.size() == 0)
    {
        // if the pool is empty, behave like a noop
        pruned_list.clear();
        return;
    }

    // If using _pq_build, over-write the PQ distances with actual distances
    // REFACTOR PQ: TODO: How to get rid of this!?
    if (_pq_dist)
    {
        for (auto &ngh : pool)
            ngh.distance = _data_store->get_distance(ngh.id, location);
    }

    // sort the pool based on distance to query and prune it with strategy
    std::sort(pool.begin(), pool.end());
    pruned_list.clear();
    pruned_list.reserve(range);

    // Strategy router: select pruning algorithm based on _pruningStrategy
    if (_pruningStrategy == PruningStrategy::GREEDY_NEAREST)
    {
        // Greedy strategy: select K nearest neighbors (no occlusion)
        greedy_nearest_prune(location, pool, range, max_candidate_size, pruned_list);
    }
    else if (_pruningStrategy == PruningStrategy::PARTITION_AWARE_PRUNE)
    {
        // Partition-aware pruning: prioritize intra-partition edges
        partition_aware_occlude_list(location, pool, alpha, range, max_candidate_size, pruned_list, scratch);
    }
    else if (_pruningStrategy == PruningStrategy::MEDIAN_EVICT)
    {
        // Median-distance eviction: keep short-range + long-range, evict middle
        median_evict_prune(location, pool, range, pruned_list);
    }
    else
    {
        // Use RobustPrune (occlude_list) for ROBUST_PRUNE, RANDOM_PRUNE, and others
        occlude_list(location, pool, alpha, range, max_candidate_size, pruned_list, scratch);

        // Apply additional random reduction if RANDOM_PRUNE strategy is enabled
        if (_pruningStrategy == PruningStrategy::RANDOM_PRUNE)
        {
            apply_random_reduction(pruned_list, _randomPruneRatio);
        }
    }

    assert(pruned_list.size() <= range);

    if (_saturate_graph && alpha > 1)
    {
        for (const auto &node : pool)
        {
            if (pruned_list.size() >= range)
                break;
            if ((std::find(pruned_list.begin(), pruned_list.end(), node.id) == pruned_list.end()) &&
                node.id != location)
                pruned_list.push_back(node.id);
        }
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, const uint32_t range,
                                          InMemQueryScratch<T> *scratch)
{
    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    for (auto des : src_pool)
    {
        // des.loc is the loc of the neighbors of n
        assert(des < _max_points + _num_frozen_pts);
        // des_pool contains the neighbors of the neighbors of n
        std::vector<uint32_t> copy_of_neighbors;
        bool prune_needed = false;
        {
            LockGuard guard(_locks[des]);
            auto &des_pool = _graph_store->get_neighbours(des);
            if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end())
            {
                if (des_pool.size() < (uint64_t)(defaults::GRAPH_SLACK_FACTOR * range))
                {
                    // des_pool.emplace_back(n);
                    _graph_store->add_neighbour(des, n);
                    prune_needed = false;
                }
                else
                {
                    copy_of_neighbors.reserve(des_pool.size() + 1);
                    copy_of_neighbors = des_pool;
                    copy_of_neighbors.push_back(n);
                    prune_needed = true;
                }
            }
        } // des lock is released by this point

        if (prune_needed)
        {
            // Increment counter for inter-insert pruning statistics
            _num_inter_insert_prunes.fetch_add(1, std::memory_order_relaxed);

            // Store original neighbors BEFORE adding candidate (for trace logging)
            std::vector<uint32_t> original_neighbors;
            {
                LockGuard guard(_locks[des]);
                original_neighbors = _graph_store->get_neighbours(des);
            }

            tsl::robin_set<uint32_t> dummy_visited(0);
            std::vector<Neighbor> dummy_pool(0);

            size_t reserveSize = (size_t)(std::ceil(1.05 * defaults::GRAPH_SLACK_FACTOR * range));
            dummy_visited.reserve(reserveSize);
            dummy_pool.reserve(reserveSize);

            for (auto cur_nbr : copy_of_neighbors)
            {
                if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != des)
                {
                    float dist = _data_store->get_distance(des, cur_nbr);
                    dummy_pool.emplace_back(Neighbor(cur_nbr, dist));
                    dummy_visited.insert(cur_nbr);
                }
            }
            std::vector<uint32_t> new_out_neighbors;
            prune_neighbors(des, dummy_pool, new_out_neighbors, scratch);

            // Record pruning trace if enabled
            if (_enable_pruning_trace.load(std::memory_order_relaxed))
            {
                int thread_id = omp_get_thread_num();
                PruningTraceEvent event;
                event.node_id = des;
                event.edges_before = original_neighbors;  // Original neighbors BEFORE candidate addition
                event.edges_after = new_out_neighbors;
                _inter_insert_trace_buffers[thread_id].push_back(std::move(event));
            }

            {
                LockGuard guard(_locks[des]);

                _graph_store->set_neighbours(des, new_out_neighbors);
            }
        }
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch)
{
    inter_insert(n, pruned_list, _indexingRange, scratch);
}

template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::link()
{
    uint32_t num_threads = _indexingThreads;
    if (num_threads != 0)
        omp_set_num_threads(num_threads);

    /* visit_order is a vector that is initialized to the entire graph */
    std::vector<uint32_t> visit_order;
    std::vector<diskann::Neighbor> pool, tmp;
    tsl::robin_set<uint32_t> visited;
    visit_order.reserve(_nd + _num_frozen_pts);
    for (uint32_t i = 0; i < (uint32_t)_nd; i++)
    {
        visit_order.emplace_back(i);
    }

    // If there are any frozen points, add them all.
    for (uint32_t frozen = (uint32_t)_max_points; frozen < _max_points + _num_frozen_pts; frozen++)
    {
        visit_order.emplace_back(frozen);
    }

    // if there are frozen points, the first such one is set to be the _start
    if (_num_frozen_pts > 0)
        _start = (uint32_t)_max_points;
    else
        _start = calculate_entry_point();

    diskann::Timer link_timer;

#pragma omp parallel for schedule(dynamic, 2048)
    for (int64_t node_ctr = 0; node_ctr < (int64_t)(visit_order.size()); node_ctr++)
    {
        auto node = visit_order[node_ctr];

        // Find and add appropriate graph edges
        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
        auto scratch = manager.scratch_space();
        std::vector<uint32_t> pruned_list;
        if (_filtered_index)
        {
            search_for_point_and_prune(node, _indexingQueueSize, pruned_list, scratch, true, _filterIndexingQueueSize);
        }
        else
        {
            search_for_point_and_prune(node, _indexingQueueSize, pruned_list, scratch);
        }
        assert(pruned_list.size() > 0);

        {
            LockGuard guard(_locks[node]);

            _graph_store->set_neighbours(node, pruned_list);
            assert(_graph_store->get_neighbours((location_t)node).size() <= _indexingRange);
        }

        inter_insert(node, pruned_list, scratch);

        if (node_ctr % 100000 == 0)
        {
            diskann::cout << "\r" << (100.0 * node_ctr) / (visit_order.size()) << "% of index build completed."
                          << std::flush;
        }
    }

    if (_nd > 0)
    {
        diskann::cout << "Starting final cleanup.." << std::flush;
    }
#pragma omp parallel for schedule(dynamic, 2048)
    for (int64_t node_ctr = 0; node_ctr < (int64_t)(visit_order.size()); node_ctr++)
    {
        auto node = visit_order[node_ctr];
        if (_graph_store->get_neighbours((location_t)node).size() > _indexingRange)
        {
            ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
            auto scratch = manager.scratch_space();

            tsl::robin_set<uint32_t> dummy_visited(0);
            std::vector<Neighbor> dummy_pool(0);
            std::vector<uint32_t> new_out_neighbors;

            for (auto cur_nbr : _graph_store->get_neighbours((location_t)node))
            {
                if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != node)
                {
                    float dist = _data_store->get_distance(node, cur_nbr);
                    dummy_pool.emplace_back(Neighbor(cur_nbr, dist));
                    dummy_visited.insert(cur_nbr);
                }
            }
            prune_neighbors(node, dummy_pool, new_out_neighbors, scratch);

            _graph_store->clear_neighbours((location_t)node);
            _graph_store->set_neighbours((location_t)node, new_out_neighbors);
        }
    }
    if (_nd > 0)
    {
        diskann::cout << "done. Link time: " << ((double)link_timer.elapsed() / (double)1000000) << "s" << std::endl;
        diskann::cout << "Inter-insert pruning operations: " << _num_inter_insert_prunes.load(std::memory_order_relaxed)
                      << std::endl;
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::prune_all_neighbors(const uint32_t max_degree, const uint32_t max_occlusion_size,
                                                 const float alpha)
{
    const uint32_t range = max_degree;
    const uint32_t maxc = max_occlusion_size;

    _filtered_index = true;

    diskann::Timer timer;
#pragma omp parallel for
    for (int64_t node = 0; node < (int64_t)(_max_points + _num_frozen_pts); node++)
    {
        if ((size_t)node < _nd || (size_t)node >= _max_points)
        {
            if (_graph_store->get_neighbours((location_t)node).size() > range)
            {
                tsl::robin_set<uint32_t> dummy_visited(0);
                std::vector<Neighbor> dummy_pool(0);
                std::vector<uint32_t> new_out_neighbors;

                ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
                auto scratch = manager.scratch_space();

                for (auto cur_nbr : _graph_store->get_neighbours((location_t)node))
                {
                    if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != node)
                    {
                        float dist = _data_store->get_distance((location_t)node, (location_t)cur_nbr);
                        dummy_pool.emplace_back(Neighbor(cur_nbr, dist));
                        dummy_visited.insert(cur_nbr);
                    }
                }

                prune_neighbors((uint32_t)node, dummy_pool, range, maxc, alpha, new_out_neighbors, scratch);
                _graph_store->clear_neighbours((location_t)node);
                _graph_store->set_neighbours((location_t)node, new_out_neighbors);
            }
        }
    }

    diskann::cout << "Prune time : " << timer.elapsed() / 1000 << "ms" << std::endl;
    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _max_points + _num_frozen_pts; i++)
    {
        if (i < _nd || i >= _max_points)
        {
            const std::vector<uint32_t> &pool = _graph_store->get_neighbours((location_t)i);
            max = (std::max)(max, pool.size());
            min = (std::min)(min, pool.size());
            total += pool.size();
            if (pool.size() < 2)
                cnt++;
        }
    }
    if (min > max)
        min = max;
    if (_nd > 0)
    {
        diskann::cout << "Index built with degree: max:" << max
                      << "  avg:" << (float)total / (float)(_nd + _num_frozen_pts) << "  min:" << min
                      << "  count(deg<2):" << cnt << std::endl;
    }
}

// REFACTOR
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_start_points(const T *data, size_t data_count)
{
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    if (_nd > 0)
        throw ANNException("Can not set starting point for a non-empty index", -1, __FUNCSIG__, __FILE__, __LINE__);

    if (data_count != _num_frozen_pts * _dim)
        throw ANNException("Invalid number of points", -1, __FUNCSIG__, __FILE__, __LINE__);

    //     memcpy(_data + _aligned_dim * _max_points, data, _aligned_dim *
    //     sizeof(T) * _num_frozen_pts);
    for (location_t i = 0; i < _num_frozen_pts; i++)
    {
        _data_store->set_vector((location_t)(i + _max_points), data + i * _dim);
    }
    _has_built = true;
    diskann::cout << "Index start points set: #" << _num_frozen_pts << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::_set_start_points_at_random(DataType radius, uint32_t random_seed)
{
    try
    {
        T radius_to_use = std::any_cast<T>(radius);
        this->set_start_points_at_random(radius_to_use, random_seed);
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException(
            "Error: bad any cast while performing _set_start_points_at_random() " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error: " + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_start_points_at_random(T radius, uint32_t random_seed)
{
    std::mt19937 gen{random_seed};
    std::normal_distribution<> d{0.0, 1.0};

    std::vector<T> points_data;
    points_data.reserve(_dim * _num_frozen_pts);
    std::vector<double> real_vec(_dim);

    for (size_t frozen_point = 0; frozen_point < _num_frozen_pts; frozen_point++)
    {
        double norm_sq = 0.0;
        for (size_t i = 0; i < _dim; ++i)
        {
            auto r = d(gen);
            real_vec[i] = r;
            norm_sq += r * r;
        }

        const double norm = std::sqrt(norm_sq);
        for (auto iter : real_vec)
            points_data.push_back(static_cast<T>(iter * radius / norm));
    }

    set_start_points(points_data.data(), points_data.size());
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build_with_data_populated(const std::vector<TagT> &tags)
{
    diskann::cout << "Starting index build with " << _nd << " points... " << std::endl;

    if (_nd < 1)
        throw ANNException("Error: Trying to build an index with 0 points", -1, __FUNCSIG__, __FILE__, __LINE__);

    if (_enable_tags && tags.size() != _nd)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _nd << " points from file,"
               << "but tags vector is of size " << tags.size() << "." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (_enable_tags)
    {
        for (size_t i = 0; i < tags.size(); ++i)
        {
            _tag_to_location[tags[i]] = (uint32_t)i;
            _location_to_tag.set(static_cast<uint32_t>(i), tags[i]);
        }
    }

    uint32_t index_R = _indexingRange;
    uint32_t num_threads_index = _indexingThreads;
    uint32_t index_L = _indexingQueueSize;
    uint32_t maxc = _indexingMaxC;

    if (_query_scratch.size() == 0)
    {
        initialize_query_scratch(5 + num_threads_index, index_L, index_L, index_R, maxc,
                                 _data_store->get_aligned_dim());
    }

    generate_frozen_point();
    link();

    size_t max = 0, min = SIZE_MAX, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++)
    {
        auto &pool = _graph_store->get_neighbours((location_t)i);
        max = std::max(max, pool.size());
        min = std::min(min, pool.size());
        total += pool.size();
        if (pool.size() < 2)
            cnt++;
    }
    diskann::cout << "Index built with degree: max:" << max << "  avg:" << (float)total / (float)(_nd + _num_frozen_pts)
                  << "  min:" << min << "  count(deg<2):" << cnt << std::endl;

    _has_built = true;

    _empty_slots.clear();
    _empty_slots.reserve(_max_points);
    for (auto i = _nd; i < _max_points; i++)
    {
        _empty_slots.insert((uint32_t)i);
    }
}
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::_build(const DataType &data, const size_t num_points_to_load, TagVector &tags)
{
    try
    {
        this->build(std::any_cast<const T *>(data), num_points_to_load, tags.get<const std::vector<TagT>>());
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad any cast in while building index. " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error" + std::string(e.what()), -1);
    }
}
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build(const T *data, const size_t num_points_to_load, const std::vector<TagT> &tags)
{
    if (num_points_to_load == 0)
    {
        throw ANNException("Do not call build with 0 points", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (_pq_dist)
    {
        throw ANNException("ERROR: DO not use this build interface with PQ distance", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    }

    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);

    {
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        _nd = num_points_to_load;

        _data_store->populate_data(data, (location_t)num_points_to_load);
    }

    build_with_data_populated(tags);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build(const char *filename, const size_t num_points_to_load, const std::vector<TagT> &tags)
{
    // idealy this should call build_filtered_index based on params passed

    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);

    // error checks
    if (num_points_to_load == 0)
        throw ANNException("Do not call build with 0 points", -1, __FUNCSIG__, __FILE__, __LINE__);

    if (!file_exists(filename))
    {
        std::stringstream stream;
        stream << "ERROR: Data file " << filename << " does not exist." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    size_t file_num_points, file_dim;
    if (filename == nullptr)
    {
        throw diskann::ANNException("Can not build with an empty file", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::get_bin_metadata(filename, file_num_points, file_dim);
    if (file_num_points > _max_points)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << num_points_to_load << " points and file has " << file_num_points
               << " points, but "
               << "index can support only " << _max_points << " points as specified in constructor." << std::endl;

        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (num_points_to_load > file_num_points)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << num_points_to_load << " points and file has only "
               << file_num_points << " points." << std::endl;

        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_dim != _dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;

        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // REFACTOR PQ TODO: We can remove this if and add a check in the InMemDataStore
    // to not populate_data if it has been called once.
    if (_pq_dist)
    {
#ifdef EXEC_ENV_OLS
        std::stringstream ss;
        ss << "PQ Build is not supported in DLVS environment (i.e. if EXEC_ENV_OLS is defined)" << std::endl;
        diskann::cerr << ss.str() << std::endl;
        throw ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
#else
        // REFACTOR TODO: Both in the previous code and in the current PQDataStore,
        // we are writing the PQ files in the same path as the input file. Now we
        // may not have write permissions to that folder, but we will always have
        // write permissions to the output folder. So we should write the PQ files
        // there. The problem is that the Index class gets the output folder prefix
        // only at the time of save(), by which time we are too late. So leaving it
        // as-is for now.
        _pq_data_store->populate_data(filename, 0U);
#endif
    }

    _data_store->populate_data(filename, 0U);
    diskann::cout << "Using only first " << num_points_to_load << " from file.. " << std::endl;

    {
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        _nd = num_points_to_load;
    }
    build_with_data_populated(tags);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build(const char *filename, const size_t num_points_to_load, const char *tag_filename)
{
    std::vector<TagT> tags;

    if (_enable_tags)
    {
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        if (tag_filename == nullptr)
        {
            throw ANNException("Tag filename is null, while _enable_tags is set", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        else
        {
            if (file_exists(tag_filename))
            {
                diskann::cout << "Loading tags from " << tag_filename << " for vamana index build" << std::endl;
                TagT *tag_data = nullptr;
                size_t npts, ndim;
                diskann::load_bin(tag_filename, tag_data, npts, ndim);
                if (npts < num_points_to_load)
                {
                    std::stringstream sstream;
                    sstream << "Loaded " << npts << " tags, insufficient to populate tags for " << num_points_to_load
                            << "  points to load";
                    throw diskann::ANNException(sstream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
                }
                for (size_t i = 0; i < num_points_to_load; i++)
                {
                    tags.push_back(tag_data[i]);
                }
                delete[] tag_data;
            }
            else
            {
                throw diskann::ANNException(std::string("Tag file") + tag_filename + " does not exist", -1, __FUNCSIG__,
                                            __FILE__, __LINE__);
            }
        }
    }
    build(filename, num_points_to_load, tags);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build(const std::string &data_file, const size_t num_points_to_load,
                                   IndexFilterParams &filter_params)
{
    size_t points_to_load = num_points_to_load == 0 ? _max_points : num_points_to_load;

    auto s = std::chrono::high_resolution_clock::now();
    if (filter_params.label_file == "")
    {
        this->build(data_file.c_str(), points_to_load);
    }
    else
    {
        // TODO: this should ideally happen in save()
        std::string labels_file_to_use = filter_params.save_path_prefix + "_label_formatted.txt";
        std::string mem_labels_int_map_file = filter_params.save_path_prefix + "_labels_map.txt";
        convert_labels_string_to_int(filter_params.label_file, labels_file_to_use, mem_labels_int_map_file,
                                     filter_params.universal_label);
        if (filter_params.universal_label != "")
        {
            LabelT unv_label_as_num = 0;
            this->set_universal_label(unv_label_as_num);
        }
        this->build_filtered_index(data_file.c_str(), labels_file_to_use, points_to_load);
    }
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
    std::cout << "Indexing time: " << diff.count() << "\n";
}

template <typename T, typename TagT, typename LabelT>
std::unordered_map<std::string, LabelT> Index<T, TagT, LabelT>::load_label_map(const std::string &labels_map_file)
{
    std::unordered_map<std::string, LabelT> string_to_int_mp;
    std::ifstream map_reader(labels_map_file);
    std::string line, token;
    LabelT token_as_num;
    std::string label_str;
    while (std::getline(map_reader, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        label_str = token;
        getline(iss, token, '\t');
        token_as_num = (LabelT)std::stoul(token);
        string_to_int_mp[label_str] = token_as_num;
    }
    return string_to_int_mp;
}

template <typename T, typename TagT, typename LabelT>
LabelT Index<T, TagT, LabelT>::get_converted_label(const std::string &raw_label)
{
    if (_label_map.find(raw_label) != _label_map.end())
    {
        return _label_map[raw_label];
    }
    if (_use_universal_label)
    {
        return _universal_label;
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::parse_label_file(const std::string &label_file, size_t &num_points)
{
    // Format of Label txt file: filters with comma separators

    std::ifstream infile(label_file);
    if (infile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file ") + label_file, -1);
    }

    std::string line, token;
    uint32_t line_cnt = 0;

    while (std::getline(infile, line))
    {
        line_cnt++;
    }
    _location_to_labels.resize(line_cnt, std::vector<LabelT>());

    infile.clear();
    infile.seekg(0, std::ios::beg);
    line_cnt = 0;

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::vector<LabelT> lbls(0);
        getline(iss, token, '\t');
        std::istringstream new_iss(token);
        while (getline(new_iss, token, ','))
        {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
            LabelT token_as_num = (LabelT)std::stoul(token);
            lbls.push_back(token_as_num);
            _labels.insert(token_as_num);
        }

        std::sort(lbls.begin(), lbls.end());
        _location_to_labels[line_cnt] = lbls;
        line_cnt++;
    }
    num_points = (size_t)line_cnt;
    diskann::cout << "Identified " << _labels.size() << " distinct label(s)" << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::load_partition_file(const std::string &partition_file)
{
    if (partition_file.empty())
    {
        diskann::cout << "No partition file specified, partition-aware pruning disabled" << std::endl;
        _partition_loaded = false;
        return;
    }

    std::ifstream reader(partition_file, std::ios::binary);
    if (!reader.is_open())
    {
        throw diskann::ANNException("Failed to open partition file: " + partition_file, -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Read header: [num_locations (uint32)] [dim=1 (uint32)]
    uint32_t num_locations, dim;
    reader.read(reinterpret_cast<char *>(&num_locations), sizeof(uint32_t));
    reader.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));

    if (dim != 1)
    {
        throw diskann::ANNException("Invalid partition file: dimension must be 1, got " + std::to_string(dim), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Resize and read partition IDs
    _location_to_partition.resize(num_locations);
    reader.read(reinterpret_cast<char *>(_location_to_partition.data()), num_locations * sizeof(uint32_t));

    if (reader.fail())
    {
        throw diskann::ANNException("Failed to read partition data from file: " + partition_file, -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    reader.close();
    _partition_loaded = true;

    // Count partitions for logging
    std::set<uint32_t> unique_partitions;
    const uint32_t INVALID_PARTITION = std::numeric_limits<uint32_t>::max();
    for (uint32_t p : _location_to_partition)
    {
        if (p != INVALID_PARTITION)
        {
            unique_partitions.insert(p);
        }
    }

    diskann::cout << "Loaded partition file: " << partition_file << std::endl;
    diskann::cout << "  - " << num_locations << " locations" << std::endl;
    diskann::cout << "  - " << unique_partitions.size() << " unique partitions" << std::endl;
}

template <typename T, typename TagT, typename LabelT>
uint32_t Index<T, TagT, LabelT>::get_partition(uint32_t location) const
{
    if (!_partition_loaded || location >= _location_to_partition.size())
    {
        return std::numeric_limits<uint32_t>::max();  // Invalid partition
    }
    return _location_to_partition[location];
}

template <typename T, typename TagT, typename LabelT>
bool Index<T, TagT, LabelT>::is_partition_loaded() const
{
    return _partition_loaded;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::load_centroid_file(const std::string &centroid_file)
{
    if (centroid_file.empty())
    {
        diskann::cout << "No centroid file specified, partition assignment for new nodes disabled" << std::endl;
        return;
    }

    std::ifstream reader(centroid_file, std::ios::binary);
    if (!reader.is_open())
    {
        throw diskann::ANNException("Failed to open centroid file: " + centroid_file, -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Read header: [num_centroids (uint32)] [dim (uint32)]
    uint32_t num_centroids, dim;
    reader.read(reinterpret_cast<char *>(&num_centroids), sizeof(uint32_t));
    reader.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));

    if (dim != _data_store->get_dims())
    {
        throw diskann::ANNException("Centroid dimension (" + std::to_string(dim) +
                                    ") does not match data dimension (" + std::to_string(_data_store->get_dims()) + ")",
                                    -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Resize and read centroids
    _centroids.resize(num_centroids * dim);
    reader.read(reinterpret_cast<char *>(_centroids.data()), num_centroids * dim * sizeof(float));

    if (reader.fail())
    {
        throw diskann::ANNException("Failed to read centroid data from file: " + centroid_file, -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    reader.close();
    _num_partitions = num_centroids;

    // Pre-allocate _location_to_partition to _max_points size for thread safety during inserts
    if (_partition_loaded && _location_to_partition.size() < _max_points + _num_frozen_pts)
    {
        size_t old_size = _location_to_partition.size();
        _location_to_partition.resize(_max_points + _num_frozen_pts, std::numeric_limits<uint32_t>::max());
        diskann::cout << "  - Extended partition vector from " << old_size << " to " << _location_to_partition.size() << std::endl;
    }

    diskann::cout << "Loaded centroid file: " << centroid_file << std::endl;
    diskann::cout << "  - " << num_centroids << " centroids" << std::endl;
    diskann::cout << "  - " << dim << " dimensions" << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::save_partition_file(const std::string &partition_file)
{
    if (partition_file.empty())
    {
        diskann::cerr << "No partition file path specified, skipping save" << std::endl;
        return;
    }

    if (!_partition_loaded || _location_to_partition.empty())
    {
        diskann::cerr << "No partition data to save" << std::endl;
        return;
    }

    std::ofstream writer(partition_file, std::ios::binary);
    if (!writer.is_open())
    {
        throw diskann::ANNException("Failed to open partition file for writing: " + partition_file, -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Write header: [num_locations (uint32)] [dim=1 (uint32)]
    // Include frozen points to match Python format (e.g., 500001 = 500000 data + 1 frozen)
    uint32_t num_locations = static_cast<uint32_t>(_nd + _num_frozen_pts);
    uint32_t dim = 1;
    writer.write(reinterpret_cast<const char *>(&num_locations), sizeof(uint32_t));
    writer.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));

    // Write partition IDs for data points (locations 0 to _nd-1)
    for (uint32_t loc = 0; loc < _nd; ++loc)
    {
        uint32_t partition_id = (loc < _location_to_partition.size()) ?
                                 _location_to_partition[loc] :
                                 std::numeric_limits<uint32_t>::max();
        writer.write(reinterpret_cast<const char *>(&partition_id), sizeof(uint32_t));
    }

    // Write partition IDs for frozen points (at the end, after data points)
    // Frozen points are stored at _max_points to _max_points + _num_frozen_pts - 1 internally
    // but in the file they come right after data points
    for (uint32_t i = 0; i < _num_frozen_pts; ++i)
    {
        uint32_t frozen_loc = static_cast<uint32_t>(_max_points + i);
        uint32_t partition_id = (frozen_loc < _location_to_partition.size()) ?
                                 _location_to_partition[frozen_loc] :
                                 std::numeric_limits<uint32_t>::max();
        writer.write(reinterpret_cast<const char *>(&partition_id), sizeof(uint32_t));
    }

    if (writer.fail())
    {
        throw diskann::ANNException("Failed to write partition data to file: " + partition_file, -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    writer.close();
    diskann::cout << "Saved partition file: " << partition_file << " (" << num_locations << " locations)" << std::endl;
}

template <typename T, typename TagT, typename LabelT>
uint32_t Index<T, TagT, LabelT>::assign_to_nearest_centroid(const T *point) const
{
    if (_centroids.empty() || _num_partitions == 0)
    {
        return std::numeric_limits<uint32_t>::max();  // No centroids available
    }

    size_t dim = _data_store->get_dims();
    uint32_t best_partition = 0;
    float best_distance = std::numeric_limits<float>::max();

    for (uint32_t p = 0; p < _num_partitions; ++p)
    {
        const float *centroid = _centroids.data() + p * dim;
        float distance = 0.0f;

        // Compute L2 distance
        for (size_t d = 0; d < dim; ++d)
        {
            float diff = static_cast<float>(point[d]) - centroid[d];
            distance += diff * diff;
        }

        if (distance < best_distance)
        {
            best_distance = distance;
            best_partition = p;
        }
    }

    return best_partition;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::_set_universal_label(const LabelType universal_label)
{
    this->set_universal_label(std::any_cast<const LabelT>(universal_label));
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_universal_label(const LabelT &label)
{
    _use_universal_label = true;
    _universal_label = label;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build_filtered_index(const char *filename, const std::string &label_file,
                                                  const size_t num_points_to_load, const std::vector<TagT> &tags)
{
    _filtered_index = true;
    _label_to_start_id.clear();
    size_t num_points_labels = 0;

    parse_label_file(label_file,
                     num_points_labels); // determines medoid for each label and identifies
                                         // the points to label mapping

    std::unordered_map<LabelT, std::vector<uint32_t>> label_to_points;

    for (uint32_t point_id = 0; point_id < num_points_to_load; point_id++)
    {
        for (auto label : _location_to_labels[point_id])
        {
            if (label != _universal_label)
            {
                label_to_points[label].emplace_back(point_id);
            }
            else
            {
                for (typename tsl::robin_set<LabelT>::size_type lbl = 0; lbl < _labels.size(); lbl++)
                {
                    auto itr = _labels.begin();
                    std::advance(itr, lbl);
                    auto &x = *itr;
                    label_to_points[x].emplace_back(point_id);
                }
            }
        }
    }

    uint32_t num_cands = 25;
    for (auto itr = _labels.begin(); itr != _labels.end(); itr++)
    {
        uint32_t best_medoid_count = std::numeric_limits<uint32_t>::max();
        auto &curr_label = *itr;
        uint32_t best_medoid;
        auto labeled_points = label_to_points[curr_label];
        for (uint32_t cnd = 0; cnd < num_cands; cnd++)
        {
            uint32_t cur_cnd = labeled_points[rand() % labeled_points.size()];
            uint32_t cur_cnt = std::numeric_limits<uint32_t>::max();
            if (_medoid_counts.find(cur_cnd) == _medoid_counts.end())
            {
                _medoid_counts[cur_cnd] = 0;
                cur_cnt = 0;
            }
            else
            {
                cur_cnt = _medoid_counts[cur_cnd];
            }
            if (cur_cnt < best_medoid_count)
            {
                best_medoid_count = cur_cnt;
                best_medoid = cur_cnd;
            }
        }
        _label_to_start_id[curr_label] = best_medoid;
        _medoid_counts[best_medoid]++;
    }

    this->build(filename, num_points_to_load, tags);
}

template <typename T, typename TagT, typename LabelT>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::_search(const DataType &query, const size_t K, const uint32_t L,
                                                              std::any &indices, float *distances)
{
    try
    {
        auto typed_query = std::any_cast<const T *>(query);
        if (typeid(uint32_t *) == indices.type())
        {
            auto u32_ptr = std::any_cast<uint32_t *>(indices);
            return this->search(typed_query, K, L, u32_ptr, distances);
        }
        else if (typeid(uint64_t *) == indices.type())
        {
            auto u64_ptr = std::any_cast<uint64_t *>(indices);
            return this->search(typed_query, K, L, u64_ptr, distances);
        }
        else
        {
            throw ANNException("Error: indices type can only be uint64_t or uint32_t.", -1);
        }
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad any cast while searching. " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error: " + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
template <typename IdType>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::search(const T *query, const size_t K, const uint32_t L,
                                                             IdType *indices, float *distances)
{
    if (K > (uint64_t)L)
    {
        throw ANNException("Set L to a value of at least K", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto scratch = manager.scratch_space();

    if (L > scratch->get_L())
    {
        diskann::cout << "Attempting to expand query scratch_space. Was created "
                      << "with Lsize: " << scratch->get_L() << " but search L is: " << L << std::endl;
        scratch->resize_for_new_L(L);
        diskann::cout << "Resize completed. New scratch->L is " << scratch->get_L() << std::endl;
    }

    const std::vector<LabelT> unused_filter_label;
    const std::vector<uint32_t> init_ids = get_init_ids();

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    _data_store->preprocess_query(query, scratch);

    auto retval = iterate_to_fixed_point(scratch, L, init_ids, false, unused_filter_label, true);

    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();

    size_t pos = 0;
    for (size_t i = 0; i < best_L_nodes.size(); ++i)
    {
        if (best_L_nodes[i].id < _max_points)
        {
            // safe because Index uses uint32_t ids internally
            // and IDType will be uint32_t or uint64_t
            indices[pos] = (IdType)best_L_nodes[i].id;
            if (distances != nullptr)
            {
#ifdef EXEC_ENV_OLS
                // DLVS expects negative distances
                distances[pos] = best_L_nodes[i].distance;
#else
                distances[pos] = _dist_metric == diskann::Metric::INNER_PRODUCT ? -1 * best_L_nodes[i].distance
                                                                                : best_L_nodes[i].distance;
#endif
            }
            pos++;
        }
        if (pos == K)
            break;
    }
    if (pos < K)
    {
        diskann::cerr << "Found pos: " << pos << "fewer than K elements " << K << " for query" << std::endl;
    }

    return retval;
}

// Search with detailed timing statistics for latency profiling
template <typename T, typename TagT, typename LabelT>
template <typename IdType>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::search_with_stats(const T *query, const size_t K, const uint32_t L,
                                                             IdType *indices, float *distances, QueryStats &stats)
{
    Timer total_timer;

    if (K > (uint64_t)L)
    {
        throw ANNException("Set L to a value of at least K", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto scratch = manager.scratch_space();

    if (L > scratch->get_L())
    {
        diskann::cout << "Attempting to expand query scratch_space. Was created "
                      << "with Lsize: " << scratch->get_L() << " but search L is: " << L << std::endl;
        scratch->resize_for_new_L(L);
        diskann::cout << "Resize completed. New scratch->L is " << scratch->get_L() << std::endl;
    }

    const std::vector<LabelT> unused_filter_label;
    const std::vector<uint32_t> init_ids = get_init_ids();

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    _data_store->preprocess_query(query, scratch);

    auto retval = iterate_to_fixed_point_timed(scratch, L, init_ids, false, unused_filter_label, true, stats);

    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();

    size_t pos = 0;
    for (size_t i = 0; i < best_L_nodes.size(); ++i)
    {
        if (best_L_nodes[i].id < _max_points)
        {
            indices[pos] = (IdType)best_L_nodes[i].id;
            if (distances != nullptr)
            {
#ifdef EXEC_ENV_OLS
                distances[pos] = best_L_nodes[i].distance;
#else
                distances[pos] = _dist_metric == diskann::Metric::INNER_PRODUCT ? -1 * best_L_nodes[i].distance
                                                                                : best_L_nodes[i].distance;
#endif
            }
            pos++;
        }
        if (pos == K)
            break;
    }
    if (pos < K)
    {
        diskann::cerr << "Found pos: " << pos << "fewer than K elements " << K << " for query" << std::endl;
    }

    stats.total_us = (float)total_timer.elapsed();

    return retval;
}

// Search with partition-aware timing statistics for distributed latency profiling
template <typename T, typename TagT, typename LabelT>
template <typename IdType>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::search_with_partition_stats(
    const T *query, const size_t K, const uint32_t L,
    IdType *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition,
    QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<T> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn)
{
    Timer total_timer;

    if (K > (uint64_t)L)
    {
        throw ANNException("Set L to a value of at least K", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto scratch = manager.scratch_space();

    if (L > scratch->get_L())
    {
        diskann::cout << "Attempting to expand query scratch_space. Was created "
                      << "with Lsize: " << scratch->get_L() << " but search L is: " << L << std::endl;
        scratch->resize_for_new_L(L);
        diskann::cout << "Resize completed. New scratch->L is " << scratch->get_L() << std::endl;
    }

    const std::vector<LabelT> unused_filter_label;
    const std::vector<uint32_t> init_ids = custom_init_ids.empty() ? get_init_ids() : custom_init_ids;

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    _data_store->preprocess_query(query, scratch);

    auto retval = iterate_to_fixed_point_timed_partitioned(
        scratch, L, init_ids, false, unused_filter_label, true,
        partition_ids, query_partition, stats, remote_dist_fn, remote_neighbor_fn,
        graph_access_fn);

    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();

    size_t pos = 0;
    for (size_t i = 0; i < best_L_nodes.size(); ++i)
    {
        if (best_L_nodes[i].id < _max_points)
        {
            indices[pos] = (IdType)best_L_nodes[i].id;
            if (distances != nullptr)
            {
#ifdef EXEC_ENV_OLS
                distances[pos] = best_L_nodes[i].distance;
#else
                distances[pos] = _dist_metric == diskann::Metric::INNER_PRODUCT ? -1 * best_L_nodes[i].distance
                                                                                : best_L_nodes[i].distance;
#endif
            }
            pos++;
        }
        if (pos == K)
            break;
    }
    if (pos < K)
    {
        diskann::cerr << "Found pos: " << pos << "fewer than K elements " << K << " for query" << std::endl;
    }

    stats.total_us = (float)total_timer.elapsed();

    return retval;
}

template <typename T, typename TagT, typename LabelT>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::_search_with_trace(const DataType &query, const size_t K, const uint32_t L,
                                                             std::any &indices,
                                                             const std::vector<uint32_t>& partition_labels,
                                                             std::vector<TraceInfo>& trace_path,
                                                             float *distances)
{
    try
    {
        auto typed_query = std::any_cast<const T *>(query);
        if (typeid(uint32_t *) == indices.type())
        {
            auto u32_ptr = std::any_cast<uint32_t *>(indices);
            return this->search_with_trace(typed_query, K, L, u32_ptr, partition_labels, trace_path, distances);
        }
        else if (typeid(uint64_t *) == indices.type())
        {
            auto u64_ptr = std::any_cast<uint64_t *>(indices);
            return this->search_with_trace(typed_query, K, L, u64_ptr, partition_labels, trace_path, distances);
        }
        else
        {
            throw ANNException("Error: indices type can only be uint64_t or uint32_t.", -1);
        }
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad any cast while searching with trace. " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error: " + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
template <typename IdType>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::search_with_trace(const T *query, const size_t K, const uint32_t L,
                                                             IdType *indices,
                                                             const std::vector<uint32_t>& partition_labels,
                                                             std::vector<TraceInfo>& trace_path,
                                                             float *distances)
{
    if (K > (uint64_t)L)
    {
        throw ANNException("Set L to a value of at least K", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto scratch = manager.scratch_space();

    if (L > scratch->get_L())
    {
        diskann::cout << "Attempting to expand query scratch_space. Was created "
                      << "with Lsize: " << scratch->get_L() << " but search with trace L is: " << L << std::endl;
        scratch->resize_for_new_L(L);
        diskann::cout << "Resize completed. New scratch->L is " << scratch->get_L() << std::endl;
    }

    const std::vector<LabelT> unused_filter_label;
    const std::vector<uint32_t> init_ids = get_init_ids();

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    _data_store->preprocess_query(query, scratch);

    // This is the only modified line inside the function:
    auto retval = iterate_to_fixed_point_with_trace(scratch, L, init_ids, false, unused_filter_label, true,
                                                    partition_labels, trace_path); // Call the new iterator

    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();

    size_t pos = 0;
    for (size_t i = 0; i < best_L_nodes.size(); ++i)
    {
        if (best_L_nodes[i].id < _max_points)
        {
            // safe because Index uses uint32_t ids internally
            // and IDType will be uint32_t or uint64_t
            indices[pos] = (IdType)best_L_nodes[i].id;
            if (distances != nullptr)
            {
#ifdef EXEC_ENV_OLS
                // DLVS expects negative distances
                distances[pos] = best_L_nodes[i].distance;
#else
                distances[pos] = _dist_metric == diskann::Metric::INNER_PRODUCT ? -1 * best_L_nodes[i].distance
                                                                                : best_L_nodes[i].distance;
#endif
            }
            pos++;
        }
        if (pos == K)
            break;
    }
    if (pos < K)
    {
        diskann::cerr << "Found pos: " << pos << "fewer than K elements " << K << " for query" << std::endl;
    }

    return retval;
}


template <typename T, typename TagT, typename LabelT>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::_search_with_filters(const DataType &query,
                                                                           const std::string &raw_label, const size_t K,
                                                                           const uint32_t L, std::any &indices,
                                                                           float *distances)
{
    auto converted_label = this->get_converted_label(raw_label);
    if (typeid(uint64_t *) == indices.type())
    {
        auto ptr = std::any_cast<uint64_t *>(indices);
        return this->search_with_filters(std::any_cast<T *>(query), converted_label, K, L, ptr, distances);
    }
    else if (typeid(uint32_t *) == indices.type())
    {
        auto ptr = std::any_cast<uint32_t *>(indices);
        return this->search_with_filters(std::any_cast<T *>(query), converted_label, K, L, ptr, distances);
    }
    else
    {
        throw ANNException("Error: Id type can only be uint64_t or uint32_t.", -1);
    }
}

template <typename T, typename TagT, typename LabelT>
template <typename IdType>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::search_with_filters(const T *query, const LabelT &filter_label,
                                                                          const size_t K, const uint32_t L,
                                                                          IdType *indices, float *distances)
{
    if (K > (uint64_t)L)
    {
        throw ANNException("Set L to a value of at least K", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto scratch = manager.scratch_space();

    if (L > scratch->get_L())
    {
        diskann::cout << "Attempting to expand query scratch_space. Was created "
                      << "with Lsize: " << scratch->get_L() << " but search L is: " << L << std::endl;
        scratch->resize_for_new_L(L);
        diskann::cout << "Resize completed. New scratch->L is " << scratch->get_L() << std::endl;
    }

    std::vector<LabelT> filter_vec;
    std::vector<uint32_t> init_ids = get_init_ids();

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock, std::defer_lock);
    if (_dynamic_index)
        tl.lock();

    if (_label_to_start_id.find(filter_label) != _label_to_start_id.end())
    {
        init_ids.emplace_back(_label_to_start_id[filter_label]);
    }
    else
    {
        diskann::cout << "No filtered medoid found. exitting "
                      << std::endl; // RKNOTE: If universal label found start there
        throw diskann::ANNException("No filtered medoid found. exitting ", -1);
    }
    if (_dynamic_index)
        tl.unlock();

    filter_vec.emplace_back(filter_label);

    _data_store->preprocess_query(query, scratch);
    auto retval = iterate_to_fixed_point(scratch, L, init_ids, true, filter_vec, true);

    auto best_L_nodes = scratch->best_l_nodes();

    size_t pos = 0;
    for (size_t i = 0; i < best_L_nodes.size(); ++i)
    {
        if (best_L_nodes[i].id < _max_points)
        {
            indices[pos] = (IdType)best_L_nodes[i].id;

            if (distances != nullptr)
            {
#ifdef EXEC_ENV_OLS
                // DLVS expects negative distances
                distances[pos] = best_L_nodes[i].distance;
#else
                distances[pos] = _dist_metric == diskann::Metric::INNER_PRODUCT ? -1 * best_L_nodes[i].distance
                                                                                : best_L_nodes[i].distance;
#endif
            }
            pos++;
        }
        if (pos == K)
            break;
    }
    if (pos < K)
    {
        diskann::cerr << "Found fewer than K elements for query" << std::endl;
    }

    return retval;
}

template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::_search_with_tags(const DataType &query, const uint64_t K, const uint32_t L,
                                                 const TagType &tags, float *distances, DataVector &res_vectors,
                                                 bool use_filters, const std::string filter_label)
{
    try
    {
        return this->search_with_tags(std::any_cast<const T *>(query), K, L, std::any_cast<TagT *>(tags), distances,
                                      res_vectors.get<std::vector<T *>>(), use_filters, filter_label);
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad any cast while performing _search_with_tags() " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error: " + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::search_with_tags(const T *query, const uint64_t K, const uint32_t L, TagT *tags,
                                                float *distances, std::vector<T *> &res_vectors, bool use_filters,
                                                const std::string filter_label)
{
    if (K > (uint64_t)L)
    {
        throw ANNException("Set L to a value of at least K", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto scratch = manager.scratch_space();

    if (L > scratch->get_L())
    {
        diskann::cout << "Attempting to expand query scratch_space. Was created "
                      << "with Lsize: " << scratch->get_L() << " but search L is: " << L << std::endl;
        scratch->resize_for_new_L(L);
        diskann::cout << "Resize completed. New scratch->L is " << scratch->get_L() << std::endl;
    }

    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);

    const std::vector<uint32_t> init_ids = get_init_ids();

    //_distance->preprocess_query(query, _data_store->get_dims(),
    // scratch->aligned_query());
    _data_store->preprocess_query(query, scratch);
    if (!use_filters)
    {
        const std::vector<LabelT> unused_filter_label;
        iterate_to_fixed_point(scratch, L, init_ids, false, unused_filter_label, true);
    }
    else
    {
        std::vector<LabelT> filter_vec;
        auto converted_label = this->get_converted_label(filter_label);
        filter_vec.push_back(converted_label);
        iterate_to_fixed_point(scratch, L, init_ids, true, filter_vec, true);
    }

    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    assert(best_L_nodes.size() <= L);

    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);

    size_t pos = 0;
    for (size_t i = 0; i < best_L_nodes.size(); ++i)
    {
        auto node = best_L_nodes[i];

        TagT tag;
        if (_location_to_tag.try_get(node.id, tag))
        {
            tags[pos] = tag;

            if (res_vectors.size() > 0)
            {
                _data_store->get_vector(node.id, res_vectors[pos]);
            }

            if (distances != nullptr)
            {
#ifdef EXEC_ENV_OLS
                distances[pos] = node.distance; // DLVS expects negative distances
#else
                distances[pos] = _dist_metric == INNER_PRODUCT ? -1 * node.distance : node.distance;
#endif
            }
            pos++;
            // If res_vectors.size() < k, clip at the value.
            if (pos == K || pos == res_vectors.size())
                break;
        }
    }

    return pos;
}

template <typename T, typename TagT, typename LabelT>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::search_with_tags_and_stats(const T *query, const uint64_t K,
                                                const uint32_t L, TagT *tags, float *distances,
                                                std::vector<T *> &res_vectors, bool use_filters,
                                                const std::string filter_label)
{
    if (K > (uint64_t)L)
    {
        throw ANNException("Set L to a value of at least K", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto scratch = manager.scratch_space();

    if (L > scratch->get_L())
    {
        diskann::cout << "Attempting to expand query scratch_space. Was created "
                      << "with Lsize: " << scratch->get_L() << " but search L is: " << L << std::endl;
        scratch->resize_for_new_L(L);
        diskann::cout << "Resize completed. New scratch->L is " << scratch->get_L() << std::endl;
    }

    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);

    const std::vector<uint32_t> init_ids = get_init_ids();

    _data_store->preprocess_query(query, scratch);

    std::pair<uint32_t, uint32_t> retval;
    if (!use_filters)
    {
        const std::vector<LabelT> unused_filter_label;
        retval = iterate_to_fixed_point(scratch, L, init_ids, false, unused_filter_label, true);
    }
    else
    {
        std::vector<LabelT> filter_vec;
        auto converted_label = this->get_converted_label(filter_label);
        filter_vec.push_back(converted_label);
        retval = iterate_to_fixed_point(scratch, L, init_ids, true, filter_vec, true);
    }

    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    assert(best_L_nodes.size() <= L);

    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);

    size_t pos = 0;
    for (size_t i = 0; i < best_L_nodes.size(); ++i)
    {
        auto node = best_L_nodes[i];

        TagT tag;
        if (_location_to_tag.try_get(node.id, tag))
        {
            tags[pos] = tag;

            if (res_vectors.size() > 0)
            {
                _data_store->get_vector(node.id, res_vectors[pos]);
            }

            if (distances != nullptr)
            {
#ifdef EXEC_ENV_OLS
                distances[pos] = node.distance;
#else
                distances[pos] = _dist_metric == INNER_PRODUCT ? -1 * node.distance : node.distance;
#endif
            }
            pos++;
            if (pos == K || pos == res_vectors.size())
                break;
        }
    }

    return retval;  // Returns (hops, cmps)
}

template <typename T, typename TagT, typename LabelT> size_t Index<T, TagT, LabelT>::get_num_points()
{
    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
    return _nd;
}

template <typename T, typename TagT, typename LabelT> size_t Index<T, TagT, LabelT>::get_max_points()
{
    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
    return _max_points;
}

template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::get_num_frozen_pts() const
{
    return _num_frozen_pts;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_num_frozen_pts(size_t num_frozen)
{
    _num_frozen_pts = num_frozen;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::inject_virtual_entry_point(
    uint32_t location, const T *vector,
    const std::vector<uint32_t> &neighbors)
{
    if (location < _max_points)
    {
        throw ANNException("inject_virtual_entry_point: location must be >= _max_points", -1,
                          __FUNCSIG__, __FILE__, __LINE__);
    }
    _data_store->set_vector((location_t)location, vector);
    std::vector<location_t> nbrs(neighbors.begin(), neighbors.end());
    _graph_store->set_neighbours((location_t)location, nbrs);
}

template <typename T, typename TagT, typename LabelT>
TagT Index<T, TagT, LabelT>::get_tag_by_location(uint32_t location) const
{
    TagT tag;
    if (_location_to_tag.try_get(location, tag))
        return tag;
    return TagT{};
}

template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::get_aligned_dim() const
{
    return _data_store->get_aligned_dim();
}

template <typename T, typename TagT, typename LabelT>
Distance<T> *Index<T, TagT, LabelT>::get_distance_fn() const
{
    return _data_store->get_dist_fn();
}

template <typename T, typename TagT, typename LabelT>
AbstractGraphStore *Index<T, TagT, LabelT>::get_graph_store() const
{
    return _graph_store.get();
}

template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::generate_frozen_point()
{
    if (_num_frozen_pts == 0)
        return;

    if (_num_frozen_pts > 1)
    {
        throw ANNException("More than one frozen point not supported in generate_frozen_point", -1, __FUNCSIG__,
                           __FILE__, __LINE__);
    }

    if (_nd == 0)
    {
        throw ANNException("ERROR: Can not pick a frozen point since nd=0", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    size_t res = calculate_entry_point();

    // REFACTOR PQ: Not sure if we should do this for both stores.
    if (_pq_dist)
    {
        // copy the PQ data corresponding to the point returned by
        // calculate_entry_point
        // memcpy(_pq_data + _max_points * _num_pq_chunks,
        //       _pq_data + res * _num_pq_chunks,
        //       _num_pq_chunks * DIV_ROUND_UP(NUM_PQ_BITS, 8));
        _pq_data_store->copy_vectors((location_t)res, (location_t)_max_points, 1);
    }
    else
    {
        _data_store->copy_vectors((location_t)res, (location_t)_max_points, 1);
    }
    _frozen_pts_used++;
}

template <typename T, typename TagT, typename LabelT> int Index<T, TagT, LabelT>::enable_delete()
{
    assert(_enable_tags);

    if (!_enable_tags)
    {
        diskann::cerr << "Tags must be instantiated for deletions" << std::endl;
        return -2;
    }

    if (this->_deletes_enabled)
    {
        return 0;
    }

    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    if (_data_compacted)
    {
        for (uint32_t slot = (uint32_t)_nd; slot < _max_points; ++slot)
        {
            _empty_slots.insert(slot);
        }
    }
    this->_deletes_enabled = true;
    return 0;
}

template <typename T, typename TagT, typename LabelT>
inline void Index<T, TagT, LabelT>::process_delete(const tsl::robin_set<uint32_t> &old_delete_set, size_t loc,
                                                   const uint32_t range, const uint32_t maxc, const float alpha,
                                                   const float delete_degree_ratio, InMemQueryScratch<T> *scratch)
{
    tsl::robin_set<uint32_t> &expanded_nodes_set = scratch->expanded_nodes_set();
    std::vector<Neighbor> &expanded_nghrs_vec = scratch->expanded_nodes_vec();

    // If this condition were not true, deadlock could result
    assert(old_delete_set.find((uint32_t)loc) == old_delete_set.end());

    std::vector<uint32_t> adj_list;
    {
        // Acquire and release lock[loc] before acquiring locks for neighbors
        std::unique_lock<non_recursive_mutex> adj_list_lock;
        if (_conc_consolidate)
            adj_list_lock = std::unique_lock<non_recursive_mutex>(_locks[loc]);
        adj_list = _graph_store->get_neighbours((location_t)loc);
    }

    // --- LITS filtering setup (ported from the diskann-lits fork) ---
    // When the LITS build filter is on, candidates injected from a deleted
    // neighbor's out-edges are dropped if they are long+inter(+to-start).
    auto lits_tag_to_uint32 = [](const TagT &tag) -> uint32_t {
        if constexpr (std::is_same_v<TagT, tag_uint128>)
            return static_cast<uint32_t>(tag._data1);
        else
            return static_cast<uint32_t>(tag);
    };

    bool lits_filter_active = false;
    uint32_t lits_loc_cluster = 0;
    float lits_mean_dist = 0.0f;
    size_t lits_dim = 0;
    std::vector<T> lits_loc_vec, lits_start_vec, lits_nbr_vec;

    // Frozen points (loc >= _max_points, incl. _start) are exempt from LITS:
    // the start is the global search entry point and must keep cross-cluster
    // fan-out.
    if (loc < _max_points && _lits_build_filter_enabled && !_lits_build_cluster_labels.empty())
    {
        TagT lits_loc_tag;
        if (_location_to_tag.try_get((uint32_t)loc, lits_loc_tag))
        {
            uint32_t lits_loc_tag_u32 = lits_tag_to_uint32(lits_loc_tag);
            if (lits_loc_tag_u32 < _lits_build_cluster_labels.size())
            {
                lits_loc_cluster = _lits_build_cluster_labels[lits_loc_tag_u32];

                // Same semantics as the insert side: tau base = mean of ALL
                // original neighbors (including deleted), or their median when
                // _lits_tau_use_median is set.
                double sum_dist = 0.0;
                std::vector<float> nbr_dists;
                if (_lits_tau_use_median)
                    nbr_dists.reserve(adj_list.size());
                for (auto ngh : adj_list)
                {
                    float d = _data_store->get_distance((location_t)loc, (location_t)ngh);
                    sum_dist += d;
                    if (_lits_tau_use_median)
                        nbr_dists.push_back(d);
                }
                lits_mean_dist = adj_list.empty() ? 0.0f : static_cast<float>(sum_dist / adj_list.size());
                if (_lits_tau_use_median && !nbr_dists.empty())
                {
                    auto mid = nbr_dists.begin() + nbr_dists.size() / 2;
                    std::nth_element(nbr_dists.begin(), mid, nbr_dists.end());
                    lits_mean_dist = *mid;
                }

                lits_dim = _data_store->get_dims();
                lits_loc_vec.resize(lits_dim);
                lits_start_vec.resize(lits_dim);
                lits_nbr_vec.resize(lits_dim);
                _data_store->get_vector((location_t)loc, lits_loc_vec.data());
                _data_store->get_vector((location_t)_start, lits_start_vec.data());
                lits_filter_active = true;
            }
        }
    }

    // Returns true if candidate j should be excluded (LITS / NOT-to-start /
    // long+inter depending on mode).
    auto is_lits = [&](uint32_t j) -> bool {
        if (!lits_filter_active)
            return false;
        // 1. Long: dist(loc, j) > c * mean_dist (c = _lits_tau_scale)
        float dist_j = _data_store->get_distance((location_t)loc, (location_t)j);
        if (dist_j <= _lits_tau_scale * lits_mean_dist)
            return false;
        // 2. Inter: different cluster
        TagT nbr_tag;
        bool got = (j == _start) ? (nbr_tag = 0, true) : _location_to_tag.try_get(j, nbr_tag);
        if (!got)
            return false;
        uint32_t nbr_tag_u32 = lits_tag_to_uint32(nbr_tag);
        if (nbr_tag_u32 >= _lits_build_cluster_labels.size())
            return false;
        if (_lits_build_cluster_labels[nbr_tag_u32] == lits_loc_cluster)
            return false;
        // Long+Inter only mode: skip to-start check
        if (_lits_build_long_inter_only)
        {
            _lits_build_filtered_total.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        // 3. To-start: dot(loc->start, loc->j) >= 0
        _data_store->get_vector((location_t)j, lits_nbr_vec.data());
        double dot = 0.0;
        for (size_t d = 0; d < lits_dim; d++)
        {
            double to_start = (double)lits_start_vec[d] - (double)lits_loc_vec[d];
            double to_nbr = (double)lits_nbr_vec[d] - (double)lits_loc_vec[d];
            dot += to_start * to_nbr;
        }
        bool excluded = _lits_build_notostart_mode ? (dot < 0.0) : (dot >= 0.0);
        if (excluded)
            _lits_build_filtered_total.fetch_add(1, std::memory_order_relaxed);
        return excluded;
    };

    // Step 1: Separate non-deleted neighbors and deleted neighbors with distances
    std::vector<Neighbor> deleted_nbrs_with_dist;

    bool modify = false;
    for (auto ngh : adj_list)
    {
        if (old_delete_set.find(ngh) == old_delete_set.end())
        {
            // Non-deleted neighbor: keep it
            expanded_nodes_set.insert(ngh);
        }
        else
        {
            // Deleted neighbor: record with distance for locality-aware processing
            modify = true;
            float dist = _data_store->get_distance((location_t)loc, (location_t)ngh);
            deleted_nbrs_with_dist.push_back(Neighbor(ngh, dist));
        }
    }

    // Pruning ablation: skip expansion and pruning, just remove deleted edges
    if (_skip_consolidation_pruning.load(std::memory_order_relaxed) && modify)
    {
        std::unique_lock<non_recursive_mutex> adj_list_lock(_locks[loc]);
        _graph_store->clear_neighbours((location_t)loc);
        for (auto &ngh : expanded_nodes_set)
            _graph_store->add_neighbour((location_t)loc, ngh);
        return;
    }

    // Step 2: Process deleted neighbors with locality-aware expansion
    std::vector<uint32_t> short_range_neighbors;
    std::vector<uint32_t> long_range_neighbors;

    // For delete detail trace: store neighbors of each deleted neighbor
    std::vector<std::vector<uint32_t>> short_range_neighbor_lists;
    std::vector<std::vector<uint32_t>> long_range_neighbor_lists;
    bool record_detail_trace = _enable_delete_detail_trace.load(std::memory_order_relaxed);

    if (modify && !deleted_nbrs_with_dist.empty())
    {
        // Check if locality-aware pruning is enabled
        bool use_locality_aware = _enable_locality_aware_pruning.load(std::memory_order_relaxed);

        if (use_locality_aware)
        {
            // DEBUG: Print first 20 events to verify sorting
            static std::atomic<int> debug_counter{0};
            bool should_debug = (debug_counter.fetch_add(1, std::memory_order_relaxed) < 20);

            if (should_debug && deleted_nbrs_with_dist.size() > 1)
            {
                diskann::cout << "\n[DEBUG] Loc=" << loc << " Before sort: ";
                for (const auto& nbr : deleted_nbrs_with_dist)
                    diskann::cout << nbr.id << "(" << nbr.distance << ") ";
                diskann::cout << std::endl;
            }

            // Sort deleted neighbors by distance (ascending)
            std::sort(deleted_nbrs_with_dist.begin(), deleted_nbrs_with_dist.end(),
                      [](const Neighbor& a, const Neighbor& b) {
                          return a.distance < b.distance;
                      });

            if (should_debug && deleted_nbrs_with_dist.size() > 1)
            {
                diskann::cout << "[DEBUG] Loc=" << loc << " After sort:  ";
                for (const auto& nbr : deleted_nbrs_with_dist)
                    diskann::cout << nbr.id << "(" << nbr.distance << ") ";
                diskann::cout << std::endl;
            }

            // Split based on short_range_ratio setting
            float ratio = _short_range_ratio.load(std::memory_order_relaxed);
            size_t split_pos = static_cast<size_t>(deleted_nbrs_with_dist.size() * ratio);

            if (should_debug)
            {
                diskann::cout << "[DEBUG] Loc=" << loc << " ratio=" << ratio << " split_pos=" << split_pos << std::endl;
            }

            // Ensure at least one neighbor is kept if there's only one deleted neighbor
            if (split_pos == 0 && deleted_nbrs_with_dist.size() > 0)
                split_pos = 1;

            // Expand only SHORT-RANGE deleted neighbors
            for (size_t i = 0; i < split_pos; i++)
            {
                uint32_t ngh = deleted_nbrs_with_dist[i].id;
                short_range_neighbors.push_back(ngh);

                std::unique_lock<non_recursive_mutex> ngh_lock;
                if (_conc_consolidate)
                    ngh_lock = std::unique_lock<non_recursive_mutex>(_locks[ngh]);

                std::vector<uint32_t> ngh_neighbors;
                for (auto j : _graph_store->get_neighbours((location_t)ngh))
                {
                    if (j != loc && old_delete_set.find(j) == old_delete_set.end() && !is_lits(j))
                        expanded_nodes_set.insert(j);
                    if (record_detail_trace)
                        ngh_neighbors.push_back(j);
                }
                if (record_detail_trace)
                    short_range_neighbor_lists.push_back(std::move(ngh_neighbors));
            }

            // Long-range neighbors are NOT expanded - they are dropped
            for (size_t i = split_pos; i < deleted_nbrs_with_dist.size(); i++)
            {
                uint32_t ngh = deleted_nbrs_with_dist[i].id;
                long_range_neighbors.push_back(ngh);

                if (record_detail_trace)
                {
                    std::unique_lock<non_recursive_mutex> ngh_lock;
                    if (_conc_consolidate)
                        ngh_lock = std::unique_lock<non_recursive_mutex>(_locks[ngh]);

                    std::vector<uint32_t> ngh_neighbors;
                    for (auto j : _graph_store->get_neighbours((location_t)ngh))
                        ngh_neighbors.push_back(j);
                    long_range_neighbor_lists.push_back(std::move(ngh_neighbors));
                }
            }
        }
        else
        {
            // Original behavior: expand ALL deleted neighbors (treat all as short-range)
            for (const auto& deleted_nbr : deleted_nbrs_with_dist)
            {
                uint32_t ngh = deleted_nbr.id;
                short_range_neighbors.push_back(ngh);

                std::unique_lock<non_recursive_mutex> ngh_lock;
                if (_conc_consolidate)
                    ngh_lock = std::unique_lock<non_recursive_mutex>(_locks[ngh]);

                std::vector<uint32_t> ngh_neighbors;
                for (auto j : _graph_store->get_neighbours((location_t)ngh))
                {
                    if (j != loc && old_delete_set.find(j) == old_delete_set.end() && !is_lits(j))
                        expanded_nodes_set.insert(j);
                    if (record_detail_trace)
                        ngh_neighbors.push_back(j);
                }
                if (record_detail_trace)
                    short_range_neighbor_lists.push_back(std::move(ngh_neighbors));
            }
        }
    }

    // Prepare delete detail trace event if enabled
    // Note: We use locations directly as they correspond to tags in most cases
    // The mapping location -> tag can be reconstructed using index.tags file
    LocalityAwareTraceEvent detail_trace_event;
    bool should_record_detail_trace = record_detail_trace && modify && !deleted_nbrs_with_dist.empty();

    // Helper lambda to convert tag to uint32_t (handles both uint32_t and tag_uint128)
    auto tag_to_uint32 = [](const TagT &tag) -> uint32_t {
        if constexpr (std::is_same_v<TagT, uint32_t> || std::is_same_v<TagT, int32_t> ||
                      std::is_same_v<TagT, uint64_t> || std::is_same_v<TagT, int64_t>)
        {
            return static_cast<uint32_t>(tag);
        }
        else
        {
            // For tag_uint128, use _data1
            return static_cast<uint32_t>(tag._data1);
        }
    };

    if (should_record_detail_trace)
    {
        // Convert location to dataset index (tag - 1) for consistent trace across iterations
        // Tags are 1-based, so dataset_index = tag - 1
        TagT source_tag;
        if (_location_to_tag.try_get(loc, source_tag))
        {
            detail_trace_event.source_node_tag = tag_to_uint32(source_tag) - 1;  // 0-based dataset index
        }
        else
        {
            // DEBUG: Log why try_get failed
            static std::atomic<int> fallback_counter{0};
            if (fallback_counter.fetch_add(1, std::memory_order_relaxed) < 10)
            {
                diskann::cout << "[DEBUG] Trace fallback: loc=" << loc
                              << " not in _location_to_tag (size=" << _location_to_tag.size() << ")" << std::endl;
            }
            detail_trace_event.source_node_tag = loc;  // Fallback to location if tag not found
        }

        // Record original neighbors as dataset indices
        for (auto ngh_loc : adj_list)
        {
            TagT ngh_tag;
            if (_location_to_tag.try_get(ngh_loc, ngh_tag))
            {
                detail_trace_event.original_neighbors.push_back(tag_to_uint32(ngh_tag) - 1);
            }
            else
            {
                detail_trace_event.original_neighbors.push_back(ngh_loc);  // Fallback
            }
        }

        // Record short-range deleted neighbors as dataset indices
        for (size_t i = 0; i < short_range_neighbors.size(); i++)
        {
            DeletedNeighborInfo info;
            uint32_t ngh_loc = short_range_neighbors[i];
            TagT ngh_tag;
            if (_location_to_tag.try_get(ngh_loc, ngh_tag))
            {
                info.deleted_node_tag = tag_to_uint32(ngh_tag) - 1;  // 0-based dataset index
            }
            else if (_deleted_location_to_tag.find(ngh_loc) != _deleted_location_to_tag.end())
            {
                // Use preserved deleted tag mapping
                info.deleted_node_tag = tag_to_uint32(_deleted_location_to_tag[ngh_loc]) - 1;
            }
            else
            {
                info.deleted_node_tag = ngh_loc;  // Fallback
            }

            // Record neighbor list as dataset indices
            if (i < short_range_neighbor_lists.size())
            {
                for (auto nbr_loc : short_range_neighbor_lists[i])
                {
                    TagT nbr_tag;
                    if (_location_to_tag.try_get(nbr_loc, nbr_tag))
                    {
                        info.neighbors.push_back(tag_to_uint32(nbr_tag) - 1);
                    }
                    else if (_deleted_location_to_tag.find(nbr_loc) != _deleted_location_to_tag.end())
                    {
                        info.neighbors.push_back(tag_to_uint32(_deleted_location_to_tag[nbr_loc]) - 1);
                    }
                    else
                    {
                        info.neighbors.push_back(nbr_loc);  // Fallback
                    }
                }
            }
            detail_trace_event.short_range_deleted.push_back(std::move(info));
        }

        // Record long-range dropped neighbors as dataset indices
        for (size_t i = 0; i < long_range_neighbors.size(); i++)
        {
            DeletedNeighborInfo info;
            uint32_t ngh_loc = long_range_neighbors[i];
            TagT ngh_tag;
            if (_location_to_tag.try_get(ngh_loc, ngh_tag))
            {
                info.deleted_node_tag = tag_to_uint32(ngh_tag) - 1;  // 0-based dataset index
            }
            else if (_deleted_location_to_tag.find(ngh_loc) != _deleted_location_to_tag.end())
            {
                // Use preserved deleted tag mapping
                info.deleted_node_tag = tag_to_uint32(_deleted_location_to_tag[ngh_loc]) - 1;
            }
            else
            {
                info.deleted_node_tag = ngh_loc;  // Fallback
            }

            // Record neighbor list as dataset indices
            if (i < long_range_neighbor_lists.size())
            {
                for (auto nbr_loc : long_range_neighbor_lists[i])
                {
                    TagT nbr_tag;
                    if (_location_to_tag.try_get(nbr_loc, nbr_tag))
                    {
                        info.neighbors.push_back(tag_to_uint32(nbr_tag) - 1);
                    }
                    else if (_deleted_location_to_tag.find(nbr_loc) != _deleted_location_to_tag.end())
                    {
                        info.neighbors.push_back(tag_to_uint32(_deleted_location_to_tag[nbr_loc]) - 1);
                    }
                    else
                    {
                        info.neighbors.push_back(nbr_loc);  // Fallback
                    }
                }
            }
            detail_trace_event.long_range_dropped.push_back(std::move(info));
        }
        // Note: neighbors_after_pruning will be filled after pruning is complete
    }

    if (modify)
    {
        // Apply delete_degree_ratio to reduce target degree during delete processing
        const uint32_t target_degree = static_cast<uint32_t>(range * delete_degree_ratio);

        // Always compare with max_degree (range) to determine if pruning is needed
        if (expanded_nodes_set.size() <= range)
        {
            std::unique_lock<non_recursive_mutex> adj_list_lock(_locks[loc]);
            _graph_store->clear_neighbours((location_t)loc);
            for (auto &ngh : expanded_nodes_set)
                _graph_store->add_neighbour((location_t)loc, ngh);

            // Record neighbors after pruning for detail trace (no actual pruning in this case)
            if (should_record_detail_trace)
            {
                for (auto &ngh : expanded_nodes_set)
                {
                    TagT ngh_tag;
                    if (_location_to_tag.try_get(ngh, ngh_tag))
                    {
                        detail_trace_event.neighbors_after_pruning.push_back(tag_to_uint32(ngh_tag) - 1);
                    }
                    else
                    {
                        detail_trace_event.neighbors_after_pruning.push_back(ngh);  // Fallback
                    }
                }
            }
        }
        else
        {
            // Pruning is needed because expanded set exceeds max_degree
            // But we prune to target_degree (which may be lower than max_degree)
            _num_delete_consolidate_prunes.fetch_add(1, std::memory_order_relaxed);

            // Create a pool of Neighbor candidates from the expanded_nodes_set
            expanded_nghrs_vec.reserve(expanded_nodes_set.size());
            for (auto &ngh : expanded_nodes_set)
            {
                expanded_nghrs_vec.emplace_back(ngh, _data_store->get_distance((location_t)loc, (location_t)ngh));
            }
            std::sort(expanded_nghrs_vec.begin(), expanded_nghrs_vec.end());

            // Extract expanded candidate set node IDs (before pruning)
            std::vector<uint32_t> expanded_node_ids;
            expanded_node_ids.reserve(expanded_nghrs_vec.size());
            for (const auto &neighbor : expanded_nghrs_vec)
            {
                expanded_node_ids.push_back(neighbor.id);
            }

            std::vector<uint32_t> &occlude_list_output = scratch->occlude_list_output();
            // Use prune_neighbors with target_degree for aggressive pruning during delete
            prune_neighbors((uint32_t)loc, expanded_nghrs_vec, target_degree, maxc, alpha, occlude_list_output, scratch);

            // Record pruning trace if enabled
            if (_enable_pruning_trace.load(std::memory_order_relaxed))
            {
                int thread_id = omp_get_thread_num();
                PruningTraceEvent event;
                event.node_id = loc;
                event.edges_before = adj_list;             // Original neighbors before expansion
                event.edges_expanded = expanded_node_ids;  // Candidate set after expansion, before pruning
                event.edges_after = occlude_list_output;   // Final neighbors after pruning

                // Store short-range and long-range info for locality-aware analysis
                event.short_range_neighbors = short_range_neighbors;
                event.long_range_neighbors = long_range_neighbors;

                _delete_consolidate_trace_buffers[thread_id].push_back(std::move(event));
            }

            // Record neighbors after pruning for detail trace
            if (should_record_detail_trace)
            {
                for (auto ngh : occlude_list_output)
                {
                    TagT ngh_tag;
                    if (_location_to_tag.try_get(ngh, ngh_tag))
                    {
                        detail_trace_event.neighbors_after_pruning.push_back(tag_to_uint32(ngh_tag) - 1);
                    }
                    else
                    {
                        detail_trace_event.neighbors_after_pruning.push_back(ngh);  // Fallback
                    }
                }
            }

            std::unique_lock<non_recursive_mutex> adj_list_lock(_locks[loc]);
            _graph_store->set_neighbours((location_t)loc, occlude_list_output);
        }

        // Push detail trace event after pruning is complete
        if (should_record_detail_trace)
        {
            int thread_id = omp_get_thread_num();
            _delete_detail_trace_buffers[thread_id].push_back(std::move(detail_trace_event));
        }
    }
}

// Statistics access methods
template <typename T, typename TagT, typename LabelT>
uint64_t Index<T, TagT, LabelT>::get_num_inter_insert_prunes() const
{
    return _num_inter_insert_prunes.load(std::memory_order_relaxed);
}

template <typename T, typename TagT, typename LabelT>
uint64_t Index<T, TagT, LabelT>::get_num_delete_consolidate_prunes() const
{
    return _num_delete_consolidate_prunes.load(std::memory_order_relaxed);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::reset_prune_counter()
{
    _num_inter_insert_prunes.store(0, std::memory_order_relaxed);
    _num_delete_consolidate_prunes.store(0, std::memory_order_relaxed);
}

// Pruning trace management functions
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::enable_pruning_trace(bool enable)
{
    _enable_pruning_trace.store(enable, std::memory_order_relaxed);
}

// Locality-aware pruning functions
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::enable_locality_aware_pruning(bool enable)
{
    _enable_locality_aware_pruning.store(enable, std::memory_order_relaxed);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_short_range_ratio(float ratio)
{
    _short_range_ratio.store(ratio, std::memory_order_relaxed);
}

// Pruning ablation: runtime strategy switch + consolidation skip
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_pruning_strategy(PruningStrategy strategy)
{
    _pruningStrategy = strategy;
}

template <typename T, typename TagT, typename LabelT>
PruningStrategy Index<T, TagT, LabelT>::get_pruning_strategy() const
{
    return _pruningStrategy;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_skip_consolidation_pruning(bool skip)
{
    _skip_consolidation_pruning.store(skip, std::memory_order_relaxed);
}

template <typename T, typename TagT, typename LabelT>
bool Index<T, TagT, LabelT>::get_skip_consolidation_pruning() const
{
    return _skip_consolidation_pruning.load(std::memory_order_relaxed);
}

// Delete detail trace functions (JSON output with tags)
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::enable_delete_detail_trace(bool enable)
{
    _enable_delete_detail_trace.store(enable, std::memory_order_relaxed);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::clear_delete_detail_trace()
{
    for (auto &buffer : _delete_detail_trace_buffers)
        buffer.clear();
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::save_delete_detail_trace(const std::string &output_dir, int iteration)
{
    if (!_enable_delete_detail_trace.load(std::memory_order_relaxed))
        return;

    // Save delete detail trace using trace_writer
    std::string json_file = output_dir + "/detailed_consolidate_delete_trace.json";
    trace::save_delete_detail_trace_json(json_file, _delete_detail_trace_buffers);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::clear_pruning_trace()
{
    for (auto &buffer : _inter_insert_trace_buffers)
        buffer.clear();
    for (auto &buffer : _delete_consolidate_trace_buffers)
        buffer.clear();
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::save_pruning_trace(const std::string &output_dir, int iteration)
{
    if (!_enable_pruning_trace.load(std::memory_order_relaxed))
        return;

    // Save inter-insert pruning trace using trace_writer
    std::string inter_file = output_dir + "/pruning_trace_inter_insert_iter_" + std::to_string(iteration) + ".csv";
    trace::save_inter_insert_trace_csv(inter_file, _inter_insert_trace_buffers);

    // Save delete consolidate pruning trace using trace_writer
    std::string delete_file =
        output_dir + "/pruning_trace_delete_consolidate_iter_" + std::to_string(iteration) + ".csv";
    trace::save_delete_consolidate_trace_csv(delete_file, _delete_consolidate_trace_buffers);
}

// Returns number of live points left after consolidation
template <typename T, typename TagT, typename LabelT>
consolidation_report Index<T, TagT, LabelT>::consolidate_deletes(const IndexWriteParameters &params)
{
    if (!_enable_tags)
        throw diskann::ANNException("Point tag array not instantiated", -1, __FUNCSIG__, __FILE__, __LINE__);

    {
        std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
        std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);
        if (_empty_slots.size() + _nd != _max_points)
        {
            std::string err = "#empty slots + nd != max points";
            diskann::cerr << err << std::endl;
            throw ANNException(err, -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        if (_location_to_tag.size() + _delete_set->size() != _nd)
        {
            diskann::cerr << "Error: _location_to_tag.size (" << _location_to_tag.size() << ")  + _delete_set->size ("
                          << _delete_set->size() << ") != _nd(" << _nd << ") ";
            return consolidation_report(diskann::consolidation_report::status_code::INCONSISTENT_COUNT_ERROR, 0, 0, 0,
                                        0, 0, 0, 0);
        }

        if (_location_to_tag.size() != _tag_to_location.size())
        {
            throw diskann::ANNException("_location_to_tag and _tag_to_location not of same size", -1, __FUNCSIG__,
                                        __FILE__, __LINE__);
        }
    }

    std::unique_lock<std::shared_timed_mutex> update_lock(_update_lock, std::defer_lock);
    if (!_conc_consolidate)
        update_lock.lock();

    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock, std::defer_lock);
    if (!cl.try_lock())
    {
        diskann::cerr << "Consildate delete function failed to acquire consolidate lock" << std::endl;
        return consolidation_report(diskann::consolidation_report::status_code::LOCK_FAIL, 0, 0, 0, 0, 0, 0, 0);
    }

    diskann::cout << "Starting consolidate_deletes... ";

    std::unique_ptr<tsl::robin_set<uint32_t>> old_delete_set(new tsl::robin_set<uint32_t>);
    {
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
        std::swap(_delete_set, old_delete_set);
    }

    if (old_delete_set->find(_start) != old_delete_set->end())
    {
        throw diskann::ANNException("ERROR: start node has been deleted", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    const uint32_t range = params.max_degree;
    const uint32_t maxc = params.max_occlusion_size;
    const float alpha = params.alpha;
    const float delete_degree_ratio = params.delete_degree_ratio;
    const uint32_t num_threads = params.num_threads == 0 ? omp_get_num_procs() : params.num_threads;

    uint32_t num_calls_to_process_delete = 0;
    diskann::Timer timer;
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8192) reduction(+ : num_calls_to_process_delete)
    for (int64_t loc = 0; loc < (int64_t)_max_points; loc++)
    {
        if (old_delete_set->find((uint32_t)loc) == old_delete_set->end() && !_empty_slots.is_in_set((uint32_t)loc))
        {
            ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
            auto scratch = manager.scratch_space();
            process_delete(*old_delete_set, loc, range, maxc, alpha, delete_degree_ratio, scratch);
            num_calls_to_process_delete += 1;
        }
    }
    for (int64_t loc = _max_points; loc < (int64_t)(_max_points + _num_frozen_pts); loc++)
    {
        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
        auto scratch = manager.scratch_space();
        process_delete(*old_delete_set, loc, range, maxc, alpha, delete_degree_ratio, scratch);
        num_calls_to_process_delete += 1;
    }

    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    size_t ret_nd = release_locations(*old_delete_set);
    size_t max_points = _max_points;
    size_t empty_slots_size = _empty_slots.size();

    std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);
    size_t delete_set_size = _delete_set->size();
    size_t old_delete_set_size = old_delete_set->size();

    if (!_conc_consolidate)
    {
        update_lock.unlock();
    }

    double duration = timer.elapsed() / 1000000.0;
    diskann::cout << " done in " << duration << " seconds." << std::endl;

    // Clear deleted location to tag mapping after consolidation is complete
    _deleted_location_to_tag.clear();

    return consolidation_report(diskann::consolidation_report::status_code::SUCCESS, ret_nd, max_points,
                                empty_slots_size, old_delete_set_size, delete_set_size, num_calls_to_process_delete,
                                duration);
}

// Consolidate with detailed timing statistics for latency profiling
template <typename T, typename TagT, typename LabelT>
consolidation_report Index<T, TagT, LabelT>::consolidate_deletes_with_stats(const IndexWriteParameters &params,
                                                                             ConsolidateStats &stats)
{
    Timer total_timer, phase_timer;

    if (!_enable_tags)
        throw diskann::ANNException("Point tag array not instantiated", -1, __FUNCSIG__, __FILE__, __LINE__);

    {
        std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
        std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);
        if (_empty_slots.size() + _nd != _max_points)
        {
            std::string err = "#empty slots + nd != max points";
            diskann::cerr << err << std::endl;
            throw ANNException(err, -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        if (_location_to_tag.size() + _delete_set->size() != _nd)
        {
            diskann::cerr << "Error: _location_to_tag.size (" << _location_to_tag.size() << ")  + _delete_set->size ("
                          << _delete_set->size() << ") != _nd(" << _nd << ") ";
            stats.total_us = (float)total_timer.elapsed();
            return consolidation_report(diskann::consolidation_report::status_code::INCONSISTENT_COUNT_ERROR, 0, 0, 0,
                                        0, 0, 0, 0);
        }

        if (_location_to_tag.size() != _tag_to_location.size())
        {
            throw diskann::ANNException("_location_to_tag and _tag_to_location not of same size", -1, __FUNCSIG__,
                                        __FILE__, __LINE__);
        }
    }

    std::unique_lock<std::shared_timed_mutex> update_lock(_update_lock, std::defer_lock);
    if (!_conc_consolidate)
        update_lock.lock();

    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock, std::defer_lock);
    if (!cl.try_lock())
    {
        diskann::cerr << "Consildate delete function failed to acquire consolidate lock" << std::endl;
        stats.total_us = (float)total_timer.elapsed();
        return consolidation_report(diskann::consolidation_report::status_code::LOCK_FAIL, 0, 0, 0, 0, 0, 0, 0);
    }

    diskann::cout << "Starting consolidate_deletes... ";

    std::unique_ptr<tsl::robin_set<uint32_t>> old_delete_set(new tsl::robin_set<uint32_t>);
    {
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
        std::swap(_delete_set, old_delete_set);
    }

    if (old_delete_set->find(_start) != old_delete_set->end())
    {
        throw diskann::ANNException("ERROR: start node has been deleted", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    const uint32_t range = params.max_degree;
    const uint32_t maxc = params.max_occlusion_size;
    const float alpha = params.alpha;
    const float delete_degree_ratio = params.delete_degree_ratio;
    const uint32_t num_threads = params.num_threads == 0 ? omp_get_num_procs() : params.num_threads;

    // Relink phase
    phase_timer.reset();
    uint32_t num_calls_to_process_delete = 0;
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8192) reduction(+ : num_calls_to_process_delete)
    for (int64_t loc = 0; loc < (int64_t)_max_points; loc++)
    {
        if (old_delete_set->find((uint32_t)loc) == old_delete_set->end() && !_empty_slots.is_in_set((uint32_t)loc))
        {
            ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
            auto scratch = manager.scratch_space();
            process_delete(*old_delete_set, loc, range, maxc, alpha, delete_degree_ratio, scratch);
            num_calls_to_process_delete += 1;
        }
    }
    for (int64_t loc = _max_points; loc < (int64_t)(_max_points + _num_frozen_pts); loc++)
    {
        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
        auto scratch = manager.scratch_space();
        process_delete(*old_delete_set, loc, range, maxc, alpha, delete_degree_ratio, scratch);
        num_calls_to_process_delete += 1;
    }
    stats.relink_us = (float)phase_timer.elapsed();

    // Compaction phase
    phase_timer.reset();
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    size_t ret_nd = release_locations(*old_delete_set);
    stats.compaction_us = (float)phase_timer.elapsed();

    size_t max_points = _max_points;
    size_t empty_slots_size = _empty_slots.size();

    std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);
    size_t delete_set_size = _delete_set->size();
    size_t old_delete_set_size = old_delete_set->size();

    if (!_conc_consolidate)
    {
        update_lock.unlock();
    }

    double duration = (double)total_timer.elapsed() / 1000000.0;
    diskann::cout << " done in " << duration << " seconds." << std::endl;

    // Clear deleted location to tag mapping after consolidation is complete
    _deleted_location_to_tag.clear();

    stats.total_us = (float)total_timer.elapsed();
    stats.points_removed = (unsigned)old_delete_set_size;
    stats.points_remaining = (unsigned)ret_nd;

    return consolidation_report(diskann::consolidation_report::status_code::SUCCESS, ret_nd, max_points,
                                empty_slots_size, old_delete_set_size, delete_set_size, num_calls_to_process_delete,
                                duration);
}

template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::compact_frozen_point()
{
    if (_nd < _max_points && _num_frozen_pts > 0)
    {
        reposition_points((uint32_t)_max_points, (uint32_t)_nd, (uint32_t)_num_frozen_pts);
        _start = (uint32_t)_nd;

        if (_filtered_index && _dynamic_index)
        {
            //  update medoid id's as frozen points are treated as medoid
            for (auto &[label, medoid_id] : _label_to_start_id)
            {
                /*  if (label == _universal_label)
                      continue;*/
                _label_to_start_id[label] = (uint32_t)_nd + (medoid_id - (uint32_t)_max_points);
            }
        }
    }
}

// Should be called after acquiring _update_lock
template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::compact_data()
{
    if (!_dynamic_index)
        throw ANNException("Can not compact a non-dynamic index", -1, __FUNCSIG__, __FILE__, __LINE__);

    if (_data_compacted)
    {
        diskann::cerr << "Warning! Calling compact_data() when _data_compacted is true!" << std::endl;
        return;
    }

    if (_delete_set->size() > 0)
    {
        throw ANNException("Can not compact data when index has non-empty _delete_set of "
                           "size: " +
                               std::to_string(_delete_set->size()),
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::Timer timer;

    std::vector<uint32_t> new_location = std::vector<uint32_t>(_max_points + _num_frozen_pts, UINT32_MAX);

    uint32_t new_counter = 0;
    std::set<uint32_t> empty_locations;
    for (uint32_t old_location = 0; old_location < _max_points; old_location++)
    {
        if (_location_to_tag.contains(old_location))
        {
            new_location[old_location] = new_counter;
            new_counter++;
        }
        else
        {
            empty_locations.insert(old_location);
        }
    }
    for (uint32_t old_location = (uint32_t)_max_points; old_location < _max_points + _num_frozen_pts; old_location++)
    {
        new_location[old_location] = old_location;
    }

    // If start node is removed, throw an exception
    if (_start < _max_points && !_location_to_tag.contains(_start))
    {
        throw diskann::ANNException("ERROR: Start node deleted.", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    size_t num_dangling = 0;
    for (uint32_t old = 0; old < _max_points + _num_frozen_pts; ++old)
    {
        // compact _final_graph
        std::vector<uint32_t> new_adj_list;

        if ((new_location[old] < _max_points) // If point continues to exist
            || (old >= _max_points && old < _max_points + _num_frozen_pts))
        {
            new_adj_list.reserve(_graph_store->get_neighbours((location_t)old).size());
            for (auto ngh_iter : _graph_store->get_neighbours((location_t)old))
            {
                if (empty_locations.find(ngh_iter) != empty_locations.end())
                {
                    ++num_dangling;
                    diskann::cerr << "Error in compact_data(). _final_graph[" << old << "] has neighbor " << ngh_iter
                                  << " which is a location not associated with any tag." << std::endl;
                }
                else
                {
                    new_adj_list.push_back(new_location[ngh_iter]);
                }
            }
            //_graph_store->get_neighbours((location_t)old).swap(new_adj_list);
            _graph_store->set_neighbours((location_t)old, new_adj_list);

            // Move the data and adj list to the correct position
            if (new_location[old] != old)
            {
                assert(new_location[old] < old);
                _graph_store->swap_neighbours(new_location[old], (location_t)old);

                if (_filtered_index)
                {
                    _location_to_labels[new_location[old]].swap(_location_to_labels[old]);
                }

                _data_store->copy_vectors(old, new_location[old], 1);
            }
        }
        else
        {
            _graph_store->clear_neighbours((location_t)old);
        }
    }
    diskann::cerr << "#dangling references after data compaction: " << num_dangling << std::endl;

    // Update partition mapping: remap old locations to new locations
    if (_partition_loaded && !_location_to_partition.empty())
    {
        // Allocate space for data points + frozen points
        std::vector<uint32_t> new_partitions(_max_points + _num_frozen_pts, std::numeric_limits<uint32_t>::max());

        // Remap data points (0 to _max_points-1)
        for (uint32_t old = 0; old < _max_points; ++old)
        {
            if (new_location[old] < _nd && old < _location_to_partition.size())
            {
                new_partitions[new_location[old]] = _location_to_partition[old];
            }
        }

        // Copy frozen points partition (they don't move)
        for (uint32_t i = 0; i < _num_frozen_pts; ++i)
        {
            uint32_t frozen_loc = static_cast<uint32_t>(_max_points + i);
            if (frozen_loc < _location_to_partition.size())
            {
                new_partitions[frozen_loc] = _location_to_partition[frozen_loc];
            }
        }

        _location_to_partition = std::move(new_partitions);
        diskann::cout << "Updated partition mapping for " << _nd << " data points + " << _num_frozen_pts << " frozen points" << std::endl;
    }

    _tag_to_location.clear();
    for (auto pos = _location_to_tag.find_first(); pos.is_valid(); pos = _location_to_tag.find_next(pos))
    {
        const auto tag = _location_to_tag.get(pos);
        _tag_to_location[tag] = new_location[pos._key];
    }
    _location_to_tag.clear();
    for (const auto &iter : _tag_to_location)
    {
        _location_to_tag.set(iter.second, iter.first);
    }
    // remove all cleared up old
    // Skip frozen point if it was already moved to _nd position
    size_t clear_start = _nd;
    if (_start == _nd && _num_frozen_pts > 0)
    {
        // Frozen point already at _nd, skip it during cleanup
        clear_start = _nd + _num_frozen_pts;
    }

    for (size_t old = clear_start; old < _max_points; ++old)
    {
        _graph_store->clear_neighbours((location_t)old);
    }
    if (_filtered_index)
    {
        for (size_t old = _nd; old < _max_points; old++)
        {
            _location_to_labels[old].clear();
        }
    }

    _empty_slots.clear();
    // mark all slots after _nd as empty
    for (auto i = _nd; i < _max_points; i++)
    {
        _empty_slots.insert((uint32_t)i);
    }
    _data_compacted = true;
    diskann::cout << "Time taken for compact_data: " << timer.elapsed() / 1000000. << "s." << std::endl;
}

//
// Caller must hold unique _tag_lock and _delete_lock before calling this
//
template <typename T, typename TagT, typename LabelT> int Index<T, TagT, LabelT>::reserve_location()
{
    if (_nd >= _max_points)
    {
        return -1;
    }
    uint32_t location;
    if (_data_compacted && _empty_slots.is_empty())
    {
        // This code path is encountered when enable_delete hasn't been
        // called yet, so no points have been deleted and _empty_slots
        // hasn't been filled in. In that case, just keep assigning
        // consecutive locations.
        location = (uint32_t)_nd;
    }
    else
    {
        assert(_empty_slots.size() != 0);
        assert(_empty_slots.size() + _nd == _max_points);

        location = _empty_slots.pop_any();
        _delete_set->erase(location);
    }
    ++_nd;
    return location;
}

template <typename T, typename TagT, typename LabelT> size_t Index<T, TagT, LabelT>::release_location(int location)
{
    if (_empty_slots.is_in_set(location))
        throw ANNException("Trying to release location, but location already in empty slots", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    _empty_slots.insert(location);

    _nd--;
    return _nd;
}

template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::release_locations(const tsl::robin_set<uint32_t> &locations)
{
    for (auto location : locations)
    {
        if (_empty_slots.is_in_set(location))
            throw ANNException("Trying to release location, but location "
                               "already in empty slots",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        _empty_slots.insert(location);

        _nd--;
    }

    if (_empty_slots.size() + _nd != _max_points)
        throw ANNException("#empty slots + nd != max points", -1, __FUNCSIG__, __FILE__, __LINE__);

    return _nd;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::reposition_points(uint32_t old_location_start, uint32_t new_location_start,
                                               uint32_t num_locations)
{
    if (num_locations == 0 || old_location_start == new_location_start)
    {
        return;
    }

    // Update pointers to the moved nodes. Note: the computation is correct even
    // when new_location_start < old_location_start given the C++ uint32_t
    // integer arithmetic rules.
    const uint32_t location_delta = new_location_start - old_location_start;

    std::vector<location_t> updated_neighbours_location;
    for (uint32_t i = 0; i < _max_points + _num_frozen_pts; i++)
    {
        auto &i_neighbours = _graph_store->get_neighbours((location_t)i);
        std::vector<location_t> i_neighbours_copy(i_neighbours.begin(), i_neighbours.end());
        for (auto &loc : i_neighbours_copy)
        {
            if (loc >= old_location_start && loc < old_location_start + num_locations)
                loc += location_delta;
        }
        _graph_store->set_neighbours(i, i_neighbours_copy);
    }

    // The [start, end) interval which will contain obsolete points to be
    // cleared.
    uint32_t mem_clear_loc_start = old_location_start;
    uint32_t mem_clear_loc_end_limit = old_location_start + num_locations;

    // Move the adjacency lists. Make sure that overlapping ranges are handled
    // correctly.
    if (new_location_start < old_location_start)
    {
        // New location before the old location: copy the entries in order
        // to avoid modifying locations that are yet to be copied.
        for (uint32_t loc_offset = 0; loc_offset < num_locations; loc_offset++)
        {
            assert(_graph_store->get_neighbours(new_location_start + loc_offset).empty());
            _graph_store->swap_neighbours(new_location_start + loc_offset, old_location_start + loc_offset);
            if (_dynamic_index && _filtered_index)
            {
                _location_to_labels[new_location_start + loc_offset].swap(
                    _location_to_labels[old_location_start + loc_offset]);
            }
        }
        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_start < new_location_start + num_locations)
        {
            // Clear only after the end of the new range.
            mem_clear_loc_start = new_location_start + num_locations;
        }
    }
    else
    {
        // Old location after the new location: copy from the end of the range
        // to avoid modifying locations that are yet to be copied.
        for (uint32_t loc_offset = num_locations; loc_offset > 0; loc_offset--)
        {
            assert(_graph_store->get_neighbours(new_location_start + loc_offset - 1u).empty());
            _graph_store->swap_neighbours(new_location_start + loc_offset - 1u, old_location_start + loc_offset - 1u);
            if (_dynamic_index && _filtered_index)
            {
                _location_to_labels[new_location_start + loc_offset - 1u].swap(
                    _location_to_labels[old_location_start + loc_offset - 1u]);
            }
        }

        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_end_limit > new_location_start)
        {
            // Clear only up to the beginning of the new range.
            mem_clear_loc_end_limit = new_location_start;
        }
    }
    _data_store->move_vectors(old_location_start, new_location_start, num_locations);
}

template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::reposition_frozen_point_to_end()
{
    if (_num_frozen_pts == 0)
        return;

    if (_nd == _max_points)
    {
        diskann::cout << "Not repositioning frozen point as it is already at the end." << std::endl;
        return;
    }

    reposition_points((uint32_t)_nd, (uint32_t)_max_points, (uint32_t)_num_frozen_pts);
    _start = (uint32_t)_max_points;

    // update medoid id's as frozen points are treated as medoid
    if (_filtered_index && _dynamic_index)
    {
        for (auto &[label, medoid_id] : _label_to_start_id)
        {
            /*if (label == _universal_label)
                continue;*/
            _label_to_start_id[label] = (uint32_t)_max_points + (medoid_id - (uint32_t)_nd);
        }
    }
}

template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::resize(size_t new_max_points)
{
    const size_t new_internal_points = new_max_points + _num_frozen_pts;
    auto start = std::chrono::high_resolution_clock::now();
    assert(_empty_slots.size() == 0); // should not resize if there are empty slots.

    _data_store->resize((location_t)new_internal_points);
    _graph_store->resize_graph(new_internal_points);
    _locks = std::vector<non_recursive_mutex>(new_internal_points);

    if (_num_frozen_pts != 0)
    {
        reposition_points((uint32_t)_max_points, (uint32_t)new_max_points, (uint32_t)_num_frozen_pts);
        _start = (uint32_t)new_max_points;
    }

    _max_points = new_max_points;
    _empty_slots.reserve(_max_points);
    for (auto i = _nd; i < _max_points; i++)
    {
        _empty_slots.insert((uint32_t)i);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    diskann::cout << "Resizing took: " << std::chrono::duration<double>(stop - start).count() << "s" << std::endl;
}

template <typename T, typename TagT, typename LabelT>
int Index<T, TagT, LabelT>::_insert_point(const DataType &point, const TagType tag)
{
    try
    {
        return this->insert_point(std::any_cast<const T *>(point), std::any_cast<const TagT>(tag));
    }
    catch (const std::bad_any_cast &anycast_e)
    {
        throw new ANNException("Error:Trying to insert invalid data type" + std::string(anycast_e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw new ANNException("Error:" + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
int Index<T, TagT, LabelT>::_insert_point(const DataType &point, const TagType tag, Labelvector &labels)
{
    try
    {
        return this->insert_point(std::any_cast<const T *>(point), std::any_cast<const TagT>(tag),
                                  labels.get<const std::vector<LabelT>>());
    }
    catch (const std::bad_any_cast &anycast_e)
    {
        throw new ANNException("Error:Trying to insert invalid data type" + std::string(anycast_e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw new ANNException("Error:" + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
int Index<T, TagT, LabelT>::insert_point(const T *point, const TagT tag)
{
    std::vector<LabelT> no_labels{0};
    return insert_point(point, tag, no_labels);
}

template <typename T, typename TagT, typename LabelT>
int Index<T, TagT, LabelT>::insert_point(const T *point, const TagT tag, const std::vector<LabelT> &labels)
{

    assert(_has_built);
    if (tag == 0)
    {
        throw diskann::ANNException("Do not insert point with tag 0. That is "
                                    "reserved for points hidden "
                                    "from the user.",
                                    -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::shared_lock<std::shared_timed_mutex> shared_ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    // Mark data as not compacted since we're inserting a new point
    _data_compacted = false;

    auto location = reserve_location();
    if (_filtered_index)
    {
        if (labels.empty())
        {
            release_location(location);
            std::cerr << "Error: Can't insert point with tag " + get_tag_string(tag) +
                             " . there are no labels for the point."
                      << std::endl;
            return -1;
        }

        _location_to_labels[location] = labels;

        for (LabelT label : labels)
        {
            if (_labels.find(label) == _labels.end())
            {
                if (_frozen_pts_used >= _num_frozen_pts)
                {
                    throw ANNException(
                        "Error: For dynamic filtered index, the number of frozen points should be atleast equal "
                        "to number of unique labels.",
                        -1);
                }

                auto fz_location = (int)(_max_points) + _frozen_pts_used; // as first _fz_point
                _labels.insert(label);
                _label_to_start_id[label] = (uint32_t)fz_location;
                _location_to_labels[fz_location] = {label};
                _data_store->set_vector((location_t)fz_location, point);
                _frozen_pts_used++;
            }
        }
    }

    if (location == -1)
    {
#if EXPAND_IF_FULL
        dl.unlock();
        tl.unlock();
        shared_ul.unlock();

        {
            std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
            tl.lock();
            dl.lock();

            if (_nd >= _max_points)
            {
                auto new_max_points = (size_t)(_max_points * INDEX_GROWTH_FACTOR);
                resize(new_max_points);
            }

            dl.unlock();
            tl.unlock();
            ul.unlock();
        }

        shared_ul.lock();
        tl.lock();
        dl.lock();

        location = reserve_location();
        if (location == -1)
        {
            throw diskann::ANNException("Cannot reserve location even after "
                                        "expanding graph. Terminating.",
                                        -1, __FUNCSIG__, __FILE__, __LINE__);
        }
#else
        return -1;
#endif
    } // cant insert as active pts >= max_pts
    dl.unlock();

    // Insert tag and mapping to location
    if (_enable_tags)
    {
        // if tags are enabled and tag is already inserted. so we can't reuse that tag.
        if (_tag_to_location.find(tag) != _tag_to_location.end())
        {
            release_location(location);
            return -1;
        }

        _tag_to_location[tag] = location;
        _location_to_tag.set(location, tag);
    }
    tl.unlock();

    _data_store->set_vector(location, point); // update datastore

    // Assign partition for the new point based on nearest centroid
    if (_partition_loaded && !_centroids.empty() && static_cast<size_t>(location) < _location_to_partition.size())
    {
        uint32_t partition = assign_to_nearest_centroid(point);
        _location_to_partition[location] = partition;
    }

    // Find and add appropriate graph edges
    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto scratch = manager.scratch_space();
    std::vector<uint32_t> pruned_list; // it is the set best candidates to connect to this point
    if (_filtered_index)
    {
        // when filtered the best_candidates will share the same label ( label_present > distance)
        search_for_point_and_prune(location, _indexingQueueSize, pruned_list, scratch, true, _filterIndexingQueueSize);
    }
    else
    {
        search_for_point_and_prune(location, _indexingQueueSize, pruned_list, scratch);
    }
    assert(pruned_list.size() > 0); // should find atleast one neighbour (i.e frozen point acting as medoid)

    {
        std::shared_lock<std::shared_timed_mutex> tlock(_tag_lock, std::defer_lock);
        if (_conc_consolidate)
            tlock.lock();

        LockGuard guard(_locks[location]);
        _graph_store->clear_neighbours(location);

        std::vector<uint32_t> neighbor_links;
        for (auto link : pruned_list)
        {
            if (_conc_consolidate)
                if (!_location_to_tag.contains(link))
                    continue;
            neighbor_links.emplace_back(link);
        }
        _graph_store->set_neighbours(location, neighbor_links);
        assert(_graph_store->get_neighbours(location).size() <= _indexingRange);

        if (_conc_consolidate)
            tlock.unlock();
    }

    inter_insert(location, pruned_list, scratch);

    return 0;
}

// Insert with detailed timing statistics for latency profiling
template <typename T, typename TagT, typename LabelT>
int Index<T, TagT, LabelT>::insert_point_with_stats(const T *point, const TagT tag, InsertStats &stats)
{
    Timer total_timer, phase_timer;

    assert(_has_built);
    if (tag == 0)
    {
        throw diskann::ANNException("Do not insert point with tag 0. That is "
                                    "reserved for points hidden "
                                    "from the user.",
                                    -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Lock acquisition
    phase_timer.reset();
    std::shared_lock<std::shared_timed_mutex> shared_ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
    stats.lock_us = (float)phase_timer.elapsed();

    // Mark data as not compacted since we're inserting a new point
    _data_compacted = false;

    // Reserve location
    phase_timer.reset();
    auto location = reserve_location();

    if (location == -1)
    {
#if EXPAND_IF_FULL
        dl.unlock();
        tl.unlock();
        shared_ul.unlock();

        {
            std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
            tl.lock();
            dl.lock();

            if (_nd >= _max_points)
            {
                auto new_max_points = (size_t)(_max_points * INDEX_GROWTH_FACTOR);
                resize(new_max_points);
            }

            dl.unlock();
            tl.unlock();
            ul.unlock();
        }

        shared_ul.lock();
        tl.lock();
        dl.lock();

        location = reserve_location();
        if (location == -1)
        {
            throw diskann::ANNException("Cannot reserve location even after "
                                        "expanding graph. Terminating.",
                                        -1, __FUNCSIG__, __FILE__, __LINE__);
        }
#else
        stats.total_us = (float)total_timer.elapsed();
        return -1;
#endif
    }
    stats.reserve_us = (float)phase_timer.elapsed();
    dl.unlock();

    // Insert tag and mapping to location
    if (_enable_tags)
    {
        if (_tag_to_location.find(tag) != _tag_to_location.end())
        {
            release_location(location);
            stats.total_us = (float)total_timer.elapsed();
            return -1;
        }

        _tag_to_location[tag] = location;
        _location_to_tag.set(location, tag);
    }
    tl.unlock();

    // Data store write
    phase_timer.reset();
    _data_store->set_vector(location, point);

    // Assign partition for the new point based on nearest centroid
    if (_partition_loaded && !_centroids.empty() && static_cast<size_t>(location) < _location_to_partition.size())
    {
        uint32_t partition = assign_to_nearest_centroid(point);
        _location_to_partition[location] = partition;
    }
    stats.data_store_us = (float)phase_timer.elapsed();

    // Find and add appropriate graph edges - Search and Prune
    phase_timer.reset();
    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto scratch = manager.scratch_space();
    std::vector<uint32_t> pruned_list;
    search_for_point_and_prune(location, _indexingQueueSize, pruned_list, scratch);
    assert(pruned_list.size() > 0);
    stats.search_prune_us = (float)phase_timer.elapsed();

    // Graph linking
    phase_timer.reset();
    {
        std::shared_lock<std::shared_timed_mutex> tlock(_tag_lock, std::defer_lock);
        if (_conc_consolidate)
            tlock.lock();

        LockGuard guard(_locks[location]);
        _graph_store->clear_neighbours(location);

        std::vector<uint32_t> neighbor_links;
        for (auto link : pruned_list)
        {
            if (_conc_consolidate)
                if (!_location_to_tag.contains(link))
                    continue;
            neighbor_links.emplace_back(link);
        }
        _graph_store->set_neighbours(location, neighbor_links);
        assert(_graph_store->get_neighbours(location).size() <= _indexingRange);

        if (_conc_consolidate)
            tlock.unlock();
    }

    inter_insert(location, pruned_list, scratch);
    stats.link_us = (float)phase_timer.elapsed();

    stats.total_us = (float)total_timer.elapsed();
    return 0;
}

template <typename T, typename TagT, typename LabelT> int Index<T, TagT, LabelT>::_lazy_delete(const TagType &tag)
{
    try
    {
        return lazy_delete(std::any_cast<const TagT>(tag));
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException(std::string("Error: ") + e.what(), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::_lazy_delete(TagVector &tags, TagVector &failed_tags)
{
    try
    {
        this->lazy_delete(tags.get<const std::vector<TagT>>(), failed_tags.get<std::vector<TagT>>());
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad any cast while performing _lazy_delete() " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error: " + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT> int Index<T, TagT, LabelT>::lazy_delete(const TagT &tag)
{
    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
    _data_compacted = false;

    if (_tag_to_location.find(tag) == _tag_to_location.end())
    {
        diskann::cerr << "Delete tag not found " << get_tag_string(tag) << std::endl;
        return -1;
    }
    assert(_tag_to_location[tag] < _max_points);

    const auto location = _tag_to_location[tag];
    _delete_set->insert(location);
    // Preserve deleted location->tag mapping for trace recording
    _deleted_location_to_tag[location] = tag;
    _location_to_tag.erase(location);
    _tag_to_location.erase(tag);
    return 0;
}

// Lazy delete with detailed timing statistics for latency profiling
template <typename T, typename TagT, typename LabelT>
int Index<T, TagT, LabelT>::lazy_delete_with_stats(const TagT &tag, DeleteStats &stats)
{
    Timer total_timer, phase_timer;

    // Lock acquisition
    phase_timer.reset();
    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
    stats.lock_us = (float)phase_timer.elapsed();

    _data_compacted = false;

    // Tag lookup
    phase_timer.reset();
    if (_tag_to_location.find(tag) == _tag_to_location.end())
    {
        diskann::cerr << "Delete tag not found " << get_tag_string(tag) << std::endl;
        stats.total_us = (float)total_timer.elapsed();
        return -1;
    }
    assert(_tag_to_location[tag] < _max_points);
    const auto location = _tag_to_location[tag];
    stats.tag_lookup_us = (float)phase_timer.elapsed();

    // Mark as deleted
    phase_timer.reset();
    _delete_set->insert(location);
    _deleted_location_to_tag[location] = tag;
    _location_to_tag.erase(location);
    _tag_to_location.erase(tag);
    stats.mark_us = (float)phase_timer.elapsed();

    stats.total_us = (float)total_timer.elapsed();
    return 0;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::lazy_delete(const std::vector<TagT> &tags, std::vector<TagT> &failed_tags)
{
    if (failed_tags.size() > 0)
    {
        throw ANNException("failed_tags should be passed as an empty list", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
    _data_compacted = false;

    for (auto tag : tags)
    {
        if (_tag_to_location.find(tag) == _tag_to_location.end())
        {
            failed_tags.push_back(tag);
        }
        else
        {
            const auto location = _tag_to_location[tag];
            _delete_set->insert(location);
            // Preserve deleted location->tag mapping for trace recording
            _deleted_location_to_tag[location] = tag;
            _location_to_tag.erase(location);
            _tag_to_location.erase(tag);
        }
    }
}

template <typename T, typename TagT, typename LabelT> bool Index<T, TagT, LabelT>::is_index_saved()
{
    return _is_saved;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::_get_active_tags(TagRobinSet &active_tags)
{
    try
    {
        this->get_active_tags(active_tags.get<tsl::robin_set<TagT>>());
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad_any cast while performing _get_active_tags() " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error :" + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::get_active_tags(tsl::robin_set<TagT> &active_tags)
{
    active_tags.clear();
    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
    for (auto iter : _tag_to_location)
    {
        active_tags.insert(iter.first);
    }
}

template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::print_status()
{
    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
    std::shared_lock<std::shared_timed_mutex> cl(_consolidate_lock);
    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);

    diskann::cout << "------------------- Index object: " << (uint64_t)this << " -------------------" << std::endl;
    diskann::cout << "Number of points: " << _nd << std::endl;
    diskann::cout << "Max points: " << _max_points << std::endl;
    diskann::cout << "Graph size: " << _graph_store->get_total_points() << std::endl;
    diskann::cout << "Location to tag size: " << _location_to_tag.size() << std::endl;
    diskann::cout << "Tag to location size: " << _tag_to_location.size() << std::endl;
    diskann::cout << "Number of empty slots: " << _empty_slots.size() << std::endl;
    diskann::cout << std::boolalpha << "Data compacted: " << this->_data_compacted << std::endl;
    diskann::cout << "---------------------------------------------------------"
                     "------------"
                  << std::endl;
}

template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::count_nodes_at_bfs_levels()
{
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);

    boost::dynamic_bitset<> visited(_max_points + _num_frozen_pts);

    size_t MAX_BFS_LEVELS = 32;
    auto bfs_sets = new tsl::robin_set<uint32_t>[MAX_BFS_LEVELS];

    bfs_sets[0].insert(_start);
    visited.set(_start);

    for (uint32_t i = (uint32_t)_max_points; i < _max_points + _num_frozen_pts; ++i)
    {
        if (i != _start)
        {
            bfs_sets[0].insert(i);
            visited.set(i);
        }
    }

    for (size_t l = 0; l < MAX_BFS_LEVELS - 1; ++l)
    {
        diskann::cout << "Number of nodes at BFS level " << l << " is " << bfs_sets[l].size() << std::endl;
        if (bfs_sets[l].size() == 0)
            break;
        for (auto node : bfs_sets[l])
        {
            for (auto nghbr : _graph_store->get_neighbours((location_t)node))
            {
                if (!visited.test(nghbr))
                {
                    visited.set(nghbr);
                    bfs_sets[l + 1].insert(nghbr);
                }
            }
        }
    }

    delete[] bfs_sets;
}

// REFACTOR: This should be an OptimizedDataStore class
template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::optimize_index_layout()
{ // use after build or load
    if (_dynamic_index)
    {
        throw diskann::ANNException("Optimize_index_layout not implemented for dyanmic indices", -1, __FUNCSIG__,
                                    __FILE__, __LINE__);
    }

    float *cur_vec = new float[_data_store->get_aligned_dim()];
    std::memset(cur_vec, 0, _data_store->get_aligned_dim() * sizeof(float));
    _data_len = (_data_store->get_aligned_dim() + 1) * sizeof(float);
    _neighbor_len = (_graph_store->get_max_observed_degree() + 1) * sizeof(uint32_t);
    _node_size = _data_len + _neighbor_len;
    _opt_graph = new char[_node_size * _nd];
    auto dist_fast = (DistanceFastL2<T> *)(_data_store->get_dist_fn());
    for (uint32_t i = 0; i < _nd; i++)
    {
        char *cur_node_offset = _opt_graph + i * _node_size;
        _data_store->get_vector(i, (T *)cur_vec);
        float cur_norm = dist_fast->norm((T *)cur_vec, (uint32_t)_data_store->get_aligned_dim());
        std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
        std::memcpy(cur_node_offset + sizeof(float), cur_vec, _data_len - sizeof(float));

        cur_node_offset += _data_len;
        uint32_t k = (uint32_t)_graph_store->get_neighbours(i).size();
        std::memcpy(cur_node_offset, &k, sizeof(uint32_t));
        std::memcpy(cur_node_offset + sizeof(uint32_t), _graph_store->get_neighbours(i).data(), k * sizeof(uint32_t));
        // std::vector<uint32_t>().swap(_graph_store->get_neighbours(i));
        _graph_store->clear_neighbours(i);
    }
    _graph_store->clear_graph();
    _graph_store->resize_graph(0);
    delete[] cur_vec;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::_search_with_optimized_layout(const DataType &query, size_t K, size_t L, uint32_t *indices)
{
    try
    {
        return this->search_with_optimized_layout(std::any_cast<const T *>(query), K, L, indices);
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad any cast while performing "
                           "_search_with_optimized_layout() " +
                               std::string(e.what()),
                           -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error: " + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::search_with_optimized_layout(const T *query, size_t K, size_t L, uint32_t *indices)
{
    DistanceFastL2<T> *dist_fast = (DistanceFastL2<T> *)(_data_store->get_dist_fn());

    NeighborPriorityQueue retset(L);
    std::vector<uint32_t> init_ids(L);

    boost::dynamic_bitset<> flags{_nd, 0};
    uint32_t tmp_l = 0;
    uint32_t *neighbors = (uint32_t *)(_opt_graph + _node_size * _start + _data_len);
    uint32_t MaxM_ep = *neighbors;
    neighbors++;

    for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++)
    {
        init_ids[tmp_l] = neighbors[tmp_l];
        flags[init_ids[tmp_l]] = true;
    }

    while (tmp_l < L)
    {
        uint32_t id = rand() % _nd;
        if (flags[id])
            continue;
        flags[id] = true;
        init_ids[tmp_l] = id;
        tmp_l++;
    }

    for (uint32_t i = 0; i < init_ids.size(); i++)
    {
        uint32_t id = init_ids[i];
        if (id >= _nd)
            continue;
        _mm_prefetch(_opt_graph + _node_size * id, _MM_HINT_T0);
    }
    L = 0;
    for (uint32_t i = 0; i < init_ids.size(); i++)
    {
        uint32_t id = init_ids[i];
        if (id >= _nd)
            continue;
        T *x = (T *)(_opt_graph + _node_size * id);
        float norm_x = *x;
        x++;
        float dist = dist_fast->compare(x, query, norm_x, (uint32_t)_data_store->get_aligned_dim());
        retset.insert(Neighbor(id, dist));
        flags[id] = true;
        L++;
    }

    while (retset.has_unexpanded_node())
    {
        auto nbr = retset.closest_unexpanded();
        auto n = nbr.id;
        _mm_prefetch(_opt_graph + _node_size * n + _data_len, _MM_HINT_T0);
        neighbors = (uint32_t *)(_opt_graph + _node_size * n + _data_len);
        uint32_t MaxM = *neighbors;
        neighbors++;
        for (uint32_t m = 0; m < MaxM; ++m)
            _mm_prefetch(_opt_graph + _node_size * neighbors[m], _MM_HINT_T0);
        for (uint32_t m = 0; m < MaxM; ++m)
        {
            uint32_t id = neighbors[m];
            if (flags[id])
                continue;
            flags[id] = 1;
            T *data = (T *)(_opt_graph + _node_size * id);
            float norm = *data;
            data++;
            float dist = dist_fast->compare(query, data, norm, (uint32_t)_data_store->get_aligned_dim());
            Neighbor nn(id, dist);
            retset.insert(nn);
        }
    }

    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retset[i].id;
    }
}

// Graph metrics: degree distribution + BFS connectivity
template <typename T, typename TagT, typename LabelT>
typename Index<T, TagT, LabelT>::GraphMetrics Index<T, TagT, LabelT>::compute_graph_metrics()
{
    GraphMetrics metrics{};

    // Collect active locations (from _location_to_tag) + frozen points
    std::vector<uint32_t> active_locs;
    active_locs.reserve(_location_to_tag.size() + _num_frozen_pts);

    for (uint32_t loc = 0; loc < _max_points; loc++)
    {
        TagT tag;
        if (_location_to_tag.try_get(loc, tag))
            active_locs.push_back(loc);
    }
    // Frozen points
    for (size_t fz = _max_points; fz < _max_points + _num_frozen_pts; fz++)
    {
        active_locs.push_back((uint32_t)fz);
    }

    if (active_locs.empty())
        return metrics;

    // Compute degrees
    std::vector<uint32_t> degrees;
    degrees.reserve(active_locs.size());
    for (auto loc : active_locs)
    {
        auto &nbrs = _graph_store->get_neighbours((location_t)loc);
        degrees.push_back((uint32_t)nbrs.size());
    }

    std::sort(degrees.begin(), degrees.end());
    size_t n = degrees.size();

    // Basic stats
    double sum = 0;
    for (auto d : degrees)
        sum += d;
    metrics.mean_degree = sum / n;
    metrics.min_degree = degrees.front();
    metrics.max_degree = degrees.back();
    metrics.median_degree = degrees[n / 2];
    metrics.p95_degree = degrees[(size_t)(n * 0.95)];
    metrics.p99_degree = degrees[(size_t)(n * 0.99)];

    double sq_sum = 0;
    for (auto d : degrees)
        sq_sum += (d - metrics.mean_degree) * (d - metrics.mean_degree);
    metrics.std_degree = std::sqrt(sq_sum / n);

    metrics.num_active_nodes = (uint32_t)n;

    // Histogram (50 bins)
    const size_t num_bins = 50;
    metrics.histogram_edges.resize(num_bins + 1);
    metrics.histogram_counts.resize(num_bins, 0);
    double bin_width = (double)(metrics.max_degree - metrics.min_degree) / num_bins;
    if (bin_width < 1e-9)
        bin_width = 1.0;
    for (size_t i = 0; i <= num_bins; i++)
        metrics.histogram_edges[i] = metrics.min_degree + i * bin_width;
    for (auto d : degrees)
    {
        size_t bin = (size_t)((d - metrics.min_degree) / bin_width);
        if (bin >= num_bins)
            bin = num_bins - 1;
        metrics.histogram_counts[bin]++;
    }

    // BFS from _start for connectivity
    tsl::robin_set<uint32_t> active_set(active_locs.begin(), active_locs.end());
    tsl::robin_set<uint32_t> visited;
    visited.reserve(active_locs.size());
    std::queue<uint32_t> queue;

    queue.push(_start);
    visited.insert(_start);

    while (!queue.empty())
    {
        uint32_t node = queue.front();
        queue.pop();
        auto &nbrs = _graph_store->get_neighbours((location_t)node);
        for (auto nbr : nbrs)
        {
            if (visited.find(nbr) == visited.end() && active_set.find(nbr) != active_set.end())
            {
                visited.insert(nbr);
                queue.push(nbr);
            }
        }
    }

    uint32_t reachable = 0;
    for (auto loc : active_locs)
    {
        if (visited.find(loc) != visited.end())
            reachable++;
    }

    metrics.reachable_count = reachable;
    metrics.unreachable_count = (uint32_t)n - reachable;
    metrics.unreachable_fraction = (double)metrics.unreachable_count / n;

    return metrics;
}

/*  Internals of the library */
template <typename T, typename TagT, typename LabelT> const float Index<T, TagT, LabelT>::INDEX_GROWTH_FACTOR = 1.5f;

// EXPORTS
template DISKANN_DLLEXPORT class Index<float, int32_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<int8_t, int32_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, int32_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<float, uint32_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<int8_t, uint32_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, uint32_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<float, int64_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<int8_t, int64_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, int64_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<float, uint64_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<int8_t, uint64_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, uint64_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<float, tag_uint128, uint32_t>;
template DISKANN_DLLEXPORT class Index<int8_t, tag_uint128, uint32_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, tag_uint128, uint32_t>;
// Label with short int 2 byte
template DISKANN_DLLEXPORT class Index<float, int32_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<int8_t, int32_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, int32_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<float, uint32_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<int8_t, uint32_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, uint32_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<float, int64_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<int8_t, int64_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, int64_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<float, uint64_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<int8_t, uint64_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, uint64_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<float, tag_uint128, uint16_t>;
template DISKANN_DLLEXPORT class Index<int8_t, tag_uint128, uint16_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, tag_uint128, uint16_t>;

template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
// TagT==uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);

template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search_with_filters<
    uint64_t>(const float *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search_with_filters<
    uint32_t>(const float *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search_with_filters<
    uint64_t>(const uint8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search_with_filters<
    uint32_t>(const uint8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search_with_filters<
    uint64_t>(const int8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search_with_filters<
    uint32_t>(const int8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
// TagT==uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search_with_filters<
    uint64_t>(const float *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search_with_filters<
    uint32_t>(const float *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search_with_filters<
    uint64_t>(const uint8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search_with_filters<
    uint32_t>(const uint8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search_with_filters<
    uint64_t>(const int8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search_with_filters<
    uint32_t>(const int8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);

template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
// TagT==uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);

template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search_with_filters<
    uint64_t>(const float *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search_with_filters<
    uint32_t>(const float *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search_with_filters<
    uint64_t>(const uint8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search_with_filters<
    uint32_t>(const uint8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search_with_filters<
    uint64_t>(const int8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search_with_filters<
    uint32_t>(const int8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
// TagT==uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search_with_filters<
    uint64_t>(const float *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search_with_filters<
    uint32_t>(const float *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search_with_filters<
    uint64_t>(const uint8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search_with_filters<
    uint32_t>(const uint8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search_with_filters<
    uint64_t>(const int8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search_with_filters<
    uint32_t>(const int8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);

// =================================================================
// REPLACE your previous search_with_trace instantiations with this CORRECTED block
// =================================================================

//
// === Explicit instantiations for search_with_trace ===
//

// TagT = uint64_t, LabelT = uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search_with_trace<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search_with_trace<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search_with_trace<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search_with_trace<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search_with_trace<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search_with_trace<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
// TagT = uint32_t, LabelT = uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search_with_trace<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search_with_trace<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search_with_trace<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search_with_trace<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search_with_trace<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search_with_trace<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
// TagT = uint64_t, LabelT = uint16_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search_with_trace<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search_with_trace<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search_with_trace<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search_with_trace<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search_with_trace<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search_with_trace<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
// TagT = uint32_t, LabelT = uint16_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search_with_trace<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search_with_trace<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search_with_trace<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search_with_trace<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search_with_trace<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search_with_trace<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, const std::vector<uint32_t>& partition_labels, std::vector<TraceInfo>& trace_path, float *distances);


//
// === Explicit instantiations for search_with_stats ===
//

// TagT = uint64_t, LabelT = uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search_with_stats<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search_with_stats<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search_with_stats<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search_with_stats<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search_with_stats<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search_with_stats<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances, QueryStats &stats);
// TagT = uint32_t, LabelT = uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search_with_stats<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search_with_stats<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search_with_stats<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search_with_stats<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search_with_stats<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search_with_stats<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances, QueryStats &stats);
// TagT = uint64_t, LabelT = uint16_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search_with_stats<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search_with_stats<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search_with_stats<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search_with_stats<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search_with_stats<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search_with_stats<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances, QueryStats &stats);
// TagT = uint32_t, LabelT = uint16_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search_with_stats<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search_with_stats<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search_with_stats<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search_with_stats<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search_with_stats<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances, QueryStats &stats);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search_with_stats<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances, QueryStats &stats);

//
// === Explicit instantiations for search_with_partition_stats ===
//

// TagT = uint64_t, LabelT = uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search_with_partition_stats<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<float> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search_with_partition_stats<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<float> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search_with_partition_stats<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<uint8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search_with_partition_stats<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<uint8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search_with_partition_stats<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<int8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search_with_partition_stats<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<int8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);

// TagT = uint32_t, LabelT = uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search_with_partition_stats<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<float> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search_with_partition_stats<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<float> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search_with_partition_stats<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<uint8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search_with_partition_stats<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<uint8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search_with_partition_stats<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<int8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search_with_partition_stats<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<int8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);

// TagT = uint64_t, LabelT = uint16_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search_with_partition_stats<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<float> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search_with_partition_stats<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<float> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search_with_partition_stats<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<uint8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search_with_partition_stats<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<uint8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search_with_partition_stats<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<int8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search_with_partition_stats<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<int8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);

// TagT = uint32_t, LabelT = uint16_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search_with_partition_stats<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<float> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search_with_partition_stats<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<float> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search_with_partition_stats<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<uint8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search_with_partition_stats<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<uint8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search_with_partition_stats<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<int8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search_with_partition_stats<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances,
    const uint32_t *partition_ids, uint32_t query_partition, QueryStats &stats,
    const std::vector<uint32_t> &custom_init_ids,
    const RemoteDistanceFn<int8_t> &remote_dist_fn,
    const RemoteNeighborFetchFn &remote_neighbor_fn,
    const GraphAccessFn &graph_access_fn);

} // namespace diskann
