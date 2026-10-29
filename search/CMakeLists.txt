# search/ — Search algorithms
add_library(danns_search STATIC
    src/beam_search.cpp
    src/pq_filter.cpp
)

target_include_directories(danns_search PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# DiskANN for PQ codebook (distance.h, pq.h)
target_include_directories(danns_search PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/diskann/include
)

set_target_properties(danns_search PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(danns_search PUBLIC
    danns_core
    danns_store
    danns_trace
    danns_cluster
    diskann_core
    OpenMP::OpenMP_CXX
)
