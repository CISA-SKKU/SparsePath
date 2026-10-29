# bindings/ — Python bindings (thin wrappers)

# 1. Distributed bindings (main RDMA-based distributed search)
pybind11_add_module(distributed_bindings distributed_bindings.cpp)

target_link_libraries(distributed_bindings PRIVATE
    danns_core
    danns_rdma
    danns_store
    danns_search
    danns_update
    danns_cluster
    danns_trace
    diskann_core
    ${IBVERBS_LIBRARY}
    ${RDMACM_LIBRARY}
    mkl_intel_ilp64
    mkl_intel_thread
    mkl_core
    iomp5
    pthread
    m
    dl
    OpenMP::OpenMP_CXX
)

# 2. DiskANN bindings (standalone graph index for pipeline build stage)
pybind11_add_module(diskann_bindings diskann_bindings.cpp)

target_include_directories(diskann_bindings PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/diskann/include
)

target_link_libraries(diskann_bindings PRIVATE
    diskann_core
    mkl_intel_ilp64
    mkl_intel_thread
    mkl_core
    iomp5
    pthread
    m
    dl
    OpenMP::OpenMP_CXX
)

# 3. PQ single-node search bindings (M11 false-filtering measurement harness).
#    Separate module: reuses danns::PQFilter (in danns_search) + DiskANN Index
#    (diskann_core) but runs its own single-node beam search with the PQ filter.
pybind11_add_module(pq_search_bindings pq_search_bindings.cpp)

target_include_directories(pq_search_bindings PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/diskann/include
    ${CMAKE_SOURCE_DIR}/search/include
)

target_link_libraries(pq_search_bindings PRIVATE
    danns_search          # brings danns::PQFilter (+ transitively diskann_core, danns_*)
    diskann_core
    mkl_intel_ilp64
    mkl_intel_thread
    mkl_core
    iomp5
    pthread
    m
    dl
    OpenMP::OpenMP_CXX
)

# 4. Unified single-node engine: build/update (diskann_bindings.cpp) + search
#    (pq_search_bindings.cpp) in ONE module. The legacy modules above stay for
#    compatibility; classes are py::module_local() so both can coexist.
pybind11_add_module(single_engine
    single_engine.cpp
    diskann_bindings.cpp
    pq_search_bindings.cpp
)

target_include_directories(single_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/diskann/include
    ${CMAKE_SOURCE_DIR}/search/include
)

target_link_libraries(single_engine PRIVATE
    danns_search          # brings danns::PQFilter (+ transitively diskann_core, danns_*)
    diskann_core
    mkl_intel_ilp64
    mkl_intel_thread
    mkl_core
    iomp5
    pthread
    m
    dl
    OpenMP::OpenMP_CXX
)

# Output .so files to project root lib/ for easy import
set_target_properties(distributed_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
)
set_target_properties(diskann_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
)
set_target_properties(pq_search_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
)
set_target_properties(single_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
)
