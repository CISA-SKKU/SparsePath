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

# Output .so files to project root lib/ for easy import
set_target_properties(distributed_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
)
set_target_properties(diskann_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
)
