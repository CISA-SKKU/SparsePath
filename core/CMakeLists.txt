# core/ — Pure C++ utilities (no RDMA/DiskANN dependency)
add_library(danns_core STATIC
    src/priority_queue.cpp
    src/latency_stats.cpp
    src/vector_id.cpp
)

target_include_directories(danns_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Optional: DiskANN distance.h for AVX2 L2
target_include_directories(danns_core PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/diskann/include
)

set_target_properties(danns_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(danns_core PUBLIC
    OpenMP::OpenMP_CXX
)
