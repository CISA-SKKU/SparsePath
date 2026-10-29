# cluster/ — Multi-node orchestration
add_library(danns_cluster STATIC
    src/cluster_setup.cpp
    src/partition_mapper.cpp
)

target_include_directories(danns_cluster PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(danns_cluster PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(danns_cluster PUBLIC
    danns_core
    danns_rdma
    danns_store
)
