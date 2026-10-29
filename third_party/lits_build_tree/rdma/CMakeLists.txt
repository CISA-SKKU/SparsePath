# rdma/ — RDMA transport layer (no DiskANN dependency)
add_library(danns_rdma STATIC
    src/connection.cpp
    src/odp.cpp
    src/umr.cpp
)

target_include_directories(danns_rdma PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${IBVERBS_INCLUDE_DIR}
    ${RDMACM_INCLUDE_DIR}
)

set_target_properties(danns_rdma PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(RDMA_LINK_LIBS
    ${IBVERBS_LIBRARY}
    ${RDMACM_LIBRARY}
    Threads::Threads
    danns_core
)

if(HAVE_MLX5_DV)
    target_include_directories(danns_rdma PRIVATE ${MLX5_INCLUDE_DIR})
    list(APPEND RDMA_LINK_LIBS ${MLX5_LIBRARY})
    target_compile_definitions(danns_rdma PRIVATE HAVE_MLX5_DV=1)
endif()

target_link_libraries(danns_rdma PUBLIC ${RDMA_LINK_LIBS})
