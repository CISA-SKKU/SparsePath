# store/ — Colocation store + RDMA fetcher
add_library(danns_store STATIC
    src/colocation_store.cpp
    src/colocation_fetcher.cpp
    src/peer_store.cpp
    src/node_store.cpp
    src/redirect_alloc.cpp
)

target_include_directories(danns_store PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(danns_store PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(danns_store PUBLIC
    danns_core
    danns_rdma
)
