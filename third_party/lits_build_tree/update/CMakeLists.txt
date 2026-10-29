# update/ — Graph update and consolidation
add_library(danns_update STATIC
    src/consolidation.cpp
    src/insertion.cpp
    src/lits_policy.cpp
)

target_include_directories(danns_update PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(danns_update PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/diskann/include
)

set_target_properties(danns_update PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(danns_update PUBLIC
    danns_core
    danns_store
    danns_search
    danns_trace
    diskann_core
    OpenMP::OpenMP_CXX
)
