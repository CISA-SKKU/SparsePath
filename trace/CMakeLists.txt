# trace/ — Instrumentation and profiling
add_library(danns_trace STATIC
    src/search_trace.cpp
    src/consolidation_trace.cpp
)

target_include_directories(danns_trace PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(danns_trace PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(danns_trace PUBLIC
    danns_core
)
