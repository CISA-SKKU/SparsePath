# third_party/diskann/ — Trimmed DiskANN build (no tcmalloc for pybind11 safety)
cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# MKL (inherited from parent CMake)
add_definitions(-DMKL_ILP64)

# Compiler flags — NOTE: no -fno-builtin-malloc (tcmalloc removed for pybind11 compat)
# DANNS_ARCH (parent cache var) raises the ISA baseline; see the top-level
# CMakeLists for why "native" must never be used here.
if(DANNS_ARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${DANNS_ARCH} -ftree-vectorize -fopenmp -fopenmp-simd -funroll-loops -DUSE_AVX2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fopenmp -fopenmp-simd -funroll-loops -DUSE_AVX2")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -Ofast")

# Source files
set(DISKANN_SOURCES
    src/abstract_data_store.cpp
    src/ann_exception.cpp
    src/disk_utils.cpp
    src/distance.cpp
    src/distance_avx512_int.cpp
    src/index.cpp
    src/in_mem_graph_store.cpp
    src/in_mem_data_store.cpp
    src/linux_aligned_file_reader.cpp
    src/math_utils.cpp
    src/natural_number_map.cpp
    src/natural_number_set.cpp
    src/memory_mapper.cpp
    src/partition.cpp
    src/pq.cpp
    src/pq_flash_index.cpp
    src/scratch.cpp
    src/logger.cpp
    src/utils.cpp
    src/filter_utils.cpp
    src/index_factory.cpp
    src/abstract_index.cpp
    src/pq_l2_distance.cpp
    src/pq_data_store.cpp
    src/trace_writer.cpp
)

# Static library with PIC (for pybind11 linking)
add_library(diskann_core STATIC ${DISKANN_SOURCES})
set_target_properties(diskann_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# P1b (2026-07-27): the integer L2 kernels are the only AVX-512 code in the
# build. Scoping the flags to this one TU keeps every other object at the
# project-wide -mavx2 baseline, so a non-AVX-512 host can still load the
# library -- get_distance_function() probes the CPU before selecting them.
set_source_files_properties(src/distance_avx512_int.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl")

target_include_directories(diskann_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link MKL + OpenMP (no tcmalloc!)
target_link_libraries(diskann_core PUBLIC
    mkl_intel_ilp64
    mkl_intel_thread
    mkl_core
    iomp5
    pthread
    m
    dl
    aio
    OpenMP::OpenMP_CXX
)

# Optional: GT computation tools
if(Boost_FOUND)
    add_subdirectory(apps/utils)
endif()

message(STATUS "DiskANN: built from source (tcmalloc removed, PIC enabled)")
