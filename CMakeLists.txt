cmake_minimum_required(VERSION 3.15)
project(danns LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# =============================================================================
# Common compiler flags
# =============================================================================
# ISA baseline. Default "" keeps the historical -mavx2 -mfma -msse2 build.
#
# The comparison baselines both raise this and we do not: CoTra builds with
# "-Ofast -march=native" and RED-ANNS with "-O3 -march=native", so on worker
# (Sapphire Rapids) they auto-vectorize the whole codebase at AVX-512 width
# while we stay at AVX2. P1b's hand-written AVX-512 integer distance kernel
# does not cover this -- that is one memory-bound kernel, whereas the ISA
# baseline governs every auto-vectorized loop (neighbor scan, PQ LUT walk,
# hash probing, the nbr_vids copies, the beam memmove).
#
# NEVER set this to "native": the build host orchestrator-host is Sapphire Rapids and has
# amx_tile/amx_int8/amx_bf16/avx512_fp16/avx512_bf16/avx_vnni, none of which
# exist on worker-host (Ice Lake, Xeon 6338N) which loads the same .so -> SIGILL.
# "icelake-server" is the common ceiling of both hosts and still carries
# avx512f/bw/dq/cd/vl/vnni/vbmi/vbmi2/bitalg/vpopcntdq/ifma.
set(DANNS_ARCH "" CACHE STRING
    "Target ISA for -march (e.g. icelake-server). Empty = legacy -mavx2 baseline.")
# Promote -O3 to -Ofast (adds -ffast-math). Changes FP semantics -> recall must
# be re-verified against a paired run; keep as a separate A/B arm.
option(DANNS_FAST_MATH "Build Release with -Ofast instead of -O3" OFF)

if(DANNS_ARCH)
    message(STATUS "DANNS_ARCH:   -march=${DANNS_ARCH} (raised ISA baseline)")
    add_compile_options(-m64 -march=${DANNS_ARCH} -ftree-vectorize
                        -fopenmp -fopenmp-simd -funroll-loops -DUSE_AVX2)
else()
    add_compile_options(-m64 -mavx2 -mfma -msse2 -ftree-vectorize
                        -fopenmp -fopenmp-simd -funroll-loops -DUSE_AVX2)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")
if(DANNS_FAST_MATH)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# =============================================================================
# Find packages
# =============================================================================
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Boost (for DiskANN CLI apps only)
find_package(Boost COMPONENTS program_options)

# RDMA libraries
find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
find_library(RDMACM_LIBRARY rdmacm REQUIRED)
find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h)
find_path(RDMACM_INCLUDE_DIR rdma/rdma_cma.h)

if(NOT IBVERBS_INCLUDE_DIR)
    message(FATAL_ERROR "libibverbs headers not found. Install libibverbs-dev")
endif()
if(NOT RDMACM_INCLUDE_DIR)
    message(FATAL_ERROR "librdmacm headers not found. Install librdmacm-dev")
endif()

# MLX5 Direct Verbs (optional, for UMR)
option(ENABLE_MLX5_DV "Enable NVIDIA mlx5 Direct Verbs (UMR support)" ON)
set(HAVE_MLX5_DV FALSE)
if(ENABLE_MLX5_DV)
    find_library(MLX5_LIBRARY mlx5)
    find_path(MLX5_INCLUDE_DIR infiniband/mlx5dv.h)
    if(MLX5_LIBRARY AND MLX5_INCLUDE_DIR)
        set(HAVE_MLX5_DV TRUE)
        message(STATUS "Found libmlx5: ${MLX5_LIBRARY} — UMR enabled")
    else()
        message(STATUS "libmlx5 not found — UMR disabled")
    endif()
endif()

# =============================================================================
# MKL paths (same search logic as legacy CMake)
# =============================================================================
set(POSSIBLE_OMP_PATHS
    "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so"
    "/usr/lib/x86_64-linux-gnu/libiomp5.so"
    "/opt/intel/lib/intel64_lin/libiomp5.so")
foreach(p ${POSSIBLE_OMP_PATHS})
    if(EXISTS ${p})
        get_filename_component(OMP_PATH ${p} DIRECTORY)
    endif()
endforeach()

set(POSSIBLE_MKL_LIB_PATHS
    "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so"
    "/usr/lib/x86_64-linux-gnu/libmkl_core.so"
    "/opt/intel/mkl/lib/intel64/libmkl_core.so")
foreach(p ${POSSIBLE_MKL_LIB_PATHS})
    if(EXISTS ${p})
        get_filename_component(MKL_PATH ${p} DIRECTORY)
    endif()
endforeach()

set(POSSIBLE_MKL_INCLUDE_PATHS
    "/opt/intel/oneapi/mkl/latest/include"
    "/usr/include/mkl"
    "/opt/intel/mkl/include/")
foreach(p ${POSSIBLE_MKL_INCLUDE_PATHS})
    if(EXISTS ${p})
        set(MKL_INCLUDE_PATH ${p})
    endif()
endforeach()

if(OMP_PATH)
    link_directories(${OMP_PATH})
endif()
if(MKL_PATH)
    link_directories(${MKL_PATH})
endif()
if(MKL_INCLUDE_PATH)
    include_directories(${MKL_INCLUDE_PATH})
endif()

add_definitions(-DMKL_ILP64)

# =============================================================================
# Subdirectories (dependency order)
# =============================================================================
add_subdirectory(third_party/diskann)  # libdiskann_core.a
add_subdirectory(core)                 # libdanns_core.a
add_subdirectory(rdma)                 # libdanns_rdma.a
add_subdirectory(trace)                # libdanns_trace.a
add_subdirectory(store)                # libdanns_store.a
add_subdirectory(search)               # libdanns_search.a
add_subdirectory(update)               # libdanns_update.a
add_subdirectory(cluster)              # libdanns_cluster.a
add_subdirectory(bindings)             # distributed_bindings.so, diskann_bindings.so

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== DANNS Build Configuration ===")
message(STATUS "Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler:  ${CMAKE_CXX_COMPILER}")
message(STATUS "MKL:           ${MKL_PATH}")
message(STATUS "RDMA:          ${IBVERBS_LIBRARY}")
message(STATUS "MLX5 (UMR):    ${HAVE_MLX5_DV}")
message(STATUS "pybind11:      ${pybind11_VERSION}")
message(STATUS "")
