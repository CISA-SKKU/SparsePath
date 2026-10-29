# Ground truth computation utilities (only files present in this project)

set(CMAKE_CXX_STANDARD 17)

add_executable(compute_groundtruth compute_groundtruth.cpp)
target_include_directories(compute_groundtruth PRIVATE ${DISKANN_MKL_INCLUDE_DIRECTORIES})
target_link_libraries(compute_groundtruth diskann_core ${DISKANN_MKL_LINK_LIBRARIES} ${DISKANN_ASYNC_LIB} Boost::program_options)

add_executable(compute_groundtruth_with_tags compute_groundtruth_with_tags.cpp)
target_include_directories(compute_groundtruth_with_tags PRIVATE ${DISKANN_MKL_INCLUDE_DIRECTORIES})
target_link_libraries(compute_groundtruth_with_tags diskann_core ${DISKANN_MKL_LINK_LIBRARIES} ${DISKANN_ASYNC_LIB} Boost::program_options)

add_executable(compute_groundtruth_batched compute_groundtruth_batched.cpp)
target_include_directories(compute_groundtruth_batched PRIVATE ${DISKANN_MKL_INCLUDE_DIRECTORIES})
target_link_libraries(compute_groundtruth_batched diskann_core ${DISKANN_MKL_LINK_LIBRARIES} ${DISKANN_ASYNC_LIB} Boost::program_options)
