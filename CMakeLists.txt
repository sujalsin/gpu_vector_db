cmake_minimum_required(VERSION 3.18)
project(gpu_vector_db CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find SQLite3 for persistence
find_package(SQLite3 REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Add source files
add_executable(vector_db
    src/main.cu
    src/index/ivf_index.cu
    src/index/hnsw_index.cu
    src/index/pq_index.cu
    src/utils/cuda_utils.cu
    src/utils/persistence.cu
    src/metrics/distance_metrics.cu
    src/batch/batch_processor.cu
)

# Include directories
target_include_directories(vector_db PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SQLite3_INCLUDE_DIRS}
)

# Set CUDA specific flags
set_target_properties(vector_db PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link against CUDA libraries and SQLite3
target_link_libraries(vector_db
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    SQLite::SQLite3
)
