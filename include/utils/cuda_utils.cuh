#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace gpu_vector_db {

// CUDA error checking
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + \
                               cudaGetErrorString(err)); \
    } \
}

// cuBLAS error checking
#define CUBLAS_CHECK(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS error"); \
    } \
}

// Compute L2 distance between vectors on GPU
__global__ void computeL2Distance(const float* vectors_a,
                                const float* vectors_b,
                                float* distances,
                                int num_a,
                                int num_b,
                                int dim);

// Compute cosine similarity between vectors on GPU
__global__ void computeCosineSimilarity(const float* vectors_a,
                                      const float* vectors_b,
                                      float* similarities,
                                      int num_a,
                                      int num_b,
                                      int dim);

// K-means update centroids kernel
__global__ void updateCentroids(float* centroids,
                               const float* vectors,
                               const int* assignments,
                               int* cluster_sizes,
                               int num_vectors,
                               int num_clusters,
                               int dim);

} // namespace gpu_vector_db
