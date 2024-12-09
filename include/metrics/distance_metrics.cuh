#pragma once

#include <cuda_runtime.h>

namespace gpu_vector_db {

enum class MetricType {
    L2,
    COSINE,
    INNER_PRODUCT,
    HAMMING,
    JACCARD
};

class DistanceMetrics {
public:
    static void computeDistance(MetricType metric_type,
                              const float* vectors_a,
                              const float* vectors_b,
                              float* distances,
                              int num_a,
                              int num_b,
                              int dim,
                              cudaStream_t stream = nullptr);

private:
    static __global__ void l2DistanceKernel(const float* vectors_a,
                                          const float* vectors_b,
                                          float* distances,
                                          int num_a,
                                          int num_b,
                                          int dim);

    static __global__ void cosineDistanceKernel(const float* vectors_a,
                                              const float* vectors_b,
                                              float* distances,
                                              int num_a,
                                              int num_b,
                                              int dim);

    static __global__ void innerProductKernel(const float* vectors_a,
                                           const float* vectors_b,
                                           float* distances,
                                           int num_a,
                                           int num_b,
                                           int dim);

    static __global__ void hammingDistanceKernel(const float* vectors_a,
                                              const float* vectors_b,
                                              float* distances,
                                              int num_a,
                                              int num_b,
                                              int dim);

    static __global__ void jaccardDistanceKernel(const float* vectors_a,
                                              const float* vectors_b,
                                              float* distances,
                                              int num_a,
                                              int num_b,
                                              int dim);
};

} // namespace gpu_vector_db
