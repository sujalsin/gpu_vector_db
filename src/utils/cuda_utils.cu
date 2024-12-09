#include "utils/cuda_utils.cuh"

namespace gpu_vector_db {

__global__ void computeL2Distance(const float* vectors_a,
                                const float* vectors_b,
                                float* distances,
                                int num_a,
                                int num_b,
                                int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_a * num_b) return;
    
    const int i = idx / num_b;
    const int j = idx % num_b;
    
    float dist = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float diff = vectors_a[i * dim + d] - vectors_b[j * dim + d];
        dist += diff * diff;
    }
    
    distances[idx] = sqrtf(dist);
}

__global__ void computeCosineSimilarity(const float* vectors_a,
                                      const float* vectors_b,
                                      float* similarities,
                                      int num_a,
                                      int num_b,
                                      int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_a * num_b) return;
    
    const int i = idx / num_b;
    const int j = idx % num_b;
    
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (int d = 0; d < dim; ++d) {
        float a = vectors_a[i * dim + d];
        float b = vectors_b[j * dim + d];
        dot_product += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    
    similarities[idx] = dot_product / (sqrtf(norm_a) * sqrtf(norm_b));
}

__global__ void updateCentroids(float* centroids,
                               const float* vectors,
                               const int* assignments,
                               int* cluster_sizes,
                               int num_vectors,
                               int num_clusters,
                               int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vectors) return;
    
    const int cluster_id = assignments[idx];
    atomicAdd(&cluster_sizes[cluster_id], 1);
    
    for (int d = 0; d < dim; ++d) {
        atomicAdd(&centroids[cluster_id * dim + d], vectors[idx * dim + d]);
    }
}

} // namespace gpu_vector_db
