#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "../utils/cuda_utils.cuh"

namespace gpu_vector_db {

struct IVFParams {
    int num_lists;           // Number of clusters/lists
    int max_vectors;         // Maximum number of vectors to store
    int vector_dim;          // Dimension of vectors
    int max_iter;           // Maximum iterations for k-means
    float threshold;        // Convergence threshold for k-means
};

class IVFIndex {
public:
    IVFIndex(const IVFParams& params);
    ~IVFIndex();

    // Build index from input vectors
    void build(const float* vectors, int num_vectors);
    
    // Search for k nearest neighbors
    void search(const float* query_vectors, int num_queries, int k,
                int* labels, float* distances);

private:
    // GPU memory management
    void allocateMemory();
    void freeMemory();
    
    // K-means clustering for coarse quantization
    void trainCentroids(const float* vectors, int num_vectors);
    
    // Assign vectors to clusters
    void assignToClusters();

    IVFParams params_;
    
    // Device memory pointers
    float* d_centroids_;        // Cluster centroids
    float* d_vectors_;          // Vector data
    int* d_assignments_;        // Cluster assignments
    int* d_list_offsets_;      // Offsets for each cluster's vectors
    
    // Host memory
    std::vector<int> list_sizes_;  // Number of vectors in each cluster
    
    // CUDA stream for async operations
    cudaStream_t stream_;
};

} // namespace gpu_vector_db
