#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "../metrics/distance_metrics.cuh"

namespace gpu_vector_db {

struct PQParams {
    int num_subvectors;    // Number of subvectors
    int bits_per_code;     // Number of bits per subvector code
    int max_elements;      // Maximum number of elements
    int vector_dim;        // Dimension of input vectors
    MetricType metric;     // Distance metric to use
};

class PQIndex {
public:
    PQIndex(const PQParams& params);
    ~PQIndex();

    // Train the product quantizer
    void train(const float* vectors, int num_vectors);
    
    // Add vectors to the index
    void add(const float* vectors, int num_vectors);
    
    // Search for k nearest neighbors
    void search(const float* query_vectors, int num_queries, int k,
                int* labels, float* distances);

    // Save/load index
    void save(const char* filename);
    void load(const char* filename);

private:
    // Compute subvector centroids using k-means
    void trainSubvectorCentroids(const float* vectors, int num_vectors);
    
    // Encode vectors into PQ codes
    void encode(const float* vectors, int num_vectors, uint8_t* codes);
    
    // Compute distances between query and database vectors
    void computeDistances(const float* query, const uint8_t* codes,
                         int num_codes, float* distances);

    PQParams params_;
    float* d_centroids_;        // Device memory for centroids
    uint8_t* d_codes_;          // Device memory for PQ codes
    float* d_lookup_tables_;    // Device memory for distance lookup tables
    cudaStream_t stream_;
};

} // namespace gpu_vector_db
