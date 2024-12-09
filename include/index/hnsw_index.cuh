#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "../metrics/distance_metrics.cuh"

namespace gpu_vector_db {

struct HNSWParams {
    int max_elements;      // Maximum number of elements
    int M;                // Number of connections per layer
    int ef_construction;  // Size of the dynamic candidate list
    int ef_search;       // Size of the dynamic candidate list for search
    int num_layers;      // Maximum number of layers
    MetricType metric;   // Distance metric to use
};

class HNSWIndex {
public:
    HNSWIndex(const HNSWParams& params);
    ~HNSWIndex();

    // Build index from input vectors
    void build(const float* vectors, int num_vectors);
    
    // Search for k nearest neighbors
    void search(const float* query_vectors, int num_queries, int k,
                int* labels, float* distances);

    // Add new elements to the index
    void addBatch(const float* vectors, int num_vectors);

    // Save/load index
    void save(const char* filename);
    void load(const char* filename);

private:
    struct Level {
        std::vector<int> neighbors;
        std::vector<float> distances;
    };

    struct Node {
        std::vector<Level> levels;
        float* vector;
        int id;
    };

    void initializeNode(Node& node, const float* vector, int id);
    void searchLayer(const float* query, int entry_point,
                    int ef, int layer, std::vector<int>& result);

    HNSWParams params_;
    std::vector<Node> nodes_;
    float* d_vectors_;        // Device memory for vectors
    int* d_neighbors_;        // Device memory for neighbor lists
    cudaStream_t stream_;
};

} // namespace gpu_vector_db
