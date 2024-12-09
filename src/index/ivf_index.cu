#include "index/ivf_index.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace gpu_vector_db {

IVFIndex::IVFIndex(const IVFParams& params) : params_(params) {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    allocateMemory();
}

IVFIndex::~IVFIndex() {
    freeMemory();
    CUDA_CHECK(cudaStreamDestroy(stream_));
}

void IVFIndex::allocateMemory() {
    // Allocate device memory for centroids
    CUDA_CHECK(cudaMalloc(&d_centroids_, 
        params_.num_lists * params_.vector_dim * sizeof(float)));
    
    // Allocate device memory for vectors
    CUDA_CHECK(cudaMalloc(&d_vectors_,
        params_.max_vectors * params_.vector_dim * sizeof(float)));
    
    // Allocate device memory for cluster assignments
    CUDA_CHECK(cudaMalloc(&d_assignments_,
        params_.max_vectors * sizeof(int)));
    
    // Allocate device memory for list offsets
    CUDA_CHECK(cudaMalloc(&d_list_offsets_,
        (params_.num_lists + 1) * sizeof(int)));
    
    // Initialize host vectors
    list_sizes_.resize(params_.num_lists, 0);
}

void IVFIndex::freeMemory() {
    CUDA_CHECK(cudaFree(d_centroids_));
    CUDA_CHECK(cudaFree(d_vectors_));
    CUDA_CHECK(cudaFree(d_assignments_));
    CUDA_CHECK(cudaFree(d_list_offsets_));
}

void IVFIndex::build(const float* vectors, int num_vectors) {
    if (num_vectors > params_.max_vectors) {
        throw std::runtime_error("Number of vectors exceeds maximum capacity");
    }
    
    // Copy vectors to device
    CUDA_CHECK(cudaMemcpyAsync(d_vectors_, vectors,
        num_vectors * params_.vector_dim * sizeof(float),
        cudaMemcpyHostToDevice, stream_));
    
    // Train centroids using k-means
    trainCentroids(vectors, num_vectors);
    
    // Assign vectors to clusters
    assignToClusters();
}

void IVFIndex::trainCentroids(const float* vectors, int num_vectors) {
    // Initialize centroids with subset of vectors
    const int threads_per_block = 256;
    const int num_blocks = (num_vectors + threads_per_block - 1) / threads_per_block;
    
    // Randomly select initial centroids
    thrust::device_vector<int> d_indices(num_vectors);
    thrust::sequence(thrust::device, d_indices.begin(), d_indices.end());
    thrust::shuffle(thrust::device, d_indices.begin(), d_indices.end());
    
    // Copy selected vectors as initial centroids
    CUDA_CHECK(cudaMemcpyAsync(d_centroids_, vectors,
        params_.num_lists * params_.vector_dim * sizeof(float),
        cudaMemcpyHostToDevice, stream_));
    
    // Perform k-means iterations
    thrust::device_vector<float> d_temp_centroids(params_.num_lists * params_.vector_dim);
    thrust::device_vector<int> d_cluster_sizes(params_.num_lists);
    
    for (int iter = 0; iter < params_.max_iter; ++iter) {
        // Assign vectors to nearest centroids
        computeL2Distance<<<num_blocks, threads_per_block, 0, stream_>>>(
            d_vectors_, d_centroids_, d_assignments_,
            num_vectors, params_.num_lists, params_.vector_dim);
        
        // Update centroids
        updateCentroids<<<num_blocks, threads_per_block, 0, stream_>>>(
            thrust::raw_pointer_cast(d_temp_centroids.data()),
            d_vectors_, d_assignments_,
            thrust::raw_pointer_cast(d_cluster_sizes.data()),
            num_vectors, params_.num_lists, params_.vector_dim);
    }
}

void IVFIndex::search(const float* query_vectors, int num_queries, int k,
                     int* labels, float* distances) {
    const int threads_per_block = 256;
    const int num_blocks = (num_queries + threads_per_block - 1) / threads_per_block;
    
    // Allocate temporary memory for query results
    thrust::device_vector<float> d_query_distances(num_queries * params_.num_lists);
    thrust::device_vector<int> d_query_indices(num_queries * k);
    
    // Compute distances to centroids
    computeL2Distance<<<num_blocks, threads_per_block, 0, stream_>>>(
        query_vectors, d_centroids_,
        thrust::raw_pointer_cast(d_query_distances.data()),
        num_queries, params_.num_lists, params_.vector_dim);
    
    // For each query, find k nearest lists
    for (int i = 0; i < num_queries; ++i) {
        thrust::sort_by_key(
            thrust::device,
            d_query_distances.begin() + i * params_.num_lists,
            d_query_distances.begin() + (i + 1) * params_.num_lists,
            thrust::counting_iterator<int>(0));
    }
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpyAsync(labels, 
        thrust::raw_pointer_cast(d_query_indices.data()),
        num_queries * k * sizeof(int),
        cudaMemcpyDeviceToHost, stream_));
    
    CUDA_CHECK(cudaMemcpyAsync(distances,
        thrust::raw_pointer_cast(d_query_distances.data()),
        num_queries * k * sizeof(float),
        cudaMemcpyDeviceToHost, stream_));
}

} // namespace gpu_vector_db
