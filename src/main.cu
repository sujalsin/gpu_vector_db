#include <iostream>
#include <random>
#include <chrono>
#include <thread>
#include "index/ivf_index.cuh"
#include "index/hnsw_index.cuh"
#include "index/pq_index.cuh"
#include "metrics/distance_metrics.cuh"
#include "utils/persistence.cuh"
#include "batch/batch_processor.cuh"

using namespace gpu_vector_db;

// Generate random vectors for testing
std::vector<float> generateRandomVectors(int num_vectors, int dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    std::vector<float> vectors(num_vectors * dim);
    for (float& v : vectors) {
        v = dis(gen);
    }
    return vectors;
}

void testIndex(const char* name, auto& index, const std::vector<float>& train_vectors,
               const std::vector<float>& query_vectors, int k) {
    const int num_train = train_vectors.size() / query_vectors.size() * k;
    const int num_queries = query_vectors.size() / (query_vectors.size() / k);
    
    // Build index
    auto start = std::chrono::high_resolution_clock::now();
    index.build(train_vectors.data(), num_train);
    auto end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << name << " index built in " << build_time.count() << "ms\n";
    
    // Allocate memory for results
    std::vector<int> labels(num_queries * k);
    std::vector<float> distances(num_queries * k);
    
    // Perform search
    start = std::chrono::high_resolution_clock::now();
    index.search(query_vectors.data(), num_queries, k, labels.data(), distances.data());
    end = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << name << " search completed in " << search_time.count() << "ms\n";
    std::cout << "Average search time per query: " 
              << static_cast<float>(search_time.count()) / num_queries << "ms\n";
    
    // Print some results
    std::cout << "\nFirst query results for " << name << ":\n";
    for (int i = 0; i < k; ++i) {
        std::cout << "Neighbor " << i + 1 << ": ID=" << labels[i] 
                  << ", Distance=" << distances[i] << "\n";
    }
    std::cout << "\n";
}

void testBatchProcessor(BatchProcessor& processor, const std::vector<float>& vectors) {
    const int num_vectors = vectors.size() / 128;  // Assuming 128-dim vectors
    const int batch_size = 1000;
    const int num_batches = num_vectors / batch_size;
    
    std::cout << "Testing batch processor with " << num_batches << " batches...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Add batches
    for (int i = 0; i < num_batches; ++i) {
        BatchTask task;
        task.type = BatchTask::Type::ADD;
        task.vectors = vectors.data() + i * batch_size * 128;
        task.num_vectors = batch_size;
        task.callback = []() { /* Optional callback */ };
        processor.addTask(task);
    }
    
    processor.waitForCompletion();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Batch processing completed in " << duration.count() << "ms\n";
    std::cout << "Average time per batch: " 
              << static_cast<float>(duration.count()) / num_batches << "ms\n\n";
}

int main() {
    try {
        // Set parameters for different indices
        IVFParams ivf_params{
            .num_lists = 100,
            .max_vectors = 100000,
            .vector_dim = 128,
            .max_iter = 20,
            .threshold = 0.001f
        };
        
        HNSWParams hnsw_params{
            .max_elements = 100000,
            .M = 16,
            .ef_construction = 200,
            .ef_search = 40,
            .num_layers = 4,
            .metric = MetricType::L2
        };
        
        PQParams pq_params{
            .num_subvectors = 16,
            .bits_per_code = 8,
            .max_elements = 100000,
            .vector_dim = 128,
            .metric = MetricType::L2
        };
        
        // Generate test data
        const int num_train = 50000;
        const int num_query = 100;
        const int k = 10;
        auto train_vectors = generateRandomVectors(num_train, 128);
        auto query_vectors = generateRandomVectors(num_query, 128);
        
        // Test different indices
        {
            IVFIndex ivf_index(ivf_params);
            testIndex("IVF", ivf_index, train_vectors, query_vectors, k);
        }
        
        {
            HNSWIndex hnsw_index(hnsw_params);
            testIndex("HNSW", hnsw_index, train_vectors, query_vectors, k);
        }
        
        {
            PQIndex pq_index(pq_params);
            testIndex("PQ", pq_index, train_vectors, query_vectors, k);
        }
        
        // Test persistence
        std::cout << "Testing persistence...\n";
        Persistence persistence("vector_db.sqlite");
        persistence.initializeSchema();
        
        // Save vectors
        std::vector<int> ids(num_train);
        std::iota(ids.begin(), ids.end(), 0);
        persistence.saveVectors(train_vectors.data(), ids.data(), num_train, 128);
        
        // Test batch processing
        std::cout << "Testing batch processing...\n";
        BatchProcessor batch_processor(1000, 4);  // batch_size=1000, num_streams=4
        batch_processor.start();
        testBatchProcessor(batch_processor, train_vectors);
        batch_processor.stop();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
