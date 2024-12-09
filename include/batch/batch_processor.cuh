#pragma once

#include <cuda_runtime.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>

namespace gpu_vector_db {

struct BatchTask {
    enum class Type {
        ADD,
        SEARCH,
        UPDATE,
        DELETE
    };

    Type type;
    const float* vectors;
    int num_vectors;
    int k;  // For search tasks
    void* results;
    std::function<void(void)> callback;
};

class BatchProcessor {
public:
    BatchProcessor(int batch_size, int num_streams);
    ~BatchProcessor();

    // Add a task to the queue
    void addTask(BatchTask task);
    
    // Wait for all tasks to complete
    void waitForCompletion();
    
    // Start processing tasks
    void start();
    
    // Stop processing tasks
    void stop();

private:
    void processingThread();
    void processAddBatch(const std::vector<BatchTask>& batch);
    void processSearchBatch(const std::vector<BatchTask>& batch);
    
    int batch_size_;
    std::vector<cudaStream_t> streams_;
    bool running_;
    
    std::queue<BatchTask> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::thread worker_thread_;
    
    // Memory pools for batched operations
    float* d_batch_vectors_;
    float* d_batch_results_;
    int* d_batch_labels_;
};

} // namespace gpu_vector_db
