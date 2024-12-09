#pragma once

#include <string>
#include <vector>
#include <sqlite3.h>

namespace gpu_vector_db {

class Persistence {
public:
    Persistence(const std::string& db_path);
    ~Persistence();

    // Initialize database schema
    void initializeSchema();

    // Vector operations
    void saveVectors(const float* vectors, const int* ids, int num_vectors, int dim);
    void loadVectors(std::vector<float>& vectors, std::vector<int>& ids);
    
    // Index metadata operations
    void saveIndexMetadata(const std::string& index_type,
                          const std::string& params_json);
    void loadIndexMetadata(std::string& index_type,
                          std::string& params_json);

    // Index structure operations
    void saveIndexStructure(const std::string& index_type,
                           const void* data,
                           size_t size);
    void loadIndexStructure(const std::string& index_type,
                           void* data,
                           size_t& size);

    // Batch operation support
    void beginTransaction();
    void commitTransaction();
    void rollbackTransaction();

private:
    sqlite3* db_;
    std::string db_path_;

    // Helper functions
    void executeQuery(const std::string& query);
    void checkError(int rc, const std::string& operation);
};

} // namespace gpu_vector_db
