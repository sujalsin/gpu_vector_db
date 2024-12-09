# GPU-Accelerated Vector Database

A high-performance vector similarity search engine that leverages GPU acceleration for building and querying vector indices. This project implements multiple indexing structures and similarity metrics, with support for batch processing and persistence.

## 🚀 Features

- **Multiple Index Types**
  - IVF (Inverted File Index)
  - HNSW (Hierarchical Navigable Small World)
  - PQ (Product Quantization)

- **Diverse Distance Metrics**
  - L2 (Euclidean) Distance
  - Cosine Similarity
  - Inner Product
  - Hamming Distance
  - Jaccard Distance

- **Advanced Features**
  - GPU Acceleration
  - Batch Processing
  - Persistence Layer
  - Asynchronous Operations

## 🏗️ Architecture

### System Overview
```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Client Layer   │     │   Index Layer    │     │    GPU Layer     │
│                  │     │                  │     │                  │
│  - Batch Queue   │ ──► │  - IVF Index    │ ──► │ - CUDA Kernels  │
│  - API Interface │     │  - HNSW Index   │     │ - Memory Mgmt   │
│  - Persistence   │ ◄── │  - PQ Index     │ ◄── │ - Multi-Stream  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

### Index Structures

#### IVF (Inverted File Index)
```
                    ┌─── Cluster 1 ───┐
Query Vector ───►   ├─── Cluster 2 ───┤
                    ├─── Cluster 3 ───┤
                    └─── Cluster 4 ───┘
```

#### HNSW (Hierarchical Navigable Small World)
```
Layer 2:    O ─── O
            │     │
Layer 1:  O─┼─O─O─┼─O
          │ │ │ │ │ │
Layer 0:  O─O─O─O─O─O
```

#### PQ (Product Quantization)
```
Vector: [x1, x2, x3, x4] ──► [Q1(x1), Q2(x2), Q3(x3), Q4(x4)]
```

## 🔧 Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compatible compiler
- SQLite3
- NVIDIA GPU with Compute Capability 7.5+

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpu_vector_db.git
cd gpu_vector_db
```

2. Build the project:
```bash
./build.sh
```

## 💡 Example Usage Scenarios

### 1. Image Similarity Search
```cpp
// Initialize HNSW index for image feature vectors
HNSWParams params{
    .max_elements = 1000000,
    .M = 16,
    .ef_construction = 200,
    .metric = MetricType::COSINE
};
HNSWIndex index(params);

// Add image feature vectors
std::vector<float> image_features = extract_image_features();
index.build(image_features.data(), num_images);

// Search for similar images
float query_vector[128];  // Feature vector of query image
int k = 10;  // Find top 10 similar images
std::vector<int> labels(k);
std::vector<float> distances(k);
index.search(query_vector, 1, k, labels.data(), distances.data());
```

### 2. Real-time Recommendation System
```cpp
// Initialize batch processor
BatchProcessor processor(1000, 4);  // batch_size=1000, num_streams=4
processor.start();

// Add user interaction vectors in real-time
BatchTask task;
task.type = BatchTask::Type::ADD;
task.vectors = user_interaction_vector;
task.num_vectors = 1;
task.callback = []() { update_recommendations(); };
processor.addTask(task);
```

### 3. Document Similarity with Persistence
```cpp
// Initialize persistence layer
Persistence db("vectors.db");
db.initializeSchema();

// Save document vectors
std::vector<float> doc_vectors = process_documents();
std::vector<int> doc_ids(num_docs);
db.saveVectors(doc_vectors.data(), doc_ids.data(), num_docs, vector_dim);

// Initialize PQ index for memory-efficient storage
PQParams params{
    .num_subvectors = 16,
    .bits_per_code = 8,
    .max_elements = 1000000
};
PQIndex index(params);
index.train(doc_vectors.data(), num_docs);
```

## 📊 Performance Benchmarks

| Index Type | Build Time (1M vectors) | Query Time (ms) | Memory Usage |
|------------|------------------------|-----------------|--------------|
| IVF        | 45s                   | 0.5            | High         |
| HNSW       | 180s                  | 0.2            | Medium       |
| PQ         | 120s                  | 0.3            | Low          |

## 🔍 Advanced Features

### Batch Processing
- Asynchronous operation handling
- Multiple CUDA streams for parallel processing
- Automatic batch size optimization
- Priority queue support

### Persistence Layer
- SQLite-based storage
- Transaction support
- Incremental updates
- Index structure persistence

### Distance Metrics
- GPU-accelerated distance computations
- Multiple similarity measures
- Custom metric support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)
- [Product Quantization Paper](https://hal.inria.fr/inria-00514462v2/document)
