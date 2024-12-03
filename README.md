# 🔍 ChromaDB Examples

A comprehensive collection of examples demonstrating how to use ChromaDB for vector search and semantic similarity operations. This repository provides practical examples and in-depth explanations of core ChromaDB concepts.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.20-green.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 Overview

ChromaDB is an open-source embedding database that allows you to store and search vector embeddings for semantic search applications. This repository contains examples that demonstrate:

- 🎯 Semantic search capabilities
- 📊 Vector operations and understanding
- 🔄 Different embedding functions and their applications
- 📏 Distance metrics and their comparative analysis

## 📂 Project Structure

```
.
├── .gitignore
├── example01-semantic-search.py
├── example02-understand-vectors.py
├── example03-embeddings-functions.py
├── example04-distance-functions.py
├── LICENSE
├── README.md
└── requirements.txt
```

## 💻 Examples

### 1. Semantic Search (`example01-semantic-search.py`)
Demonstrates basic semantic search functionality using ChromaDB:
- Text-to-vector conversion (embeddings)
- Simple similarity search
- Query result analysis

### 2. Vector Understanding (`example02-understand-vectors.py`)
Deep dive into vector operations:
- Vector creation and analysis
- Dimensionality exploration
- Vector properties and normalization
- Practical similarity search demonstrations

### 3. Embedding Functions (`example03-embeddings-functions.py`)
Comprehensive guide to embedding functions:
- Different types of embedding models
- Language specialization
- Domain-specific embeddings
- Content type specialization
- Best practices for mixing embeddings

### 4. Distance Functions (`example04-distance-functions.py`)
Analysis of different distance metrics:
- Cosine distance
- Euclidean (L2) distance
- Inner Product distance
- Comparative analysis with real-world examples

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chromadb-examples.git
cd chromadb-examples

# Install dependencies
pip install -r requirements.txt
```

### System Requirements
- Python 3.8 or higher
- 64-bit operating system
- At least 4GB RAM (8GB recommended)

### Key Dependencies
- `chromadb==0.5.20`
- `numpy==2.1.3`
- `scipy==1.14.1`
- Additional dependencies in `requirements.txt`

## 🚀 Usage

Each example can be run independently:

```bash
python example01-semantic-search.py
python example02-understand-vectors.py
python example03-embeddings-functions.py
python example04-distance-functions.py
```

## 🧠 Key Concepts

### Vector Embeddings
- Fixed-length arrays representing data mathematically
- Example: Text "I love dogs" → Vector [0.2, -0.5, 0.8, ...]
- Normalized vectors (magnitude ≈ 1.0) for consistent scaling

### Distance Metrics
1. **Cosine Distance** (Recommended for semantic search)
   - Best for capturing meaning similarities
   - Immune to text length variations
   - Range: 0-2 (0 = identical, 2 = opposite)

2. **Euclidean (L2) Distance**
   - Good for structural similarities
   - Influenced by vector magnitude
   - Unbounded range

3. **Inner Product Distance**
   - Sensitive to vector magnitude
   - Useful for specific recommendation systems
   - Requires careful normalization

### Embedding Models

1. **Language Models**
   - [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 dimensions) - Default
   - [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) (384 dimensions)
   - [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) (768 dimensions)

2. **Domain-Specific Models**
   - [pritamdeka/S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO) (768 dimensions)
   - [nlpaueb/legal-bert-base-uncased](https://huggingface.co/nlpaueb/legal-bert-base-uncased) (768 dimensions)
   - [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) (768 dimensions)

3. **Special Purpose Models**
   - [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base) (768 dimensions)
   - [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) (512 dimensions)
   - [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) (768 dimensions)

## 📝 Best Practices

### Embedding Selection
1. Choose based on:
   - Primary use case (general, domain-specific, multilingual)
   - Resource constraints
   - Required languages
   - Performance requirements

2. Consider trade-offs:
   - Model size vs. accuracy
   - Processing speed vs. quality
   - Memory usage vs. feature richness

### Collection Management
1. Data Organization:
   - One embedding model per collection
   - Consistent dimension sizes
   - Proper metadata tagging

2. Performance Optimization:
   - Index appropriate fields
   - Batch similar operations
   - Monitor memory usage

### Query Optimization
1. Search Strategies:
   - Use appropriate distance metrics
   - Implement proper filtering
   - Consider result count limits

2. Performance Tips:
   - Cache common queries
   - Use batch operations
   - Implement proper error handling

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [ChromaDB Documentation](https://docs.trychroma.com/getting-started)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Vector Similarity Search Guide](https://www.pinecone.io/learn/vector-similarity/)
