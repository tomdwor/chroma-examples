import chromadb
from chromadb.utils import embedding_functions

# Understanding Embedding Functions in Vector Databases
# =====================================================
#
# Embedding Function: A method that converts data (text, images, etc.) into numerical vectors.
# Different embedding functions are optimized for different use cases:
# 1. Language Specialization:
#    - Language-specific models for better understanding of specific languages
#    - Multilingual models for cross-language applications
#
# 2. Domain Specialization:
#    - Scientific text understanding (scientific papers, research)
#    - Legal document processing (contracts, laws, regulations)
#    - Medical text analysis (clinical notes, medical literature)
#    - Financial text processing (reports, news, analysis)
#
# 3. Content Type Specialization:
#    - Code embeddings (for programming languages)
#    - Image embeddings (for visual content)
#    - Audio embeddings (for speech and sound)
#    - Multi-modal embeddings (combining text, images, etc.)
#
# Important Notes About Embedding Functions:
# 1. Vector Dimensions:
#    - Each embedding function produces vectors with a fixed number of dimensions
#    - You cannot mix embeddings of different dimensions in the same collection
#    - When switching embedding functions, you need to recreate the collection
#
# 2. Mixing Embeddings:
#    - Within a collection: All documents must use the same embedding function
#    - Between collections: You can have different collections with different embedding functions
#    - Same dimensions, different models: Can cause inconsistent results due to different vector spaces
#
# Default Embedding Function in ChromaDB:
# - Model: 'sentence-transformers/all-MiniLM-L6-v2'
# - Dimensions: 384
# - Language: Optimized for English, but works with many languages
# - Use case: General purpose text embeddings

# Initialize the client
chroma_client = chromadb.Client()

# Sample texts for different specializations
texts = {
    # Language examples
    "english": "The rapid advancement of artificial intelligence has transformed many industries.",
    "spanish": "El rápido avance de la inteligencia artificial ha transformado muchas industrias.",

    # Domain-specific examples
    "scientific": "The quaternionic Kähler manifold exhibits holonomy group Sp(n)Sp(1).",
    "legal": "The party of the first part hereby agrees to indemnify and hold harmless the party of the second part.",
    "medical": "Patient presents with acute myocardial infarction with ST elevation in leads V1-V4.",
    "financial": "The company's EBITDA grew by 15% YoY, with a corresponding increase in operating margin.",

    # Content type examples
    "code": "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
    "casual": "Hey! Have you tried that new coffee shop downtown? It's amazing! 😊"
}


def demonstrate_embedding_models_comparison():
    """Compare different embedding models and their effectiveness for specific use cases"""
    print("\nEmbedding Models Comparison")
    print("==========================")
    print("""
Distance Metrics in ChromaDB:
- By default, ChromaDB uses L2 (Euclidean) distance
- For clearer comparison, we'll use cosine distance:
  * 0 means vectors are identical (perfect similarity)
  * 2 means vectors are opposite (complete dissimilarity)
  * Lower distance = higher similarity
""")

    # Initialize different specialized embedding functions
    default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # General purpose
    )

    scientific_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO"  # Scientific/medical text
    )

    multilingual_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual
    )

    # Create test cases for different scenarios
    test_cases = {
        "Scientific Understanding": {
            "query": "What are the symptoms of myocardial infarction?",
            "relevant": texts["medical"],
            "irrelevant": texts["casual"],
            "expected_best": "Scientific"
        },
        "Multilingual Capability": {
            "query": texts["spanish"],
            "relevant": texts["english"],  # Same meaning in English
            "irrelevant": texts["code"],
            "expected_best": "Multilingual"
        },
        "General Usage": {
            "query": "Tell me about coffee shops",
            "relevant": texts["casual"],
            "irrelevant": texts["scientific"],
            "expected_best": "General"
        }
    }

    # Create collections for each model with cosine distance
    collections = {
        "General": chroma_client.create_collection(
            name="general_collection",
            embedding_function=default_ef,
            metadata={"hnsw:space": "cosine"}  # Explicitly set distance metric
        ),
        "Scientific": chroma_client.create_collection(
            name="scientific_collection",
            embedding_function=scientific_ef,
            metadata={"hnsw:space": "cosine"}
        ),
        "Multilingual": chroma_client.create_collection(
            name="multilingual_collection",
            embedding_function=multilingual_ef,
            metadata={"hnsw:space": "cosine"}
        )
    }

    # Run tests and analyze results
    print("\nModel Performance Analysis")
    print("=========================")

    for test_name, test_case in test_cases.items():
        print(f"\nTest: {test_name}")
        print(f"Query: '{test_case['query']}'")
        print("\nRelevant document:")
        print(f"'{test_case['relevant'][:100]}...'\n")
        print("Irrelevant document:")
        print(f"'{test_case['irrelevant'][:100]}...'\n")

        results = {}
        for model_name, collection in collections.items():
            # Add test documents
            collection.add(
                documents=[test_case['relevant'], test_case['irrelevant']],
                ids=["relevant", "irrelevant"]
            )

            # Query and get results
            query_results = collection.query(
                query_texts=[test_case['query']],
                n_results=2
            )

            # Get distances
            distances = {
                id_: dist for id_, dist in zip(query_results['ids'][0],
                                               query_results['distances'][0])
            }

            # Store results
            results[model_name] = {
                "distance_to_relevant": distances["relevant"],
                "distance_to_irrelevant": distances["irrelevant"]
            }

            # Delete documents for next test
            collection.delete(ids=["relevant", "irrelevant"])

        # Analyze and print results
        print("Results:")
        for model_name, result in results.items():
            dist_relevant = result["distance_to_relevant"]
            dist_irrelevant = result["distance_to_irrelevant"]

            # Determine if model correctly identifies relevant document
            better_similarity = dist_relevant < dist_irrelevant
            status = "✓" if better_similarity else "✗"

            print(f"\n{status} {model_name} Model:")
            print(f"   Distance to relevant document:   {dist_relevant:.4f}")
            print(f"   Distance to irrelevant document: {dist_irrelevant:.4f}")

            if model_name == test_case["expected_best"]:
                print("   📌 Expected to perform best for this case")

        # Print conclusion
        best_model = min(results.items(),
                         key=lambda x: x[1]["distance_to_relevant"])[0]
        print(f"\nConclusion: {best_model} model found the relevant document with lowest distance")
        if best_model == test_case["expected_best"]:
            print("✓ This matches our expectations")
        else:
            print("⚠ This differs from our expectations")


def demonstrate_mixed_embeddings():
    """Demonstrate mixing different embedding models"""
    print("\nMixed Embeddings Demo")
    print("====================")
    print("Demonstrating mixing embedding models with same and different dimensions\n")

    # Initialize embedding functions with same dimensions
    default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # 384 dimensions
    )

    similar_dim_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"  # 384 dimensions
    )

    # Custom function with different dimensions
    class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
        def __call__(self, texts):
            return [[0.1] * 512 for _ in texts]

    different_dim_ef = CustomEmbeddingFunction()

    print("Test 1: Different Models with Same Dimensions (384)")
    collection_same_dim = chroma_client.create_collection(
        name="same_dim_collection",
        embedding_function=default_ef
    )

    # Add documents with first embedding function
    collection_same_dim.add(
        documents=[texts["english"]],
        ids=["text_1"]
    )

    # Try to add document with different model but same dimensions
    try:
        # Note: This might work but could lead to inconsistent results
        collection_same_dim._embedding_function = similar_dim_ef
        collection_same_dim.add(
            documents=[texts["spanish"]],
            ids=["text_2"]
        )
        print("✓ Added documents with different models (same dimensions)")
        print("⚠ Warning: While this works technically, it may lead to inconsistent results")
        print("  because different models create different vector spaces")
    except Exception as e:
        print(f"✗ Error: {str(e)}")

    print("\nTest 2: Different Models with Different Dimensions (384 vs 512)")
    try:
        collection_same_dim._embedding_function = different_dim_ef
        collection_same_dim.add(
            documents=[texts["technical"]],
            ids=["text_3"]
        )
        print("Added document with different dimensions")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        print("  This is expected - cannot mix different dimensions")


print("\nEmbedding Functions Guide")
print("========================")
print("""
Available Embedding Functions in ChromaDB and Their Specializations:

1. Language Specialization (Text):
   - all-MiniLM-L6-v2 (384 dimensions) - General English
   - paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions) - Multiple languages
   - LaBSE (768 dimensions) - 109 languages
   - XLM-RoBERTa (768 dimensions) - 100 languages

2. Domain Specialization:
   - Scientific/Medical:
     * pritamdeka/S-PubMedBert-MS-MARCO (768 dimensions) - optimized for scientific/medical text
     * biobert-v1.1 (768 dimensions)
     * covid-scibert (768 dimensions)
   - Legal:
     * legal-bert-base-uncased (768 dimensions)
     * law-bert (768 dimensions)
   - Medical:
     * clinical-bert (768 dimensions)
     * biosyn-bert-large (1024 dimensions)
   - Financial:
     * finbert (768 dimensions)
     * fin-bert-tone (768 dimensions)

3. Content Type Specialization:
   - Code:
     * codebert-base (768 dimensions)
     * graphcodebert-base (768 dimensions)
   - Multi-modal:
     * clip-ViT-B-32 (512 dimensions) - text and images
     * align-base (768 dimensions) - text and images
   - Audio:
     * wav2vec2 (768 dimensions)
     * hubert-base (768 dimensions)

4. Hosted Solutions:
   - OpenAI's text-embedding-ada-002 (1536 dimensions)
   - Cohere's embed-multilingual-v3.0 (1024 dimensions)

Important Notes About Mixing Embeddings:
1. Different Dimensions:
   - Cannot mix in same collection
   - Must create separate collections
   - No direct comparison possible

2. Same Dimensions, Different Models:
   - Technically possible but not recommended
   - Models create different vector spaces
   - May lead to inconsistent similarity results
   - Better to use separate collections

3. Best Practices:
   - Use one embedding model per collection
   - Choose model based on primary use case
   - Consider computational resources and speed
   - Test different models for your specific needs
""")

# Run demonstrations
demonstrate_embedding_models_comparison()
demonstrate_mixed_embeddings()
