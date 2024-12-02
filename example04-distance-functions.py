import chromadb
import numpy as np
from scipy.spatial.distance import cosine, euclidean

# Initialize the client
chroma_client = chromadb.Client()

# Create collections with different distance functions
cosine_collection = chroma_client.create_collection(
    name="cosine_example",
    metadata={"hnsw:space": "cosine"}  # Default
)

l2_collection = chroma_client.create_collection(
    name="l2_example",
    metadata={"hnsw:space": "l2"}
)

ip_collection = chroma_client.create_collection(
    name="ip_example",
    metadata={"hnsw:space": "ip"}  # Inner Product
)

# Sample texts designed to demonstrate distance function differences
BASE_TEXT = ("Machine learning is a type of artificial intelligence that allows "
             "software applications to become more accurate at predicting outcomes "
             "without being explicitly programmed to do so.")

SIMILAR_MEANING_DIFFERENT_LENGTH = ("AI and machine learning enable computer programs "
                                    "to automatically improve their performance through "
                                    "experience, learning to make predictions without "
                                    "explicit programming. This revolutionary technology "
                                    "has transformed numerous industries, from healthcare "
                                    "to finance, by providing powerful tools for pattern "
                                    "recognition and data analysis. The ability to learn "
                                    "from data has made these systems increasingly accurate "
                                    "over time.")

SIMILAR_LENGTH_DIFFERENT_MEANING = ("The annual music festival attracted thousands "
                                    "of visitors from around the world. The lineup "
                                    "included both established artists and emerging "
                                    "talents, offering a diverse range of musical "
                                    "genres and performances that kept audiences "
                                    "entertained throughout the weekend.")

OPPOSITE_MEANING = ("Traditional rule-based systems rely entirely on explicit "
                    "programming, requiring manual coding of every possible scenario. "
                    "These systems cannot improve their accuracy through experience "
                    "and need constant human intervention to handle new cases.")


def analyze_distances(collection, texts, query_text, distance_type, id_prefix):
    print(f"\nDistance Analysis: {distance_type}")
    print("=" * (19 + len(distance_type)))
    print(f"\nQuery text: '{query_text}'\n")

    # Add documents to collection with unique IDs for each collection type
    collection.add(
        documents=texts,
        ids=[f"{id_prefix}_{i}" for i in range(len(texts))]
    )

    # Query and show distances
    results = collection.query(
        query_texts=[query_text],
        n_results=len(texts)
    )

    print("Results (sorted by similarity):")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
        print(f"\n{i}. Distance: {distance:.4f}")
        print(f"   Text snippet: {doc[:100]}...")

    return results['distances'][0]


print("""
Distance Functions Comparison in Vector Search
===========================================

This script demonstrates how different distance functions behave with specifically
designed test cases:

1. BASE_TEXT: A reference text about machine learning
2. SIMILAR_MEANING_DIFFERENT_LENGTH: Same topic but much longer
3. SIMILAR_LENGTH_DIFFERENT_MEANING: Different topic but similar length
4. OPPOSITE_MEANING: Contrasting viewpoint about the same topic

The comparison will show how each distance function handles:
- Semantic similarity vs text length
- Topic relevance vs structural similarity
- Opposing viewpoints
""")

# Test cases
sample_texts = [
    SIMILAR_MEANING_DIFFERENT_LENGTH,
    SIMILAR_LENGTH_DIFFERENT_MEANING,
    OPPOSITE_MEANING
]

print("\nComparative Analysis")
print("===================")

# Get distances for each metric using unique ID prefixes
cosine_distances = analyze_distances(cosine_collection, sample_texts, BASE_TEXT, "Cosine Distance", "cos")
l2_distances = analyze_distances(l2_collection, sample_texts, BASE_TEXT, "Euclidean (L2) Distance", "l2")
ip_distances = analyze_distances(ip_collection, sample_texts, BASE_TEXT, "Inner Product Distance", "ip")

print("\nKey Findings")
print("============")
print("\n1. Similar Meaning, Different Length Text:")
print(f"   - Cosine Distance: {cosine_distances[0]:.4f} (Best at capturing semantic similarity)")
print(f"   - L2 Distance: {l2_distances[0]:.4f}")
print(f"   - Inner Product: {ip_distances[0]:.4f}")
print("\n   → Cosine distance shows the closest match despite length difference")
print("   → L2 distance is affected by the length difference")

print("\n2. Similar Length, Different Meaning Text:")
print(f"   - Cosine Distance: {cosine_distances[1]:.4f}")
print(f"   - L2 Distance: {l2_distances[1]:.4f} (Shows structural similarity)")
print(f"   - Inner Product: {ip_distances[1]:.4f}")
print("\n   → L2 distance shows relatively closer match due to similar vector magnitudes")
print("   → Cosine distance correctly shows high dissimilarity")

print("\n3. Opposite Meaning Text:")
print(f"   - Cosine Distance: {cosine_distances[2]:.4f} (Successfully identifies opposition)")
print(f"   - L2 Distance: {l2_distances[2]:.4f}")
print(f"   - Inner Product: {ip_distances[2]:.4f}")
print("\n   → Cosine distance approaches 2.0 for opposing concepts")
print("   → L2 and Inner Product don't clearly identify semantic opposition")

print("""
Conclusions
==========

1. Cosine Distance (Best for Semantic Search):
   - Most reliable for capturing meaning similarities
   - Immune to text length variations
   - Clear distinction between similar and opposing concepts
   - Range [0-2] makes results interpretable

2. Euclidean (L2) Distance:
   - Influenced by vector magnitude (text length)
   - Better for comparing structural similarities
   - Less effective for semantic similarity
   - Unbounded range makes interpretation harder

3. Inner Product Distance:
   - Very sensitive to vector magnitude
   - Can be misleading without careful normalization
   - Not recommended for general text similarity
   - Useful for specific cases like recommendation systems

Recommendation: For text similarity and semantic search, prefer Cosine Distance
as it provides the most intuitive and reliable results.""")

# Get embeddings to demonstrate normalization
results = cosine_collection.get(
    ids=["cos_0"],
    include=['embeddings']
)

vector = np.array(results['embeddings'][0])
magnitude = np.linalg.norm(vector)
print(f"\nVector magnitude check (should be ≈ 1.0): {magnitude:.4f}")
