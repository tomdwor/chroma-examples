import chromadb
import numpy as np

# Initialize the client
chroma_client = chromadb.Client()

# Create a collection - ChromaDB will use default embedding function
collection = chroma_client.create_collection(name="vector_example")

# Sample texts to analyze
TEXT_ABOUT_PHYSICS = ("The equivalence principle is the hypothesis that the observed equivalence of gravitational and "
                     "inertial mass is a consequence of nature. The weak form, known for centuries, relates to "
                     "masses of any composition in free fall taking the same trajectories and landing at identical "
                     "times.")

TEXT_ABOUT_IT = ("Some online sites offer customers the ability to use a six-digit code which randomly changes every "
                 "30–60 seconds on a physical security token. The token has built-in computations and manipulates "
                 "numbers based on the current time. This means that every thirty seconds only a certain array of "
                 "numbers validate access.")

SIMILAR_PHYSICS_TEXT = ("Gravity makes objects fall at the same speed regardless of their mass. This principle of "
                       "equivalence is fundamental to our understanding of gravitational forces.")

SIMILAR_IT_TEXT = ("Two-factor authentication often uses time-based tokens that generate temporary codes. "
                   "These security devices create new passwords every minute using cryptographic algorithms "
                   "synchronized with authentication servers.")

# Add all documents to collection
collection.add(
    documents=[TEXT_ABOUT_PHYSICS, TEXT_ABOUT_IT, SIMILAR_PHYSICS_TEXT, SIMILAR_IT_TEXT],
    ids=["physics_text", "it_text", "similar_physics", "similar_it"]
)

# Get embeddings for all texts
results = collection.get(
    ids=["physics_text", "it_text", "similar_physics", "similar_it"],
    include=['embeddings', 'documents']
)

# Extract and analyze vectors
vectors = np.array(results['embeddings'])

print("\nVector Analysis")
print("===============")
print("\nChromaDB uses 'sentence-transformers/all-MiniLM-L6-v2' as the default embedding function.")
print("This model creates vectors with 384 dimensions, where each dimension represents different")
print("semantic aspects of the text. The model is optimized for semantic similarity tasks.\n")

print(f"Vector dimensions: {vectors.shape[1]}")
print(f"\nExample - first 10 dimensions of vectors:")
for idx, name in enumerate(["Physics", "IT", "Similar Physics", "Similar IT"]):
    print(f"\n{name} text vector:")
    print(f"    {vectors[idx, :10].round(4)}")
    print(f"    (... and {vectors.shape[1] - 10} more dimensions)")

# Demonstrate similarity search for both domains
def print_search_results(query_text, label):
    print(f"\nSimilarity Search Results for {label}")
    print("==============================" + len(label)*"=")
    print("\nChromaDB uses cosine similarity as the default distance metric for searching.")
    print("Cosine similarity measures the angle between vectors, with smaller angles (lower distance)")
    print("indicating greater similarity. The distance ranges from 0 (identical) to 2 (opposite).\n")
    results = collection.query(
        query_texts=[query_text],
        n_results=4
    )
    print(f"\nQuery text: {query_text[:100]}...")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
        print(f"\n{i}. Distance: {distance:.4f}")
        print(f"   {doc[:100]}...")

# Search with both texts to show how distance correlates with meaning
print_search_results(TEXT_ABOUT_PHYSICS, "Physics Text")
print_search_results(TEXT_ABOUT_IT, "IT Text")

print("\nVector Properties")
print("=================")
print("\nThese properties help understand how the embedding model represents text in vector space.")
print("Normalized vectors (magnitude ≈ 1.0) ensure consistent scaling across different text lengths.")
print("Mean and standard deviation show how the semantic information is distributed across dimensions.\n")
for idx, name in enumerate(["Physics", "IT", "Similar Physics", "Similar IT"]):
    vector = vectors[idx]
    magnitude = np.linalg.norm(vector)
    print(f"\n{name} text vector:")
    print(f"    Magnitude (length): {magnitude:.4f}")  # Should be close to 1.0
    print(f"    Mean value: {vector.mean():.4f}")
    print(f"    Standard deviation: {vector.std():.4f}")
