import chromadb

# Understanding Vectors and Embeddings in Vector Databases
# ========================================================
#
# Vector: A fixed-length array of numbers (e.g., [0.2, -0.5, 0.8, ...]) that represents data
#        mathematically in high-dimensional space.
#
# Embedding: The process and result of converting data (like text) into vectors. For example:
#           Text: "I love dogs" -> Vector: [0.2, -0.5, 0.8, ...]
#           ChromaDB uses the 'sentence-transformers/all-MiniLM-L6-v2' model to create
#           384-dimensional vectors for each text.

# Initialize the client
chroma_client = chromadb.Client()

# Create a collection - ChromaDB will use default embedding function
collection = chroma_client.create_collection(name="example01_collection")

# Sample texts to analyze
TEXT_ABOUT_PHYSICS = ("The equivalence principle is the hypothesis that the observed equivalence of gravitational and "
                      "inertial mass is a consequence of nature. The weak form, known for centuries, relates to "
                      "masses of any composition in free fall taking the same trajectories and landing at identical "
                      "times.")

TEXT_ABOUT_IT = ("Some online sites offer customers the ability to use a six-digit code which randomly changes every "
                 "30–60 seconds on a physical security token. The token has built-in computations and manipulates "
                 "numbers based on the current time. This means that every thirty seconds only a certain array of "
                 "numbers validate access.")

# Add documents to collection
collection.add(
    documents=[TEXT_ABOUT_PHYSICS, TEXT_ABOUT_IT],
    ids=["physics_text", "it_text"]
)


def print_search_results(query_text):
    print("\nSemantic Search Demo")
    print("===================")
    print("\nChromaDB performs semantic search using text embeddings.")
    print("It finds similar documents based on meaning, not just keyword matching.\n")

    print(f"Query: '{query_text}'")

    results = collection.query(
        query_texts=[query_text],
        n_results=2
    )

    print("\nResults:")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
        print(f"\n{i}. Distance: {distance:.4f}")
        print(f"   Text: {doc[:100]}...")


# Try different queries to demonstrate semantic search
print_search_results("Bill Gates")
print("\n" + "=" * 50 + "\n")
print_search_results("Richard Feynman")
