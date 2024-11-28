import chromadb

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="example01_collection")

TEXT_ABOUT_PHYSICS = ("The equivalence principle is the hypothesis that the observed equivalence of gravitational and "
                      "inertial mass is a consequence of nature. The weak form, known for centuries, relates to "
                      "masses of any composition in free fall taking the same trajectories and landing at identical "
                      "times.")

TEXT_ABOUT_IT = ("Some online sites offer customers the ability to use a six-digit code which randomly changes every "
                 "30–60 seconds on a physical security token. The token has built-in computations and manipulates "
                 "numbers based on the current time. This means that every thirty seconds only a certain array of "
                 "numbers validate access.")

collection.add(
    documents=[
        TEXT_ABOUT_PHYSICS,
        TEXT_ABOUT_IT
    ],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["Bill Gates"],
    # query_texts=["Richard Feynman"],
    n_results=2  # how many results to return
)
print(results)
