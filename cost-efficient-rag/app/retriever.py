from app.vector_store import collection
from app.embeddings import generate_embedding


def retrieve(query, k=3, metadata_filter=None):
    """
    Retrieve top-k most relevant text chunks, with optional metadata filtering.
    """

    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=metadata_filter
    )

    return results