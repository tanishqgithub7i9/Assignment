from app.ingest import load_documents
from app.embeddings import generate_embedding
from app.vector_store import collection

def ingest_all_documents():
    documents = load_documents()
    count = 0

    for doc in documents:
        filename = doc["filename"]

        # Ensure idempotency by deleting any existing chunks of this document
        try:
            collection.delete(where={"filename": filename})
        except Exception as e:
            # Silence error if delete fails (e.g. collection is empty or not initialized)
            pass

        for i, chunk in enumerate(doc["chunks"]):
            vector = generate_embedding(chunk)
            collection.add(
                ids=[f"{filename}_{i}"],
                documents=[chunk],
                embeddings=[vector],
                metadatas=[
                    {
                        "filename": filename,
                        "chunk": i
                    }
                ]
            )
            count += 1

    print(f"\nStored {count} chunks successfully!")
    print("Total vectors in database:", collection.count())
    return count

if __name__ == "__main__":
    ingest_all_documents()