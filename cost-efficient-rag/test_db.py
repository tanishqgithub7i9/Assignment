from app.vector_store import collection

print("=" * 50)
print("ChromaDB Verification")
print("=" * 50)

print(f"Collection Name: {collection.name}")
print(f"Total Vectors: {collection.count()}")

results = collection.get()

print("\nStored IDs:")
print(results["ids"])

print("\nStored Metadata:")
print(results["metadatas"])