from app.ingest import load_documents

documents = load_documents()

print(f"\nFound {len(documents)} document(s)\n")

for doc in documents:
    print("=" * 60)
    print(f"File: {doc['filename']}")
    print(f"Number of Chunks: {len(doc['chunks'])}")

    print("\nFirst Chunk:\n")

    print(doc["chunks"][0])

    print()