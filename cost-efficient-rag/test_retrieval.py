from app.retriever import retrieve

query = "How many sick leaves does an employee get?"

results = retrieve(query, k=3)

print("=" * 60)
print("USER QUESTION")
print("=" * 60)
print(query)

print("\n" + "=" * 60)
print("TOP RETRIEVED CHUNKS")
print("=" * 60)

documents = results["documents"][0]
metadatas = results["metadatas"][0]

for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
    print(f"\nResult {i}")
    print(f"File: {meta['filename']}")
    print("-" * 50)
    print(doc)