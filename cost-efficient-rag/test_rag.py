from app.retriever import retrieve
from app.llm import generate_answer

question = "How many sick leaves does an employee get?"

results = retrieve(question, k=3)

chunks = results["documents"][0]

answer = generate_answer(question, chunks)

print("=" * 60)
print("QUESTION")
print("=" * 60)
print(question)

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)
print(answer)

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for meta in results["metadatas"][0]:
    print(meta["filename"])