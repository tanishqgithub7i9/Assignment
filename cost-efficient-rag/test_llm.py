from app.llm import generate_answer

context = """
Employee leave policy: Employees get 12 paid leaves per year.
Unused leaves can be carried forward.
"""

question = "How many paid leaves do employees get?"

response = generate_answer(question, context)

print("\n===== LLM RESPONSE =====\n")
print(response)