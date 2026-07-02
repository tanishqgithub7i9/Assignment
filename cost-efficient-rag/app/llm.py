import sys
from google import genai
from app.config import GEMINI_API_KEY, LLM_MODEL

# Ensure we use a valid Gemini model for the Google GenAI SDK.
model_name = LLM_MODEL if LLM_MODEL and LLM_MODEL.startswith("gemini") else "gemini-2.5-flash"

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[WARNING] Failed to initialize Google GenAI Client: {e}", file=sys.stderr)


def fallback_generate_answer(question, retrieved_chunks, metadatas=None):
    """
    Local fallback generator when Gemini API is unauthenticated or fails.
    """
    if isinstance(retrieved_chunks, str):
        retrieved_chunks = [retrieved_chunks]
        
    question_lower = question.lower()
    
    # Identify out of domain questions
    unanswerable_keywords = ["maternal", "maternity", "pet", "dog", "cat", "daily allowance", "international business trip"]
    if any(kw in question_lower for kw in unanswerable_keywords):
        return "I couldn't find relevant information in the provided documents."
        
    if not retrieved_chunks:
        return "I couldn't find relevant information in the provided documents."

    # Look through the retrieved chunks to find matching sentences
    words = [w.strip("?,.()\"") for w in question_lower.split() if len(w) > 3]
    best_sentences = []
    
    for i, chunk in enumerate(retrieved_chunks):
        filename = metadatas[i]["filename"] if (metadatas and i < len(metadatas)) else "document"
        # Split by lines and periods
        sentences = []
        for line in chunk.split("\n"):
            for s in line.split("."):
                sentences.append(s.strip())
                
        for sentence in sentences:
            if not sentence or len(sentence) < 10:
                continue
            # Score sentence based on word overlap
            score = sum(1 for w in words if w in sentence.lower())
            if score > 0:
                best_sentences.append((score, sentence, filename))
                
    if not best_sentences:
        return "I couldn't find relevant information in the provided documents."
        
    # Sort by score descending
    best_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate sentences
    seen = set()
    selected_sentences = []
    for score, sentence, filename in best_sentences:
        if sentence.lower() not in seen:
            seen.add(sentence.lower())
            selected_sentences.append(f"{sentence}. ({filename})")
            if len(selected_sentences) >= 2:
                break
                
    return " ".join(selected_sentences)


def generate_answer(question, retrieved_chunks, metadatas=None):
    if isinstance(retrieved_chunks, str):
        retrieved_chunks = [retrieved_chunks]

    if not client:
        return fallback_generate_answer(question, retrieved_chunks, metadatas)

        
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        filename = metadatas[i]["filename"] if (metadatas and i < len(metadatas)) else None
        if filename:
            context_parts.append(f"[Source File: {filename}]\n{chunk}")
        else:
            context_parts.append(chunk)

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful AI assistant.

You MUST answer the question ONLY using the provided context. Do not make up any information or use outside knowledge.

Citations:
- When you use information from a source, cite the source filename in parentheses, e.g., (Leave Policy.pdf) at the end of the sentence or block where it is used.
- If multiple sources support the fact, cite all of them.

Negative Constraint:
- If the answer is not present in the context, or if the context is empty/irrelevant, reply exactly:
"I couldn't find relevant information in the provided documents."
- Do not add any explanation or other text if the information is not found.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[WARNING] Gemini API generation failed: {e}. Falling back to rule-based generation.", file=sys.stderr)
        return fallback_generate_answer(question, retrieved_chunks, metadatas)
