from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL

# Load embedding model once
model = SentenceTransformer(EMBEDDING_MODEL)

def generate_embedding(text):
    return model.encode(text).tolist()