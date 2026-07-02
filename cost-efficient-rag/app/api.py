import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

from app.retriever import retrieve
from app.llm import generate_answer
from app.store import ingest_all_documents
from app.config import TOP_K

app = FastAPI(title="Cost-Efficient RAG Application API")

class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = TOP_K
    filename_filter: Optional[str] = None

class SourceInfo(BaseModel):
    filename: str
    chunk: int

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    metrics: Dict[str, Any]

@app.post("/ingest")
def trigger_ingestion():
    try:
        count = ingest_all_documents()
        return {"status": "success", "chunks_ingested": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest):
    start_time = time.perf_counter()
    
    # Setup metadata filter
    metadata_filter = None
    if request.filename_filter:
        metadata_filter = {"filename": request.filename_filter}
        
    try:
        # 1. Retrieve relevant chunks
        results = retrieve(request.query, k=request.k, metadata_filter=metadata_filter)
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        # 2. Generate answer
        answer = generate_answer(request.query, documents, metadatas)
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        # 3. Compile sources
        sources = []
        for meta in metadatas:
            sources.append(SourceInfo(
                filename=meta.get("filename", "unknown"),
                chunk=meta.get("chunk", 0)
            ))
            
        # 4. Token usage estimation (1 token ~ 4 chars)
        prompt_est = len(request.query + "\n".join(documents)) // 4
        completion_est = len(answer) // 4
        total_tokens = prompt_est + completion_est
        
        metrics = {
            "latency_ms": round(latency_ms, 2),
            "chunk_count": len(documents),
            "estimated_token_usage": total_tokens
        }
        
        # Log to stdout/console as required
        print(f"[QUERY LOG] Query: '{request.query}' | Latency: {metrics['latency_ms']}ms | Chunks: {metrics['chunk_count']} | Est. Tokens: {metrics['estimated_token_usage']}")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            metrics=metrics
        )
        
    except Exception as e:
        print(f"[ERROR LOG] Failed query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
