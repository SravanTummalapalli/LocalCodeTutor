from fastapi import FastAPI
from pydantic import BaseModel
from api.rag_pipeline import rag_pipeline, load_rag_pipeline
from api.vector_builder import build_vector_store

app = FastAPI(
    title="Offline Python RAG Chatbot API",
    description="FAISS + Ollama powered Python tutor",
    version="1.0.0"
)

class BuildRequest(BaseModel):
    pdf_path: str


@app.post("/build_vectors")
async def build_vectors(req: BuildRequest):
    """Build FAISS vector store from a PDF."""
    try:
        build_vector_store(req.pdf_path)
        return {"status": "success", "message": "Vector store created"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(req: ChatRequest):
    """Returns answer from offline RAG chatbot."""
    try:
        result = rag_pipeline.invoke(req.question)
        return {"answer": result.content}
    except Exception as e:
        return {"error": str(e)}
