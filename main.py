from contextlib import asynccontextmanager
from uuid import uuid4
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api import load_rag_pipeline, build_vector_store
import api

# ✅ PDF path — uses environment variable on Render, falls back to local path
PDF_PATH = os.getenv("PDF_PATH", "data/Fluent_Python_by_Luciano_Ramalho.pdf")

# In-memory session store: { session_id: [ {role, content}, ... ] }
chat_sessions: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once at startup."""
    try:
        load_rag_pipeline()
        print("✅ RAG pipeline loaded successfully.")
    except Exception as e:
        print(f"⚠️  RAG pipeline failed to load: {e}")
        print("   /chat will be unavailable until /build_vectors is called.")
    yield


# ── CORS — allow frontend to call backend ──────────────────────────────────
# Add your frontend URLs here (local + deployed)
ALLOWED_ORIGINS = [
    "http://localhost:5173",      # Vite local dev
    "http://localhost:3000",      # alternate local
    "https://*.netlify.app",      # Netlify deploy
    "https://*.vercel.app",       # Vercel deploy
    "*",                          # Allow all (remove in production)
]

app = FastAPI(
    title="Python RAG Chatbot API",
    description="""
## 🤖 Python Interview Prep Chatbot

A RAG-powered chatbot using FAISS + Groq to answer Python interview questions.

---

### 🚀 How to use this API (follow this order):

1. **`POST /build_vectors`** — Index your PDF (run once)
2. **`POST /session/new`** — Create a chat session, get a `session_id`
3. **`POST /chat`** — Ask questions (full response)
4. **`POST /chat/stream`** — Ask questions (streamed word by word)
5. **`GET /session/{session_id}/history`** — View your chat history
6. **`POST /session/{session_id}/clear`** — Clear chat history

---
""",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],       # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],       # Allow all headers
)


# ─────────────────────────────────────────────
# /build_vectors
# ─────────────────────────────────────────────
@app.post(
    "/build_vectors",
    summary="📄 Step 1 — Build vector store from PDF",
    tags=["Setup"],
)
async def build_vectors():
    """Build FAISS vector store from the PDF path."""
    try:
        build_vector_store(PDF_PATH)
        load_rag_pipeline()
        return {
            "status": "success",
            "message": f"Vector store created from '{PDF_PATH}' and pipeline reloaded",
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"PDF not found at path: '{PDF_PATH}'. Check PDF_PATH environment variable.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────
# /session/new
# ─────────────────────────────────────────────
@app.post(
    "/session/new",
    summary="🆕 Step 2 — Create a new chat session",
    tags=["Session"],
)
async def new_session():
    """
    Create a new chat session.
    Returns a **session_id** — copy this and use it in the **/chat** endpoint.
    """
    session_id = str(uuid4())
    chat_sessions[session_id] = []
    return {
        "session_id": session_id,
        "message": "Session created! Copy the session_id and use it in /chat or /chat/stream",
    }


# ─────────────────────────────────────────────
# /session/{session_id}/history
# ─────────────────────────────────────────────
@app.get(
    "/session/{session_id}/history",
    summary="📜 View chat history for a session",
    tags=["Session"],
)
async def get_history(session_id: str):
    """Return the full chat history for a session."""
    if session_id not in chat_sessions:
        return {"error": "Session not found. Create one via /session/new"}
    return {
        "session_id": session_id,
        "history": chat_sessions[session_id],
        "total_turns": len(chat_sessions[session_id]) // 2,
    }


# ─────────────────────────────────────────────
# /session/{session_id}/clear
# ─────────────────────────────────────────────
@app.post(
    "/session/{session_id}/clear",
    summary="🗑️ Clear chat history for a session",
    tags=["Session"],
)
async def clear_history(session_id: str):
    """Clear the chat history for a session."""
    if session_id not in chat_sessions:
        return {"error": "Session not found. Create one via /session/new"}
    chat_sessions[session_id] = []
    return {"status": "success", "message": "Chat history cleared"}


# ─────────────────────────────────────────────
# Shared request model
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        json_schema_extra={"example": "What is Python inheritance?"},
        description="The Python topic or question you want to ask",
    )
    session_id: str = Field(
        ...,
        json_schema_extra={"example": "paste-your-session-id-here"},
        description="Session ID from /session/new — required to maintain chat history",
    )


# ─────────────────────────────────────────────
# /chat — full response
# ─────────────────────────────────────────────
@app.post(
    "/chat",
    summary="💬 Step 3 — Ask a question (full response)",
    tags=["Chat"],
)
async def chat(req: ChatRequest):
    """
    Ask a Python interview question and get the full response at once.
    1. Call `/session/new` to get a `session_id`
    2. Paste that `session_id` in the request body below
    """
    if api.rag_pipeline is None:
        return {"error": "RAG pipeline is not initialized. Please call /build_vectors first."}
    if req.session_id not in chat_sessions:
        return {"error": "Invalid session_id. Please create a session via /session/new first."}

    history = chat_sessions[req.session_id]
    try:
        result = api.rag_pipeline.invoke({
            "question": req.question,
            "history": history,
        })
        answer = result.content
        history.append({"role": "user",      "content": req.question})
        history.append({"role": "assistant", "content": answer})
        return {
            "session_id": req.session_id,
            "question": req.question,
            "answer": answer,
            "total_turns": len(history) // 2,
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "detail": traceback.format_exc()}


# ─────────────────────────────────────────────
# /chat/stream — streaming response
# ─────────────────────────────────────────────
@app.post(
    "/chat/stream",
    summary="⚡ Step 3 (alt) — Ask a question (streamed word by word)",
    tags=["Chat"],
)
async def chat_stream(req: ChatRequest):
    """
    Ask a Python interview question and get the answer streamed word by word like ChatGPT.
    1. Call `/session/new` to get a `session_id`
    2. Paste that `session_id` in the request body below
    """
    if api.rag_pipeline is None:
        async def error_stream():
            yield "RAG pipeline is not initialized. Please call /build_vectors first."
        return StreamingResponse(error_stream(), media_type="text/plain")

    if req.session_id not in chat_sessions:
        async def error_stream():
            yield "Invalid session_id. Please create a session via /session/new first."
        return StreamingResponse(error_stream(), media_type="text/plain")

    history = chat_sessions[req.session_id]

    async def stream_generator():
        full_answer = ""
        try:
            async for chunk in api.rag_pipeline.astream({
                "question": req.question,
                "history": history,
            }):
                token = chunk.content
                full_answer += token
                yield token
            history.append({"role": "user",      "content": req.question})
            history.append({"role": "assistant", "content": full_answer})
        except Exception as e:
            import traceback
            yield f"\n\n[Error]: {str(e)}\n\n[Traceback]:\n{traceback.format_exc()}"

    return StreamingResponse(stream_generator(), media_type="text/plain")