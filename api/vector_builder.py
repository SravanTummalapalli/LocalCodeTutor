import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import httpx


VECTOR_STORE_DIR = "vector_store"

# HuggingFace Inference API — free, no local model loading
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"


class HFEmbeddings(Embeddings):
    """
    HuggingFace Inference API embeddings — runs in the cloud.
    No local model loading, no torch, no 400MB memory usage.
    """

    def _embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
        response = httpx.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        # API returns list of vectors directly
        if isinstance(result[0], list) and isinstance(result[0][0], float):
            return result

        # Sometimes returns nested list — flatten one level
        return [r[0] if isinstance(r[0], list) else r for r in result]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Process in batches of 32 to avoid API limits
        all_vectors = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            all_vectors.extend(self._embed(batch))
        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def build_vector_store(pdf_path: str):
    """Build a FAISS vector store from a PDF and save it to disk."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs = PyPDFLoader(pdf_path).load()

    if not docs:
        raise ValueError(f"No content could be extracted from PDF: {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError(
            "PDF was loaded but produced no text chunks. "
            "It may be a scanned image PDF without OCR text."
        )

    embeddings = HFEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    db.save_local(VECTOR_STORE_DIR)

    print(f"✅ Vector store built: {len(chunks)} chunks saved to '{VECTOR_STORE_DIR}'")
    return db