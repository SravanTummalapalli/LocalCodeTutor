import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

VECTOR_STORE_DIR = "vector_store"

# Local, ONNX-based embedding model — no API key, no network calls per-request.
# Matches the model used in rag_pipeline.py; keep these in sync or the FAISS
# index built here won't match the vectors produced at query time.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Free tier: 100 requests/minute for embed_content. Stay comfortably under that.
BATCH_SIZE = 20          # chunks embedded per batch
SECONDS_BETWEEN_BATCHES = 15  # 4 batches/min max => well under 100 req/min


def get_embeddings():
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(multiplier=1, min=31, max=120),
    stop=stop_after_attempt(5),
)
def _embed_batch_with_retry(db_or_none, embeddings, batch_chunks):
    """Embed one batch, retrying on 429 with backoff. Returns a FAISS store for this batch."""
    return FAISS.from_documents(batch_chunks, embeddings)


def build_vector_store(pdf_path: str):
    """Build a FAISS vector store from a PDF using local embeddings.

    No API key, no rate limits, no daily quota — so unlike the old
    Google-embeddings version, this runs as a single pass with no batching,
    retry-on-429 logic, or resumable progress file needed.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs = PyPDFLoader(pdf_path).load()
    if not docs:
        raise ValueError(f"No content could be extracted from PDF: {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("PDF produced no text chunks. May be a scanned PDF.")

    print(f"Embedding {len(chunks)} chunks locally with '{EMBEDDING_MODEL}' (fastembed)...")
    embeddings = get_embeddings()

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_STORE_DIR)

    print(f"✅ Vector store built: {len(chunks)} chunks saved to '{VECTOR_STORE_DIR}'")
    return db