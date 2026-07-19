import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.api_core.exceptions import ResourceExhausted

VECTOR_STORE_DIR = "vector_store"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Free tier: 100 requests/minute for embed_content. Stay comfortably under that.
BATCH_SIZE = 20          # chunks embedded per batch
SECONDS_BETWEEN_BATCHES = 15  # 4 batches/min max => well under 100 req/min


def get_embeddings():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )


@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(multiplier=1, min=31, max=120),
    stop=stop_after_attempt(5),
)
def _embed_batch_with_retry(db_or_none, embeddings, batch_chunks):
    """Embed one batch, retrying on 429 with backoff. Returns a FAISS store for this batch."""
    return FAISS.from_documents(batch_chunks, embeddings)


def build_vector_store(pdf_path: str):
    """Build a FAISS vector store from a PDF and save it to disk, respecting free-tier rate limits."""
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
        raise ValueError("PDF produced no text chunks. May be a scanned PDF.")

    embeddings = get_embeddings()

    db = None
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        batch_db = _embed_batch_with_retry(db, embeddings, batch)

        if db is None:
            db = batch_db
        else:
            db.merge_from(batch_db)

        # Throttle between batches (skip the wait after the very last batch)
        if batch_num < total_batches:
            time.sleep(SECONDS_BETWEEN_BATCHES)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    db.save_local(VECTOR_STORE_DIR)

    print(f"✅ Vector store built: {len(chunks)} chunks saved to '{VECTOR_STORE_DIR}'")
    return db