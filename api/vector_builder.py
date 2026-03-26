import os
import nomic
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nomic import NomicEmbeddings


VECTOR_STORE_DIR = "vector_store"
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY", "")


def get_embeddings():
    # ✅ Explicitly authenticate before using the API
    nomic.login(NOMIC_API_KEY)
    return NomicEmbeddings(
        model="nomic-embed-text-v1.5",
        inference_mode="remote",
    )


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

    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    db.save_local(VECTOR_STORE_DIR)

    print(f"✅ Vector store built: {len(chunks)} chunks saved to '{VECTOR_STORE_DIR}'")
    return db