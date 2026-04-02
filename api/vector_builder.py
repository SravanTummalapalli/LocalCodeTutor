import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


VECTOR_STORE_DIR = "vector_store"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
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
        raise ValueError("PDF produced no text chunks. May be a scanned PDF.")

    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    db.save_local(VECTOR_STORE_DIR)

    print(f"✅ Vector store built: {len(chunks)} chunks saved to '{VECTOR_STORE_DIR}'")
    return db