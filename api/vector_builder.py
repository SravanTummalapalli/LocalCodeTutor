import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


def build_vector_store(pdf_path: str, store_dir="vector_store"):
    """Loads PDF, splits into chunks, generates FAISS DB."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("PDF not found: " + pdf_path)

    print("📚 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    print("🔤 Embedding (Offline Ollama)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("🧠 Building FAISS vector store...")
    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(store_dir, exist_ok=True)
    db.save_local(store_dir)

    print("✅ Vector store created:", store_dir)
    return True


if __name__ == "__main__":
    print("🔥 MAIN BLOCK RUNNING")

    PDF_PATH = r"C:\Users\sravan\Documents\LocalCodeTutor\LocalCodeTutor\data\python_full_notes.pdf"
    
    print(f"📄 Using PDF: {PDF_PATH}")
    build_vector_store(PDF_PATH)
