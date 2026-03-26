import os
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class HFEmbeddings(Embeddings):
    """
    HuggingFace SentenceTransformer embeddings
    compatible with LangChain + FAISS.
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_documents(self, texts):
        vectors = self.model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text):
        vector = self.model.encode([text], show_progress_bar=False)
        return vector[0].tolist()

    def __call__(self, texts):
        """Backward-compatible callable interface expected by some vectorstore code."""
        if isinstance(texts, str):
            return self.embed_query(texts)
        if isinstance(texts, (list, tuple)):
            return self.embed_documents(texts)
        try:
            return self.embed_documents(list(texts))
        except Exception:
            raise TypeError("HFEmbeddings input must be str or list-like")


def build_vector_store(pdf_path: str):
    """Build a FAISS vector store from a PDF and save it to disk.

    Returns the created FAISS DB object.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs = PyPDFLoader(pdf_path).load()

    if not docs:
        raise ValueError(f"No content could be extracted from PDF: {pdf_path}")

    # ✅ Increased chunk_size from 300 → 800 for richer technical context
    # ✅ Increased chunk_overlap from 50 → 100 to preserve context across chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    # ✅ Guard against empty chunks (e.g. scanned/image-only PDFs)
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