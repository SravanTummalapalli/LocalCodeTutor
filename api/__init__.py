from api.vector_builder import build_vector_store
from api.rag_pipeline import get_rag_pipeline
 
rag_pipeline = None
 
 
def load_rag_pipeline():
    """Initialize the RAG pipeline and store it globally."""
    global rag_pipeline
    rag_pipeline = get_rag_pipeline()
 