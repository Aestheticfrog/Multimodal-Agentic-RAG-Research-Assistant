"""ChromaDB Vector Store Management and Hybrid Retriever Initialization."""
import os
import logging
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from backend.app.models.llm import get_gemini_embeddings

logger = logging.getLogger("researchpilot.vectorstore")

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./database/chroma_db")
_vector_store_instance: Optional[Chroma] = None


def get_vector_store() -> Chroma:
    """Returns singleton instance of ChromaDB vector store with zero-crash fallback."""
    global _vector_store_instance
    if _vector_store_instance is None:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        try:
            embeddings = get_gemini_embeddings()
            _vector_store_instance = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings,
                collection_name="research_papers",
            )
            logger.info("ChromaDB vector store initialized with Gemini embeddings.")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini embeddings: {e}. Falling back to default ONNX embeddings.")
            _vector_store_instance = Chroma(
                persist_directory=f"{PERSIST_DIR}_default",
                collection_name="research_papers_default",
            )
            logger.info("ChromaDB vector store initialized with default ONNX embeddings.")

    return _vector_store_instance


def add_documents_to_vector_store(documents: List[Document]) -> int:
    """Ingests list of LangChain Document objects into ChromaDB."""
    vs = get_vector_store()
    try:
        vs.add_documents(documents)
        logger.info(f"Successfully added {len(documents)} document chunks to ChromaDB.")
        return len(documents)
    except Exception as e:
        logger.warning(f"Embedding error during Chroma ingestion: {e}. Switching to Chroma default ONNX embeddings...")
        
        # Zero-dependency ONNX fallback
        fallback_vs = Chroma(
            persist_directory=f"{PERSIST_DIR}_default",
            collection_name="research_papers_default",
        )
        fallback_vs.add_documents(documents)
        global _vector_store_instance
        _vector_store_instance = fallback_vs
        logger.info(f"Successfully added {len(documents)} chunks to default ONNX vector store.")
        return len(documents)


def get_retriever(k: int = 16) -> Optional[VectorStoreRetriever]:
    """Returns LangChain retriever interface for vector store."""
    try:
        vs = get_vector_store()
        return vs.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        logger.error(f"Error getting retriever: {e}")
        return None
