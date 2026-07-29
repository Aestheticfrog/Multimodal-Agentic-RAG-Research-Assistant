"""ChromaDB Vector Store Management with Per-Source Metadata Filtered Retrieval (GitHub Gold-Standard Pattern)."""
import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional, Dict

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger("researchpilot.vectorstore")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(ROOT_DIR / "database" / "chroma_store"))
_vector_store_instance: Optional[Chroma] = None


def get_vector_store() -> Chroma:
    """Returns singleton instance of ChromaDB vector store using unified local persistence."""
    global _vector_store_instance
    if _vector_store_instance is None:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        _vector_store_instance = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name="unified_research_papers",
        )
        logger.info("ChromaDB vector store initialized with unified local ONNX persistence.")

    return _vector_store_instance


def add_documents_to_vector_store(documents: List[Document]) -> int:
    """Ingests list of LangChain Document objects into ChromaDB unified store with unique IDs."""
    vs = get_vector_store()
    try:
        ids = [
            f"{doc.metadata.get('source', 'paper')}_p{doc.metadata.get('page', 1)}_{i}_{uuid.uuid4().hex[:6]}"
            for i, doc in enumerate(documents)
        ]
        vs.add_documents(documents, ids=ids)
        logger.info(f"Successfully added {len(documents)} document chunks to unified ChromaDB.")
        return len(documents)
    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB: {e}")
        return len(documents)


def get_indexed_sources_summary() -> Dict[str, int]:
    """Inspects ChromaDB and returns a dict mapping source filenames to chunk counts."""
    vs = get_vector_store()
    try:
        data = vs._collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        counts: Dict[str, int] = {}
        for m in metadatas:
            if m and "source" in m:
                src = str(m["source"])
                counts[src] = counts.get(src, 0) + 1
        return counts
    except Exception as e:
        logger.error(f"Error getting indexed sources summary: {e}")
        return {}


def clear_vector_store():
    """Resets the ChromaDB collection completely."""
    global _vector_store_instance
    try:
        vs = get_vector_store()
        vs.delete_collection()
        _vector_store_instance = None
        logger.info("ChromaDB vector store cleared.")
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        _vector_store_instance = None


def retrieve_adaptive_documents(query: str) -> List[Document]:
    """Performs per-source metadata filtered retrieval for multi-paper comparison (GitHub LangChain pattern)."""
    vs = get_vector_store()
    if vs is None:
        return []

    comp_keywords = [
        "compare", "comparison", "difference", "between", "versus", "vs",
        "both", "all papers", "literature review", "synthesis", "different", "two", "differnce", "papers"
    ]
    is_comparative = any(kw in query.lower() for kw in comp_keywords)

    try:
        summary = get_indexed_sources_summary()
        sources = list(summary.keys())

        # If multiple papers are indexed and query is comparative, query EACH source PDF independently
        if is_comparative and len(sources) >= 2:
            combined_docs: List[Document] = []
            for src in sources:
                try:
                    # Filtered search for exact paper source
                    src_docs = vs.similarity_search(query, k=6, filter={"source": src})
                    if src_docs:
                        combined_docs.extend(src_docs)
                    else:
                        # Fallback: get top 6 chunks for this source directly
                        raw_data = vs._collection.get(where={"source": src}, limit=6, include=["documents", "metadatas"])
                        for txt, meta in zip(raw_data.get("documents", []), raw_data.get("metadatas", [])):
                            if txt:
                                combined_docs.append(Document(page_content=txt, metadata=meta or {}))
                except Exception as e:
                    logger.warning(f"Error querying source '{src}': {e}")
            
            if combined_docs:
                logger.info(f"Per-Source Filtered Retrieval: Retracted {len(combined_docs)} balanced chunks across {len(sources)} sources: {sources}")
                return combined_docs

        # Fallback for single paper or non-comparative query: fetch all or top-k
        raw_docs = vs.similarity_search(query, k=10)
        return raw_docs if raw_docs else []
    except Exception as e:
        logger.error(f"Error in document retrieval: {e}")
        return []


def get_retriever(k: int = 16) -> Optional[VectorStoreRetriever]:
    """Returns LangChain retriever interface for vector store."""
    try:
        vs = get_vector_store()
        return vs.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        logger.error(f"Error getting retriever: {e}")
        return None
