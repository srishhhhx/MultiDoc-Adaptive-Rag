"""
Document reranking using cross-encoder models

This module implements document reranking using a cross-encoder model from
sentence-transformers. It takes a larger pool of candidate documents and
reranks them based on their relevance to the query, returning only the
top-k most relevant documents.
"""

import torch
from sentence_transformers import CrossEncoder
from typing import List
import logging

logger = logging.getLogger(__name__)


# --- Singleton Model Initialization ---
# Initialize the reranker model once in the global scope to avoid reloading on every call.
# This addresses the "cold start" performance issue.

_reranker_model = None


def _initialize_reranker():
    """Initializes the CrossEncoder model and caches it globally."""
    global _reranker_model
    if _reranker_model is None:
        try:
            model_name = "BAAI/bge-reranker-base"
            # Check for Apple Silicon GPU and set device
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info(
                    "MPS backend is available. Using Apple Silicon GPU for reranker."
                )
            else:
                device = "cpu"
                logger.info(
                    "MPS backend not available. Falling back to CPU for reranker."
                )

            logger.info(f"Loading reranker model: {model_name} on device: {device}")
            _reranker_model = CrossEncoder(model_name, device=device)
            logger.info("Reranker model loaded successfully.")
            print(f"✅ CrossEncoder is running on device: {_reranker_model.device}")
        except Exception as e:
            logger.error(f"Fatal error: Failed to load reranker model: {e}")
            # If the model fails to load, the application cannot proceed.
            raise RuntimeError("Could not initialize the reranker model.") from e


# Call the initialization function when the module is loaded.
_initialize_reranker()

# --- Reranking Function ---


def rerank_documents(query: str, documents: List, top_k: int = 5) -> List:
    """
    Reranks documents based on their relevance to the query using the pre-initialized model.

    Args:
        query: The user's query.
        documents: A list of document objects with a `page_content` attribute.
        top_k: The number of top documents to return.

    Returns:
        A list of the top-k reranked documents.
    """
    if not documents:
        logger.warning("No documents provided for reranking.")
        return []

    # If the number of documents is less than or equal to top_k, no reranking is needed.
    if len(documents) <= top_k:
        logger.info(
            f"Document count ({len(documents)}) is less than or equal to top_k ({top_k}). Skipping reranking."
        )
        return documents

    try:
        logger.info(f"Reranking {len(documents)} documents to return the top {top_k}.")

        # Prepare query-document pairs for the cross-encoder
        query_doc_pairs = [[query, doc.page_content] for doc in documents]

        # Get relevance scores from the pre-initialized cross-encoder model
        scores = _reranker_model.predict(query_doc_pairs)

        # Combine documents with their scores and sort in descending order
        doc_score_pairs = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )

        # Extract the top-k documents
        top_documents = [doc for doc, score in doc_score_pairs[:top_k]]

        # Log the reranking results for traceability
        logger.info("Top 5 reranking results:")
        for i, (doc, score) in enumerate(doc_score_pairs[:5]):  # Log top 5 for brevity
            doc_preview = doc.page_content[:120].replace("\n", " ")
            logger.info(
                f"  Rank {i + 1}: Score={score:.4f} | Preview: '{doc_preview}...'"
            )

        return top_documents

    except Exception as e:
        logger.error(f"An error occurred during document reranking: {e}")
        # As a fallback, return the original top_k documents if reranking fails
        logger.warning(
            f"Fallback: Returning the first {top_k} documents without reranking."
        )
        return documents[:top_k]
