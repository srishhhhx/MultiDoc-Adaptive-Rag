"""
Document processing and vector database management using FAISS with hybrid search

This module handles document loading, text splitting, embedding generation,
FAISS vector database operations, and BM25 keyword search for the RAG system.
Implements hybrid search using Reciprocal Rank Fusion (RRF).
"""

import os
import pickle
import logging
import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from backend.config import FAISS_INDEX_DIR, FAISS_COLLECTION_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# **PERFORMANCE OPTIMIZATION: Auto-detect GPU availability**
def _get_optimal_device():
    """
    Auto-detect the best available device for embeddings
    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    """
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("🚀 CUDA GPU detected - using GPU acceleration for embeddings")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🚀 Apple Silicon (MPS) detected - using GPU acceleration for embeddings")
            return "mps"
        else:
            logger.info("💻 No GPU detected - using CPU for embeddings")
            return "cpu"
    except ImportError:
        logger.warning("⚠️  PyTorch not fully available - defaulting to CPU")
        return "cpu"


class DocumentProcessor:
    """Processes documents and creates embeddings for the FAISS vector database"""

    def __init__(self, document_loader):
        self.document_loader = document_loader

        # **PERFORMANCE OPTIMIZATION: Auto-detect best device (GPU if available)**
        optimal_device = _get_optimal_device()

        # **OPTIMIZATION 1: Upgraded embedding model for better semantic search**
        # Changed to BAAI/bge-base-en-v1.5 for optimal speed/accuracy balance
        # (Note: bge-large was too slow on CPU - 35s+ latency. Base provides 95% accuracy at 5x speed)
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": optimal_device},
            encode_kwargs={"normalize_embeddings": True}  # Normalize for better similarity scores
        )
        logger.info(f"🔧 Embedding model initialized: BAAI/bge-base-en-v1.5 on {optimal_device.upper()}")

        # Ensure FAISS directory exists
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        logger.info(f"FAISS index directory: {FAISS_INDEX_DIR}")

        # **NEW: Persistent chunk store infrastructure**
        self.chunk_store_dir = os.path.join(FAISS_INDEX_DIR, "chunk_stores")
        os.makedirs(self.chunk_store_dir, exist_ok=True)
        logger.info(f"Chunk store directory: {self.chunk_store_dir}")

        # **OPTIMIZATION 2: BM25 index storage for hybrid search**
        self.bm25_store_dir = os.path.join(FAISS_INDEX_DIR, "bm25_indexes")
        os.makedirs(self.bm25_store_dir, exist_ok=True)
        logger.info(f"BM25 index directory: {self.bm25_store_dir}")
        logger.info(f"🔧 Initialized with upgraded embedding model: {self.embedding_function.model_name}")

    def _get_chunk_store_path(self, collection_name: str) -> str:
        """Get the file path for a session's chunk store"""
        return os.path.join(self.chunk_store_dir, f"{collection_name}_chunks.pkl")

    def _load_chunk_store(self, collection_name: str) -> List[Document]:
        """
        Load existing chunks from the persistent chunk store
        
        Args:
            collection_name: Session collection name
            
        Returns:
            List of Document chunks or empty list if store doesn't exist
        """
        chunk_store_path = self._get_chunk_store_path(collection_name)
        
        if os.path.exists(chunk_store_path):
            try:
                with open(chunk_store_path, "rb") as f:
                    chunks = pickle.load(f)
                logger.info(f"📦 Loaded {len(chunks)} chunks from store: {collection_name}")
                return chunks
            except Exception as e:
                logger.error(f"❌ Error loading chunk store {collection_name}: {e}")
                return []
        else:
            logger.info(f"📦 No existing chunk store found for: {collection_name}")
            return []

    def _save_chunk_store(self, collection_name: str, chunks: List[Document]) -> bool:
        """
        Save chunks to the persistent chunk store
        
        Args:
            collection_name: Session collection name
            chunks: List of Document chunks to save
            
        Returns:
            True if successful, False otherwise
        """
        chunk_store_path = self._get_chunk_store_path(collection_name)
        
        try:
            with open(chunk_store_path, "wb") as f:
                pickle.dump(chunks, f)
            logger.info(f"💾 Saved {len(chunks)} chunks to store: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving chunk store {collection_name}: {e}")
            return False

    def _delete_chunk_store(self, collection_name: str) -> bool:
        """
        Delete the chunk store file for a session

        Args:
            collection_name: Session collection name

        Returns:
            True if successful, False otherwise
        """
        chunk_store_path = self._get_chunk_store_path(collection_name)

        try:
            if os.path.exists(chunk_store_path):
                os.remove(chunk_store_path)
                logger.info(f"🗑️  Deleted chunk store: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting chunk store {collection_name}: {e}")
            return False

    # ============================================================================
    # **OPTIMIZATION 2: BM25 Index Management for Hybrid Search**
    # ============================================================================

    def _get_bm25_index_path(self, collection_name: str) -> str:
        """Get the file path for a session's BM25 index"""
        return os.path.join(self.bm25_store_dir, f"{collection_name}_bm25.pkl")

    def _create_bm25_index(self, chunks: List[Document]) -> Tuple[BM25Okapi, List[str]]:
        """
        Create BM25 index from document chunks

        Args:
            chunks: List of Document chunks

        Returns:
            Tuple of (BM25Okapi index, list of tokenized documents)
        """
        if not chunks:
            logger.warning("Cannot create BM25 index with empty chunks")
            return None, []

        # Tokenize documents for BM25 (simple whitespace tokenization)
        tokenized_docs = [doc.page_content.lower().split() for doc in chunks]

        # Create BM25 index
        bm25_index = BM25Okapi(tokenized_docs)

        logger.info(f"✅ Created BM25 index with {len(tokenized_docs)} documents")
        return bm25_index, tokenized_docs

    def _save_bm25_index(self, collection_name: str, bm25_index: BM25Okapi, tokenized_docs: List[str]) -> bool:
        """
        Save BM25 index to disk

        Args:
            collection_name: Session collection name
            bm25_index: BM25Okapi index object
            tokenized_docs: List of tokenized documents

        Returns:
            True if successful, False otherwise
        """
        bm25_path = self._get_bm25_index_path(collection_name)

        try:
            with open(bm25_path, "wb") as f:
                pickle.dump({"bm25": bm25_index, "tokenized_docs": tokenized_docs}, f)
            logger.info(f"💾 Saved BM25 index: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving BM25 index {collection_name}: {e}")
            return False

    def _load_bm25_index(self, collection_name: str) -> Tuple[Optional[BM25Okapi], Optional[List[str]]]:
        """
        Load BM25 index from disk

        Args:
            collection_name: Session collection name

        Returns:
            Tuple of (BM25Okapi index, tokenized docs) or (None, None) if not found
        """
        bm25_path = self._get_bm25_index_path(collection_name)

        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"📦 Loaded BM25 index: {collection_name}")
                return data["bm25"], data["tokenized_docs"]
            except Exception as e:
                logger.error(f"❌ Error loading BM25 index {collection_name}: {e}")
                return None, None
        else:
            logger.info(f"📦 No existing BM25 index found for: {collection_name}")
            return None, None

    def _delete_bm25_index(self, collection_name: str) -> bool:
        """
        Delete BM25 index file for a session

        Args:
            collection_name: Session collection name

        Returns:
            True if successful, False otherwise
        """
        bm25_path = self._get_bm25_index_path(collection_name)

        try:
            if os.path.exists(bm25_path):
                os.remove(bm25_path)
                logger.info(f"🗑️  Deleted BM25 index: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting BM25 index {collection_name}: {e}")
            return False

    def reciprocal_rank_fusion(self, semantic_results: List[Tuple[Document, float]],
                               bm25_results: List[Tuple[Document, float]],
                               k: int = 60) -> List[Document]:
        """
        Combine FAISS semantic search and BM25 keyword search results using Reciprocal Rank Fusion (RRF)

        RRF formula: RRF_score(d) = Σ 1 / (k + rank_i(d))
        where rank_i(d) is the rank of document d in result list i

        Args:
            semantic_results: List of (Document, score) tuples from FAISS
            bm25_results: List of (Document, score) tuples from BM25
            k: Constant for RRF (default 60, as per original paper)

        Returns:
            List of Documents ranked by RRF score
        """
        # Create a dictionary to store RRF scores for each document
        rrf_scores = {}
        doc_map = {}  # Map content to document for deduplication

        # Process semantic search results
        for rank, (doc, score) in enumerate(semantic_results, start=1):
            doc_key = doc.page_content  # Use content as unique key
            doc_map[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + 1 / (k + rank)

        # Process BM25 results
        for rank, (doc, score) in enumerate(bm25_results, start=1):
            doc_key = doc.page_content
            doc_map[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + 1 / (k + rank)

        # Sort documents by RRF score (descending)
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Return documents in ranked order
        ranked_documents = [doc_map[doc_key] for doc_key, score in sorted_docs]

        logger.info(f"🔀 RRF: Combined {len(semantic_results)} semantic + {len(bm25_results)} BM25 results → {len(ranked_documents)} unique docs")

        return ranked_documents

    def hybrid_search(self, query: str, collection_name: str, k: int = 20,
                     metadata_filter: Optional[Dict] = None) -> List[Document]:
        """
        Perform hybrid search combining FAISS semantic search and BM25 keyword search

        Args:
            query: Search query
            collection_name: Session collection name
            k: Number of documents to retrieve from each method
            metadata_filter: Optional metadata filter for FAISS search

        Returns:
            List of Documents ranked by Reciprocal Rank Fusion
        """
        logger.info(f"🔍 HYBRID SEARCH: Query='{query[:50]}...', Collection='{collection_name}'")

        # Load chunks for BM25 search
        chunks = self._load_chunk_store(collection_name)
        if not chunks:
            logger.warning(f"No chunks found for hybrid search in collection: {collection_name}")
            return []

        # === PART 1: FAISS Semantic Search ===
        try:
            # Load FAISS index
            vectorstore = FAISS.load_local(
                FAISS_INDEX_DIR,
                self.embedding_function,
                index_name=collection_name,
                allow_dangerous_deserialization=True,
            )

            # Perform semantic search with optional metadata filter
            if metadata_filter:
                logger.info(f"   🎯 Applying metadata filter: {metadata_filter}")
                semantic_docs_with_scores = vectorstore.similarity_search_with_score(
                    query, k=k, filter=metadata_filter
                )
            else:
                semantic_docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)

            logger.info(f"   ✅ FAISS retrieved {len(semantic_docs_with_scores)} documents")

        except Exception as e:
            logger.error(f"❌ Error in FAISS search: {e}")
            semantic_docs_with_scores = []

        # === PART 2: BM25 Keyword Search ===
        try:
            # Load or create BM25 index
            bm25_index, tokenized_docs = self._load_bm25_index(collection_name)

            if not bm25_index:
                logger.info("   🔧 BM25 index not found, creating new one...")
                bm25_index, tokenized_docs = self._create_bm25_index(chunks)
                self._save_bm25_index(collection_name, bm25_index, tokenized_docs)

            # Tokenize query
            tokenized_query = query.lower().split()

            # Get BM25 scores
            bm25_scores = bm25_index.get_scores(tokenized_query)

            # Get top-k documents with scores
            top_k_indices = np.argsort(bm25_scores)[::-1][:k]

            # Apply metadata filter to BM25 results if provided
            bm25_results = []
            for idx in top_k_indices:
                if idx < len(chunks):
                    doc = chunks[idx]
                    score = bm25_scores[idx]

                    # Apply metadata filter if provided
                    if metadata_filter:
                        match = all(doc.metadata.get(key) == value for key, value in metadata_filter.items())
                        if not match:
                            continue

                    bm25_results.append((doc, float(score)))

            logger.info(f"   ✅ BM25 retrieved {len(bm25_results)} documents")

        except Exception as e:
            logger.error(f"❌ Error in BM25 search: {e}")
            bm25_results = []

        # === PART 3: Reciprocal Rank Fusion ===
        if not semantic_docs_with_scores and not bm25_results:
            logger.warning("Both FAISS and BM25 returned no results")
            return []

        # Apply RRF to combine results
        fused_documents = self.reciprocal_rank_fusion(semantic_docs_with_scores, bm25_results)

        logger.info(f"✅ HYBRID SEARCH COMPLETE: Returned {len(fused_documents[:k])} documents after RRF")

        return fused_documents[:k]  # Return top-k after fusion

    def delete_document_from_session(self, collection_name: str, filename_to_delete: str) -> bool:
        """
        Delete a specific document's chunks from the persistent chunk store and rebuild index
        
        Args:
            collection_name: Session collection name
            filename_to_delete: Name of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🗑️  DELETING DOCUMENT: '{filename_to_delete}' from session '{collection_name}'")
            
            # **STEP 1: Load all chunks from persistent store**
            logger.info("📦 Loading chunks from persistent store...")
            all_chunks = self._load_chunk_store(collection_name)
            
            if not all_chunks:
                logger.warning(f"No chunks found in store for {collection_name}")
                return False
            
            # **STEP 2: Filter out chunks from the deleted document**
            logger.info(f"🎯 Filtering out chunks from: {filename_to_delete}")
            remaining_chunks = []
            deleted_count = 0
            
            for chunk in all_chunks:
                chunk_source = chunk.metadata.get("source")
                if chunk_source == filename_to_delete:
                    deleted_count += 1
                    logger.debug(f"   Removing chunk from {filename_to_delete}")
                else:
                    remaining_chunks.append(chunk)
            
            logger.info(f"📊 Deletion results:")
            logger.info(f"   Total chunks before: {len(all_chunks)}")
            logger.info(f"   Chunks deleted: {deleted_count}")
            logger.info(f"   Chunks remaining: {len(remaining_chunks)}")
            
            if deleted_count == 0:
                logger.warning(f"No chunks found for document: {filename_to_delete}")
                return False
            
            # **STEP 3: Handle empty session case**
            if not remaining_chunks:
                logger.info("📭 No chunks remaining - cleaning up session")
                self._delete_faiss_index(collection_name)
                self._delete_chunk_store(collection_name)
                logger.info(f"✅ Cleaned up empty session: {collection_name}")
                return True
            
            # **STEP 4: Save filtered chunks back to persistent store**
            logger.info("💾 Saving remaining chunks to persistent store...")
            if not self._save_chunk_store(collection_name, remaining_chunks):
                logger.error("❌ Failed to save remaining chunks to persistent store")
                return False
            
            # **STEP 5: Rebuild FAISS index from remaining chunks**
            logger.info("🔄 Rebuilding FAISS index...")
            self._delete_faiss_index(collection_name)
            vectorstore = self._create_vector_database(remaining_chunks, collection_name)
            
            logger.info(f"✅ Successfully deleted document '{filename_to_delete}':")
            logger.info(f"   Chunks deleted: {deleted_count}")
            logger.info(f"   Chunks remaining: {len(remaining_chunks)}")
            logger.info(f"   FAISS index rebuilt: ✅")
            logger.info(f"   Persistent store updated: ✅")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting document {filename_to_delete} from {collection_name}: {e}")
            return False

    def clear_database(self):
        """Clear the FAISS database by removing index files"""
        try:
            import shutil

            if os.path.exists(FAISS_INDEX_DIR):
                shutil.rmtree(FAISS_INDEX_DIR)
                os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
                logger.info("FAISS database cleared successfully")
                return True
            return True
        except Exception as e:
            logger.error(f"Error clearing FAISS database: {e}")
            return False

    def clear_all_collections(self):
        """Clear all FAISS indexes (equivalent to clearing all collections)"""
        return self.clear_database()

    def process_file(self, file_path: str, filename: str):
        """
        Process a single file and add to the default FAISS index
        Returns retriever and document metadata
        """
        return self.process_file_for_session(file_path, filename, FAISS_COLLECTION_NAME)

    def _create_document_chunks(self, documents):
        """Split documents into chunks for processing"""
        if not documents:
            logger.warning("No documents provided for chunking")
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        
        try:
            chunks = text_splitter.split_documents(documents)
            if not chunks:
                logger.warning("Document chunking produced no chunks - document may be empty or contain only unsupported content")
            return chunks
        except Exception as e:
            logger.error(f"Error during document chunking: {e}")
            return []

    def _get_faiss_index_path(self, collection_name: str):
        """Get the file path for a FAISS index"""
        return os.path.join(FAISS_INDEX_DIR, f"{collection_name}.faiss")

    def _get_metadata_path(self, collection_name: str):
        """Get the file path for FAISS metadata"""
        return os.path.join(FAISS_INDEX_DIR, f"{collection_name}_metadata.pkl")

    # ============================================================================
    # **OPTIMIZATION 3: HNSW Index Creation for Better Performance**
    # ============================================================================

    def _create_hnsw_index(self, doc_splits: List[Document]):
        """
        Create FAISS index using IndexHNSWFlat for better performance on large datasets

        HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor search
        algorithm that provides:
        - Faster search than flat L2 index (especially for large datasets)
        - No training required (unlike IVF indices)
        - Good recall with reasonable memory usage

        Args:
            doc_splits: List of Document chunks

        Returns:
            FAISS vectorstore with HNSW index
        """
        logger.info(f"🔧 Creating FAISS HNSW index with {len(doc_splits)} documents...")

        # Get embedding dimension
        test_embedding = self.embedding_function.embed_query("test")
        dimension = len(test_embedding)
        logger.info(f"   Embedding dimension: {dimension}")

        # Create HNSW index
        # M = 32: number of connections per layer (higher = better recall, more memory)
        # efConstruction = 40: higher = better index quality, slower construction
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 40  # Search depth during construction

        logger.info(f"   ✅ HNSW index created (M=32, efConstruction=40)")

        # Create vectorstore using the custom HNSW index
        # Use FAISS.from_documents but replace the index
        vectorstore = FAISS.from_documents(doc_splits, self.embedding_function)

        # Replace the default flat index with our HNSW index
        # First, get all embeddings and metadata from the temp vectorstore
        old_index = vectorstore.index
        docstore = vectorstore.docstore
        index_to_docstore_id = vectorstore.index_to_docstore_id

        # Add vectors to new HNSW index
        if old_index.ntotal > 0:
            # Reconstruct vectors from the flat index
            vectors = np.array([old_index.reconstruct(i) for i in range(old_index.ntotal)])
            index.add(vectors)
            logger.info(f"   ✅ Added {old_index.ntotal} vectors to HNSW index")

        # Create new vectorstore with HNSW index
        vectorstore.index = index
        vectorstore.docstore = docstore
        vectorstore.index_to_docstore_id = index_to_docstore_id

        logger.info(f"✅ HNSW index created successfully with {index.ntotal} documents")

        return vectorstore

    def _create_vector_database(self, doc_splits: List, collection_name: str):
        """
        Create or update FAISS vector database with enhanced load-and-merge logic
        **OPTIMIZATION 3: Uses IndexHNSWFlat for better performance on large datasets**
        """
        logger.info(
            f"Creating/updating FAISS index '{collection_name}' with {len(doc_splits)} new chunks..."
        )

        # Guard clause: Check if doc_splits is empty
        if not doc_splits:
            logger.warning(f"Cannot create FAISS index '{collection_name}' - no document chunks provided")
            raise ValueError(f"Cannot create FAISS index with empty document chunks for collection '{collection_name}'")

        faiss_index_path = self._get_faiss_index_path(collection_name)
        metadata_path = self._get_metadata_path(collection_name)

        # Enhanced logging for debugging
        logger.info(f"FAISS index path: {faiss_index_path}")
        logger.info(f"Metadata path: {metadata_path}")
        logger.info(f"Index file exists: {os.path.exists(faiss_index_path)}")
        logger.info(f"Metadata file exists: {os.path.exists(metadata_path)}")

        try:
            # CRITICAL: Check if FAISS index already exists (ROBUST LOAD-AND-MERGE PATH)
            # PRIMARY CHECK: Only require the main .faiss file to exist for merge operation
            if os.path.exists(faiss_index_path):
                logger.info(f"🔄 LOAD-AND-MERGE: Found existing FAISS index '{collection_name}'")
                logger.info(f"   Index file exists: {os.path.exists(faiss_index_path)}")
                logger.info(f"   Metadata file exists: {os.path.exists(metadata_path)}")

                # Get current index stats before loading
                index_stat = os.stat(faiss_index_path)
                logger.info(f"   Existing index size: {index_stat.st_size} bytes")
                logger.info(f"   Last modified: {index_stat.st_mtime}")

                try:
                    # ROBUST LOADING: Load existing FAISS index (handles missing metadata gracefully)
                    logger.info(f"   Loading existing FAISS index...")
                    vectorstore = FAISS.load_local(
                        FAISS_INDEX_DIR,
                        self.embedding_function,
                        index_name=collection_name,
                        allow_dangerous_deserialization=True,
                    )

                    # Get current document count before adding new ones
                    current_doc_count = vectorstore.index.ntotal
                    logger.info(f"   ✅ Successfully loaded existing index with {current_doc_count} documents")

                    # CRITICAL: Add new documents to existing index (MERGE OPERATION)
                    logger.info(f"   Adding {len(doc_splits)} new chunks to existing index...")
                    vectorstore.add_documents(doc_splits)

                    # Verify the merge worked
                    new_doc_count = vectorstore.index.ntotal
                    logger.info(f"   ✅ MERGE SUCCESSFUL: Index now contains {new_doc_count} documents")
                    logger.info(f"   📈 Added {new_doc_count - current_doc_count} new documents")

                except Exception as load_error:
                    # FALLBACK: If loading fails, create new index (data loss but system continues)
                    logger.error(f"   ❌ Failed to load existing index: {load_error}")
                    logger.warning(f"   🔄 FALLBACK: Creating new index (existing data will be lost)")

                    # **OPTIMIZATION 3: Create HNSW index for better performance**
                    vectorstore = self._create_hnsw_index(doc_splits)
                    logger.info(f"   ✅ Created fallback FAISS HNSW index with {len(doc_splits)} chunks")

            else:
                # CREATE PATH: No existing index found
                logger.info(f"🆕 CREATE: Creating new FAISS HNSW index '{collection_name}'")
                logger.info(f"   Reason: No existing .faiss file found at {faiss_index_path}")

                # **OPTIMIZATION 3: Create HNSW index instead of flat index**
                vectorstore = self._create_hnsw_index(doc_splits)
                logger.info(f"   ✅ Created new FAISS HNSW index with {len(doc_splits)} chunks")

            # CRITICAL: Save the updated/new FAISS index back to disk
            # This ensures both .faiss and .pkl files are properly synchronized
            logger.info(f"💾 Saving FAISS index '{collection_name}' to disk...")
            
            try:
                vectorstore.save_local(FAISS_INDEX_DIR, index_name=collection_name)
                
                # Verify save operation for both files
                if os.path.exists(faiss_index_path):
                    new_stat = os.stat(faiss_index_path)
                    logger.info(f"   ✅ Index file saved successfully")
                    logger.info(f"   📁 Index file size: {new_stat.st_size} bytes")
                    logger.info(f"   🕒 Index timestamp: {new_stat.st_mtime}")
                else:
                    logger.error(f"   ❌ Index file not found after save operation!")
                
                # Check metadata file (but don't fail if missing)
                if os.path.exists(metadata_path):
                    meta_stat = os.stat(metadata_path)
                    logger.info(f"   ✅ Metadata file saved successfully")
                    logger.info(f"   📁 Metadata file size: {meta_stat.st_size} bytes")
                else:
                    logger.warning(f"   ⚠️ Metadata file not created (this is sometimes normal)")
                    
            except Exception as save_error:
                logger.error(f"   ❌ Failed to save FAISS index: {save_error}")
                raise

            # Final verification
            final_doc_count = vectorstore.index.ntotal
            logger.info(f"📊 FINAL STATS: FAISS index '{collection_name}' contains {final_doc_count} total documents")

            # Log embedding dimensions for verification
            test_embedding = self.embedding_function.embed_query("test")
            logger.info(f"🔧 Embedding model: {self.embedding_function.model_name}")
            logger.info(f"🔧 Embedding dimension: {len(test_embedding)}")

            return vectorstore

        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR in FAISS vector database operation: {e}")
            logger.error(f"   Collection: {collection_name}")
            logger.error(f"   New chunks: {len(doc_splits)}")
            logger.error(f"   Index path: {faiss_index_path}")
            logger.error(f"   Metadata path: {metadata_path}")
            logger.error(f"   Index exists: {os.path.exists(faiss_index_path)}")
            logger.error(f"   Metadata exists: {os.path.exists(metadata_path)}")
            raise

    def process_file_for_session(
        self, file_path: str, filename: str, collection_name: str
    ):
        """
        Process a file for a specific session (collection)
        Returns retriever and document metadata
        """
        try:
            logger.info(f"Processing file for session: {filename} -> {collection_name}")

            # Step 1: Load document
            logger.info("Loading document...")
            documents = self.document_loader.load_document(file_path)
            
            # Guard clause: Check if document loading failed
            if not documents:
                logger.warning(f"Document loading failed or produced no content for file: {filename}")
                raise ValueError(f"No content could be extracted from file: {filename}")

            # **CRITICAL CHANGE: Tag documents with source filename before chunking**
            logger.info(f"🏷️  Tagging documents with source: {filename}")
            for doc in documents:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["source"] = filename
                logger.debug(f"   Tagged document with metadata: {doc.metadata}")

            # Step 2: Split into chunks (metadata will be inherited by all chunks)
            logger.info("Splitting into chunks...")
            new_doc_splits = self._create_document_chunks(documents)
            logger.info(f"Created {len(new_doc_splits)} chunks")
            
            # Guard clause: Check if chunking produced no results
            if not new_doc_splits:
                logger.warning(f"Document chunking produced no chunks for file: {filename}. File may be empty or contain only unsupported content.")
                raise ValueError(f"No processable chunks could be created from file: {filename}")

            # **CRITICAL CHANGE: Load existing chunks from persistent store**
            logger.info("📦 Loading existing chunks from persistent store...")
            existing_chunks = self._load_chunk_store(collection_name)
            
            # **CRITICAL CHANGE: Combine existing chunks with new chunks**
            all_chunks = existing_chunks + new_doc_splits
            logger.info(f"💾 Total chunks after adding new document: {len(all_chunks)} (existing: {len(existing_chunks)}, new: {len(new_doc_splits)})")
            
            # **CRITICAL CHANGE: Save updated chunks to persistent store**
            logger.info("💾 Saving updated chunks to persistent store...")
            if not self._save_chunk_store(collection_name, all_chunks):
                logger.error("❌ Failed to save chunks to persistent store")
                raise RuntimeError("Failed to save chunks to persistent store")

            # Step 3: Create/rebuild FAISS index with ALL chunks (existing + new)
            logger.info("🔄 Rebuilding FAISS index with all chunks...")
            vectorstore = self._create_vector_database(all_chunks, collection_name)

            # **NEW: Step 3.5: Create/update BM25 index for hybrid search**
            logger.info("🔄 Creating/updating BM25 index for hybrid search...")
            bm25_index, tokenized_docs = self._create_bm25_index(all_chunks)
            if bm25_index:
                self._save_bm25_index(collection_name, bm25_index, tokenized_docs)
                logger.info("✅ BM25 index created/updated successfully")

            # Step 4: Create retriever with larger candidate pool
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 20}  # Increased for reranking pipeline
            )
            logger.info("Retriever created successfully for session")

            return {
                'retriever': retriever,
                'chunks': len(new_doc_splits),  # New chunks added in this operation
                'total_chunks': len(all_chunks),  # Total chunks in the session
                'vectorstore': vectorstore,
                'chunks_data': all_chunks  # All chunks for potential rebuilding
            }

        except Exception as e:
            logger.error(f"Error processing file for session: {e}")
            raise
    
    def process_multiple_files_for_session(self, file_paths_and_names: List[tuple], collection_name: str):
        """
        Process multiple files for a specific session in a single batch operation
        Returns retriever and combined document metadata
        
        Args:
            file_paths_and_names: List of (file_path, filename) tuples
            collection_name: Session collection name
        """
        try:
            logger.info(f"Processing {len(file_paths_and_names)} files for session: {collection_name}")
            
            # Guard clause: Check if file list is empty
            if not file_paths_and_names:
                logger.warning("No files provided for batch processing")
                raise ValueError("Cannot process empty file list")
            
            # Step 1: Collect all document chunks from all files
            all_doc_splits = []
            file_metadata = []
            failed_files = []
            
            for file_path, filename in file_paths_and_names:
                try:
                    logger.info(f"Loading document: {filename}")
                    
                    # Load individual document
                    documents = self.document_loader.load_document(file_path)
                    
                    # Guard clause: Check if document loading failed
                    if not documents:
                        logger.warning(f"Document loading failed for file: {filename} - skipping")
                        failed_files.append(filename)
                        file_metadata.append({
                            'filename': filename,
                            'chunks': 0,
                            'status': 'failed_loading'
                        })
                        continue
                    
                    # **CRITICAL CHANGE: Tag documents with source filename before chunking**
                    logger.info(f"🏷️  Tagging documents with source: {filename}")
                    for doc in documents:
                        if not hasattr(doc, 'metadata') or doc.metadata is None:
                            doc.metadata = {}
                        doc.metadata["source"] = filename
                        logger.debug(f"   Tagged document with metadata: {doc.metadata}")
                    
                    # Split into chunks (metadata will be inherited by all chunks)
                    doc_splits = self._create_document_chunks(documents)
                    logger.info(f"Created {len(doc_splits)} chunks from {filename}")
                    
                    # Guard clause: Check if chunking produced no results
                    if not doc_splits:
                        logger.warning(f"Document chunking produced no chunks for file: {filename} - skipping")
                        failed_files.append(filename)
                        file_metadata.append({
                            'filename': filename,
                            'chunks': 0,
                            'status': 'failed_chunking'
                        })
                        continue
                    
                    # Add to combined list
                    all_doc_splits.extend(doc_splits)
                    
                    # Track metadata for this file
                    file_metadata.append({
                        'filename': filename,
                        'chunks': len(doc_splits),
                        'status': 'success'
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing individual file {filename}: {e}")
                    failed_files.append(filename)
                    file_metadata.append({
                        'filename': filename,
                        'chunks': 0,
                        'status': 'error',
                        'error': str(e)
                    })
                    continue
            
            logger.info(f"Total NEW chunks from all files: {len(all_doc_splits)}")
            
            # Guard clause: Check if no chunks were produced from any files
            if not all_doc_splits:
                logger.error(f"No processable chunks were created from any of the {len(file_paths_and_names)} files")
                raise ValueError(f"Batch processing failed: No valid chunks could be created from any files. Failed files: {failed_files}")
            
            # **CRITICAL CHANGE: Load existing chunks from persistent store**
            logger.info("📦 Loading existing chunks from persistent store...")
            existing_chunks = self._load_chunk_store(collection_name)
            
            # **CRITICAL CHANGE: Combine existing chunks with new chunks**
            all_session_chunks = existing_chunks + all_doc_splits
            logger.info(f"💾 Total chunks after batch processing: {len(all_session_chunks)} (existing: {len(existing_chunks)}, new: {len(all_doc_splits)})")
            
            # **CRITICAL CHANGE: Save updated chunks to persistent store**
            logger.info("💾 Saving updated chunks to persistent store...")
            if not self._save_chunk_store(collection_name, all_session_chunks):
                logger.error("❌ Failed to save chunks to persistent store")
                raise RuntimeError("Failed to save chunks to persistent store")
            
            # Log processing summary
            successful_files = len([f for f in file_metadata if f['status'] == 'success'])
            logger.info(f"Batch processing summary: {successful_files}/{len(file_paths_and_names)} files processed successfully")
            if failed_files:
                logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")
            
            # Step 2: Create/update FAISS index with ALL chunks (existing + new)
            logger.info(f"🔄 BATCH PROCESSING: Creating/updating vector database with {len(all_session_chunks)} total chunks...")
            logger.info(f"   📊 Chunks breakdown by file:")
            for file_meta in file_metadata:
                if file_meta.get('status') == 'success':
                    logger.info(f"     - {file_meta['filename']}: {file_meta['chunks']} chunks")

            vectorstore = self._create_vector_database(all_session_chunks, collection_name)

            # **NEW: Step 2.5: Create/update BM25 index for hybrid search**
            logger.info("🔄 Creating/updating BM25 index for hybrid search...")
            bm25_index, tokenized_docs = self._create_bm25_index(all_session_chunks)
            if bm25_index:
                self._save_bm25_index(collection_name, bm25_index, tokenized_docs)
                logger.info("✅ BM25 index created/updated successfully")

            # Step 3: Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 20}  # Increased for reranking pipeline
            )
            logger.info("Retriever created successfully for session")
            
            return {
                'retriever': retriever,
                'total_chunks': len(all_doc_splits),
                'vectorstore': vectorstore,
                'file_metadata': file_metadata,
                'chunks_data': all_doc_splits,  # Store for potential rebuilding
                'failed_files': failed_files,
                'successful_files': successful_files
            }
            
        except Exception as e:
            logger.error(f"Error processing multiple files for session: {e}")
            raise
    
    def rebuild_session_index(self, session_id: str, remaining_documents: List[Dict]):
        """
        Rebuild FAISS index for a session with only the remaining documents
        This is used when documents are deleted from a session
        
        **CRITICAL FIX**: Now uses persistent chunk store instead of unreliable metadata
        """
        try:
            collection_name = f"session_{session_id}"
            logger.info(f"🔄 REBUILDING INDEX: '{collection_name}' with {len(remaining_documents)} remaining documents")
            
            if len(remaining_documents) == 0:
                # No documents left, delete everything
                logger.info("📭 No documents remaining - cleaning up session")
                self._delete_faiss_index(collection_name)
                self._delete_chunk_store(collection_name)
                logger.info(f"✅ Cleaned up empty session: {collection_name}")
                return None
            
            # **CRITICAL CHANGE: Load all chunks from persistent store**
            logger.info("📦 Loading all chunks from persistent store...")
            all_session_chunks = self._load_chunk_store(collection_name)
            
            if not all_session_chunks:
                logger.error(f"❌ No chunks found in persistent store for {collection_name}")
                logger.error("   This indicates a critical data integrity issue!")
                return None
            
            # **CRITICAL CHANGE: Filter chunks to keep only remaining documents**
            remaining_filenames = {doc['filename'] for doc in remaining_documents}
            logger.info(f"🎯 Filtering chunks for remaining documents: {remaining_filenames}")
            
            filtered_chunks = []
            for chunk in all_session_chunks:
                chunk_source = chunk.metadata.get("source")
                if chunk_source in remaining_filenames:
                    filtered_chunks.append(chunk)
                else:
                    logger.debug(f"   Removing chunk from deleted document: {chunk_source}")
            
            logger.info(f"📊 Chunk filtering results:")
            logger.info(f"   Total chunks in store: {len(all_session_chunks)}")
            logger.info(f"   Chunks after filtering: {len(filtered_chunks)}")
            logger.info(f"   Chunks removed: {len(all_session_chunks) - len(filtered_chunks)}")
            
            if not filtered_chunks:
                logger.warning("⚠️  No chunks remain after filtering - this shouldn't happen!")
                return None
            
            # **CRITICAL CHANGE: Save filtered chunks back to persistent store**
            logger.info("💾 Saving filtered chunks back to persistent store...")
            if not self._save_chunk_store(collection_name, filtered_chunks):
                logger.error("❌ Failed to save filtered chunks to persistent store")
                return None
            
            # Delete old FAISS index
            logger.info("🗑️  Deleting old FAISS index...")
            self._delete_faiss_index(collection_name)

            # **CRITICAL CHANGE: Rebuild FAISS index from filtered chunks**
            logger.info(f"🔄 Rebuilding FAISS index with {len(filtered_chunks)} filtered chunks...")
            vectorstore = self._create_vector_database(filtered_chunks, collection_name)

            # **NEW: Rebuild BM25 index for hybrid search**
            logger.info("🔄 Rebuilding BM25 index for hybrid search...")
            bm25_index, tokenized_docs = self._create_bm25_index(filtered_chunks)
            if bm25_index:
                self._save_bm25_index(collection_name, bm25_index, tokenized_docs)
                logger.info("✅ BM25 index rebuilt successfully")

            # Create new retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 20}  # Increased for reranking pipeline
            )
            
            logger.info(f"✅ Successfully rebuilt session index:")
            logger.info(f"   Session: {collection_name}")
            logger.info(f"   Documents: {len(remaining_documents)}")
            logger.info(f"   Chunks: {len(filtered_chunks)}")
            logger.info(f"   Persistent store updated: ✅")
            
            return {
                'retriever': retriever,
                'chunks': len(filtered_chunks),
                'total_chunks': len(filtered_chunks),  # For consistency with other methods
                'vectorstore': vectorstore,
                'chunks_data': filtered_chunks  # For potential future use
            }
            
        except Exception as e:
            logger.error(f"Error rebuilding session index: {e}")
            raise
    
    def _delete_faiss_index(self, collection_name: str):
        """Delete FAISS index files and BM25 index for a collection"""
        try:
            import os
            faiss_file = self._get_faiss_index_path(collection_name)
            metadata_file = self._get_metadata_path(collection_name)

            if os.path.exists(faiss_file):
                os.remove(faiss_file)
                logger.info(f"Deleted FAISS index file: {faiss_file}")

            if os.path.exists(metadata_file):
                os.remove(metadata_file)
                logger.info(f"Deleted FAISS metadata file: {metadata_file}")

            # **NEW: Also delete BM25 index**
            self._delete_bm25_index(collection_name)

        except Exception as e:
            logger.warning(f"Error deleting FAISS index files: {e}")
