"""
Document processing and vector database management using FAISS

This module handles document loading, text splitting, embedding generation,
and FAISS vector database operations for the RAG system.
"""

import os
import logging
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import FAISS_INDEX_DIR, FAISS_COLLECTION_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documents and creates embeddings for the FAISS vector database"""

    def __init__(self, document_loader):
        self.document_loader = document_loader
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
        )

        # Ensure FAISS directory exists
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        logger.info(f"FAISS index directory: {FAISS_INDEX_DIR}")

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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    def _get_faiss_index_path(self, collection_name: str):
        """Get the file path for a FAISS index"""
        return os.path.join(FAISS_INDEX_DIR, f"{collection_name}.faiss")

    def _get_metadata_path(self, collection_name: str):
        """Get the file path for FAISS metadata"""
        return os.path.join(FAISS_INDEX_DIR, f"{collection_name}_metadata.pkl")

    def _create_vector_database(self, doc_splits: List, collection_name: str):
        """Create or update FAISS vector database"""
        logger.info(
            f"Creating/updating FAISS index '{collection_name}' with {len(doc_splits)} chunks..."
        )

        faiss_index_path = self._get_faiss_index_path(collection_name)
        metadata_path = self._get_metadata_path(collection_name)

        try:
            # Check if FAISS index already exists
            if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
                logger.info(f"Loading existing FAISS index: {collection_name}")

                # Load existing FAISS index
                vectorstore = FAISS.load_local(
                    FAISS_INDEX_DIR,
                    self.embedding_function,
                    index_name=collection_name,
                    allow_dangerous_deserialization=True,
                )

                # Add new documents to existing index
                vectorstore.add_documents(doc_splits)
                logger.info(f"Added {len(doc_splits)} chunks to existing FAISS index")

            else:
                logger.info(f"Creating new FAISS index: {collection_name}")

                # Create new FAISS index from documents
                vectorstore = FAISS.from_documents(doc_splits, self.embedding_function)
                logger.info(f"Created new FAISS index with {len(doc_splits)} chunks")

            # Save the FAISS index
            vectorstore.save_local(FAISS_INDEX_DIR, index_name=collection_name)
            logger.info(f"FAISS index '{collection_name}' saved successfully")

            # Log embedding dimensions for verification
            test_embedding = self.embedding_function.embed_query("test")
            logger.info(f"Embedding model: {self.embedding_function.model_name}")
            logger.info(f"Embedding dimension: {len(test_embedding)}")

            return vectorstore

        except Exception as e:
            logger.error(f"Error creating FAISS vector database: {e}")
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

            # Step 2: Split into chunks
            logger.info("Splitting into chunks...")
            doc_splits = self._create_document_chunks(documents)
            logger.info(f"Created {len(doc_splits)} chunks")

            # Step 3: Add to existing or create new FAISS index
            logger.info("Adding to vector database...")
            vectorstore = self._create_vector_database(doc_splits, collection_name)

            # Step 4: Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            )
            logger.info("Retriever created successfully for session")

            return {
                'retriever': retriever,
                'chunks': len(doc_splits),
                'vectorstore': vectorstore,
                'chunks_data': doc_splits  # Store for potential rebuilding
            }

        except Exception as e:
            logger.error(f"Error processing file for session: {e}")
            raise
    
    def rebuild_session_index(self, session_id: str, remaining_documents: List[Dict]):
        """
        Rebuild FAISS index for a session with only the remaining documents
        This is used when documents are deleted from a session
        """
        try:
            collection_name = f"session_{session_id}"
            logger.info(f"Rebuilding FAISS index '{collection_name}' with {len(remaining_documents)} documents")
            
            if len(remaining_documents) == 0:
                # No documents left, delete the index
                self._delete_faiss_index(collection_name)
                logger.info(f"Deleted empty FAISS index: {collection_name}")
                return None
            
            # Recreate documents from metadata
            all_chunks = []
            for doc_meta in remaining_documents:
                # We need to reload and rechunk the documents
                # For now, we'll store the original chunks in metadata if available
                if 'chunks_data' in doc_meta:
                    all_chunks.extend(doc_meta['chunks_data'])
                else:
                    logger.warning(f"No chunk data available for {doc_meta['filename']}, skipping rebuild")
                    continue
            
            if not all_chunks:
                logger.warning("No valid chunks found for rebuilding index")
                return None
            
            # Delete old index
            self._delete_faiss_index(collection_name)
            
            # Create new index with remaining chunks
            vectorstore = self._create_vector_database(all_chunks, collection_name)
            
            # Create new retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            )
            
            logger.info(f"Successfully rebuilt FAISS index '{collection_name}' with {len(all_chunks)} chunks")
            return {
                'retriever': retriever,
                'chunks': len(all_chunks),
                'vectorstore': vectorstore
            }
            
        except Exception as e:
            logger.error(f"Error rebuilding session index: {e}")
            raise
    
    def _delete_faiss_index(self, collection_name: str):
        """Delete FAISS index files for a collection"""
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
                
        except Exception as e:
            logger.warning(f"Error deleting FAISS index files: {e}")
