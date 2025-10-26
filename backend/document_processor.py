"""
Document processing and vector database management using FAISS

This module handles document loading, text splitting, embedding generation,
and FAISS vector database operations for the RAG system.
"""

import os
import pickle
import logging
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from backend.config import FAISS_INDEX_DIR, FAISS_COLLECTION_NAME

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
        
        # **NEW: Persistent chunk store infrastructure**
        self.chunk_store_dir = os.path.join(FAISS_INDEX_DIR, "chunk_stores")
        os.makedirs(self.chunk_store_dir, exist_ok=True)
        logger.info(f"Chunk store directory: {self.chunk_store_dir}")

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

    def _create_vector_database(self, doc_splits: List, collection_name: str):
        """Create or update FAISS vector database with enhanced load-and-merge logic"""
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
                    
                    # Create new FAISS index from documents as fallback
                    vectorstore = FAISS.from_documents(doc_splits, self.embedding_function)
                    logger.info(f"   ✅ Created fallback FAISS index with {len(doc_splits)} chunks")

            else:
                # CREATE PATH: No existing index found
                logger.info(f"🆕 CREATE: Creating new FAISS index '{collection_name}'")
                logger.info(f"   Reason: No existing .faiss file found at {faiss_index_path}")

                # Create new FAISS index from documents
                vectorstore = FAISS.from_documents(doc_splits, self.embedding_function)
                logger.info(f"   ✅ Created new FAISS index with {len(doc_splits)} chunks")

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
