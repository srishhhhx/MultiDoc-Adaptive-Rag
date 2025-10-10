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

            # Step 2: Split into chunks
            logger.info("Splitting into chunks...")
            doc_splits = self._create_document_chunks(documents)
            logger.info(f"Created {len(doc_splits)} chunks")
            
            # Guard clause: Check if chunking produced no results
            if not doc_splits:
                logger.warning(f"Document chunking produced no chunks for file: {filename}. File may be empty or contain only unsupported content.")
                raise ValueError(f"No processable chunks could be created from file: {filename}")

            # Step 3: Add to existing or create new FAISS index
            logger.info("Adding to vector database...")
            vectorstore = self._create_vector_database(doc_splits, collection_name)

            # Step 4: Create retriever with larger candidate pool
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 20}  # Increased for reranking pipeline
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
                    
                    # Split into chunks
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
            
            logger.info(f"Total chunks from all files: {len(all_doc_splits)}")
            
            # Guard clause: Check if no chunks were produced from any files
            if not all_doc_splits:
                logger.error(f"No processable chunks were created from any of the {len(file_paths_and_names)} files")
                raise ValueError(f"Batch processing failed: No valid chunks could be created from any files. Failed files: {failed_files}")
            
            # Log processing summary
            successful_files = len([f for f in file_metadata if f['status'] == 'success'])
            logger.info(f"Batch processing summary: {successful_files}/{len(file_paths_and_names)} files processed successfully")
            if failed_files:
                logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")
            
            # Step 2: Create/update FAISS index with ALL chunks at once (CRITICAL CUMULATIVE OPERATION)
            logger.info(f"🔄 BATCH PROCESSING: Creating/updating vector database with {len(all_doc_splits)} total chunks...")
            logger.info(f"   📊 Chunks breakdown by file:")
            for file_meta in file_metadata:
                if file_meta.get('status') == 'success':
                    logger.info(f"     - {file_meta['filename']}: {file_meta['chunks']} chunks")
            
            vectorstore = self._create_vector_database(all_doc_splits, collection_name)
            
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
