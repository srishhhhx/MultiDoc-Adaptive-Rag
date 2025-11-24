"""
FastAPI Backend for Advanced RAG System with LangGraph

This API provides endpoints for:
- Document upload and processing
- Question answering with RAG workflow
- Real-time status updates
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel
from typing import Optional, Dict, List
import os
import tempfile
import logging
import json
from datetime import datetime
# import uuid  # Currently unused

# Local imports
from backend.document_loader import MultiModalDocumentLoader
from backend.document_processor import DocumentProcessor
from backend.rag_workflow import RAGWorkflow
from backend.session_manager import session_manager, ConversationExchange
from backend.evaluation.runtime_metrics_tracker import RuntimeMetricsTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced RAG API",
    description="API for document processing and question answering using LangGraph RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # Allow all headers including Accept: text/event-stream
)

# Initialize components
document_loader = MultiModalDocumentLoader()

class SessionAwareDocumentProcessor(DocumentProcessor):
    """Document processor with session management support"""
    
    def __init__(self, document_loader):
        super().__init__(document_loader)
    
    def process_file_for_session_api(self, file_path: str, filename: str, session_id: str):
        """Process file for a specific session"""
        try:
            # Get session data
            session = session_manager.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Process file for session
            result = self.process_file_for_session(file_path, filename, session.collection_name)
            
            session_manager.set_session_retriever(session_id, result['retriever'])
            
            file_metadata = {
                'filename': filename,
                'chunks': result['chunks'],
                'processed_at': datetime.now().isoformat(),
                'status': 'success'
            }
            session_manager.add_document_to_session(session_id, file_metadata)
            
            logger.info(f"File processed successfully for session {session_id}: {filename}")
            return {
                'session_id': session_id,
                'filename': filename,
                'chunks': result['chunks'],
                'success': True
            }
            
        except ValueError as e:
            # Handle cases where file couldn't be processed (empty chunks, etc.)
            logger.error(f"Validation error processing file {filename} for session {session_id}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"File processing validation failed for {filename}: {str(e)}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing file for session {session_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file for session {session_id}: {str(e)}")
    
    def process_multiple_files_for_session_api(self, file_paths_and_names: List[tuple], session_id: str):
        """Process multiple files for a specific session in batch"""
        try:
            # Get session data
            session = session_manager.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Process all files in batch
            result = self.process_multiple_files_for_session(file_paths_and_names, session.collection_name)
            
            # Update session with retriever
            session_manager.set_session_retriever(session_id, result['retriever'])
            
            # Add metadata for each file to session (only successful ones)
            successful_files = 0
            for file_meta in result['file_metadata']:
                file_metadata = {
                    'filename': file_meta['filename'],
                    'chunks': file_meta['chunks'],
                    'processed_at': datetime.now().isoformat(),
                    'status': file_meta.get('status', 'unknown')
                }
                # Only add successful files to session
                if file_meta.get('status') == 'success':
                    session_manager.add_document_to_session(session_id, file_metadata)
                    successful_files += 1
            
            # Log processing results
            failed_count = len(result.get('failed_files', []))
            if failed_count > 0:
                logger.warning(f"Batch processing completed with {failed_count} failed files for session {session_id}")
            else:
                logger.info(f"All files processed successfully for session {session_id}: {len(file_paths_and_names)} files")
            
            return {
                'session_id': session_id,
                'total_files': len(file_paths_and_names),
                'successful_files': successful_files,
                'failed_files': failed_count,
                'total_chunks': result['total_chunks'],
                'file_metadata': result['file_metadata'],
                'failed_file_names': result.get('failed_files', []),
                'success': True,
                'partial_success': failed_count > 0
            }
            
        except ValueError as e:
            # Handle cases where no files could be processed
            logger.error(f"Validation error processing files for session {session_id}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"File processing validation failed: {str(e)}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing multiple files for session {session_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

document_processor = SessionAwareDocumentProcessor(document_loader)

class SessionAwareRAGWorkflow(RAGWorkflow):
    """RAG workflow with session management support"""

    def __init__(self, document_processor=None):
        # **OPTIMIZATION: Pass document_processor to enable hybrid search (FAISS + BM25)**
        super().__init__(document_processor=document_processor)
        self.graph = None
    
    def _create_graph(self):
        """Create the graph instance - alias for create_workflow for compatibility"""
        return self.create_workflow()
    
    def get_graph(self):
        """Get or create the graph instance"""
        if self.graph is None:
            self.graph = self._create_graph()
        return self.graph
    
    def process_question_for_session(self, question: str, session_id: str):
        """Process a question through the RAG workflow for a specific session"""
        logger.info(f"Processing question for session {session_id}: {question}")
        
        # Get session retriever
        retriever = session_manager.get_session_retriever(session_id)
        if not retriever:
            raise HTTPException(status_code=400, detail="No documents found in session")
        
        # Debug: Test retriever before using it
        logger.info(f"Testing retriever for session {session_id}")
        try:
            test_docs = retriever.invoke("test query")
            logger.info(f"Retriever test successful: {len(test_docs)} documents available")
        except Exception as e:
            logger.error(f"Retriever test failed: {e}")
            raise HTTPException(status_code=500, detail=f"Retriever error: {str(e)}")
        
        # Set retriever and session ID for this request
        self.set_retriever(retriever)
        self.set_session_id(session_id)  # NEW: Set session ID for document access
        
        # Get conversation context (for future use)
        # context = session_manager.get_conversation_context(session_id)
        
        # Process through graph with proper state initialization
        graph = self.get_graph()
        result = graph.invoke(input={
            "question": question, 
            "original_question": question,  # Preserve original for context assessment
            "rewrite_attempts": 0
        })
        
        # Create conversation exchange
        exchange = ConversationExchange(
            question=question,
            answer=result.get('solution', 'No answer generated'),
            timestamp=datetime.now(),
            search_method=result.get('search_method', 'unknown'),
            online_search=result.get('online_search', False),
            document_evaluations=result.get('document_evaluations'),
            question_relevance_score=result.get('question_relevance_score'),
            document_relevance_score=result.get('document_relevance_score')
        )
        
        # Add to session conversation history
        session_manager.add_conversation_exchange(session_id, exchange)
        
        logger.info(f"Question processed successfully for session {session_id}")
        return result

# **OPTIMIZATION: Initialize RAG workflow with document_processor for hybrid search support**
rag_workflow = SessionAwareRAGWorkflow(document_processor=document_processor)

# Initialize global metrics tracker for Phase 1
metrics_tracker = RuntimeMetricsTracker()
logger.info("✅ RuntimeMetricsTracker initialized for Phase 1 metrics collection")

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    session_id: str

class UploadRequest(BaseModel):
    session_id: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    search_method: str
    online_search: bool
    document_evaluations: Optional[List[Dict]] = None
    question_relevance_score: Optional[Dict] = None
    document_relevance_score: Optional[Dict] = None

# Session Management Endpoints
@app.post("/api/start-session")
async def start_new_session():
    """Start a new document session - clears previous data"""
    session_id = session_manager.create_session()
    return {
        "session_id": session_id,
        "message": "New session started",
        "created_at": datetime.now().isoformat()
    }

@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a session"""
    session_info = session_manager.get_session_info(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return session_info

@app.delete("/api/session/{session_id}")
async def end_session(session_id: str):
    """End session and cleanup resources"""
    session_manager.cleanup_session(session_id)
    return {"message": "Session ended successfully"}

@app.delete("/api/session/{session_id}/document/{document_id}")
async def delete_document(session_id: str, document_id: str):
    """Delete a specific document from a session"""
    try:
        # Remove document from session
        removed_doc = session_manager.remove_document_from_session(session_id, document_id)
        if not removed_doc:
            raise HTTPException(status_code=404, detail="Document not found in session")
        
        # Get remaining documents
        remaining_docs = session_manager.get_session_documents(session_id)
        
        # Rebuild FAISS index with remaining documents
        if remaining_docs:
            result = document_processor.rebuild_session_index(session_id, remaining_docs)
            if result:
                # Update session retriever
                session_manager.set_session_retriever(session_id, result['retriever'])
                logger.info(f"Rebuilt index for session {session_id} with {result['chunks']} chunks")
            else:
                logger.warning(f"Failed to rebuild index for session {session_id}")
        else:
            # No documents left, clear retriever
            session_manager.set_session_retriever(session_id, None)
            logger.info(f"No documents left in session {session_id}, cleared retriever")
        
        return {
            "success": True,
            "message": f"Document {removed_doc['filename']} deleted successfully",
            "removed_document": removed_doc,
            "remaining_documents": len(remaining_docs),
            "remaining_chunks": sum(doc.get('chunks', 0) for doc in remaining_docs)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Advanced RAG API is running",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "ask": "/api/ask",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Upload and process a document for a session
    
    Supports: PDF, DOCX, TXT, CSV, XLSX
    If no session_id provided, creates a new session
    """
    try:
        # Validate file type
        filename = file.filename
        file_extension = filename.split('.')[-1].lower()
        
        if not document_loader.is_supported_file(filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: .{file_extension}. Supported: {document_loader.get_supported_extensions_display()}"
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Create session if not provided
            if not session_id:
                session_id = session_manager.create_session()
                logger.info(f"Created new session for upload: {session_id}")
            
            # Process the file for session
            result = document_processor.process_file_for_session_api(tmp_file_path, filename, session_id)
            
            # Format size for display
            size_mb = file_size / (1024 * 1024)
            size_display = f"{size_mb:.1f}MB" if size_mb >= 1 else f"{file_size / 1024:.0f}KB"
            
            # Add document metadata to session
            document_metadata = {
                'filename': file.filename,
                'size': size_display,
                'chunks': result['chunks'],
                'uploadedAt': datetime.now().isoformat(),
                'chunks_data': result.get('chunks_data', [])  # Store chunks for rebuilding
            }
            session_manager.add_document_to_session(session_id, document_metadata)
            
            return JSONResponse(content={
                "success": True,
                "message": "File uploaded and processed successfully",
                "session_id": session_id,
                "metadata": document_metadata
            })
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_file_path)
            except OSError:
                logger.warning(f"Could not delete temp file: {tmp_file_path}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Upload and process multiple documents for a session using batch processing
    
    Supports: PDF, DOCX, TXT, CSV, XLSX
    If no session_id provided, creates a new session
    """
    try:
        # Create session if not provided
        if not session_id:
            session_id = session_manager.create_session()
            logger.info(f"Created new session for multi-upload: {session_id}")
        
        # Step 1: Validate all files and create temporary files
        temp_files = []
        failed_files = []
        file_paths_and_names = []
        
        for file in files:
            try:
                # Validate file type
                filename = file.filename
                file_extension = filename.split('.')[-1].lower()
                
                if not document_loader.is_supported_file(filename):
                    failed_files.append({
                        "filename": filename,
                        "error": f"Unsupported file type: .{file_extension}"
                    })
                    continue
                
                # Read file content
                content = await file.read()
                file_size = len(content)
                
                # Create temporary file
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
                tmp_file.write(content)
                tmp_file.close()
                
                # Store temp file info for cleanup
                temp_files.append({
                    'path': tmp_file.name,
                    'filename': filename,
                    'size': file_size
                })
                
                # Add to processing list
                file_paths_and_names.append((tmp_file.name, filename))
                
            except Exception as e:
                logger.error(f"Error preparing file {filename}: {str(e)}")
                failed_files.append({
                    "filename": filename,
                    "error": str(e)
                })
        
        # Step 2: Process all valid files in batch
        results = []
        total_chunks = 0
        
        if file_paths_and_names:
            try:
                # Process all files in a single batch operation
                batch_result = document_processor.process_multiple_files_for_session_api(
                    file_paths_and_names, session_id
                )
                
                # Format results for response
                for i, temp_file_info in enumerate(temp_files):
                    if temp_file_info['filename'] in [name for _, name in file_paths_and_names]:
                        # Find matching file metadata
                        file_meta = next(
                            (fm for fm in batch_result['file_metadata'] if fm['filename'] == temp_file_info['filename']),
                            {'filename': temp_file_info['filename'], 'chunks': 0}
                        )
                        
                        # Format size for display
                        size_mb = temp_file_info['size'] / (1024 * 1024)
                        size_display = f"{size_mb:.1f}MB" if size_mb >= 1 else f"{temp_file_info['size'] / 1024:.0f}KB"
                        
                        results.append({
                            "filename": temp_file_info['filename'],
                            "size": size_display,
                            "chunks": file_meta['chunks'],
                            "success": True
                        })
                
                total_chunks = batch_result['total_chunks']
                logger.info(f"Batch processing completed: {len(results)} files, {total_chunks} total chunks")
                
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                # Add all files to failed list if batch processing fails
                for _, filename in file_paths_and_names:
                    failed_files.append({
                        "filename": filename,
                        "error": f"Batch processing failed: {str(e)}"
                    })
        
        # Step 3: Clean up temporary files
        for temp_file_info in temp_files:
            try:
                os.unlink(temp_file_info['path'])
            except OSError:
                logger.warning(f"Could not delete temp file: {temp_file_info['path']}")
        
        return JSONResponse(content={
            "success": True,
            "message": f"Processed {len(results)} files successfully using batch processing",
            "session_id": session_id,
            "total_chunks": total_chunks,
            "successful_uploads": results,
            "failed_uploads": failed_files,
            "uploadedAt": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Multi-upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question about uploaded documents in a session
    
    Returns answer with evaluation metrics
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if not request.session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        # Check if session exists and has documents
        session_info = session_manager.get_session_info(request.session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        if len(session_info['documents']) == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded in this session. Please upload documents first."
            )

        # PHASE 1: Track metrics - start timing
        import time
        start_time = time.time()

        # Process the question for session
        result = rag_workflow.process_question_for_session(request.question, request.session_id)

        # PHASE 1: Extract and log metrics from final state
        try:
            query_metrics = metrics_tracker.extract_from_state(
                state=result,
                session_id=request.session_id,
                start_time=start_time
            )
            logger.info(f"📊 Metrics collected for query: {request.question[:50]}...")
            logger.info(f"   Query Type: {query_metrics.query_type}")
            logger.info(f"   Latency: {query_metrics.latency_ms:.0f}ms")
            logger.info(f"   Answer Relevance: {query_metrics.answer_relevance}")
            logger.info(f"   Faithfulness: {query_metrics.faithfulness}")
        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")

        # Format response
        response = {
            "answer": result.get('solution', 'No answer generated'),
            "search_method": result.get('search_method', 'unknown'),
            "online_search": result.get('online_search', False),
        }
        
        # Add evaluation data if available
        if 'document_evaluations' in result and result['document_evaluations']:
            response['document_evaluations'] = []
            for eval in result['document_evaluations']:
                # Handle both old object format and new dictionary format
                if isinstance(eval, dict):
                    # New batch analysis format (dictionary)
                    eval_data = {
                        'score': eval.get('score', 'NO'),
                        'relevance_score': eval.get('relevance_score', None),
                        'coverage_assessment': eval.get('coverage_assessment', None),
                        'missing_information': eval.get('missing_information', None)
                    }
                else:
                    # Old individual evaluation format (object)
                    eval_data = {
                        'score': getattr(eval, 'score', 'NO'),
                        'relevance_score': getattr(eval, 'relevance_score', None),
                        'coverage_assessment': getattr(eval, 'coverage_assessment', None),
                        'missing_information': getattr(eval, 'missing_information', None)
                    }
                response['document_evaluations'].append(eval_data)
        
        # Add relevance scores
        if 'question_relevance_score' in result:
            q_score = result['question_relevance_score']
            response['question_relevance_score'] = {
                'binary_score': getattr(q_score, 'binary_score', None),
                'relevance_score': getattr(q_score, 'confidence_score', None),  # Fixed: use confidence_score
                'completeness': getattr(q_score, 'completeness', None),
                'reasoning': getattr(q_score, 'explanation', None),  # Fixed: use explanation
                'missing_aspects': getattr(q_score, 'missing_aspects', None)
            }
        
        if 'document_relevance_score' in result:
            d_score = result['document_relevance_score']
            response['document_relevance_score'] = {
                'binary_score': getattr(d_score, 'binary_score', None),
                'confidence': getattr(d_score, 'confidence_score', None),  # Fixed: use confidence_score
                'reasoning': getattr(d_score, 'explanation', None)  # Fixed: use explanation
            }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask-stream")
async def ask_question_stream(request: QuestionRequest):
    """
    Ask a question with Server-Sent Events (SSE) streaming

    This endpoint streams events in real-time as the RAG workflow progresses:
    - Stage updates (analyzing, retrieving, generating, validating)
    - Provisional answer tokens as they're generated
    - Rewrite events if hallucinations are detected
    - Final answer with quality metrics

    Frontend should use EventSource to consume these events.
    """
    async def event_generator():
        logger.info("=" * 80)
        logger.info(f"🚀 EVENT GENERATOR STARTED for session {request.session_id}")
        logger.info(f"   Question: {request.question[:100]}...")
        logger.info("=" * 80)

        try:
            # Validate request
            if not request.question or not request.question.strip():
                error_event = {"type": "error", "message": "Question cannot be empty", "recoverable": False}
                yield {"data": json.dumps(error_event)}
                return

            if not request.session_id:
                error_event = {"type": "error", "message": "Session ID is required", "recoverable": False}
                yield {"data": json.dumps(error_event)}
                return

            # Check if session exists and has documents
            session_info = session_manager.get_session_info(request.session_id)
            if not session_info:
                error_event = {"type": "error", "message": "Session not found or expired", "recoverable": False}
                yield {"data": json.dumps(error_event)}
                return

            if len(session_info['documents']) == 0:
                error_event = {"type": "error", "message": "No documents uploaded in this session. Please upload documents first.", "recoverable": False}
                yield {"data": json.dumps(error_event)}
                return

            # Get retriever for this session
            retriever = session_manager.get_session_retriever(request.session_id)
            if not retriever:
                error_event = {"type": "error", "message": "Session retriever not found. Please re-upload documents.", "recoverable": False}
                yield {"data": json.dumps(error_event)}
                return

            # CRITICAL: Set session_id BEFORE retriever (multi_tool_executor depends on it)
            rag_workflow.set_session_id(request.session_id)
            rag_workflow.set_retriever(retriever)

            # Test retriever before streaming
            logger.info(f"Testing retriever for session {request.session_id}")
            try:
                test_docs = retriever.invoke("test")
                logger.info(f"Retriever test successful: {len(test_docs)} documents available")
            except Exception as e:
                logger.error(f"Retriever test failed: {e}")
                error_event = {"type": "error", "message": f"Retriever error: {str(e)}", "recoverable": False}
                yield {"data": json.dumps(error_event)}
                return

            # Stream the RAG workflow
            logger.info(f"Starting streaming workflow for session {request.session_id}")
            event_count = 0
            last_event = None  # Track last event for conversation saving

            async for event in rag_workflow.process_question_stream(request.question):
                event_count += 1
                event_type = event.get("type", "unknown")
                logger.info(f"📤 Streaming event #{event_count}: {event_type}")

                # Format event for SSE - yield dict and let EventSourceResponse format it
                logger.info(f"   Yielding SSE event: {event_type}")
                logger.info(f"   Event data: {json.dumps(event)[:200]}...")
                yield {"data": json.dumps(event)}

                # Track last event for conversation saving
                last_event = event

                # Add conversation to history if this is the final answer
                if event.get("type") == "final_answer":
                    # Extract quality metrics from event
                    grounding_source = event.get("grounding_source", "DOCUMENT_ONLY")

                    logger.info(f"Saving conversation to session {request.session_id}")
                    session_manager.add_conversation_exchange(
                        request.session_id,
                        ConversationExchange(
                            question=request.question,
                            answer=event.get("content", ""),
                            timestamp=datetime.now(),
                            search_method=grounding_source,
                            online_search=(grounding_source in ["WEB_ONLY", "HYBRID"]),
                            document_evaluations=None,  # Not tracked in streaming
                            question_relevance_score=event.get("question_relevance"),
                            document_relevance_score=event.get("document_relevance")
                        )
                    )
                    logger.info(f"Conversation saved successfully")

            # Send success end event after stream completes
            logger.info(f"✅ Stream complete. Total events sent: {event_count}")
            end_event = {
                "type": "end",
                "success": True,
                "total_events": event_count
            }
            yield {"data": json.dumps(end_event)}

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            error_event = {
                "type": "error",
                "message": f"An error occurred: {str(e)}",
                "recoverable": False
            }
            yield {"data": json.dumps(error_event)}

            end_event = {
                "type": "end",
                "success": False
            }
            yield {"data": json.dumps(end_event)}

    return EventSourceResponse(
        event_generator(),
        ping=15,  # Send ping every 15 seconds to keep connection alive
        sep="\r\n"  # Explicit separator for better compatibility
    )

@app.get("/api/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "formats": document_loader.get_supported_extensions(),
        "display": document_loader.get_supported_extensions_display()
    }

@app.post("/api/clear-database")
async def clear_database():
    """
    Clear the FAISS database
    
    This removes all previously uploaded documents and their embeddings.
    Useful for starting fresh or managing storage.
    """
    try:
        success = document_processor.clear_database()
        if success:
            return {"success": True, "message": "Database cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear database")
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear-all-collections")
async def clear_all_collections():
    """
    Clear ALL FAISS indexes (for debugging purposes)
    
    This removes all indexes and their embeddings.
    Use when you need a complete reset.
    """
    try:
        success = document_processor.clear_all_collections()
        if success:
            return {"success": True, "message": "All collections cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear all collections")
    except Exception as e:
        logger.error(f"Error clearing all collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debug/collections")
async def debug_collections():
    """
    Debug endpoint to check FAISS indexes and their contents
    """
    try:
        import os
        from config import FAISS_INDEX_DIR
        
        result = {
            "total_indexes": 0,
            "indexes": []
        }
        
        if not os.path.exists(FAISS_INDEX_DIR):
            return result
            
        # List all FAISS index files
        faiss_files = [f for f in os.listdir(FAISS_INDEX_DIR) if f.endswith('.faiss')]
        result["total_indexes"] = len(faiss_files)
        
        for faiss_file in faiss_files:
            index_name = faiss_file.replace('.faiss', '')
            index_info = {
                "name": index_name,
                "faiss_file": faiss_file,
                "metadata_file": f"{index_name}_metadata.pkl",
                "file_size": os.path.getsize(os.path.join(FAISS_INDEX_DIR, faiss_file))
            }
            
            # Try to load and get basic info
            try:
                from langchain_community.vectorstores import FAISS
                # Test loading the FAISS index
                FAISS.load_local(
                    FAISS_INDEX_DIR,
                    document_processor.embedding_function,
                    index_name=index_name,
                    allow_dangerous_deserialization=True
                )
                index_info["status"] = "loaded_successfully"
                index_info["sample_search"] = "Available"
            except Exception as e:
                index_info["status"] = f"load_error: {str(e)}"
            
            result["indexes"].append(index_info)
        
        return result
    except Exception as e:
        logger.error(f"Error debugging collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/nuclear-reset")
async def nuclear_reset():
    """
    Nuclear option: Completely delete and recreate FAISS database
    Use this when you need a complete fresh start
    """
    try:
        import shutil
        import os
        from config import FAISS_INDEX_DIR
        
        # Stop all sessions
        session_manager.active_sessions.clear()
        
        # Delete the entire FAISS directory
        if os.path.exists(FAISS_INDEX_DIR):
            shutil.rmtree(FAISS_INDEX_DIR)
            logger.info(f"Deleted FAISS directory: {FAISS_INDEX_DIR}")
        
        # Recreate document processor with fresh FAISS setup
        global document_processor
        document_processor = SessionAwareDocumentProcessor(document_loader)
        logger.info("Recreated document processor with fresh FAISS setup")
        
        return {"success": True, "message": "FAISS database completely reset - all data deleted"}
    except Exception as e:
        logger.error(f"Error in nuclear reset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ======================
# PHASE 1: METRICS ENDPOINTS
# ======================

@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """
    Get human-readable summary of all collected metrics
    """
    try:
        summary = metrics_tracker.get_metrics_summary()
        return {"summary": summary, "success": True}
    except Exception as e:
        logger.error(f"Error getting metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/aggregated")
async def get_aggregated_metrics():
    """
    Get aggregated metrics across all queries as structured JSON
    """
    try:
        from dataclasses import asdict
        agg_metrics = metrics_tracker.get_aggregated_metrics()
        return asdict(agg_metrics)
    except Exception as e:
        logger.error(f"Error getting aggregated metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/session/{session_id}")
async def get_session_metrics(session_id: str):
    """
    Get metrics for a specific session
    """
    try:
        from dataclasses import asdict
        session_metrics = metrics_tracker.get_session_metrics(session_id)
        if session_metrics is None:
            raise HTTPException(status_code=404, detail=f"No metrics found for session {session_id}")
        return asdict(session_metrics)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/sessions")
async def get_all_session_metrics():
    """
    Get metrics for all sessions
    """
    try:
        from dataclasses import asdict
        session_ids = metrics_tracker.get_all_session_ids()

        sessions_data = {}
        for session_id in session_ids:
            session_metrics = metrics_tracker.get_session_metrics(session_id)
            if session_metrics:
                sessions_data[session_id] = asdict(session_metrics)

        return {
            "sessions": sessions_data,
            "overall": asdict(metrics_tracker.get_aggregated_metrics())
        }
    except Exception as e:
        logger.error(f"Error getting all session metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/queries")
async def get_all_query_metrics():
    """
    Get detailed metrics for all individual queries
    """
    try:
        from dataclasses import asdict
        queries = [asdict(m) for m in metrics_tracker.query_metrics]
        return {
            "total_queries": len(queries),
            "queries": queries
        }
    except Exception as e:
        logger.error(f"Error getting query metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/metrics/export")
async def export_metrics(filepath: Optional[str] = None):
    """
    Export all metrics to a JSON file
    """
    try:
        if filepath is None:
            filepath = f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        metrics_tracker.export_to_json(filepath)
        return {"success": True, "filepath": filepath}
    except Exception as e:
        logger.error(f"Error exporting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/metrics/clear")
async def clear_metrics():
    """
    Clear all collected metrics (useful for testing)
    """
    try:
        metrics_tracker.clear()
        return {"success": True, "message": "All metrics cleared"}
    except Exception as e:
        logger.error(f"Error clearing metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ======================
# PHASE 2: EVALUATION ENDPOINTS
# ======================

@app.post("/api/evaluate/ground-truth/{session_id}")
async def generate_ground_truth(
    session_id: str,
    num_questions: int = 20,
    document_name: Optional[str] = None
):
    """
    Generate ground truth Q&A pairs for evaluation

    Args:
        session_id: Session ID
        num_questions: Number of Q&A pairs to generate (default: 20)
        document_name: Optional specific document name

    Cost: ~$0.002 per 20 questions (Gemini Flash)
    """
    try:
        from backend.evaluation.ground_truth_generator import GroundTruthGenerator
        from backend.document_loader import MultiModalDocumentLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # Get session info
        session_info = session_manager.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")

        if not session_info['documents']:
            raise HTTPException(status_code=400, detail="No documents in session")

        # Get document to process (first one if not specified)
        target_doc = document_name or session_info['documents'][0]['filename']

        # Load and chunk the document
        # Note: In production, you'd retrieve this from your stored chunks
        # For now, we'll need to re-process
        logger.info(f"Generating ground truth for {target_doc} in session {session_id}")

        # This is a simplified version - in production you'd get chunks from storage
        raise HTTPException(
            status_code=501,
            detail="Please use the test script for now. Full integration coming in next update."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ground truth generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate/retrieval/{session_id}")
async def evaluate_retrieval_metrics(
    session_id: str,
    use_ground_truth: bool = True,
    max_queries: int = 10
):
    """
    Evaluate retrieval metrics (Recall, Precision, MRR, nDCG)

    Args:
        session_id: Session ID
        use_ground_truth: Use stored ground truth (if available)
        max_queries: Maximum number of queries to evaluate

    Returns:
        Retrieval metrics (Recall@5/10, Precision@5/10, MRR, nDCG@5/10)

    Cost: $0 (uses embeddings only)
    """
    try:
        from backend.evaluation.ground_truth_generator import GroundTruthGenerator
        from backend.evaluation.retrieval_metrics import RetrievalMetrics
        from dataclasses import asdict

        # Load ground truth
        gt_generator = GroundTruthGenerator()
        ground_truth = gt_generator.load_ground_truth(session_id)

        if not ground_truth:
            raise HTTPException(
                status_code=404,
                detail="No ground truth found. Generate it first using /api/evaluate/ground-truth"
            )

        # Get retriever
        retriever = session_manager.get_session_retriever(session_id)
        if not retriever:
            raise HTTPException(status_code=404, detail="No retriever found for session")

        # Evaluate
        evaluator = RetrievalMetrics()

        qa_pairs = ground_truth['qa_pairs'][:max_queries]
        logger.info(f"Evaluating {len(qa_pairs)} queries for retrieval metrics")

        for idx, qa_pair in enumerate(qa_pairs, 1):
            # Retrieve documents
            retrieved_docs = retriever.invoke(qa_pair['question'])

            # Extract chunk IDs from retrieved documents
            retrieved_ids = []
            logger.info(f"\n🔍 Query {idx}: {qa_pair['question'][:60]}...")
            logger.info(f"   Expected chunks: {qa_pair['relevant_chunk_ids']}")
            logger.info(f"   Retrieved {len(retrieved_docs)} documents:")

            for i, doc in enumerate(retrieved_docs[:10], 1):  # Log top 10
                chunk_id = doc.metadata.get('chunk_id', f"chunk_{doc.metadata.get('chunk_index', 0)}")
                retrieved_ids.append(chunk_id)
                logger.info(f"      #{i}: {chunk_id} | metadata: {doc.metadata}")

            # Get relevant chunk IDs from ground truth
            relevant_ids = set(qa_pair['relevant_chunk_ids'])

            # Log matching
            matches = set(retrieved_ids[:10]) & relevant_ids
            logger.info(f"   ✅ Matched chunks in top-10: {matches if matches else 'NONE'}")

            # Add query result
            evaluator.add_query_result(
                retrieved_docs=retrieved_ids,
                relevant_docs=relevant_ids
            )

        # Compute results
        metrics = evaluator.compute_metrics()

        logger.info(f"✅ Retrieval evaluation complete: {metrics.num_queries} queries")

        return {
            "success": True,
            "session_id": session_id,
            "num_queries_evaluated": metrics.num_queries,
            "metrics": asdict(metrics),
            "evaluation_time_s": 0,
            "cost_estimate": "$0.00"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrieval evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate/ragas/{session_id}")
async def evaluate_ragas_metrics(
    session_id: str,
    sample_size: int = 5,
    enable_all_metrics: bool = True
):
    """
    Run RAGAS evaluation (Context Precision, Context Recall, Faithfulness, Relevancy)

    Args:
        session_id: Session ID
        sample_size: Number of queries to sample (default: 5)
        enable_all_metrics: Enable all RAGAS metrics

    Returns:
        RAGAS evaluation results

    Cost: ~$0.004 per 5 queries (uses Groq)
    """
    try:
        from backend.evaluation.ground_truth_generator import GroundTruthGenerator
        from backend.evaluation.enhanced_ragas_evaluator import EnhancedRAGASEvaluator
        from dataclasses import asdict
        import random

        # Load ground truth
        gt_generator = GroundTruthGenerator()
        ground_truth = gt_generator.load_ground_truth(session_id)

        if not ground_truth:
            raise HTTPException(
                status_code=404,
                detail="No ground truth found. Generate it first using /api/evaluate/ground-truth"
            )

        # Sample Q&A pairs
        qa_pairs = random.sample(
            ground_truth['qa_pairs'],
            min(sample_size, len(ground_truth['qa_pairs']))
        )

        logger.info(f"Running RAGAS evaluation on {len(qa_pairs)} queries")

        # Run queries through RAG pipeline
        questions = []
        ground_truth_answers = []
        generated_answers = []
        contexts_list = []

        for qa_pair in qa_pairs:
            # Process question
            result = rag_workflow.process_question_for_session(
                qa_pair['question'],
                session_id
            )

            questions.append(qa_pair['question'])
            ground_truth_answers.append(qa_pair['answer'])
            generated_answers.append(result.get('solution', ''))

            # Get contexts
            docs = result.get('documents', [])
            contexts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs]
            contexts_list.append(contexts)

        # Run RAGAS evaluation in thread pool to avoid nested async
        import asyncio

        def run_ragas_eval():
            evaluator = EnhancedRAGASEvaluator(
                use_groq=True,
                enable_all_metrics=enable_all_metrics
            )
            return evaluator.evaluate_batch(
                questions=questions,
                answers=generated_answers,
                contexts_list=contexts_list,
                ground_truths=ground_truth_answers
            )

        ragas_result = await asyncio.to_thread(run_ragas_eval)

        logger.info(f"✅ RAGAS evaluation complete")

        return {
            "success": True,
            "session_id": session_id,
            "num_queries": len(questions),
            "metrics": {
                "faithfulness": ragas_result.faithfulness,
                "answer_relevance": ragas_result.answer_relevance,
                "context_precision": ragas_result.context_precision,
                "context_recall": ragas_result.context_recall,
                "context_relevancy": ragas_result.context_relevancy,
                "answer_correctness": ragas_result.answer_correctness,
                "answer_similarity": ragas_result.answer_similarity,
            },
            "evaluation_time_s": ragas_result.evaluation_time_s,
            "has_ground_truth": ragas_result.has_ground_truth
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAGAS evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evaluate/complete/{session_id}")
async def get_complete_evaluation(session_id: str):
    """
    Get complete evaluation: Phase 1 (runtime) + Phase 2 (retrieval + RAGAS)

    Returns all available metrics for a session in one call
    """
    try:
        from dataclasses import asdict

        # Phase 1: Runtime metrics
        phase1_metrics = metrics_tracker.get_session_metrics(session_id)

        # Phase 2: Check if ground truth exists
        from backend.evaluation.ground_truth_generator import GroundTruthGenerator
        gt_generator = GroundTruthGenerator()
        ground_truth = gt_generator.load_ground_truth(session_id)

        response = {
            "session_id": session_id,
            "has_ground_truth": ground_truth is not None,
            "phase1_runtime_metrics": asdict(phase1_metrics) if phase1_metrics else None,
            "phase2_available": ground_truth is not None
        }

        if ground_truth:
            response["ground_truth_info"] = {
                "num_qa_pairs": ground_truth['num_qa_pairs'],
                "generated_at": ground_truth['generated_at'],
                "document_name": ground_truth['document_name']
            }

        return response

    except Exception as e:
        logger.error(f"Complete evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
