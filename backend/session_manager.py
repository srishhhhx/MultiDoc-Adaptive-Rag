"""
Session Management for Advanced RAG System

This module handles session-based document management, conversation memory,
and multi-document support. Each session maintains its own FAISS index
and conversation history for complete isolation.
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from backend.config import FAISS_INDEX_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationExchange:
    """Single question-answer exchange in a conversation"""
    question: str
    answer: str
    timestamp: datetime
    search_method: str
    online_search: bool
    document_evaluations: Optional[List[Dict]] = None
    question_relevance_score: Optional[Dict] = None
    document_relevance_score: Optional[Dict] = None


@dataclass
class SessionData:
    """Data structure for a single RAG session"""
    session_id: str
    collection_name: str
    created_at: datetime
    last_activity: datetime
    documents: List[Dict] = field(default_factory=list)  # File metadata
    conversation_history: List[ConversationExchange] = field(default_factory=list)
    retriever: Optional[Any] = None
    total_chunks: int = 0


class SessionManager:
    """
    Manages RAG sessions with isolated document indexes and conversation memory
    
    Features:
    - Session-based FAISS indexes for document isolation
    - Conversation memory within sessions
    - Multi-document support per session
    - Automatic session cleanup
    """
    
    def __init__(self, session_timeout_hours: int = 24):
        self.active_sessions: Dict[str, SessionData] = {}
        self.session_timeout = timedelta(hours=session_timeout_hours)
        # FAISS doesn't need a persistent client like ChromaDB
        # self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        logger.info("SessionManager initialized")
    
    def create_session(self) -> str:
        """Create a new RAG session with isolated document index"""
        session_id = str(uuid.uuid4())
        collection_name = f"session_{session_id}"
        
        # Create session data
        session_data = SessionData(
            session_id=session_id,
            collection_name=collection_name,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        # Store session
        self.active_sessions[session_id] = session_data
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data if it exists and is not expired"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # Check if session is expired
        if datetime.now() - session.last_activity > self.session_timeout:
            logger.info(f"Session {session_id} expired, cleaning up")
            self.cleanup_session(session_id)
            return None
        
        # Update last activity
        session.last_activity = datetime.now()
        return session
    
    def add_document_to_session(self, session_id: str, file_metadata: Dict):
        """Add document metadata to session"""
        session = self.get_session(session_id)
        if session:
            # Add unique document ID if not present
            if 'document_id' not in file_metadata:
                file_metadata['document_id'] = f"doc_{len(session.documents)}_{file_metadata['filename']}"
            session.documents.append(file_metadata)
            session.total_chunks += file_metadata.get('chunks', 0)
            logger.info(f"Added document {file_metadata['filename']} to session {session_id}")
    
    def remove_document_from_session(self, session_id: str, document_id: str):
        """Remove document from session by document ID"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Find and remove the document
        document_to_remove = None
        for i, doc in enumerate(session.documents):
            if doc.get('document_id') == document_id or doc.get('filename') == document_id:
                document_to_remove = session.documents.pop(i)
                break
        
        if document_to_remove:
            # Update total chunks
            session.total_chunks -= document_to_remove.get('chunks', 0)
            logger.info(f"Removed document {document_to_remove['filename']} from session {session_id}")
            return document_to_remove
        
        logger.warning(f"Document {document_id} not found in session {session_id}")
        return None
    
    def get_session_documents(self, session_id: str):
        """Get all documents in a session"""
        session = self.get_session(session_id)
        return session.documents if session else []
    
    
    def add_conversation_exchange(self, session_id: str, exchange: ConversationExchange):
        """Add a question-answer exchange to session conversation history"""
        session = self.get_session(session_id)
        if session:
            session.conversation_history.append(exchange)
            logger.info(f"Added conversation exchange to session {session_id}")
    
    def get_conversation_context(self, session_id: str, last_n: int = 3) -> str:
        """Get recent conversation context for better responses"""
        session = self.get_session(session_id)
        if not session or not session.conversation_history:
            return ""
        
        recent_exchanges = session.conversation_history[-last_n:]
        context_parts = []
        
        for exchange in recent_exchanges:
            context_parts.append(f"Previous Q: {exchange.question}")
            context_parts.append(f"Previous A: {exchange.answer[:200]}...")  # Truncate long answers
        
        return "\n".join(context_parts)
    
    def set_session_retriever(self, session_id: str, retriever):
        """Set the retriever for a session"""
        session = self.get_session(session_id)
        if session:
            session.retriever = retriever
            logger.info(f"Set retriever for session {session_id}")
    
    def get_session_retriever(self, session_id: str):
        """Get the retriever for a session"""
        session = self.get_session(session_id)
        return session.retriever if session else None
    
    def cleanup_session(self, session_id: str):
        """Clean up session and its FAISS index"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        try:
            # Delete FAISS index files
            import os
            faiss_file = os.path.join(FAISS_INDEX_DIR, f"{session.collection_name}.faiss")
            metadata_file = os.path.join(FAISS_INDEX_DIR, f"{session.collection_name}_metadata.pkl")
            
            if os.path.exists(faiss_file):
                os.remove(faiss_file)
                logger.info(f"Deleted FAISS index: {faiss_file}")
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
                logger.info(f"Deleted FAISS metadata: {metadata_file}")
        except Exception as e:
            logger.warning(f"Failed to delete FAISS files for {session.collection_name}: {e}")
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        logger.info(f"Cleaned up session: {session_id}")
    
    def cleanup_expired_sessions(self):
        """Clean up all expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if current_time - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.cleanup_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_session_info(self, session_id: str):
        """Get detailed session information"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "collection_name": session.collection_name,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "documents": session.documents,
            "total_chunks": session.total_chunks,
            "conversation_count": len(session.conversation_history)
        }
    
    def list_active_sessions(self) -> List[Dict]:
        """List all active sessions (for debugging/admin)"""
        return [self.get_session_info(sid) for sid in self.active_sessions.keys()]

# Global session manager instance
session_manager = SessionManager()
