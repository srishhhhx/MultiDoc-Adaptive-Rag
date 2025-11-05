"""
State management for LangGraph RAG workflow

This module defines the state structure used throughout the LangGraph RAG
workflow. The GraphState tracks all the information needed as the system
processes questions and generates answers.

The state includes:
- Question and answer data
- Document retrieval results
- Search method tracking
- Evaluation scores and metrics
- Error handling information
- Streaming event types for real-time updates

This state management approach is essential for LangGraph RAG implementations,
providing clear data flow and enabling complex workflow orchestration.
"""
from typing import List, TypedDict, Optional, Dict, Any, Literal

class GraphState(TypedDict):
    """
    State structure for LangGraph RAG workflow
    
    This defines all the data that flows through the LangGraph RAG pipeline.
    Each step in the workflow can read from and write to this state, allowing
    for complex decision-making and proper error handling.
    
    Used throughout the RAG workflow to maintain context and enable
    conditional logic based on processing results.
    """
    
    question: str
    original_question: Optional[str]  # Preserve user's original query for context assessment
    solution: str
    online_search: bool
    documents: List[str]
    search_method: Optional[str]  # 'documents' or 'online'
    document_evaluations: Optional[List[Dict[str, Any]]]  # Store document evaluation results
    document_relevance_score: Optional[Dict[str, Any]]  # Store document relevance check
    question_relevance_score: Optional[Dict[str, Any]]  # Store question relevance check
    
    # New fields for Query Analysis Router
    execution_plan: Optional[List[Dict[str, str]]]  # List of tasks with tool and query
    vectorstore_results: Optional[List[str]]  # Results from vectorstore retrieval tasks
    web_search_results: Optional[List[str]]  # Results from web search tasks
    combined_context: Optional[str]  # Combined context from all tools
    metadata: Optional[Dict[str, Any]]  # Extracted metadata from query analysis
    
    # Circuit breaker for infinite loop prevention
    generation_attempts: Optional[int]  # Track number of answer generation attempts
    rewrite_attempts: Optional[int]  # Track number of query rewrite attempts
    
    # Query rewriting loop fields
    context_assessment: Optional[str]  # Result of context sufficiency assessment ("sufficient" or "insufficient")
    context_assessment_json: Optional[Dict[str, Any]]  # Full JSON gap analysis report from assessment
    rerank_completed: Optional[bool]  # Flag to indicate if reranking was successfully completed
    
    # Relevance score caching for API optimization
    cached_document_relevance: Optional[Dict[str, Any]]  # Cached document relevance score
    cached_question_relevance: Optional[Dict[str, Any]]  # Cached question relevance score
    cached_grounding_source: Optional[str]  # Cached grounding source from batch validation (DOCUMENT_ONLY/WEB_ONLY/HYBRID/NONE)
    cache_context_signature: Optional[str]  # Hash of context used for cache validation


# ========== STREAMING EVENT TYPES ==========
# These types define the structure of events sent to the frontend during streaming

class StreamEventBase(TypedDict):
    """Base class for all streaming events"""
    type: str
    timestamp: Optional[float]


class ProvisionalTokenEvent(StreamEventBase):
    """Event for streaming answer tokens as they're generated"""
    type: Literal["provisional_token"]
    content: str
    attempt: int


class StageEvent(StreamEventBase):
    """Event for pipeline stage updates"""
    type: Literal["stage"]
    stage: Literal["analyzing", "retrieving", "generating", "validating"]
    message: str


class ValidationSuccessEvent(StreamEventBase):
    """Event when answer passes quality checks"""
    type: Literal["validation_success"]
    message: str


class RewriteEvent(StreamEventBase):
    """Event when hallucination is detected and rewrite is triggered"""
    type: Literal["rewrite"]
    reason: str
    attempt: int
    max_attempts: int


class FinalAnswerEvent(StreamEventBase):
    """Event containing the final validated answer"""
    type: Literal["final_answer"]
    content: str
    total_attempts: int
    document_relevance: Optional[Dict[str, Any]]
    question_relevance: Optional[Dict[str, Any]]


class ErrorEvent(StreamEventBase):
    """Event when an error occurs"""
    type: Literal["error"]
    message: str
    recoverable: bool


class EndEvent(StreamEventBase):
    """Event signaling the end of the stream"""
    type: Literal["end"]
    success: bool


# Union type for all possible streaming events
StreamEvent = ProvisionalTokenEvent | StageEvent | ValidationSuccessEvent | RewriteEvent | FinalAnswerEvent | ErrorEvent | EndEvent