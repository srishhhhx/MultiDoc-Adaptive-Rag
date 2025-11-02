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

This state management approach is essential for LangGraph RAG implementations,
providing clear data flow and enabling complex workflow orchestration.
"""
from typing import List, TypedDict, Optional, Dict, Any

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