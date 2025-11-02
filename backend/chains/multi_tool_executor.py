"""
Multi-Tool Execution Engine for Query Analysis Router

This module executes the structured plans created by the Query Analysis Router.
It can run multiple tools (vectorstore retrieval and web search) in parallel
or sequence, then combine the results for final answer generation.
"""

import logging
import concurrent.futures
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from backend.config import TAVILY_SEARCH_RESULTS
from backend.chains.rerank_documents import rerank_documents

# Configure logging
logger = logging.getLogger(__name__)


class MultiToolExecutor:
    """
    Executes multi-tool plans from the Query Analysis Router
    
    This class takes an execution plan (list of tasks with tools and queries)
    and executes each task using the appropriate tool, then combines the results.
    """
    
    def __init__(self, retriever=None, self_query_retriever=None):
        """
        Initialize the multi-tool executor
        
        Args:
            retriever: The basic vectorstore retriever for document searches
            self_query_retriever: The self-query retriever for metadata-aware searches
        """
        self.retriever = retriever
        self.self_query_retriever = self_query_retriever
        self.web_search_tool = TavilySearchResults(max_results=TAVILY_SEARCH_RESULTS)
        # Per-task reranking configuration
        self.RERANK_TOP_K = 3  # Return top 3 docs per task for balanced multi-topic context
        
    def execute_plan(self, execution_plan: List[Dict[str, str]], metadata: Optional[Dict[str, Any]] = None, 
                    state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a complete execution plan with PARALLEL per-task reranking and context accumulation
        
        This method now runs all tasks in parallel using ThreadPoolExecutor.
        Each vectorstore_retrieval task performs: Retrieve (20 docs) → Rerank (top 3)
        
        OPTIMIZATION: Accumulates context from previous rewrite loops to preserve successful retrievals
        
        Args:
            execution_plan: List of tasks with 'tool' and 'query' fields
            metadata: Extracted metadata from query analysis for optimization
            state: Optional state dict for context accumulation across rewrite loops
            
        Returns:
            Dict containing results from all tools and combined context
        """
        logger.info(f"🚀 PARALLEL EXECUTION: Running {len(execution_plan)} tasks concurrently")
        if metadata:
            logger.info(f"Using metadata for optimization: {metadata}")
        
        final_docs = []
        final_web_results = []
        
        # Use ThreadPoolExecutor to run tasks concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks to the pool
            futures = [
                executor.submit(self._process_task, task, metadata) 
                for task in execution_plan
            ]
            
            # Collect results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    if result["type"] == "docs":
                        final_docs.extend(result["data"])
                        logger.info(f"✅ Task {i+1} complete: Added {len(result['data'])} reranked docs")
                    elif result["type"] == "web":
                        final_web_results.extend(result["data"])
                        logger.info(f"✅ Task {i+1} complete: Added {len(result['data'])} web results")
                except Exception as e:
                    logger.error(f"❌ Error in parallel task {i+1}: {e}")
                    continue
        
        logger.info(f"--- PARALLEL EXECUTION COMPLETE ---")
        logger.info(f"  Total reranked docs collected: {len(final_docs)}")
        logger.info(f"  Total web results collected: {len(final_web_results)}")
        
        # --- CONTEXT ACCUMULATION OPTIMIZATION ---
        # Get context from previous loops (if any)
        existing_docs = state.get("documents", []) if state else []
        existing_web = state.get("web_search_results", []) if state else []
        
        # Combine old and new. Use new docs first for relevance.
        accumulated_docs = final_docs + existing_docs
        accumulated_web = final_web_results + existing_web
        
        # De-duplicate lists based on page_content
        final_docs_deduped = list({doc.page_content: doc for doc in accumulated_docs}.values())
        final_web_deduped = list({result.page_content if hasattr(result, 'page_content') else str(result): result 
                                  for result in accumulated_web}.values())
        
        logger.info(f"  Accumulated context: {len(final_docs_deduped)} docs (deduped), {len(final_web_deduped)} web (deduped)")
        # --- END OPTIMIZATION ---
        
        # Combine all results
        combined_context = self._combine_results(final_docs_deduped, final_web_deduped)
        
        return {
            "vectorstore_results": final_docs_deduped,  # Return accumulated docs
            "web_search_results": final_web_deduped,  # Return accumulated web results
            "combined_context": combined_context,
            "total_sources": len(final_docs_deduped) + len(final_web_deduped),
            "rerank_completed": True  # Flag that reranking was done per-task
        }
    
    def _process_task(self, task: Dict[str, str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a single task from the execution plan (with per-task reranking)
        
        For vectorstore_retrieval tasks:
        1. Retrieve initial documents (k=20)
        2. Immediately rerank to top-k (default 3)
        3. Return small, hyper-relevant set
        
        For web_search tasks:
        1. Execute search
        2. Return results directly
        
        Args:
            task: Task dictionary with 'tool', 'query', and optional 'source_document'
            metadata: Extracted metadata for optimization
            
        Returns:
            Dict with 'type' ('docs' or 'web') and 'data' (list of documents)
        """
        tool = task.get("tool")
        query = task.get("query")
        task_source_document = task.get("source_document")
        
        if tool == "vectorstore_retrieval":
            logger.info(f"📚 Processing vectorstore task: {query[:50]}...")
            if task_source_document:
                logger.info(f"   🎯 Task-specific source filter: {task_source_document}")
            
            # Step 1: Retrieve initial documents
            initial_docs = self._execute_vectorstore_task(query, metadata, task_source_document)
            
            if not initial_docs:
                logger.warning(f"   ⚠️  No documents retrieved for task")
                return {"type": "docs", "data": []}
            
            # Step 2: Rerank ONLY these results
            logger.info(f"   🔄 Reranking {len(initial_docs)} docs for this task (top_k={self.RERANK_TOP_K})")
            reranked_docs = rerank_documents(
                query=query,
                documents=initial_docs,
                top_k=self.RERANK_TOP_K
            )
            logger.info(f"   ✅ Rerank complete: Selected top {len(reranked_docs)} docs for this task")
            
            return {"type": "docs", "data": reranked_docs}
            
        elif tool == "web_search":
            logger.info(f"🌐 Processing web_search task: {query[:50]}...")
            result = self._execute_web_search_task(query)
            return {"type": "web", "data": result if result else []}
        
        logger.warning(f"Unknown tool: {tool}")
        return {"type": "none", "data": []}
    
    def _execute_vectorstore_task(self, query: str, metadata: Optional[Dict[str, Any]] = None, task_source_document: Optional[str] = None) -> Optional[List[Document]]:
        """
        Execute a vectorstore retrieval task with metadata-aware optimization
        
        Args:
            query: The search query for the vectorstore
            metadata: Extracted metadata for optimization (legacy)
            task_source_document: Task-specific source document filter (NEW)
            
        Returns:
            List of retrieved documents or None if no retriever available
        """
        if not self.retriever:
            logger.warning("No vectorstore retriever available")
            return None
        
        # **CRITICAL CHANGE: Prioritize task-specific source_document over global metadata**
        source_document = task_source_document or (metadata.get("source_document") if metadata else None)
        
        if source_document and self.self_query_retriever:
            # OPTIMIZED PATH: Use direct metadata filter, skip Self-Query LLM call
            logger.info(f"🚀 METADATA OPTIMIZATION: Using direct filter for document '{source_document}'")
            logger.info("   Skipping Self-Query LLM call - using pre-extracted metadata")
            
            try:
                # Construct metadata filter directly
                metadata_filter = {"source": source_document}
                
                # Use the underlying vectorstore with direct filter
                vectorstore = self.self_query_retriever.vectorstore
                documents = vectorstore.similarity_search(
                    query, 
                    k=20,  # Get more documents for reranking
                    filter=metadata_filter
                )
                
                logger.info(f"✅ Retrieved {len(documents)} documents using metadata filter")
                logger.info(f"   Filter applied: {metadata_filter}")
                
                # Log document previews for debugging
                for i, doc in enumerate(documents[:3]):  # Show first 3 docs
                    preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    source = doc.metadata.get("source", "unknown")
                    logger.info(f"   Doc {i+1} (source: {source}): {preview}")
                
                return documents
                
            except Exception as e:
                logger.error(f"❌ Error in metadata-optimized retrieval: {e}")
                logger.info("   Falling back to Self-Query retriever")
                # Fall through to Self-Query retriever
        
        # **NEW FALLBACK PATH: Try basic retriever with metadata filter if source_document specified**
        if source_document and not self.self_query_retriever:
            logger.info(f"🎯 BASIC RETRIEVER + FILTER: Attempting metadata filter for document '{source_document}'")
            try:
                # Try to use the basic retriever with metadata filtering
                if hasattr(self.retriever, 'vectorstore'):
                    # Access underlying vectorstore for filtering
                    vectorstore = self.retriever.vectorstore
                    metadata_filter = {"source": source_document}
                    documents = vectorstore.similarity_search(
                        query, 
                        k=20,  # Get more documents for reranking
                        filter=metadata_filter
                    )
                    logger.info(f"✅ Basic retriever with filter returned {len(documents)} documents")
                    logger.info(f"   Filter applied: {metadata_filter}")
                    
                    # Log document previews for debugging
                    for i, doc in enumerate(documents[:3]):  # Show first 3 docs
                        preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                        source = doc.metadata.get("source", "unknown")
                        logger.info(f"   Doc {i+1} (source: {source}): {preview}")
                    
                    return documents
                else:
                    logger.warning("   Basic retriever doesn't support metadata filtering - falling back to unfiltered search")
            except Exception as e:
                logger.error(f"❌ Error in basic retriever metadata filtering: {e}")
                logger.info("   Falling back to unfiltered basic retrieval")
        
        # ORIGINAL FALLBACK PATH: Use Self-Query retriever or basic retriever
        try:
            if self.self_query_retriever and not source_document:
                logger.info(f"🔍 Using Self-Query retriever for: {query}")
                documents = self.self_query_retriever.invoke(query)
                logger.info(f"   Self-Query retriever returned {len(documents)} documents")
            else:
                logger.info(f"🔍 Using basic retriever for: {query}")
                documents = self.retriever.invoke(query)
                logger.info(f"   Basic retriever returned {len(documents)} documents")
            
            # Log document previews for debugging
            for i, doc in enumerate(documents[:3]):  # Show first 3 docs
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                source = doc.metadata.get("source", "unknown")
                logger.info(f"   Doc {i+1} (source: {source}): {preview}")
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error in vectorstore retrieval: {e}")
            return None
    
    def _execute_web_search_task(self, query: str) -> Optional[List[Document]]:
        """
        Execute a web search task with robust error handling
        
        CRITICAL: This method is now fault-tolerant to handle:
        - Bad queries from Rewrite Query node (multi-paragraph text)
        - Tavily API errors (returns string instead of list)
        - AttributeError when parsing non-dict responses
        - Any other API failures
        
        Args:
            query: The search query for web search
            
        Returns:
            List of search result documents, empty list if search fails, or None
        """
        try:
            # Validate query length (Tavily has limits)
            if len(query) > 500:
                logger.warning(f"⚠️ Query too long ({len(query)} chars), truncating to 500 chars")
                logger.warning(f"   Original query preview: {query[:100]}...")
                query = query[:500]
            
            logger.info(f"Searching web for: {query[:100]}...")
            search_results = self.web_search_tool.invoke(query)
            
            # CRITICAL: Check if Tavily returned an error string instead of list
            if not isinstance(search_results, list):
                logger.error(f"❌ Web search returned non-list type: {type(search_results)}")
                logger.error(f"   Response: {str(search_results)[:200]}...")
                logger.warning("   This usually means the API returned an error message")
                return []  # Return empty list to allow pipeline to continue
            
            # Convert search results to Document objects
            documents = []
            for i, result in enumerate(search_results):
                try:
                    # CRITICAL: Validate that each result is a dict
                    if not isinstance(result, dict):
                        logger.warning(f"⚠️ Skipping web result {i+1}: Not a dict (type: {type(result)})")
                        continue
                    
                    content = result.get("content", "")
                    title = result.get("title", "")
                    url = result.get("url", "")
                    
                    # Skip results with no content
                    if not content and not title:
                        logger.warning(f"⚠️ Skipping web result {i+1}: Empty content and title")
                        continue
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=f"Title: {title}\n\nContent: {content}",
                        metadata={
                            "source": url,
                            "title": title,
                            "type": "web_search"
                        }
                    )
                    documents.append(doc)
                    
                except AttributeError as e:
                    # CRITICAL: This is the specific error we're fixing
                    logger.error(f"❌ AttributeError parsing web result {i+1}: {e}")
                    logger.error(f"   Result type: {type(result)}")
                    logger.error(f"   Result preview: {str(result)[:200]}...")
                    continue  # Skip this result, try next one
                    
                except Exception as e:
                    logger.error(f"❌ Error parsing web result {i+1}: {e}")
                    continue  # Skip this result, try next one
            
            logger.info(f"✅ Retrieved {len(documents)} valid web search results")
            
            # Log search result previews
            for i, doc in enumerate(documents[:2]):  # Show first 2 results
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                logger.info(f"   Web result {i+1} preview: {preview}")
            
            return documents if documents else []  # Return empty list instead of None
            
        except AttributeError as e:
            # CRITICAL: Catch the specific 'str' object has no attribute 'get' error
            logger.error(f"❌ CRITICAL: AttributeError in web search: {e}")
            logger.error(f"   This usually means Tavily returned a string error instead of list")
            logger.error(f"   Failed query: {query[:100]}...")
            return []  # Return empty list to allow pipeline to continue
            
        except Exception as e:
            logger.error(f"❌ ERROR in web search task: {e}")
            logger.error(f"   Failed query: {query[:100]}...")
            logger.error(f"   Error type: {type(e).__name__}")
            return []  # Return empty list to allow pipeline to continue
    
    def _combine_results(self, vectorstore_results: List[Document], web_search_results: List[Document]) -> str:
        """
        Combine results from all tools into a unified context
        
        Args:
            vectorstore_results: Documents from vectorstore searches
            web_search_results: Documents from web searches
            
        Returns:
            Combined context string for answer generation
        """
        context_parts = []
        
        # Add vectorstore results
        if vectorstore_results:
            context_parts.append("=== DOCUMENT INFORMATION ===")
            for i, doc in enumerate(vectorstore_results):
                context_parts.append(f"Document {i+1}:")
                context_parts.append(doc.page_content)
                context_parts.append("")  # Empty line for separation
        
        # Add web search results
        if web_search_results:
            context_parts.append("=== WEB SEARCH INFORMATION ===")
            for i, doc in enumerate(web_search_results):
                title = doc.metadata.get("title", f"Web Result {i+1}")
                context_parts.append(f"{title}:")
                context_parts.append(doc.page_content)
                context_parts.append("")  # Empty line for separation
        
        combined_context = "\n".join(context_parts)
        
        logger.info(f"Combined context length: {len(combined_context)} characters")
        logger.info(f"Sources: {len(vectorstore_results)} documents, {len(web_search_results)} web results")
        
        return combined_context
    
    def get_source_summary(self, vectorstore_results: List[Document], web_search_results: List[Document]) -> str:
        """
        Generate a summary of sources used
        
        Args:
            vectorstore_results: Documents from vectorstore
            web_search_results: Documents from web search
            
        Returns:
            Human-readable source summary
        """
        sources = []
        
        if vectorstore_results:
            sources.append(f"{len(vectorstore_results)} document(s)")
            
        if web_search_results:
            sources.append(f"{len(web_search_results)} web result(s)")
        
        if not sources:
            return "No sources found"
            
        return f"Sources: {', '.join(sources)}"


def create_executor_with_retriever(retriever, self_query_retriever=None) -> MultiToolExecutor:
    """
    Factory function to create executor with retrievers
    
    Args:
        retriever: The basic vectorstore retriever
        self_query_retriever: Optional self-query retriever for metadata optimization
        
    Returns:
        Configured MultiToolExecutor instance
    """
    return MultiToolExecutor(retriever=retriever, self_query_retriever=self_query_retriever)


# Test function for development
def test_multi_tool_executor():
    """Test the multi-tool executor with sample plans"""
    executor = MultiToolExecutor()  # No retriever for testing
    
    test_plans = [
        [
            {"tool": "web_search", "query": "current weather"},
            {"tool": "vectorstore_retrieval", "query": "document summary"}
        ],
        [
            {"tool": "web_search", "query": "latest AI developments"}
        ],
        [
            {"tool": "vectorstore_retrieval", "query": "main findings"}
        ]
    ]
    
    for i, plan in enumerate(test_plans):
        print(f"\nTest Plan {i+1}: {plan}")
        result = executor.execute_plan(plan)
        print(f"Result: {result}")


if __name__ == "__main__":
    test_multi_tool_executor()
