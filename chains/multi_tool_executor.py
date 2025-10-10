"""
Multi-Tool Execution Engine for Query Analysis Router

This module executes the structured plans created by the Query Analysis Router.
It can run multiple tools (vectorstore retrieval and web search) in parallel
or sequence, then combine the results for final answer generation.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from config import TAVILY_SEARCH_RESULTS

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
        
    def execute_plan(self, execution_plan: List[Dict[str, str]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a complete execution plan with metadata-aware optimization
        
        Args:
            execution_plan: List of tasks with 'tool' and 'query' fields
            metadata: Extracted metadata from query analysis for optimization
            
        Returns:
            Dict containing results from all tools and combined context
        """
        logger.info(f"Executing plan with {len(execution_plan)} tasks")
        if metadata:
            logger.info(f"Using metadata for optimization: {metadata}")
        
        vectorstore_results = []
        web_search_results = []
        
        for i, task in enumerate(execution_plan):
            tool = task["tool"]
            query = task["query"]
            
            logger.info(f"Executing task {i+1}/{len(execution_plan)}: {tool} - {query}")
            
            try:
                if tool == "vectorstore_retrieval":
                    result = self._execute_vectorstore_task(query, metadata)
                    if result:
                        vectorstore_results.extend(result)
                        
                elif tool == "web_search":
                    result = self._execute_web_search_task(query)
                    if result:
                        web_search_results.extend(result)
                        
                else:
                    logger.warning(f"Unknown tool: {tool}")
                    
            except Exception as e:
                logger.error(f"Error executing task {i+1} ({tool}): {e}")
                continue
        
        # Combine all results
        combined_context = self._combine_results(vectorstore_results, web_search_results)
        
        return {
            "vectorstore_results": vectorstore_results,
            "web_search_results": web_search_results,
            "combined_context": combined_context,
            "total_sources": len(vectorstore_results) + len(web_search_results)
        }
    
    def _execute_vectorstore_task(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[List[Document]]:
        """
        Execute a vectorstore retrieval task with metadata-aware optimization
        
        Args:
            query: The search query for the vectorstore
            metadata: Extracted metadata for optimization
            
        Returns:
            List of retrieved documents or None if no retriever available
        """
        if not self.retriever:
            logger.warning("No vectorstore retriever available")
            return None
        
        # CRITICAL: Check for metadata-based optimization
        source_document = metadata.get("source_document") if metadata else None
        
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
        
        # FALLBACK PATH: Use Self-Query retriever or basic retriever
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
        Execute a web search task
        
        Args:
            query: The search query for web search
            
        Returns:
            List of search result documents or None if search fails
        """
        try:
            logger.info(f"Searching web for: {query}")
            search_results = self.web_search_tool.invoke(query)
            
            # Convert search results to Document objects
            documents = []
            for result in search_results:
                content = result.get("content", "")
                title = result.get("title", "")
                url = result.get("url", "")
                
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
            
            logger.info(f"Retrieved {len(documents)} web search results")
            
            # Log search result previews
            for i, doc in enumerate(documents[:2]):  # Show first 2 results
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                logger.info(f"Web result {i+1} preview: {preview}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return None
    
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
