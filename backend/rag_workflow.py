"""
RAG workflow management using LangGraph

This module implements the core RAG workflow using LangGraph's state management
and graph-based orchestration. It handles the complete flow from question
processing to answer generation, with built-in evaluation and fallback mechanisms.

The LangGraph workflow includes:
- Document retrieval and relevance checking
- Conditional routing between local and online search
- Multi-step answer generation and validation
- Error handling and recovery strategies

This demonstrates practical LangGraph RAG patterns for building robust
question-answering systems with advanced state management and caching.
"""

import hashlib
import time
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from backend.state import GraphState
from backend.chains.generate_answer import generate_chain
from backend.chains.evaluate_groq import evaluate_documents as evaluate_docs  # Groq-optimized
from backend.chains.evaluate_batch import evaluate_documents_batch
from backend.chains.relevance_batch import evaluate_relevance_batch  # Gemini fallback
from backend.chains.relevance_groq import evaluate_relevance_batch_groq  # Groq primary (10-20x faster)
from backend.chains.query_classifier import classify_query
# CRITICAL: Query Analysis REVERTED to Gemini for reliability (Groq 70B failed on complex multi-part queries)
# The 3-11s latency is necessary to ensure the "brain" of the agent functions correctly
from backend.chains.query_analysis_router import analyze_query  # GEMINI (reliable planning)
from backend.chains.multi_tool_executor import MultiToolExecutor
from backend.chains.rerank_documents import rerank_documents
# GROQ-POWERED CHAINS: Keep Groq for evaluation/recovery tasks (proven reliable)
from backend.chains.context_assessment_groq import assess_context_sufficiency  # Groq-optimized
from langchain_community.tools.tavily_search import TavilySearchResults
from backend.config import TAVILY_SEARCH_RESULTS


def _create_context_signature(documents, web_search_results, rerank_completed):
    """
    Create a hash signature of the current context for cache validation.
    
    This helps determine if the validation context has changed since the last
    relevance evaluation, allowing us to reuse cached scores when appropriate.
    """
    context_elements = []
    
    # Add document content hashes
    if documents:
        for doc in documents:
            if hasattr(doc, 'page_content'):
                context_elements.append(doc.page_content[:200])  # First 200 chars for efficiency
            else:
                context_elements.append(str(doc)[:200])
    
    # Add web search results hashes
    if web_search_results:
        for result in web_search_results:
            if hasattr(result, 'page_content'):
                context_elements.append(result.page_content[:200])
            elif isinstance(result, dict) and 'content' in result:
                context_elements.append(result['content'][:200])
            else:
                context_elements.append(str(result)[:200])
    
    # Add rerank status
    context_elements.append(f"rerank:{rerank_completed}")
    
    # Create hash of all context elements
    context_string = "|".join(context_elements)
    return hashlib.md5(context_string.encode()).hexdigest()


class RAGWorkflow:
    """
    Manages the RAG workflow using LangGraph

    This class orchestrates the complete RAG pipeline using LangGraph's state
    management system. It handles document processing, question answering,
    and evaluation with proper error handling and fallback mechanisms.

    The workflow demonstrates key LangGraph RAG patterns:
    - State-based workflow management
    - Conditional routing based on document availability
    - Multi-step evaluation and quality checks
    - Dynamic fallback to online search when needed

    Good for understanding how to build RAG systems with LangGraph in practice.
    """

    def __init__(self):
        self.retriever = None
        self.web_search = TavilySearchResults(max_results=TAVILY_SEARCH_RESULTS)
        self.multi_tool_executor = None  # Will be initialized when retriever is set
        self.session_retriever_key = None
        self.current_session_id = None  # Track current session for document access

    def get_graph(self):
        """Get or create the graph instance (cached for performance)"""
        if self.graph is None:
            self.graph = self._create_graph()
        return self.graph

    def set_retriever(self, retriever):
        """Set the document retriever"""
        self.retriever = retriever
        # Initialize multi-tool executor with the new retriever
        self.multi_tool_executor = MultiToolExecutor(retriever=retriever)

        if retriever is not None:
            print("Retriever set")
        else:
            print("Retriever cleared")

    def get_current_retriever(self):
        """Get the current retriever"""
        return self.retriever
    
    def set_session_id(self, session_id: str):
        """Set the current session ID for document access"""
        self.current_session_id = session_id
    
    def _get_available_documents(self):
        """Get list of available document filenames from current session"""
        if not self.current_session_id:
            return None
        
        try:
            # Import here to avoid circular imports
            from backend.session_manager import session_manager
            
            # Get session documents
            session_documents = session_manager.get_session_documents(self.current_session_id)
            if not session_documents:
                return None
            
            # Extract filenames
            filenames = [doc.get('filename') for doc in session_documents if doc.get('filename')]
            return filenames if filenames else None
            
        except Exception as e:
            print(f"Error getting available documents: {e}")
            return None

    def process_question(self, question):
        """Process a question through the RAG workflow"""
        print(f"STARTING RAG WORKFLOW for question: '{question}'")

        # Ensure we have the most current retriever
        current_retriever = self.get_current_retriever()
        self.set_retriever(current_retriever)

        workflow = self.create_workflow()
        result = workflow.invoke(input={
            "question": question, 
            "original_question": question,  # Preserve original for context assessment
            "rewrite_attempts": 0
        })

        print("RAG WORKFLOW COMPLETED")
        return result

    def create_workflow(self):
        """Create and configure the state graph for handling queries with query rewriting loop"""
        workflow = StateGraph(GraphState)

        # Add nodes for Query Analysis Router pipeline with rewriting loop
        workflow.add_node("Query Analysis", self._analyze_query)
        workflow.add_node("Execute Multi-Tool Plan", self._execute_multi_tool_plan)  # Now includes per-task reranking
        workflow.add_node("Context Assessment", self._assess_context)  # Assess context sufficiency
        workflow.add_node("Rewrite Query", self._rewrite_query)  # Rewrite query for better retrieval
        workflow.add_node("Evaluate Documents", self._analyze_batch)  # Document evaluation for Quality Metrics
        workflow.add_node("Generate Answer", self._generate_answer)
        
        # Legacy nodes for fallback compatibility
        workflow.add_node("Retrieve Documents", self._retrieve)
        workflow.add_node("Search Online", self._search_online)
        # NOTE: Rerank Documents node REMOVED - reranking now happens per-task in Execute Multi-Tool Plan

        # Set entry point to Query Analysis Router
        workflow.set_entry_point("Query Analysis")
        
        # Main flow with query rewriting loop:
        # Query Analysis -> Execute Plan (with per-task reranking) -> Evaluate -> Context Assessment
        workflow.add_edge("Query Analysis", "Execute Multi-Tool Plan")
        workflow.add_edge("Execute Multi-Tool Plan", "Evaluate Documents")  # Direct edge - reranking done per-task
        workflow.add_edge("Evaluate Documents", "Context Assessment")  # Route to assessment
        
        # Context Assessment conditional routing (NEW: Self-correcting loop)
        workflow.add_conditional_edges(
            "Context Assessment",
            self._route_after_assessment,
            {
                "sufficient": "Generate Answer",  # Context is good -> generate answer
                "max_attempts": "Generate Answer",  # Max attempts reached -> generate with what we have
                "insufficient": "Rewrite Query",  # Context insufficient -> rewrite query
            },
        )
        
        # Query rewriting loop: Rewrite Query -> back to Execute Multi-Tool Plan
        workflow.add_edge("Rewrite Query", "Execute Multi-Tool Plan")  # NEW: Loop back for retry
        
        # Answer generation with quality checks
        workflow.add_conditional_edges(
            "Generate Answer",
            self._check_hallucinations,
            {
                "Hallucinations detected": "Generate Answer",
                "Answers Question": END,
                "Question not addressed": "Search Online",
            },
        )
        
        # Fallback edge from Search Online directly to Evaluate (no global rerank needed)
        workflow.add_edge("Search Online", "Evaluate Documents")

        return workflow.compile()

    def _retrieve(self, state: GraphState):
        """Retrieve documents relevant to the user's question"""
        print("GRAPH STATE: Retrieve Documents")
        question = state["question"]

        # Get the current retriever (with fallback to session state)
        current_retriever = self.get_current_retriever()

        # Debug: Print retriever status
        print(f"Current retriever status: {current_retriever is not None}")

        if current_retriever is None:
            print("No retriever available - going to online search")
            return {"documents": [], "question": question, "online_search": True}

        try:
            print(f"Attempting to retrieve documents for question: '{question}'")
            documents = current_retriever.invoke(question)
            print(f"Retrieved {len(documents)} documents from FAISS")

            # Debug: Print document details
            for i, doc in enumerate(documents):
                print(
                    f"Document {i + 1}: {len(doc.page_content)} chars, metadata: {doc.metadata}"
                )
                print(f"Document {i + 1} preview: {doc.page_content[:200]}...")

            return {"documents": documents, "question": question}
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            print("Clearing invalid retriever and falling back to online search")
            # Clear the invalid retriever
            self.retriever = None
            return {"documents": [], "question": question, "online_search": True}

    def _evaluate(self, state: GraphState):
        """Filter documents based on their relevance to the question with document-first approach"""
        print("GRAPH STATE: Grade Documents")
        question = state["question"]
        documents = state["documents"]

        # Check if online search is already required
        online_search = state.get("online_search", False)
        print(f"Evaluating {len(documents)} documents, online_search: {online_search}")

        # Classify the query to determine evaluation strategy
        query_classification = classify_query(question)
        print(f"Query classification: {query_classification}")

        # **BATCHED DOCUMENT EVALUATION OPTIMIZATION**
        print(f"🚀 BATCH EVALUATION: Processing {len(documents)} documents in single API call")
        
        try:
            # Single API call for all documents
            document_evaluations = evaluate_documents_batch(question, documents, query_classification)
            print(f"✅ Batch evaluation successful for {len(document_evaluations)} documents")
            
            # Filter relevant documents based on batch results
            filtered_docs = []
            for i, (document, evaluation) in enumerate(zip(documents, document_evaluations)):
                if evaluation.score.lower() == "yes":
                    filtered_docs.append(document)
            
            print(f"📊 Batch evaluation results: {len(filtered_docs)}/{len(documents)} documents deemed relevant")
            
        except Exception as e:
            print(f"⚠️  Batch evaluation failed: {e}")
            print("🔄 Falling back to individual document evaluation")
            
            # Fallback to individual evaluation
            filtered_docs = []
            document_evaluations = []

            for document in documents:
                response = evaluate_docs.invoke(
                    {
                        "question": question,
                        "document": document.page_content,
                        "query_classification": query_classification,
                    }
                )
                document_evaluations.append(response)

                result = response.score
                if result.lower() == "yes":
                    filtered_docs.append(document)

            print(f"📊 Individual evaluation results: {len(filtered_docs)}/{len(documents)} documents deemed relevant")

        # Determine if online search is needed based on query classification and document relevance
        if query_classification == "DOCUMENT_FIRST":
            # For document-first queries, prioritize documents heavily.
            # Only go online if NO documents were retrieved at all, or if all retrieved documents were truly irrelevant.
            if len(documents) == 0:
                online_search = True
                print(
                    "DOCUMENT_FIRST query: No documents retrieved, falling back to online search."
                )
            elif len(filtered_docs) == 0:
                # If documents were retrieved but none passed evaluation,
                # it means even with a lenient prompt, they were not relevant.
                # In this case, we should fall back to online search.
                online_search = True
                print(
                    "DOCUMENT_FIRST query: Documents retrieved but none relevant, falling back to online search."
                )
            else:
                online_search = False
                print(
                    "DOCUMENT_FIRST query: Relevant documents found, proceeding with documents."
                )
        elif query_classification == "HYBRID":
            # For hybrid queries, if documents are not sufficiently relevant, consider online search.
            if len(filtered_docs) == 0:
                online_search = True
                print(
                    "HYBRID query: No relevant documents found, falling back to online search."
                )
            else:
                online_search = False
                print(
                    "HYBRID query: Relevant documents found, proceeding with documents."
                )
        else:  # ONLINE_SEARCH classification
            # For online search queries, always prefer online search, even if some documents are found.
            online_search = True
            print(
                f"ONLINE_SEARCH query: Classified as online search, setting online_search = True. Documents found: {len(filtered_docs)}"
            )

        # The _any_doc_irrelevant function will make the final routing decision.
        # We set online_search here, but the conditional edge will use it.

        print(f"Final decision for online_search in _evaluate: {online_search}")

        # Determine search method for state tracking
        search_method = "online" if online_search else "documents"

        return {
            "documents": filtered_docs,
            "question": question,
            "online_search": online_search,
            "search_method": search_method,
            "document_evaluations": document_evaluations,
            "query_classification": query_classification,  # Pass classification for later use if needed
        }

    def _generate_answer(self, state: GraphState):
        """Single source of truth for context building and answer generation"""
        print("GRAPH STATE: Generate Answer")
        
        # Increment generation attempts for circuit breaker
        current_attempts = state.get("generation_attempts", 0)
        new_attempts = current_attempts + 1
        print(f"Answer generation attempt: {new_attempts}")
        
        question = state["question"]
        
        # Read all available sources from state
        documents = state.get("documents", [])  # These are reranked documents (top 5)
        web_search_results = state.get("web_search_results", [])
        
        print("=== CONTEXT BUILDING DEBUG INFO ===")
        print(f"Reranked documents available: {len(documents)}")
        print(f"Web search results available: {len(web_search_results)}")
        
        # BUILD FINAL CONTEXT FROM SCRATCH - Single Source of Truth
        context_sections = []
        
        # Add document information section
        if documents:
            print(f"📚 Adding DOCUMENT section with {len(documents)} reranked documents")
            doc_content = []
            for i, doc in enumerate(documents, 1):
                if hasattr(doc, 'page_content'):
                    doc_content.append(f"Document {i}:\n{doc.page_content}")
                else:
                    doc_content.append(f"Document {i}:\n{str(doc)}")
            
            context_sections.append(
                "=== DOCUMENT INFORMATION ===\n" + 
                "\n\n".join(doc_content)
            )
        
        # Add web search results section
        if web_search_results:
            print(f"🌐 Adding WEB SEARCH section with {len(web_search_results)} results")
            web_content = []
            for i, result in enumerate(web_search_results, 1):
                if hasattr(result, 'page_content'):
                    web_content.append(f"Web Result {i}:\n{result.page_content}")
                else:
                    web_content.append(f"Web Result {i}:\n{str(result)}")
            
            context_sections.append(
                "=== WEB SEARCH RESULTS ===\n" + 
                "\n\n".join(web_content)
            )
        
        # Combine all sections into final context
        if context_sections:
            final_context = "\n\n" + "\n\n".join(context_sections) + "\n\n"
            print(f"✅ Final context built: {len(final_context)} characters")
            print(f"Context sections: {len(context_sections)} ({'Documents' if documents else ''}{' + ' if documents and web_search_results else ''}{'Web' if web_search_results else ''})")
        else:
            final_context = "No relevant information found."
            print("⚠️  No context available - using fallback message")
        
        context_for_generation = final_context
        
        solution = generate_chain.invoke({"context": context_for_generation, "question": question})
        print(f"Answer generated: {len(solution)} characters")
        
        # Generate Quality Metrics for frontend display with batching optimization
        print("🔍 Generating Quality Metrics...")
        
        # Create context signature for cache validation
        rerank_completed = state.get("rerank_completed", False)
        current_context_signature = _create_context_signature(documents, web_search_results, rerank_completed)
        
        # **GROQ-POWERED BATCHED RELEVANCE EVALUATION (10-20x FASTER)**
        print("🚀 BATCH RELEVANCE: Evaluating document + question relevance in single API call (Groq)")
        
        try:
            # **CRITICAL FIX: Combine documents AND web_search_results for validation**
            validation_docs = []
            if documents:
                validation_docs.extend(documents)
            if web_search_results:
                # Ensure web results have proper metadata for detection
                from langchain_core.documents import Document
                for result in web_search_results:
                    if hasattr(result, 'page_content'):
                        # Already a Document - use as-is (preserves metadata)
                        validation_docs.append(result)
                    elif isinstance(result, dict) and 'content' in result:
                        # Dict format - create Document with web_search metadata
                        validation_docs.append(Document(
                            page_content=result['content'],
                            metadata={'type': 'web_search', 'source': result.get('url', 'web')}
                        ))
                    else:
                        # Fallback - mark as web_search
                        validation_docs.append(Document(
                            page_content=str(result),
                            metadata={'type': 'web_search'}
                        ))
            
            # **DEBUG: Log exact documents being sent to batch relevance evaluation**
            print(f"🔍 DEBUG: Groq batch relevance receiving {len(validation_docs)} total sources")
            print(f"   Documents: {len(documents)}, Web results: {len(web_search_results)}")
            for i, doc in enumerate(validation_docs[:3]):  # Show first 3 for debugging
                preview = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)[:100]
                doc_type = doc.metadata.get('type', 'document') if hasattr(doc, 'metadata') else 'unknown'
                print(f"   Source {i+1} (type: {doc_type}): {preview}...")
            
            # **CRITICAL: Use Groq-powered evaluation (10-20x faster than Gemini)**
            # This call has built-in Gemini fallback if Groq fails
            batch_relevance_result = evaluate_relevance_batch_groq(question, validation_docs, solution)
            document_relevance_score = batch_relevance_result["document_relevance_score"]
            question_relevance_score = batch_relevance_result["question_relevance_score"]
            grounding_source = batch_relevance_result.get("grounding_source", "UNKNOWN")
            
            print("✅ Groq batch relevance evaluation successful")
            print(f"   Document grounding: {document_relevance_score.binary_score}")
            print(f"   Question relevance: {question_relevance_score.binary_score}")
            print(f"   Grounding source: {grounding_source}")
            
        except Exception as e:
            print(f"⚠️  Batch relevance evaluation failed: {e}")
            print("🔄 Falling back to individual relevance evaluations")
            
            # Fallback to individual evaluations
            document_relevance_score = None
            if documents:
                try:
                    from backend.chains.document_relevance_groq import check_document_relevance  # Groq-optimized
                    document_relevance_score = check_document_relevance.invoke(
                        {"documents": documents, "solution": solution}
                    )
                    print(f"   Document grounding: {document_relevance_score.binary_score}")
                except Exception as e:
                    print(f"   Error generating document relevance score: {e}")
            
            question_relevance_score = None
            try:
                from backend.chains.question_relevance import question_relevance
                question_relevance_score = question_relevance.invoke(
                    {"question": question, "solution": solution}
                )
                print(f"   Question relevance: {question_relevance_score.binary_score}")
            except Exception as e:
                print(f"   Error generating question relevance score: {e}")
        
        print("✅ Answer generation and quality metrics completed")

        return {
            "documents": documents,
            "question": question,
            "solution": solution,
            "generation_attempts": new_attempts,  # Update generation attempts
            "document_relevance_score": document_relevance_score,  # Quality metrics
            "question_relevance_score": question_relevance_score,  # Quality metrics
            # Cache the relevance scores, grounding source, and context signature for reuse
            "cached_document_relevance": document_relevance_score,
            "cached_question_relevance": question_relevance_score,
            "cached_grounding_source": grounding_source if 'grounding_source' in locals() else "UNKNOWN",  # NEW: Cache grounding source
            "cache_context_signature": current_context_signature,
        }

    def _search_online(self, state: GraphState):
        """Search online for additional context if needed"""
        print("GRAPH STATE: Search Online")
        question = state["question"]
        documents = state["documents"]

        print(f"Searching online for: {question}")
        try:
            tavily_client = TavilySearchResults(k=TAVILY_SEARCH_RESULTS)
            response = tavily_client.invoke({"query": question})
            
            # Handle different Tavily response formats
            web_results = []
            if isinstance(response, list):
                # Response is a list of dictionaries
                for element in response:
                    if isinstance(element, dict) and "content" in element:
                        web_results.append(element["content"])
                    elif isinstance(element, str):
                        web_results.append(element)
                    else:
                        print(f"Unexpected element type in Tavily response: {type(element)}")
                        web_results.append(str(element))
            elif isinstance(response, dict):
                # Response is a single dictionary
                if "content" in response:
                    web_results.append(response["content"])
                else:
                    print(f"Unexpected dict structure in Tavily response: {response.keys()}")
                    web_results.append(str(response))
            else:
                # Response is something else (string, etc.)
                print(f"Unexpected Tavily response type: {type(response)}")
                web_results.append(str(response))
            
            # Combine all web results into a single document
            if web_results:
                combined_content = "\n\n".join(web_results)
                results = Document(page_content=combined_content)
                print(f"Successfully processed {len(web_results)} web search results")
            else:
                results = Document(page_content="No web search results found.")
                print("No valid web search results found")
            
        except Exception as e:
            print(f"Error during web search: {e}")
            results = Document(page_content=f"Web search failed: {str(e)}")

        # Add to existing documents or create new list
        if documents is not None:
            documents.append(results)
            print(f"Added online search results to {len(documents) - 1} existing documents")
        else:
            documents = [results]
            print("Using only online search results")

        # Store web search results in separate state field for proper handling
        return {
            "documents": documents, 
            "question": question, 
            "search_method": "online",
            "web_search_results": [results],  # Store web results separately
            "online_search": True
        }

    def _rerank(self, state: GraphState):
        """Specialized reranking function - ONLY reranks vectorstore documents, ignores web results"""
        print("GRAPH STATE: Rerank Documents")
        question = state["question"]
        
        # CRITICAL: Only read vectorstore documents, completely ignore web results
        vectorstore_results = state.get("vectorstore_results", [])
        web_search_results = state.get("web_search_results", [])
        
        print("=== RERANK INPUT ANALYSIS ===")
        print(f"Vectorstore documents to rerank: {len(vectorstore_results)}")
        print(f"Web search results (IGNORED by reranker): {len(web_search_results)}")
        
        # Only rerank if we have vectorstore documents
        if not vectorstore_results:
            print("No vectorstore documents to rerank - returning empty list")
            return {"documents": []}
        
        try:
            print(f"Reranking {len(vectorstore_results)} vectorstore documents, selecting top 5")
            print(f"Input documents preview: {[doc.page_content[:100] + '...' if hasattr(doc, 'page_content') and len(doc.page_content) > 100 else str(doc)[:100] for doc in vectorstore_results[:3]]}")
            
            # SPECIALIZED: Only rerank dense vectorstore documents (not noisy web snippets)
            reranked_docs = rerank_documents(question, vectorstore_results, top_k=5)
            
            print(f"Reranking completed: {len(reranked_docs)} documents selected")
            print(f"Reranked documents preview: {[doc.page_content[:100] + '...' if hasattr(doc, 'page_content') and len(doc.page_content) > 100 else str(doc)[:100] for doc in reranked_docs[:3]]}")
            
            # PURE FUNCTION: Only return the reranked vectorstore documents
            # Web search results are preserved in their original state key
            return {"documents": reranked_docs}
            
        except Exception as e:
            print(f"Error during reranking: {e}")
            # Fallback: return first 5 vectorstore documents
            fallback_docs = vectorstore_results[:5] if len(vectorstore_results) > 5 else vectorstore_results
            print(f"Falling back to first {len(fallback_docs)} vectorstore documents")
            return {"documents": fallback_docs}

    def _analyze_batch(self, state: GraphState):
        """Analyze documents in batch using single LLM call for Quality Metrics"""
        print("GRAPH STATE: Evaluate Documents")
        question = state["question"]
        documents = state["documents"]

        # Log query type for transparency
        online_search = state.get("online_search", False)
        query_type = "HYBRID" if (documents and online_search) else ("WEB_ONLY" if online_search else "DOCUMENT_ONLY")
        print(
            f"Evaluating {len(documents)} reranked documents for Quality Metrics"
        )
        print(f"Query type: {query_type}")

        # ✅ FIXED: Only skip if there are NO documents to evaluate
        # Always evaluate documents when present, regardless of web search status
        if not documents:
            print("Skipping document evaluation - no documents to evaluate")
            return {
                "documents": documents,
                "question": question,
                "online_search": online_search,
                "document_evaluations": [],
                # Preserve other state fields
                "vectorstore_results": state.get("vectorstore_results", []),
                "web_search_results": state.get("web_search_results", []),
                "combined_context": state.get("combined_context", ""),
                "execution_plan": state.get("execution_plan", []),
                "metadata": state.get("metadata", {}),
                "generation_attempts": state.get("generation_attempts", 0),
                "rewrite_attempts": state.get("rewrite_attempts", 0),
                "context_assessment": state.get("context_assessment"),
                "rerank_completed": state.get("rerank_completed", False)
            }

        try:
            # **DEBUG: Log exact documents being sent to batch evaluation**
            print(f"🔍 DEBUG: Batch evaluation receiving {len(documents)} documents")
            for i, doc in enumerate(documents[:3]):  # Show first 3 for debugging
                preview = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)[:100]
                print(f"   Doc {i+1}: {preview}...")
            
            # **BATCH EVALUATION: Single API call for all documents**
            print("="*80)
            print("🚀 BATCH DOCUMENT EVALUATION - SINGLE API CALL")
            print("="*80)
            
            from backend.chains.evaluate_groq import evaluate_documents_batch
            
            # Single batch API call replaces N sequential calls
            batch_start_time = time.time()
            batch_result = evaluate_documents_batch(
                question=question,
                documents=documents,
                query_classification="document-specific"
            )
            batch_duration = (time.time() - batch_start_time) * 1000
            
            # Convert batch results to analysis_results format
            analysis_results = []
            for eval_item in batch_result.evaluations:
                analysis_results.append({
                    "doc_id": eval_item.document_index,
                    "is_relevant": eval_item.is_relevant,
                    "relevance_score": eval_item.relevance_score,
                    "analysis": {
                        "coverage": eval_item.coverage_assessment,
                        "missing_info": eval_item.missing_information
                    }
                })
            
            print("\n" + "="*80)
            print("📊 BATCH EVALUATION SUMMARY")
            print("="*80)
            print(f"✅ Evaluated {len(documents)} documents in single API call")
            print(f"⚡ Total time: {batch_duration:.0f}ms")
            print(f"📈 Average per document: {batch_duration/len(documents):.0f}ms")
            print(f"🎯 Performance gain: ~{len(documents)}x faster than sequential")
            print("="*80)

            print(f"Batch analysis completed: {len(analysis_results)} analyses")

            # Generate evaluation data for Quality Metrics (preserve all reranked documents)
            document_evaluations = []

            for analysis in analysis_results:
                doc_id = analysis.get("doc_id", 0)
                is_relevant = analysis.get("is_relevant", "NO")

                # Ensure doc_id is within bounds
                if 0 <= doc_id < len(documents):
                    # Create evaluation object compatible with existing system
                    evaluation = {
                        "score": is_relevant,
                        "coverage_assessment": analysis.get("analysis", {}).get(
                            "coverage", ""
                        ),
                        "missing_information": analysis.get("analysis", {}).get(
                            "missing_info", ""
                        ),
                        "relevance_score": analysis.get("relevance_score", 0),
                    }
                    document_evaluations.append(evaluation)
                else:
                    print(
                        f"Warning: doc_id {doc_id} out of bounds for {len(documents)} documents"
                    )

            print(
                f"Generated evaluation data for {len(document_evaluations)} documents (Quality Metrics)"
            )

            # Preserve all reranked documents (don't filter them out)
            return {
                "documents": documents,  # Keep all reranked documents
                "question": question,
                "online_search": online_search,
                "document_evaluations": document_evaluations,
                # Preserve other state fields
                "vectorstore_results": state.get("vectorstore_results", []),
                "web_search_results": state.get("web_search_results", []),
                "combined_context": state.get("combined_context", ""),
                "execution_plan": state.get("execution_plan", []),
                "metadata": state.get("metadata", {}),
                "generation_attempts": state.get("generation_attempts", 0),
                "rewrite_attempts": state.get("rewrite_attempts", 0),
                "context_assessment": state.get("context_assessment"),
                "rerank_completed": state.get("rerank_completed", False)
            }

        except Exception as e:
            print(f"Error during batch analysis: {e}")
            # Fallback: assume all documents are relevant
            print("Falling back to accepting all documents as relevant")
            fallback_evaluations = []
            for i, doc in enumerate(documents):
                evaluation = {
                    "score": "YES",
                    "coverage_assessment": f"Document {i + 1} analysis failed, assuming relevant",
                    "missing_information": "Unable to determine due to analysis error",
                    "relevance_score": 75,
                }
                fallback_evaluations.append(evaluation)

            return {
                "documents": documents,
                "question": question,
                "online_search": False,
                "document_evaluations": fallback_evaluations,
                # Preserve other state fields
                "vectorstore_results": state.get("vectorstore_results", []),
                "web_search_results": state.get("web_search_results", []),
                "combined_context": state.get("combined_context", ""),
                "execution_plan": state.get("execution_plan", []),
                "metadata": state.get("metadata", {}),
                "generation_attempts": state.get("generation_attempts", 0),
                "rewrite_attempts": state.get("rewrite_attempts", 0),
                "context_assessment": state.get("context_assessment"),
                "rerank_completed": state.get("rerank_completed", False)
            }

    def _any_doc_irrelevant(self, state):
        """Determine whether online search is needed based on evaluation and query classification"""
        online_search = state.get("online_search", False)
        query_classification = state.get(
            "query_classification", "HYBRID"
        )  # Default to HYBRID
        documents = state.get("documents", [])

        # If online_search is already true from _evaluate, or if it's an ONLINE_SEARCH query
        # and we don't have enough documents, then go online.
        if online_search:
            next_state = "Search Online"
        elif (
            query_classification == "ONLINE_SEARCH" and len(documents) < 2
        ):  # If few documents for ONLINE_SEARCH query
            next_state = "Search Online"
        elif (
            query_classification == "HYBRID" and len(documents) == 0
        ):  # If no documents for HYBRID query
            next_state = "Search Online"
        else:
            next_state = "Generate Answer"

        print(
            f"ROUTING DECISION: Going to '{next_state}' (online_search: {online_search}, classification: {query_classification}, docs: {len(documents)})"
        )
        return next_state

    def _check_hallucinations(self, state: GraphState):
        """Enhanced hallucination checker - validates against COMPLETE context (documents + web results)"""
        print("GRAPH STATE: Check Hallucinations")
        
        # Circuit breaker: Maximum number of generation attempts
        MAX_ATTEMPTS = 2
        current_attempts = state.get("generation_attempts", 0)
        
        # Get the generated answer and all available sources
        solution = state.get("solution", "")
        question = state.get("question", "")
        reranked_docs = state.get("documents", [])  # Top 5 reranked documents
        web_results = state.get("web_search_results", [])  # Web search results
        
        print("=== HALLUCINATION CHECK INPUT ANALYSIS ===")
        print(f"Reranked documents available: {len(reranked_docs)}")
        print(f"Web search results available: {len(web_results)}")
        print(f"Answer length: {len(solution)} characters")
        
        # BUILD COMPLETE SOURCE OF TRUTH CONTEXT
        full_context_parts = []
        
        # Add reranked document content
        if reranked_docs:
            print(f"📚 Adding {len(reranked_docs)} reranked documents to validation context")
            doc_content = []
            for doc in reranked_docs:
                if hasattr(doc, 'page_content'):
                    doc_content.append(doc.page_content)
                else:
                    doc_content.append(str(doc))
            full_context_parts.extend(doc_content)
        
        # Add web search results content
        if web_results:
            print(f"🌐 Adding {len(web_results)} web search results to validation context")
            web_content = []
            for result in web_results:
                if hasattr(result, 'page_content'):
                    web_content.append(result.page_content)
                elif isinstance(result, dict) and 'content' in result:
                    web_content.append(result['content'])
                else:
                    web_content.append(str(result))
            full_context_parts.extend(web_content)
        
        # **CACHING OPTIMIZATION: Check if we can reuse cached relevance scores**
        rerank_completed = state.get("rerank_completed", False)
        current_context_signature = _create_context_signature(reranked_docs, web_results, rerank_completed)
        cached_context_signature = state.get("cache_context_signature")
        cached_document_relevance = state.get("cached_document_relevance")
        cached_question_relevance = state.get("cached_question_relevance")
        cached_grounding_source = state.get("cached_grounding_source")  # NEW: Get cached grounding source
        
        # Check cache validity conditions
        cache_valid = (
            cached_context_signature == current_context_signature and
            cached_document_relevance is not None and
            cached_question_relevance is not None
        )
        
        if cache_valid:
            print("🚀 CACHE HIT: Reusing cached relevance scores (context unchanged)")
            print(f"   Cached context signature: {cached_context_signature[:16]}...")
            doc_relevance_score = cached_document_relevance
            question_relevance_score = cached_question_relevance
            grounding_source = cached_grounding_source if cached_grounding_source else "UNKNOWN"  # NEW: Use cached grounding source
            grounding_source_details = "Cache hit: Using grounding source from Generate Answer node"
            
            print(f"   Cached document grounding: {doc_relevance_score.binary_score}")
            print(f"   Cached question relevance: {question_relevance_score.binary_score}")
            print(f"   Cached grounding source: {grounding_source}")
        else:
            print("💡 CACHE MISS: Context changed, performing fresh relevance evaluation")
            if cached_context_signature:
                print(f"   Previous signature: {cached_context_signature[:16]}...")
                print(f"   Current signature:  {current_context_signature[:16]}...")
            
            # Create comprehensive validation context and use batched evaluation
            if full_context_parts:
                full_context = " ".join(full_context_parts)
                print(f"✅ Complete validation context built: {len(full_context)} characters")
                print(f"Context sources: {len(reranked_docs)} docs + {len(web_results)} web results")
                
                # Create document-like objects for the validation
                validation_docs = []
                if reranked_docs:
                    validation_docs.extend(reranked_docs)
                if web_results:
                    # Convert web results to document format for validation
                    # **CRITICAL: Preserve metadata to enable web result detection**
                    from langchain_core.documents import Document
                    for result in web_results:
                        if hasattr(result, 'page_content'):
                            # Already a Document object - use as-is (preserves metadata)
                            validation_docs.append(result)
                        elif isinstance(result, dict) and 'content' in result:
                            # Dict format - create Document with web_search metadata
                            validation_docs.append(Document(
                                page_content=result['content'],
                                metadata={'type': 'web_search', 'source': result.get('url', 'web')}
                            ))
                        else:
                            # Fallback - mark as web_search to prevent false hallucination detection
                            validation_docs.append(Document(
                                page_content=str(result),
                                metadata={'type': 'web_search'}
                            ))
                
                # **BATCHED VALIDATION EVALUATION**
                print("🚀 BATCH VALIDATION: Evaluating both relevance scores in single API call")
                
                try:
                    # **DEBUG: Log exact documents being sent to batch validation**
                    print(f"🔍 DEBUG: batch validation receiving {len(validation_docs)} documents")
                    for i, doc in enumerate(validation_docs[:3]):  # Show first 3 for debugging
                        preview = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)[:100]
                        doc_type = doc.metadata.get('type', 'document') if hasattr(doc, 'metadata') else 'unknown'
                        print(f"   Validation Doc {i+1} (type: {doc_type}): {preview}...")
                    
                    # Single API call for both validation checks
                    batch_validation_result = evaluate_relevance_batch(question, validation_docs, solution)
                    doc_relevance_score = batch_validation_result["document_relevance_score"]
                    question_relevance_score = batch_validation_result["question_relevance_score"]
                    grounding_source = batch_validation_result.get("grounding_source", "UNKNOWN")
                    grounding_source_details = batch_validation_result.get("grounding_source_details", "")
                    
                    print("✅ Batch validation evaluation successful")
                    print(f"Document grounding (complete context): {doc_relevance_score.binary_score}")
                    print(f"Grounding source: {grounding_source}")
                    print(f"Source details: {grounding_source_details[:200]}..." if len(grounding_source_details) > 200 else f"Source details: {grounding_source_details}")
                    print(f"Question relevance: {question_relevance_score.binary_score}")
                    
                except Exception as e:
                    print(f"⚠️  Batch validation failed: {e}")
                    print("🔄 Falling back to individual validation evaluations")
                    
                    # Fallback to individual evaluations
                    from backend.chains.document_relevance_groq import check_document_relevance  # Groq-optimized
                    doc_relevance_score = check_document_relevance.invoke(
                        {"documents": validation_docs, "solution": solution}
                    )
                    print(f"Document grounding (complete context): {doc_relevance_score.binary_score}")
                    
                    from backend.chains.question_relevance import question_relevance
                    question_relevance_score = question_relevance.invoke(
                        {"question": question, "solution": solution}
                    )
                    print(f"Question relevance: {question_relevance_score.binary_score}")
                    
                    # **FALLBACK: Infer grounding source from available context**
                    if doc_relevance_score.binary_score:
                        if reranked_docs and web_results:
                            grounding_source = "HYBRID"
                            grounding_source_details = "Fallback inference: Both documents and web results available"
                        elif web_results:
                            grounding_source = "WEB_ONLY"
                            grounding_source_details = "Fallback inference: Only web results available"
                        elif reranked_docs:
                            grounding_source = "DOCUMENT_ONLY"
                            grounding_source_details = "Fallback inference: Only documents available"
                        else:
                            grounding_source = "UNKNOWN"
                            grounding_source_details = "Fallback inference: No clear context source"
                    else:
                        grounding_source = "NONE"
                        grounding_source_details = "Fallback inference: Answer not grounded in provided context"
                    
                    print(f"⚠️  Fallback grounding source inference: {grounding_source}")
                    
            else:
                print("⚠️  No context available for validation - assuming grounded")
                # If no context, assume it's grounded to avoid false positives
                class MockScore:
                    binary_score = True
                doc_relevance_score = MockScore()
                question_relevance_score = MockScore()
                grounding_source = "UNKNOWN"
                grounding_source_details = "No context available for validation"
        
        # **CRITICAL: Log grounding source for transparency**
        print("="*80)
        print("📊 GROUNDING SOURCE ANALYSIS")
        print("="*80)
        print(f"Grounding Source: {grounding_source}")
        print(f"Is Grounded: {doc_relevance_score.binary_score}")
        print(f"Question Relevance: {question_relevance_score.binary_score if question_relevance_score else 'N/A'}")
        print(f"Details: {grounding_source_details}")
        print("="*80)
        
        # **SIMPLIFIED ROUTING LOGIC: Use grounding_source directly (no inference)**
        # The grounding_source was already determined by the batch validation chain in Generate Answer
        
        if grounding_source == "NONE":
            # Answer is not grounded in any provided context - retry or give up
            print(f"❌ Answer not grounded (Source: {grounding_source}). Attempt {current_attempts}/{MAX_ATTEMPTS}")
            
            if current_attempts < MAX_ATTEMPTS:
                print("ROUTING DECISION: Going to 'Generate Answer' (Retry within limit)")
                return "Hallucinations detected"
            else:
                print(f"ROUTING DECISION: Going to 'END' (Max attempts {MAX_ATTEMPTS} reached)")
                return "Answers Question"  # End even with flawed answer
        
        elif grounding_source in ["DOCUMENT_ONLY", "WEB_ONLY", "HYBRID"]:
            # Answer is properly grounded - check if it addresses the question
            print(f"✅ Answer grounded in context (Source: {grounding_source})")
            
            if question_relevance_score and question_relevance_score.binary_score:
                print("ROUTING DECISION: Going to 'END' (Answers Question)")
                return "Answers Question"
            else:
                print("ROUTING DECISION: Going to 'Search Online' (Question not addressed)")
                return "Question not addressed"
        
        else:
            # Unknown grounding source - fallback to binary score check
            print(f"⚠️  Unknown grounding source: {grounding_source}, falling back to binary score check")
            
            if doc_relevance_score and doc_relevance_score.binary_score:
                print(f"✅ Fallback: Document grounding check passed")
                
                if question_relevance_score and question_relevance_score.binary_score:
                    print("ROUTING DECISION: Going to 'END' (Answers Question)")
                    return "Answers Question"
                else:
                    print("ROUTING DECISION: Going to 'Search Online' (Question not addressed)")
                    return "Question not addressed"
            else:
                # Hallucination detected - check circuit breaker
                print(f"❌ Fallback: Hallucinations detected. Attempt {current_attempts}/{MAX_ATTEMPTS}")
                
                if current_attempts < MAX_ATTEMPTS:
                    print("ROUTING DECISION: Going to 'Generate Answer' (Retry within limit)")
                    return "Hallucinations detected"
                else:
                    print(f"ROUTING DECISION: Going to 'END' (Max attempts {MAX_ATTEMPTS} reached)")
                    return "Answers Question"  # End even with flawed answer
    
    def _analyze_query(self, state: GraphState):
        """Analyze the user query and create execution plan with metadata extraction"""
        print("GRAPH STATE: Query Analysis")
        question = state["question"]
        
        print(f"Analyzing query: '{question}'")
        
        # Get available documents from current session
        available_documents = self._get_available_documents()
        if available_documents:
            print(f"Available documents in session: {available_documents}")
        else:
            print("No documents available in current session")
        
        # Use the Query Analysis Router to create execution plan with metadata
        analysis_result = analyze_query(question, available_documents)
        
        # Extract tasks and metadata from the new format
        execution_plan = analysis_result["tasks"]
        metadata = analysis_result["metadata"]
        
        print(f"Generated execution plan: {execution_plan}")
        print(f"Extracted metadata: {metadata}")
        print(f"Plan summary: {len(execution_plan)} task(s)")
        
        # Log each task in the plan
        for i, task in enumerate(execution_plan):
            print(f"Task {i+1}: {task['tool']} - {task['query']}")
        
        return {
            "question": question,
            "original_question": state.get("original_question", question),  # Preserve original query
            "execution_plan": execution_plan,
            "metadata": metadata,  # NEW: Include extracted metadata
            "vectorstore_results": [],
            "web_search_results": [],
            "combined_context": "",
            "generation_attempts": 0,  # Initialize circuit breaker counter
            "rewrite_attempts": state.get("rewrite_attempts", 0)  # Preserve rewrite attempts
        }
    
    def _execute_multi_tool_plan(self, state: GraphState):
        """Execute the multi-tool execution plan with metadata-aware optimization"""
        print("GRAPH STATE: Execute Multi-Tool Plan")
        
        execution_plan = state.get("execution_plan", [])
        metadata = state.get("metadata", {})  # NEW: Get extracted metadata
        question = state["question"]
        
        if not execution_plan:
            print("No execution plan found, falling back to document retrieval")
            # Fallback to simple document retrieval
            execution_plan = [{"tool": "vectorstore_retrieval", "query": question}]
        
        print(f"Executing plan with {len(execution_plan)} tasks")
        if metadata:
            print(f"Using metadata for optimization: {metadata}")
        
        # Initialize multi-tool executor if not already done
        if not self.multi_tool_executor:
            self.multi_tool_executor = MultiToolExecutor(retriever=self.retriever)
        
        # Execute the plan with metadata for optimization (NOW WITH PER-TASK RERANKING AND CONTEXT ACCUMULATION)
        try:
            results = self.multi_tool_executor.execute_plan(execution_plan, metadata, state)
            
            # Extract results - vectorstore_results are now ALREADY RERANKED (top 3 per task)
            vectorstore_results = results.get("vectorstore_results", [])
            web_search_results = results.get("web_search_results", [])
            combined_context = results.get("combined_context", "")
            total_sources = results.get("total_sources", 0)
            rerank_completed = results.get("rerank_completed", True)  # Flag from executor
            
            print(f"🎯 PARALLEL EXECUTION COMPLETED: {total_sources} total sources")
            print(f"   📚 Reranked docs collected: {len(vectorstore_results)}")
            print(f"   🌐 Web results collected: {len(web_search_results)}")
            print(f"   ✅ Per-task reranking: {rerank_completed}")
            
            # Update state with results and preserve metadata
            return {
                "question": question,
                "execution_plan": execution_plan,
                "metadata": metadata,  # Preserve metadata through workflow
                "vectorstore_results": vectorstore_results,
                "web_search_results": web_search_results,
                "documents": vectorstore_results,  # Set documents to reranked results
                "combined_context": combined_context,
                "rerank_completed": rerank_completed,  # Flag that reranking was done
                "online_search": len(web_search_results) > 0,
                "search_method": "multi_tool"
            }
            
        except Exception as e:
            print(f"Error executing multi-tool plan: {e}")
            # Fallback to empty results
            return {
                "question": question,
                "execution_plan": execution_plan,
                "metadata": metadata,  # NEW: Preserve metadata even in fallback
                "vectorstore_results": [],
                "web_search_results": [],
                "combined_context": "",
                "documents": [],
                "online_search": True,  # Trigger fallback
                "search_method": "fallback"
            }
    
    def _assess_context(self, state: GraphState):
        """
        Assess whether retrieved context is sufficient to answer the question.
        
        CRITICAL FIX: This function is now TOOL-AWARE and analyzes COMBINED context
        from both documents and web search results to prevent false negatives on hybrid queries.
        
        Previous Bug: Only assessed documents, ignored web results → false "insufficient" on hybrid queries
        Fix: Passes both documents AND web_search_results to assessment chain
        """
        print("GRAPH STATE: Assess Context")
        
        # **STEP 1: EXTRACT ALL AVAILABLE CONTEXT**
        execution_plan = state.get("execution_plan", [])
        web_search_results = state.get("web_search_results", [])
        documents = state.get("documents", [])
        original_question = state.get("original_question", state["question"])
        
        # Determine plan composition
        has_web_search = any(task.get("tool") == "web_search" for task in execution_plan)
        has_doc_search = any(task.get("tool") == "vectorstore_retrieval" for task in execution_plan)
        
        web_results_exist = bool(web_search_results)
        docs_exist = bool(documents)
        
        print("=== PLAN TYPE ANALYSIS ===")
        print(f"   Plan includes web_search: {has_web_search}")
        print(f"   Plan includes vectorstore_retrieval: {has_doc_search}")
        print(f"   Web results exist: {web_results_exist} ({len(web_search_results)} results)")
        print(f"   Documents exist: {docs_exist} ({len(documents)} docs)")
        
        # **CONDITION 1: Safe early exit ONLY for WEB-SEARCH-ONLY queries**
        if has_web_search and not has_doc_search and web_results_exist:
            print("---CONTEXT ASSESSMENT: SUFFICIENT (Successful Web-Search-Only Plan)---")
            print("   ✅ Pure web search query with results - skipping LLM assessment")
            return {
                "context_assessment": "sufficient"
            }
        
        # **CONDITION 2: Immediate failure if docs were required but not found**
        if has_doc_search and not docs_exist:
            print("---CONTEXT ASSESSMENT: INSUFFICIENT (Document search required but no docs retrieved)---")
            print("   ❌ Plan required documents but none were retrieved/reranked")
            return {
                "context_assessment": "insufficient"
            }
        
        # **CONDITION 3: TOOL-AWARE LLM Gap Analysis for DOCUMENT-ONLY or HYBRID queries**
        if has_doc_search:
            plan_type = "HYBRID" if has_web_search else "DOCUMENT-ONLY"
            print(f"---CONTEXT ASSESSMENT: Performing TOOL-AWARE LLM Gap Analysis for {plan_type} plan---")
            print(f"   📊 Analyzing {len(documents)} documents and {len(web_search_results)} web results against original question")
            
            if has_web_search:
                print("   🔧 CRITICAL FIX: Passing BOTH documents AND web results to prevent false negatives")
            
            # **CRITICAL FIX: Pass BOTH documents AND web_search_results to assessment chain**
            try:
                # Now returns (decision, gap_analysis_json)
                assessment_result, gap_analysis_json = assess_context_sufficiency(
                    original_question=original_question,
                    documents=documents,
                    web_search_results=web_search_results  # NEW: Pass web results
                )
                
                print(f"✅ LLM GAP ANALYSIS COMPLETE: '{assessment_result}'")
                
                if assessment_result == "sufficient":
                    print("   ✅ Combined context (documents + web) provides sufficient information")
                else:
                    print("   ❌ Combined context is insufficient - query rewrite may be needed")
                
                # Save both the decision and the full JSON report
                return {
                    "context_assessment": assessment_result,
                    "context_assessment_json": gap_analysis_json  # NEW: Save full report for plan correction
                }
                
            except Exception as e:
                print("---CONTEXT ASSESSMENT ERROR: LLM Gap Analysis failed---")
                print(f"   Error: {e}")
                print("   ⚠️  Defaulting to 'insufficient' for safety")
                return {
                    "context_assessment": "insufficient",
                    "error": str(e)
                }
        
        # **FALLBACK: Unhandled case (should not reach here with proper plan analysis)**
        print("---CONTEXT ASSESSMENT WARNING: Unhandled plan type---")
        print("   ⚠️  No web search or document search detected in plan")
        print("   Defaulting to 'insufficient' for safety")
        return {
            "context_assessment": "insufficient"
        }
    
    def _rewrite_query(self, state: GraphState):
        """
        Surgically corrects the execution_plan based on the gap analysis report.
        This is a "plan-based" correction, not a "query-based" rewrite.
        """
        print("GRAPH STATE: Rewrite Query (Plan Correction)")
        
        original_question = state["original_question"]
        execution_plan = state["execution_plan"]
        gap_analysis_report = state.get("context_assessment_json", {})
        rewrite_attempts = state.get("rewrite_attempts", 0) + 1

        print(f"Correcting plan (attempt #{rewrite_attempts})")
        print(f"  Original question: '{original_question[:100]}...'")
        print(f"  Original plan has {len(execution_plan)} tasks")
        
        if not gap_analysis_report:
            print("  ⚠️  No gap analysis report found. Rewrite may be suboptimal.")

        # Call the plan correction chain with the plan and the report
        try:
            from backend.chains.rewrite_query_groq import correct_execution_plan
            
            new_plan = correct_execution_plan(
                original_question=original_question,
                execution_plan=execution_plan,
                gap_analysis_report=gap_analysis_report,
                attempt_number=rewrite_attempts
            )
            print(f"✅ Generated new plan with {len(new_plan)} tasks")
        except Exception as e:
            print(f"❌ Plan correction failed: {e}")
            print("   Falling back to original plan to prevent crash.")
            new_plan = execution_plan

        # Clear old context to force fresh retrieval
        # Note: We clear context here, but the executor will accumulate
        return {
            "question": original_question,  # Keep original question in 'question'
            "original_question": original_question,
            
            # CRITICAL: Overwrite the plan with the new, corrected plan
            "execution_plan": new_plan,
            
            "rewrite_attempts": rewrite_attempts,
            "documents": state.get("documents", []),  # Keep existing docs for accumulation
            "web_search_results": state.get("web_search_results", []),  # Keep existing web for accumulation
            "combined_context": "",  # Clear previous context
            "context_assessment_json": {}  # Clear old report
        }
    
    def _route_after_assessment(self, state: GraphState):
        """Route based on context assessment and rewrite attempts"""
        print("GRAPH STATE: Route After Assessment")
        
        assessment = state.get("context_assessment", "insufficient")
        rewrite_attempts = state.get("rewrite_attempts", 0)
        MAX_REWRITE_ATTEMPTS = 2
        
        print("=== ROUTING DEBUG INFO ===")
        print(f"Raw context_assessment from state: '{assessment}' (type: {type(assessment)})")
        print(f"Rewrite attempts: {rewrite_attempts}/{MAX_REWRITE_ATTEMPTS}")
        print(f"State keys: {list(state.keys())}")
        
        # Decision logic with enhanced debugging
        if assessment == "sufficient":
            print("ROUTING DECISION: Going to 'Generate Answer' (sufficient context)")
            return "sufficient"
        elif rewrite_attempts >= MAX_REWRITE_ATTEMPTS:
            print(f"ROUTING DECISION: Going to 'Generate Answer' (max attempts {MAX_REWRITE_ATTEMPTS} reached)")
            return "max_attempts"
        else:
            print("ROUTING DECISION: Going to 'Rewrite Query' (insufficient context, attempts remaining)")
            print(f"   Reason: assessment='{assessment}' != 'sufficient' AND attempts={rewrite_attempts} < {MAX_REWRITE_ATTEMPTS}")
            return "insufficient"
