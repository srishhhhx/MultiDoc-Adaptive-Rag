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
question-answering systems with proper workflow orchestration.
"""

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from state import GraphState
from chains.document_relevance import document_relevance
from chains.evaluate import evaluate_docs
from chains.generate_answer import generate_chain
from chains.question_relevance import question_relevance
from chains.query_classifier import classify_query
from langchain_community.tools.tavily_search import TavilySearchResults
from config import TAVILY_SEARCH_RESULTS


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
        self.graph = None
        self.retriever = None
        self._current_session_retriever_key = None

    def get_graph(self):
        """Get or create the graph instance (cached for performance)"""
        if self.graph is None:
            self.graph = self._create_graph()
        return self.graph

    def set_retriever(self, retriever):
        """Set the document retriever"""
        self.retriever = retriever

        if retriever is not None:
            print(f"Retriever set")
        else:
            print("Retriever cleared")

    def get_current_retriever(self):
        """Get the current retriever"""
        return self.retriever

    def process_question(self, question):
        """Process a question through the RAG workflow"""
        print(f"STARTING RAG WORKFLOW for question: '{question}'")

        # Ensure we have the most current retriever
        current_retriever = self.get_current_retriever()
        self.set_retriever(current_retriever)

        graph = self.get_graph()
        result = graph.invoke(input={"question": question})

        print(f"RAG WORKFLOW COMPLETED")
        return result

    def _create_graph(self):
        """Create and configure the state graph for handling queries"""
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("Retrieve Documents", self._retrieve)
        workflow.add_node("Grade Documents", self._evaluate)
        workflow.add_node("Generate Answer", self._generate_answer)
        workflow.add_node("Search Online", self._search_online)

        # Set entry point and edges
        workflow.set_entry_point("Retrieve Documents")
        workflow.add_edge("Retrieve Documents", "Grade Documents")
        workflow.add_conditional_edges(
            "Grade Documents",
            self._any_doc_irrelevant,
            {
                "Search Online": "Search Online",
                "Generate Answer": "Generate Answer",
            },
        )

        workflow.add_conditional_edges(
            "Generate Answer",
            self._check_hallucinations,
            {
                "Hallucinations detected": "Generate Answer",
                "Answers Question": END,
                "Question not addressed": "Search Online",
            },
        )
        workflow.add_edge("Search Online", "Generate Answer")

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
            print(f"Retrieved {len(documents)} documents from ChromaDB")

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

        print(
            f"Evaluated {len(documents)} documents, {len(filtered_docs)} deemed relevant."
        )

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
        """Generate an answer based on the retrieved documents"""
        print("GRAPH STATE: Generate Answer")
        question = state["question"]
        documents = state["documents"]

        print(f"Generating answer using {len(documents)} documents")
        solution = generate_chain.invoke({"context": documents, "question": question})
        print(f"Answer generated: {len(solution)} characters")

        # Evaluate the generated answer
        print("Evaluating generated answer...")

        # Check document grounding (hallucination check)
        doc_relevance_score = document_relevance.invoke(
            {"documents": documents, "solution": solution}
        )
        print(f"Document grounding: {doc_relevance_score.binary_score}")

        # Check question relevance
        question_relevance_score = question_relevance.invoke(
            {"question": question, "solution": solution}
        )
        print(f"Question relevance: {question_relevance_score.binary_score}")

        return {
            "documents": documents,
            "question": question,
            "solution": solution,
            "document_relevance_score": doc_relevance_score,
            "question_relevance_score": question_relevance_score,
        }

    def _search_online(self, state: GraphState):
        """Search online for additional context if needed"""
        print("GRAPH STATE: Search Online")
        question = state["question"]
        documents = state["documents"]

        print(f"Searching online for: {question}")
        tavily_client = TavilySearchResults(k=TAVILY_SEARCH_RESULTS)
        response = tavily_client.invoke({"query": question})
        results = "\n".join([element["content"] for element in response])
        results = Document(page_content=results)

        if documents is not None:
            documents.append(results)
            print(
                f"Added online search results to {len(documents) - 1} existing documents"
            )
        else:
            documents = [results]
            print(f"Using only online search results")

        # Update search method to indicate online search was used
        return {"documents": documents, "question": question, "search_method": "online"}

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
        """Check for hallucinations in the generated answers"""
        print("GRAPH STATE: Check Hallucinations")

        # Get the scores that were already computed in _generate_answer
        doc_relevance_score = state.get("document_relevance_score")
        question_relevance_score = state.get("question_relevance_score")

        if doc_relevance_score and doc_relevance_score.binary_score:
            print("Document relevance check passed")

            if question_relevance_score and question_relevance_score.binary_score:
                print("ROUTING DECISION: Going to 'END' (Answers Question)")
                return "Answers Question"
            else:
                print(
                    "ROUTING DECISION: Going to 'Search Online' (Question not addressed)"
                )
                return "Question not addressed"
        else:
            print(
                "ROUTING DECISION: Going to 'Generate Answer' (Hallucinations detected)"
            )
            return "Hallucinations detected"
