# Multi-Doc Adaptive RAG Agent

## 1. Introduction

The Multi-Doc Adaptive RAG Agent is an advanced, AI-powered reasoning system designed to provide comprehensive answers by intelligently utilizing multiple documents and real-time web searches. This is not just a chatbot; it's a full-stack application featuring a state-of-the-art, self-correcting RAG pipeline that can deconstruct complex questions, form an execution plan, and use multiple tools to synthesize insightful, accurate, and fully-grounded responses. The system combines a modern React.js interface with a high-performance FastAPI backend, orchestrated by LangGraph to create a dynamic, adaptive workflow that represents the cutting edge of RAG technology.

## 2. Demo Video

(Placeholder for a screen recording of the application in action)

## 3. Performance & Architecture Summary

This project was built with a relentless focus on performance, accuracy, and architectural robustness. The pipeline evolved through multiple stages of optimization to achieve its current high-performance state.
* Latency Reduction: >60% reduction in end-to-end query latency (from ~56s to <20s) by eliminating architectural bottlenecks.
* GPU-Accelerated Reranking: Achieved a 2-3x speedup in the relevance reranking step by leveraging Apple Silicon's MPS backend for local cross-encoder inference.
* Efficient API Usage: Drastically reduced cost and latency by replacing N+1 LLM calls with a single, batch analysis call.
* Adaptive Multi-Tool Architecture: The system uses an LLM-powered Query Analysis Router to intelligently decide whether to use document retrieval, web search, or both, enabling it to handle a vastly wider range of questions than a standard RAG pipeline.

## 4. Features

### Core User Features

- Multi-Format Document Upload: Supports PDF, DOCX, TXT, and more for building a knowledge base.
- Session-Based Interaction: Uploaded documents and conversation history persist across a user's session.
- Hybrid Search: Seamlessly answers questions using both the content of the uploaded documents and real-time information from the web (via Tavily).
- Natural Language Queries: Ask complex, multi-part questions in plain English.
- Source-Grounded Responses: Final answers are generated based on verifiable information from the provided context.

### Advanced Pipeline Features

- Query Analysis Router: An LLM-powered first step that deconstructs user queries and creates a dynamic execution plan (e.g., "use web search for part A, use document search for part B").
- Self-Correction (Query Rewriting): A conditional loop that assesses the quality of retrieved context. If insufficient, it rewrites the query to be more specific and retries the retrieval, recovering from "needle-in-a-haystack" failures.
- Cross-Encoder Reranking: Utilizes a local BAAI/bge-reranker-base model to re-rank the initial retrieval results, ensuring only the most relevant documents are passed to the generator.
- Quality Gates & Self-Correction (Answer Regeneration): The system evaluates every generated answer for hallucinations and relevance. If a check fails, a "circuit breaker" allows for a limited number of regeneration attempts.

## 5. Architecture Diagram

(Placeholder - Replace with your generated workflow image link)

## 6. Tech Stack

- Frontend: React.js, TypeScript, Vite, Tailwind CSS, Axios
- Backend: Python 3.11, FastAPI, Uvicorn
- AI Orchestration: LangGraph
- LLMs: Google Gemini (e.g., gemini-2.5-pro for routing, gemini-2.5-flash for generation)
- Vector Database: FAISS (or ChromaDB/LanceDB)
- AI/ML Components:
    - Embeddings: sentence-transformers
    - Reranking: cross-encoder (BAAI/bge-reranker-base)
    - Document Processing: unstructured
    - Web Search: Tavily API

## 7. Project Structure

```
AdvLang/
├── frontend/                    # React TypeScript Frontend
│   ├── src/
│   │   ├── components/          # UI Components (Upload, Chat, etc.)
│   │   └── App.tsx
│   └── package.json
├── backend/                     # FastAPI Python Backend
│   ├── chains/                  # LangChain/LangGraph components
│   │   ├── query_analysis_router.py
│   │   ├── multi_tool_executor.py
│   │   ├── rerank_documents.py
│   │   ├── context_assessment.py
│   │   ├── rewrite_query.py
│   │   ├── generate_answer.py
│   │   ├── evaluate_batch.py
│   │   ├── relevance_batch.py
│   │   └── analyze_documents_batch.py
│   ├── api.py                   # FastAPI endpoints
│   ├── rag_workflow.py          # Main LangGraph workflow definition
│   ├── document_processor.py    # Document chunking & FAISS indexing
│   ├── document_loader.py       # Multi-format document loading
│   ├── session_manager.py       # Session & conversation management
│   ├── state.py                 # LangGraph state definition
│   ├── config.py                # Configuration & API keys
│   └── utils.py                 # Utility functions
├── tests/                       # Test suite
│   ├── test_batching_optimizations.py
│   ├── test_relevance_caching.py
│   ├── test_query_rewriting_loop.py
│   ├── test_metadata_extraction.py
│   └── ... (other test files)
├── faiss_indexes/               # Persistent FAISS vector stores
├── chunk_stores/                # Persistent document chunk storage
├── run_api.py                   # API startup script
├── requirements.txt             # Python dependencies
└── .env                         # API keys and configuration
```

## 8. How to Run the App

### Prerequisites
* Python 3.11+
* Node.js 18+
* An .env file with your GOOGLE_API_KEY and TAVILY_API_KEY.

### Step 1: Clone & Setup
```bash
git clone <your-repo-link>
cd adaptive-rag-agent
```

### Step 2: Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Frontend Setup
```bash
cd frontend
npm install
```

### Step 4: Run the Application
```bash
# Terminal 1: Start the Backend Server (from project root)
python run_api.py

# Terminal 2: Start the Frontend Dev Server (from frontend folder)
cd frontend
npm run dev
```
Access the application at http://localhost:5173 (or your Vite port).

## 9. Challenges Faced & Solutions

This project's development involved a rigorous process of iterative optimization and debugging to overcome significant technical challenges:
* Problem: Extreme Initial Latency (>56s)
* Diagnosis: Identified an N+1 problem where the document grading step made a separate LLM call for each document.
* Solution: Re-architected the pipeline to use a fast, local cross-encoder for reranking, eliminating the multiple API calls.
* Problem: Reranker "Cold Starts" & CPU Bottleneck
* Diagnosis: The reranker model was being reloaded on every query, and inference on the CPU was slow.
* Solution: Implemented the Singleton pattern to load the model only once at startup and refactored the code to leverage the Apple Silicon GPU (MPS), achieving a 2-3x speedup.
* Problem: Handling Complex, Hybrid Queries
* Diagnosis: The linear pipeline could not handle questions requiring both document context and real-time information.
* Solution: Engineered a Query Analysis Router to deconstruct user intent and create a dynamic, multi-tool execution plan.
* Problem: Infinite Self-Correction Loops
* Diagnosis: Quality gate failures (hallucination or relevance checks) caused the graph to get stuck in an infinite loop.
* Solution: Implemented "circuit breakers" by adding attempt counters to the graph state, ensuring the system fails gracefully after a set number of retries.
* Problem: Critical Data Handoff Bugs
* Diagnosis: The graph state was being managed incorrectly, causing downstream nodes (like the generator and hallucination checker) to use stale or incomplete data.
* Solution: Refactored the key nodes to be responsible for their own inputs, ensuring a clean and predictable data flow (e.g., the generator now builds its own context from all available sources).

## 10. Future Improvements

- Add More Specialized Tools: Integrate more tools beyond web search, such as a calculator for mathematical queries or a database agent for structured data lookup.
- Advanced Retrieval Strategies: Implement techniques like HyDE (Hypothetical Document Embeddings) or Multi-Query Retrieval to further improve the quality of the initial document retrieval.
- Enhanced UI/UX: Introduce real-time streaming for LLM responses and implement interactive citation highlighting, allowing users to click a citation to see the source document chunk.
- Formal Evaluation Suite: Build a comprehensive evaluation dataset to quantitatively measure performance (e.g., context relevance, answer accuracy, faithfulness) using frameworks like RAGAs.