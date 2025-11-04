# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Advanced Multi-Document Adaptive RAG (Retrieval-Augmented Generation) Agent built with LangGraph. It's a production-grade full-stack application featuring self-correcting retrieval, hybrid search (documents + web), and intelligent query routing.

**Stack**: Python 3.11+ (FastAPI, LangGraph, FAISS) + React.js (Vite, Tailwind CSS)

## Development Commands

### Backend

```bash
# Start backend server (from project root)
python run_api.py
# Backend runs on http://localhost:8000
# API docs available at http://localhost:8000/docs

# Run tests
pytest tests/                           # Run all tests
pytest tests/test_metadata_extraction.py  # Run specific test
pytest -v tests/                        # Verbose output
pytest -k "test_context"                # Run tests matching pattern

# Check dependencies
pip list
pip install -r requirements.txt
```

### Frontend

```bash
# Start development server (from frontend/)
cd frontend
npm run dev        # Runs on http://localhost:5173
npm run build      # Production build
npm run preview    # Preview production build
npm run lint       # ESLint check
```

### Environment Setup

Required API keys in `.env` file (project root):
```env
GOOGLE_API_KEY=your_gemini_key          # Required: Gemini Pro for reasoning/generation
GROQ_API_KEY=your_groq_key              # Required: Groq for fast evaluation tasks
TAVILY_API_KEY=your_tavily_key          # Required: Web search functionality
```

## Critical Architecture Patterns

### LangGraph Workflow Architecture

The core RAG pipeline is orchestrated via **LangGraph** state machine in `backend/rag_workflow.py`. The workflow nodes execute in sequence based on conditional routing:

**Key workflow nodes**:
1. `analyze_query` - Deconstructs user intent, identifies relevant documents via metadata, creates dynamic multi-tool execution plans (uses **Gemini Pro**)
2. `execute_plan` - Executes retrieval plan with metadata filtering to prevent context contamination
3. `rerank` - GPU-accelerated cross-encoder reranking (BAAI/bge-reranker-base on Apple Silicon MPS)
4. `assess_context` - Gap analysis to check if context logically covers all query parts (uses **Groq**)
5. `rewrite_query` - Rewrites query focusing on missing information if context insufficient (max 2 attempts)
6. `generate` - Generates final answer (uses **Gemini Pro**)
7. `evaluate` - Hallucination detection and relevance checking (uses **Groq**)

**State flow**: All nodes read/write to `GraphState` (defined in `backend/state.py`), which tracks:
- Query analysis results (`execution_plan`, `metadata`)
- Retrieved context (`documents`, `web_search_results`, `combined_context`)
- Circuit breakers (`generation_attempts`, `rewrite_attempts`)
- Quality metrics (`document_evaluations`, `relevance_scores`)
- Caching signatures for API optimization

### Hybrid LLM Strategy

**CRITICAL**: This system uses a dual-LLM approach for cost/latency optimization:

- **Gemini Pro**: Complex reasoning tasks (query analysis in `query_analysis_router.py`, answer generation in `generate_answer.py`)
- **Groq (Llama3-8B)**: Fast evaluation tasks (context assessment in `context_assessment_groq.py`, hallucination checks in `evaluate_groq.py`, relevance checks in `relevance_groq.py`)

**DO NOT** switch evaluation tasks back to Gemini - this would increase latency by 35-40%. The Groq integration is in `backend/inference_clients/groq_client.py`.

### Session-Based Multi-Document Management

**Key pattern**: Each user session maintains isolated FAISS indexes and conversation history via `backend/session_manager.py`:

- Session creation assigns unique `collection_name` (e.g., `session_<uuid>`)
- Document uploads are tagged with `source_document` metadata for filtering
- FAISS indexes stored per-session in `faiss_indexes/session_<uuid>.faiss`
- **Persistent chunk store** at `faiss_indexes/chunk_stores/<collection>_chunks.pkl` enables reliable index rebuilding after document deletion

**Metadata filtering**: Query analysis identifies relevant source documents, and `multi_tool_executor.py` applies strict FAISS metadata filters to prevent context contamination in multi-document scenarios.

### Self-Correcting Retrieval Loop

The system can retry retrieval up to 2 times if initial context is insufficient:

1. After retrieval, `assess_context_sufficiency` (in `context_assessment_groq.py`) performs **Gap Analysis** - checks if context logically covers all query parts
2. If `context_assessment == "insufficient"`, `rewrite_query` (in `rewrite_query.py`) focuses on missing information identified in gap analysis
3. Loop prevention: `rewrite_attempts` counter in state, max 2 attempts
4. Tool preservation: Query rewriter maintains original tool choice from execution plan

### Document Processing Pipeline

Document ingestion flow (`backend/document_processor.py`):

1. Load via `MultiModalDocumentLoader` (`document_loader.py`) - supports PDF, DOCX, TXT, CSV, XLSX
2. Metadata extraction via `metadata_extractor.py` - tags each chunk with `source_document`, `page`, etc.
3. Recursive text splitting (500 chars, 50 overlap) with metadata preservation
4. Embedding via `sentence-transformers/all-mpnet-base-v2`
5. FAISS indexing with metadata filtering support
6. **Persistent chunk storage** to pickle file for rebuild capability

### API Endpoints Pattern

FastAPI endpoints in `backend/api.py` follow this pattern:

- **Session lifecycle**: `/api/start-session` (POST), `/api/session/{id}` (GET), `/api/session/{id}` (DELETE)
- **Document management**: `/api/upload` (single), `/api/upload-multiple` (batch), `/api/session/{id}/document/{doc_id}` (DELETE)
- **Q&A**: `/api/ask` (POST) - requires `session_id`, streams via SSE in production
- **Admin**: `/api/clear-database`, `/api/nuclear-reset` (complete FAISS wipe)

All endpoints use session-aware wrappers (`SessionAwareDocumentProcessor`, `SessionAwareRAGWorkflow`) that bridge session management with core processing logic.

## Testing Patterns

Tests in `tests/` and `backend/tests/` follow these patterns:

- `test_metadata_*.py` - Metadata extraction and filtering logic
- `test_groq_*.py` - Groq API integration and fallback mechanisms
- `test_query_rewriting.py` - Self-correction loop behavior
- `test_hybrid_queries.py` - Multi-tool execution with docs + web
- `test_deletion_rebuild.py` - Chunk store persistence and index rebuilding

Run tests individually to debug specific components without full pipeline overhead.

## Performance Considerations

**GPU Acceleration**: Reranking uses Apple Silicon MPS if available (2-3x speedup). Check `backend/chains/rerank_documents.py` for device detection logic.

**Caching**: Relevance scores are cached in `GraphState` using context signatures (see `_create_context_signature` in `rag_workflow.py`). Avoids redundant LLM calls when context hasn't changed.

**Batch Processing**: Multiple file uploads use `process_multiple_files_for_session_api` to build single FAISS index (more efficient than per-file indexing).

**Streaming**: Production should implement SSE streaming for answer generation to reduce perceived latency (TTFT target: <2s).

## Common Pitfalls

1. **Context contamination**: Always ensure query analysis metadata filtering is working. Multi-document queries should only retrieve from relevant sources.

2. **Infinite loops**: Check `rewrite_attempts` and `generation_attempts` in state. Circuit breakers prevent runaway retries.

3. **FAISS index corruption**: After document deletion, index MUST be rebuilt from chunk store. Never modify FAISS index in-place.

4. **LLM choice**: Don't use Groq for complex reasoning (query analysis) - it fails on multi-part queries. Keep Gemini for "brain" tasks.

5. **Session isolation**: Each session has separate FAISS index. Never share retrievers across sessions.

6. **Metadata preservation**: When splitting documents, metadata MUST be copied to all chunks. See `document_processor.py` chunk creation loop.

## File Organization Logic

- `backend/chains/` - LangGraph workflow nodes (each file is a single workflow step)
- `backend/inference_clients/` - LLM client abstractions (Groq, Gemini wrappers)
- `backend/api.py` - FastAPI endpoints and session-aware wrappers
- `backend/rag_workflow.py` - LangGraph StateGraph definition and node orchestration
- `backend/state.py` - GraphState schema (single source of truth for workflow data)
- `backend/session_manager.py` - Session lifecycle and isolation logic
- `backend/document_processor.py` - Document ingestion, chunking, FAISS indexing, chunk store
- `frontend/src/App.jsx` - Main React UI component
- `faiss_indexes/` - Per-session FAISS indexes and chunk stores (gitignored, local only)

## Deployment Notes

**Docker**: Use `docker-compose up --build` for full stack. Frontend uses Nginx reverse proxy (config in `frontend/nginx.conf`).

**Production checklist**:
- Set `DEBUG=false` in environment
- Use HTTPS with proper certificates
- Implement rate limiting on `/api/ask`
- Enable CORS only for trusted domains (update `api.py` CORS middleware)
- Set up monitoring for API latency and error rates
- Configure Redis for caching relevance scores across requests

## When Making Changes

**Adding new chains**: Create new file in `backend/chains/`, implement function that takes `GraphState` and returns updated state, add to workflow in `rag_workflow.py`.

**Modifying state**: Update `GraphState` TypedDict in `backend/state.py`, ensure all workflow nodes handle new/changed fields.

**Changing LLM providers**: Update `backend/config.py` and corresponding chain files. Maintain hybrid strategy pattern (fast LLM for eval, capable LLM for reasoning).

**Adding document formats**: Extend `MultiModalDocumentLoader` in `backend/document_loader.py`, ensure metadata extraction works for new format.

**Frontend changes**: UI components are in `frontend/src/`, use Tailwind for styling, Axios for API calls. Follow existing patterns for session management and error handling.
