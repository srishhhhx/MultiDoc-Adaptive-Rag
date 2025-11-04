# Gemini Code Assistant Context

## Project Overview

This project is a full-stack web application that implements an advanced multi-document adaptive Retrieval-Augmented Generation (RAG) agent. The application allows users to upload multiple documents and ask complex questions that require synthesizing information from these documents and the web.

The backend is built with Python, FastAPI, and LangGraph. It uses a sophisticated, self-correcting RAG pipeline to provide accurate and well-grounded answers. The frontend is a single-page application built with React.js that provides a user-friendly interface for uploading documents and interacting with the RAG agent.

**Key Technologies:**

*   **Frontend:** React.js, Vite, Tailwind CSS
*   **Backend:** Python, FastAPI, Uvicorn
*   **AI Orchestration:** LangGraph
*   **LLMs:** Google Gemini Pro, Groq Llama3-8B
*   **Vector Database:** FAISS
*   **Web Search:** Tavily API

## Building and Running the Project

### Prerequisites

*   Python 3.11+
*   Node.js 18+
*   An `.env` file with `GOOGLE_API_KEY`, `TAVILY_API_KEY`, and `GROQ_API_KEY`.

### Backend Setup

1.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Start the backend server:
    ```bash
    python run_api.py
    ```

### Frontend Setup

1.  Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
2.  Install the required Node.js packages:
    ```bash
    npm install
    ```
3.  Start the frontend development server:
    ```bash
    npm run dev
    ```

The application will be accessible at `http://localhost:5173`.

## Key Files

*   `run_api.py`: The entry point for starting the backend FastAPI server.
*   `backend/api.py`: Defines all the API endpoints for the application.
*   `backend/rag_workflow.py`: Implements the core RAG pipeline using LangGraph.
*   `backend/document_processor.py`: Handles document processing, chunking, and FAISS vector database management.
*   `backend/config.py`: Contains all the configuration settings for the backend.
*   `frontend/src/App.jsx`: The main React component for the frontend application.
*   `frontend/package.json`: Defines the frontend dependencies and scripts.
*   `requirements.txt`: Defines the Python dependencies for the backend.
*   `README.md`: Provides a detailed overview of the project.

## Development Conventions

*   The backend code is located in the `backend` directory and follows a modular structure. The core logic is organized into `chains`, and the main API is defined in `backend/api.py`.
*   The frontend code is located in the `frontend` directory and uses a component-based architecture. The main application component is `frontend/src/App.jsx`.
*   The RAG pipeline is orchestrated using LangGraph, as defined in `backend/rag_workflow.py`.
*   The project uses a hybrid LLM strategy, with different models for different tasks to optimize for cost and latency.
*   The project includes a suite of tests in the `tests` directory.