#   Advanced RAG with LangGraph - React & FastAPI Edition

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Vite](https://img.shields.io/badge/Vite-5.0-646cff.svg)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38bdf8.svg)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, production-ready RAG (Retrieval-Augmented Generation) system with a beautiful React frontend and FastAPI backend, powered by LangGraph for intelligent document processing and question answering.

## 🌟 What's New

This version features a complete architectural overhaul:

- **🎨 Modern React Frontend** - Built with Vite and TailwindCSS for a sleek, responsive UI
- **⚡ FastAPI Backend** - High-performance async API with automatic documentation
- **🎯 Improved UX** - Drag-and-drop file uploads with real-time status updates
- **📊 Rich Visualizations** - Beautiful display of evaluation metrics and results
- **🔌 API-First Design** - Easily integrate with other applications

## ✨ Features

### Document Processing
- **Multi-Format Support**: PDFs, Word docs (DOCX), Excel files (XLSX, CSV), and text files
- **Drag-and-Drop Upload**: Intuitive file upload with visual feedback
- **Automatic Chunking**: Smart document splitting for optimal retrieval
- **Vector Storage**: ChromaDB for fast semantic search

### Intelligent Q&A
- **LangGraph Workflow**: Orchestrated RAG pipeline with conditional routing
- **Document Evaluation**: Quality checks on retrieved documents
- **Online Fallback**: Automatic web search when documents don't contain the answer
- **Hallucination Detection**: Built-in checks for answer accuracy

### Evaluation & Transparency
- **Relevance Scores**: Document and question-answer matching metrics
- **Confidence Levels**: Transparency in answer quality
- **Source Tracking**: Know whether answers came from documents or web search
- **Detailed Reasoning**: See the AI's evaluation process

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  React Frontend                     │
│         (Vite + TailwindCSS + Axios)               │
│  ┌──────────────────────────────────────────┐      │
│  │  FileUploader  │  QuestionAnswer  │  App  │      │
│  └──────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────┘
                         │
                    HTTP/REST API
                         │
┌─────────────────────────────────────────────────────┐
│                 FastAPI Backend                     │
│  ┌──────────────────────────────────────────┐      │
│  │   /api/upload   │   /api/ask   │  /api/* │      │
│  └──────────────────────────────────────────┘      │
│  ┌──────────────────────────────────────────┐      │
│  │  DocumentProcessor  │  RAGWorkflow        │      │
│  └──────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │ ChromaDB│    │LangGraph│    │  Tavily │
    │ Vector  │    │ Workflow│    │  Search │
    │   DB    │    │         │    │         │
    └─────────┘    └─────────┘    └─────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** - [Download here](https://www.python.org/downloads/)
- **Node.js 18+** - [Download here](https://nodejs.org/)
- **Git** - [Download here](https://git-scm.com/downloads)

### API Keys Required

- **OpenAI API Key** - For LLM operations
- **Tavily API Key** - For online search (optional but recommended)
- **LangSmith API Key** - For workflow tracing (optional)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/chitralputhran/Advanced-RAG-LangGraph.git
cd Advanced-RAG-LangGraph/AdvLang
```

#### 2. Backend Setup

Create and activate a virtual environment:

**Mac/Linux:**
```bash
python3 -m venv rag_env
source rag_env/bin/activate
```

**Windows:**
```bash
python -m venv rag_env
rag_env\Scripts\activate
```

Install Python dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Configure Environment Variables

Create a `.env` file in the `AdvLang` directory:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional but recommended
TAVILY_API_KEY=your_tavily_api_key_here

# Optional - for workflow tracing
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Advanced-RAG-LangGraph
```

#### 4. Frontend Setup

Navigate to the frontend directory:
```bash
cd frontend
```

Install Node.js dependencies:
```bash
npm install
```

### Running the Application

You'll need two terminal windows:

#### Terminal 1 - Backend (FastAPI)

```bash
# In the AdvLang directory
python api.py
```

The API will start at `http://localhost:8000`

- API Documentation: `http://localhost:8000/docs`
- Alternative Docs: `http://localhost:8000/redoc`

#### Terminal 2 - Frontend (React)

```bash
# In the AdvLang/frontend directory
npm run dev
```

The frontend will start at `http://localhost:5173`

Open your browser and navigate to `http://localhost:5173`

## 📖 Usage Guide

### 1. Upload a Document

1. Visit `http://localhost:5173`
2. Drag and drop a file or click to browse
3. Supported formats: PDF, DOCX, TXT, CSV, XLSX
4. Wait for processing to complete (you'll see a success message)

### 2. Ask Questions

1. Type your question in the text area
2. Click "Ask Question"
3. View the answer with evaluation metrics
4. Check whether the answer came from your documents or online search

### 3. Review Evaluations

- **Search Method**: See if document search or online search was used
- **Evaluation Scores**: View relevance and confidence metrics
- **Document Details**: Expand to see individual document evaluations
- **Reasoning**: Read the AI's thought process

## 🔌 API Endpoints

### POST `/api/upload`
Upload and process a document

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (PDF, DOCX, TXT, CSV, or XLSX)

**Response:**
```json
{
  "success": true,
  "message": "File uploaded and processed successfully",
  "file_id": "uuid-string",
  "metadata": {
    "filename": "document.pdf",
    "size": "2.5MB",
    "chunks": 42,
    "uploadedAt": "2025-09-30T03:51:30.000Z"
  }
}
```

### POST `/api/ask`
Ask a question about uploaded documents

**Request:**
```json
{
  "question": "What is the main topic?",
  "file_id": "uuid-string"
}
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "search_method": "documents",
  "online_search": false,
  "document_evaluations": [...],
  "question_relevance_score": {...},
  "document_relevance_score": {...}
}
```

### GET `/api/health`
Health check endpoint

### GET `/api/supported-formats`
Get list of supported file formats

## 🛠️ Development

### Project Structure

```
AdvLang/
├── api.py                      # FastAPI backend
├── rag_workflow.py             # LangGraph RAG workflow
├── document_processor.py       # Document processing logic
├── document_loader.py          # Multi-format document loader
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUploader.jsx
│   │   │   └── QuestionAnswer.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
│
└── chains/                     # LangGraph chains
    ├── document_relevance.py
    ├── evaluate.py
    ├── generate_answer.py
    └── question_relevance.py
```

### Building for Production

#### Backend
```bash
# Use a production ASGI server
pip install gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker
```

#### Frontend
```bash
cd frontend
npm run build
# Output will be in frontend/dist
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM |
| `TAVILY_API_KEY` | Recommended | Tavily API for online search |
| `LANGCHAIN_API_KEY` | Optional | LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | Optional | Enable LangSmith |
| `LANGCHAIN_PROJECT` | Optional | LangSmith project name |

## 🐛 Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
# Change the port in api.py (last line)
uvicorn.run(app, host="0.0.0.0", port=8001)
```

**CORS errors:**
- Ensure frontend URL is in `allow_origins` in `api.py`

**Module import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend Issues

**Port 5173 already in use:**
```bash
# Edit vite.config.js and change the port
server: {
  port: 3000,
  ...
}
```

**API connection refused:**
- Ensure backend is running on port 8000
- Check proxy settings in `vite.config.js`

**Module not found:**
```bash
# Clean install
rm -rf node_modules package-lock.json
npm install
```

## 🎨 Customization

### Styling
- Edit `frontend/src/index.css` for global styles
- Modify `frontend/tailwind.config.js` for Tailwind configuration
- Component styles use Tailwind utility classes

### API Configuration
- Edit `config.py` for RAG parameters
- Modify chunk size, overlap, and other settings

### LangGraph Workflow
- Customize workflow in `rag_workflow.py`
- Add or modify chains in the `chains/` directory

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangGraph** - Workflow orchestration
- **LangChain** - LLM framework
- **FastAPI** - Modern Python web framework
- **React** - UI library
- **Vite** - Build tool
- **TailwindCSS** - Utility-first CSS

## 🔗 Links

- [Original Streamlit Version](./README.md)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [React Documentation](https://react.dev/)

---

Built with ❤️ using LangGraph, FastAPI, and React
