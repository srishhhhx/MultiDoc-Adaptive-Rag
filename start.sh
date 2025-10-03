#!/bin/bash

# Advanced RAG System Startup Script
# This script starts both the FastAPI backend and React frontend

echo "🚀 Starting Advanced RAG System..."
echo ""

# Check if virtual environment exists
if [ ! -d "rag_env" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python -m venv rag_env"
    echo "   source rag_env/bin/activate  # or rag_env\\Scripts\\activate on Windows"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Frontend dependencies not found. Please run:"
    echo "   cd frontend && npm install"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please create one with your API keys."
    echo "   See README_NEW.md for required environment variables."
    echo ""
fi

# Start backend in background
echo "🐍 Starting FastAPI backend on port 8000..."
source rag_env/bin/activate
python api.py &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Start frontend in background
echo "⚛️  Starting React frontend on port 5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

echo ""
echo "✅ Both services are starting..."
echo ""
echo "📍 Access points:"
echo "   Frontend:  http://localhost:5173"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Trap Ctrl+C and cleanup
trap "echo ''; echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

# Wait for both processes
wait
