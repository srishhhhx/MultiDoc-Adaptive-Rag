@echo off
REM Advanced RAG System Startup Script for Windows
REM This script starts both the FastAPI backend and React frontend

echo 🚀 Starting Advanced RAG System...
echo.

REM Check if virtual environment exists
if not exist "rag_env" (
    echo ❌ Virtual environment not found. Please run setup first:
    echo    python -m venv rag_env
    echo    rag_env\Scripts\activate
    echo    pip install -r requirements.txt
    exit /b 1
)

REM Check if frontend dependencies are installed
if not exist "frontend\node_modules" (
    echo ❌ Frontend dependencies not found. Please run:
    echo    cd frontend ^&^& npm install
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  Warning: .env file not found. Please create one with your API keys.
    echo    See README_NEW.md for required environment variables.
    echo.
)

REM Start backend
echo 🐍 Starting FastAPI backend on port 8000...
start "Advanced RAG Backend" cmd /k "rag_env\Scripts\activate && python api.py"

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend
echo ⚛️  Starting React frontend on port 5173...
start "Advanced RAG Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ✅ Both services are starting in separate windows...
echo.
echo 📍 Access points:
echo    Frontend:  http://localhost:5173
echo    Backend:   http://localhost:8000
echo    API Docs:  http://localhost:8000/docs
echo.
echo Press any key to exit this window (services will continue running)
pause >nul
