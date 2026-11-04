"""
Startup script for the Advanced RAG API

This script ensures the backend module is properly accessible
and starts the FastAPI server.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import and run the API
if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI app
    uvicorn.run(
        "backend.api:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=8000,
        reload=os.getenv("API_RELOAD", "0") == "1",
        log_level="info"
    )
