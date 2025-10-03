# Migration Guide: From Streamlit to React + FastAPI

This guide helps you transition from the old Streamlit-based UI to the new React + FastAPI architecture.

## Overview of Changes

| Aspect | Old (Streamlit) | New (React + FastAPI) |
|--------|----------------|----------------------|
| **Frontend** | Streamlit | React + Vite + TailwindCSS |
| **Backend** | Integrated in Streamlit | Separate FastAPI server |
| **Architecture** | Monolithic | Client-Server |
| **State Management** | Streamlit session state | React state + API |
| **Styling** | Streamlit components | TailwindCSS |
| **API** | N/A | RESTful API |

## Key Benefits of New Architecture

### 🎯 Separation of Concerns
- Frontend and backend are independent
- Easier to test and maintain
- Can scale independently

### ⚡ Better Performance
- Async FastAPI for concurrent requests
- React's virtual DOM for efficient UI updates
- No page reloads

### 🎨 Modern UI/UX
- Custom-designed components
- Smooth animations and transitions
- Mobile-responsive design

### 🔌 API-First Design
- Easy integration with other services
- Can build mobile apps using the same API
- Automatic API documentation

## What Stayed the Same

✅ **Core RAG Logic**: All LangGraph workflows remain unchanged  
✅ **Document Processing**: Same multi-format support  
✅ **Evaluation System**: Identical scoring and metrics  
✅ **Environment Variables**: Same `.env` configuration  

## File Mapping

### Removed Files (Streamlit-specific)
- `app.py` - Old Streamlit app
- `ui_components.py` - Streamlit UI components
- `state.py` - Can still be used but not required for API

### New Files
- `api.py` - FastAPI backend
- `frontend/` - Entire React application
- `start.sh` / `start.bat` - Startup scripts
- `README_NEW.md` - Updated documentation

### Modified Files
- `requirements.txt` - Added FastAPI, removed Streamlit
- `rag_workflow.py` - Adapted for non-Streamlit usage
- `document_processor.py` - Adapted for non-Streamlit usage

## Migration Steps

### 1. Backup Your Current Setup
```bash
# Create a backup branch
git branch streamlit-backup
```

### 2. Update Python Dependencies
```bash
# Activate your virtual environment
source rag_env/bin/activate  # or rag_env\Scripts\activate on Windows

# Install new dependencies
pip install -r requirements.txt
```

### 3. Install Node.js and Frontend Dependencies
```bash
# Install Node.js from https://nodejs.org/ (if not already installed)

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 4. Test the New System
```bash
# Option 1: Use startup scripts
./start.sh          # Linux/Mac
start.bat           # Windows

# Option 2: Manual start
# Terminal 1: Backend
python api.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

### 5. Verify Functionality
- [ ] Upload a document
- [ ] Ask a question
- [ ] Check evaluation metrics
- [ ] Try online search fallback

## API Integration Examples

### Using cURL
```bash
# Upload a file
curl -X POST "http://localhost:8000/api/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Ask a question
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

### Using Python
```python
import requests

# Upload file
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/upload',
        files={'file': f}
    )
    result = response.json()
    print(f"File ID: {result['file_id']}")

# Ask question
response = requests.post(
    'http://localhost:8000/api/ask',
    json={'question': 'What is the main topic?'}
)
answer = response.json()
print(f"Answer: {answer['answer']}")
```

### Using JavaScript/TypeScript
```javascript
// Upload file
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const uploadResponse = await fetch('http://localhost:8000/api/upload', {
  method: 'POST',
  body: formData
});
const uploadResult = await uploadResponse.json();

// Ask question
const askResponse = await fetch('http://localhost:8000/api/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: 'What is the main topic?'
  })
});
const answer = await askResponse.json();
```

## Common Issues and Solutions

### Issue: Backend won't start
**Solution:**
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# Change port in api.py if needed
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Issue: Frontend can't connect to backend
**Solution:**
- Ensure backend is running on port 8000
- Check CORS settings in `api.py`
- Verify proxy settings in `vite.config.js`

### Issue: File upload fails
**Solution:**
- Check file size (max 10MB by default)
- Verify file format is supported
- Check backend logs for errors

### Issue: Missing environment variables
**Solution:**
```bash
# Ensure .env file exists with required keys
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

## Reverting to Streamlit

If you need to revert to the Streamlit version:

```bash
# Switch to backup branch
git checkout streamlit-backup

# Or restore requirements
pip install streamlit
pip uninstall fastapi uvicorn python-multipart

# Run old app
streamlit run app.py
```

## Performance Comparison

| Metric | Streamlit | React + FastAPI |
|--------|-----------|----------------|
| Initial Load | ~3-5s | ~1-2s |
| File Upload | Blocking | Async |
| UI Updates | Full reload | Partial update |
| Concurrent Users | Limited | High |
| Mobile Support | Basic | Excellent |

## Feature Parity Checklist

- [x] Document upload (PDF, DOCX, TXT, CSV, XLSX)
- [x] Question answering
- [x] Document evaluation
- [x] Online search fallback
- [x] Evaluation metrics display
- [x] Error handling
- [x] File size validation
- [x] Real-time status updates
- [x] Reasoning display
- [x] Document chunk information

## Next Steps

### Extend the Frontend
- Add user authentication
- Implement chat history
- Add multi-file support
- Create document management UI

### Enhance the Backend
- Add user sessions
- Implement caching
- Add rate limiting
- Create webhook support

### Deploy to Production
- Containerize with Docker
- Set up CI/CD pipeline
- Configure production database
- Add monitoring and logging

## Support

If you encounter issues:
1. Check the [README_NEW.md](README_NEW.md) documentation
2. Review backend logs in terminal
3. Check browser console for frontend errors
4. Verify all dependencies are installed correctly

## Feedback

The new architecture is designed to be more flexible and maintainable. If you have suggestions or find issues, please open an issue on GitHub.

---

Happy coding! 🚀
