# 🚀 Quick Setup Guide

This guide will help you get the Advanced RAG system up and running in minutes.

## Prerequisites

- **Python 3.11+** installed
- **Node.js 18+** installed
- **Git** installed

## Step-by-Step Setup

### 1️⃣ Install Backend Dependencies

```bash
# Create virtual environment
python -m venv rag_env

# Activate it
# On Windows:
rag_env\Scripts\activate
# On Mac/Linux:
source rag_env/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 2️⃣ Configure Environment Variables

Create a `.env` file in the `AdvLang` directory:

```env
OPENAI_API_KEY=sk-your-openai-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here

# Optional - for workflow tracing
LANGCHAIN_API_KEY=your-langsmith-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Advanced-RAG-LangGraph
```

**Where to get API keys:**
- OpenAI: https://platform.openai.com/api-keys
- Tavily: https://tavily.com/ (sign up for free)
- LangSmith: https://smith.langchain.com/ (optional)

### 3️⃣ Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 4️⃣ Start the Application

**Option A: Using Startup Scripts (Recommended)**

On Windows:
```bash
start.bat
```

On Mac/Linux:
```bash
chmod +x start.sh
./start.sh
```

**Option B: Manual Start**

Open two terminal windows:

Terminal 1 (Backend):
```bash
# Activate virtual environment first
python api.py
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

### 5️⃣ Access the Application

Open your browser and go to:
- **Frontend UI**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

## 🎯 First Time Usage

1. **Upload a Document**
   - Drag and drop or click to browse
   - Supported: PDF, DOCX, TXT, CSV, XLSX
   - Max size: 10MB

2. **Ask a Question**
   - Type your question in the text area
   - Click "Ask Question"
   - View the answer and evaluation metrics

3. **Explore Features**
   - Check document evaluations
   - Review relevance scores
   - See if online search was used

## 🔧 Troubleshooting

### Backend won't start

**Issue**: Port 8000 already in use
```bash
# Find and kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Mac/Linux:
lsof -i :8000
kill -9 <PID>
```

**Issue**: Module not found
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**Issue**: API key errors
- Check your `.env` file exists
- Verify API keys are correct
- Ensure no extra spaces or quotes

### Frontend won't start

**Issue**: Port 5173 already in use
```bash
# Edit vite.config.js and change port to 3000
```

**Issue**: Dependencies error
```bash
# Clean install
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Issue**: Can't connect to backend
- Ensure backend is running on port 8000
- Check terminal for backend errors
- Try accessing http://localhost:8000/api/health

### File upload fails

**Issue**: Unsupported file type
- Only PDF, DOCX, TXT, CSV, XLSX are supported
- Check file extension is correct

**Issue**: File too large
- Maximum size is 10MB
- Compress or split larger files

**Issue**: Processing error
- Check backend logs for details
- Verify file is not corrupted
- Try a different file

## 📊 System Requirements

### Minimum
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Internet**: Required for API calls

### Recommended
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 5GB+ free space
- **Internet**: Stable broadband connection

## 🔒 Security Notes

- Never commit `.env` files to git
- Keep API keys secure
- Use environment variables for sensitive data
- In production, use HTTPS and proper authentication

## 📚 Next Steps

- Read [README_NEW.md](README_NEW.md) for detailed documentation
- Check [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) if migrating from Streamlit
- Explore API documentation at http://localhost:8000/docs
- Customize the UI in `frontend/src/components/`

## 🆘 Getting Help

If you're stuck:
1. Check the terminal/console for error messages
2. Review the troubleshooting section above
3. Ensure all prerequisites are installed
4. Verify your `.env` file is configured correctly
5. Try restarting both frontend and backend

## ✅ Verification Checklist

Before reporting issues, verify:
- [ ] Python 3.11+ is installed (`python --version`)
- [ ] Node.js 18+ is installed (`node --version`)
- [ ] Virtual environment is activated
- [ ] All dependencies are installed
- [ ] `.env` file exists with valid API keys
- [ ] Both backend and frontend are running
- [ ] No port conflicts (8000, 5173)
- [ ] Firewall allows local connections

---

Enjoy using the Advanced RAG System! 🎉
