
#   Advanced RAG with LangGraph

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.60-green.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.13-blue.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.23-purple.svg)](https://www.trychroma.com/)
[![Code style: Python](https://img.shields.io/badge/Code%20Style-Python-black.svg)](https://www.python.org/dev/peps/pep-0008/)

This is a web application that allows you to upload documents and ask questions about them. It is built with LangGraph, Streamlit, and ChromaDB. If your document does not contain the answer, it automatically searches online to help you out.

This project demonstrates how to build a RAG (Retrieval-Augmented Generation) system using LangGraph for workflow management. LangGraph helps orchestrate the different steps in the RAG pipeline, from document processing to answer generation, with built-in error handling and state management.

If you're looking to understand how LangGraph works with RAG systems, this implementation shows practical patterns for building document question-answering applications with proper workflow orchestration.

## How It Works

Here is what happens when you use this application and the process is actually quite straightforward:

![Workflow](screenshots/graph.png)

*This diagram shows how everything works together behind the scenes.*

---

## What You Can Do

- **Upload Different Types of Files**:
  - Just drag and drop your **PDFs, Word docs, Excel files, or text files**
  - The app figures out what type of file it is and reads the content automatically
  - Everything gets converted into a searchable format and stored so you can ask questions about it later

![Document Upload Interface](screenshots/document-upload.png)

*Simple drag-and-drop file upload that works with multiple file types*

- **Ask Questions and Get Answers**:
  - Type any question about your uploaded documents
  - The application tells you exactly where it found the answer in your files
  - Cannot find it in your documents? No problem - it will search online for you
  - You will always know if the answer came from your files or from the internet

![Q&A Interface](screenshots/qa-interface.png)

*Ask questions and get answers*

- **View Evaluation Results**:
  - See detailed system information about how your answer was generated
  - Review evaluation scores for document relevance, question matching, and answer grounding
  - Understand which documents were used and their relevance scores
  - View confidence levels and reasoning behind each evaluation
  - Check if online search was used or if answers came from your documents

![Evaluation Results](screenshots/evaluations.png)

*Comprehensive evaluation metrics and system transparency*

- **Smart Answer Generation**:
  - Uses LangGraph to make sure answers are relevant and accurate
  - Has built-in checks to catch when something might be wrong
  - Falls back to different sources if needed
  - Evaluates document relevance so you get better answers

- **See What's Happening Behind the Scenes**:
  - Connects with LangSmith so you can see how it's working
  - Great for debugging or just understanding what's going on

![LangSmith Tracing](screenshots/langsmith-tracing.png)

*Optional: See exactly how your questions are being processed*

---

## LangGraph RAG Implementation

This project shows how to implement RAG with LangGraph in a practical way. LangGraph handles the workflow orchestration, managing the different steps like document retrieval, relevance checking, and answer generation.

### Why LangGraph for RAG?
- **State Management**: LangGraph manages the application state as it moves through different processing steps
- **Conditional Logic**: The workflow can decide whether to search documents or go online based on what it finds
- **Error Handling**: Built-in mechanisms to handle failures and try alternative approaches
- **Extensibility**: Easy to add new steps or modify the workflow as needed

### Key LangGraph RAG Patterns Used:
- Document evaluation before answer generation
- Conditional routing between different search methods
- Multi-step validation and quality checks
- State transitions with proper error recovery

If you want to learn LangGraph RAG implementation, this codebase provides a complete working example with real-world patterns.

---

## File Types We Support

You can upload these types of files:
- **Text Files**: `.txt`
- **PDF Documents**: `.pdf` 
- **Microsoft Word**: `.docx`
- **Excel Files**: `.csv`, `.xlsx`

---

## How It Works

The application works in a few simple steps, but there is complex processing happening behind the scenes:

1. **When You Upload a Document**:
   - You upload your files (PDFs, Word docs, Excel, or text files)
   - The app reads the content and breaks it into smaller chunks
   - These chunks get converted into a special searchable format
   - Everything gets saved in a database called ChromaDB so it can find things quickly

2. **When You Ask a Question**:
   - You type your question in the text box
   - The app checks if your question makes sense
   - It searches through your uploaded documents to find relevant information
   - Multiple checks happen to make sure the answer will be good

3. **Getting Your Answer**:
   - The application looks at what it found in your documents
   - If there is relevant information, it writes an answer based on that
   - If your documents do not contain what you need, it searches online instead
   - It performs quality checks to ensure the answer is not fabricated

4. **Ensuring Everything Works Properly**:
   - The application has several checkpoints to catch problems
   - It can identify when an answer might be incorrect or fabricated
   - If one method does not work, it tries another approach
   - You always know where your answer originated

5. **Evaluation and Transparency**:
   - After generating an answer, the system provides detailed evaluation metrics
   - Shows document relevance scores and which documents were most helpful
   - Displays question-answer matching quality and completeness ratings
   - Provides confidence levels and reasoning for each evaluation
   - Indicates whether online search was used or if answers came from your documents
   - All evaluation data is presented in easy-to-read tables for full transparency

6. **Seeing What's Happening** (Optional):
   - If you set up LangSmith, you can see exactly what the app is doing
   - Great for understanding the process or fixing issues
   - Shows you timing and performance info

### LangGraph RAG Architecture

The workflow uses LangGraph to manage the entire RAG pipeline:

- **State Management**: All data flows through a defined GraphState that tracks questions, documents, and evaluation results
- **Conditional Routing**: The system decides whether to use document search or online search based on what it finds
- **Error Recovery**: If document search fails, the workflow automatically tries online search
- **Multi-Step Validation**: Each step includes quality checks before moving to the next stage
- **Extensible Design**: Easy to add new evaluation steps or modify the workflow logic

This LangGraph RAG implementation provides a good foundation for building more complex document processing systems.

---

## What You'll Need

Before you start, make sure you have these things:

- **Python 3.11 or newer** - [Get it here](https://www.python.org/downloads/)
- **Git** - [Download here](https://git-scm.com/downloads) 
- **OpenAI API Key** - You need this to make the LLM work
- **Tavily API Key** - This is for online search (optional, but really useful)
- **LangSmith API Key** - Only if you want to see the workflow details (optional)

---

## Getting Started

### Step 1: Download the Code

```bash
git clone https://github.com/chitralputhran/Advanced-RAG-LangGraph.git
cd Advanced-RAG-LangGraph
```

### Step 2: Set Up a Virtual Environment

This keeps everything organized and will not interfere with your other Python projects.

**If you are on Mac or Linux:**
```bash
python3 -m venv rag_env
source rag_env/bin/activate
```

**If you are on Windows:**
```bash
python -m venv rag_env
rag_env\Scripts\activate
```

### Step 3: Install Everything You Need

This installs all the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Add Your API Keys

Create a file called `.env` in the main folder and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Advanced-RAG-LangGraph
```

### Step 5: Start the App

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

---

## Quick Start Guide

1. **Upload a File**: Click the file uploader and pick a PDF, Word doc, Excel file, or text file
2. **Wait a Moment**: The app will read your document and get it ready for questions
3. **Ask Away**: Type your question and hit "Ask"
4. **Get Your Answer**: You will see the answer

## How to Use It

### The Basics

1. **Start the App**:
   ```bash
   streamlit run app.py
   ```
   Then go to `http://localhost:8501` in your browser

2. **Upload Your Files**:
   - Look for the file uploader on the page
   - Pick your files (PDFs, Word docs, Excel files, or text files)
   - Wait for the little progress bar to finish

3. **Ask Questions**:
   - Type your question in the text box
   - Click "Ask" or just press Enter
   - Check your answer


## When Things Go Wrong

### Common Issues

**Application Will Not Start**
- Ensure you have installed everything: `pip install -r requirements.txt`
- Check your Python version: `python --version` (must be 3.11 or newer)
- Make sure your virtual environment is active

**API Key Issues**
- Double-check your `.env` file contains the correct API keys
- Ensure your OpenAI account has sufficient credits
- Verify that your Tavily API key works (if you are using online search)

**Cannot Upload Files**
- Ensure your file type is supported (PDF, Word, Excel, or text)
- Large files take longer - please be patient!

**Running Slowly**
- Large documents can be slow - this is normal

## What's Inside

Here is how the code is organized:

```
Advanced-RAG-LangGraph/
├── app.py                # The main app file
├── config.py             # All the settings
├── utils.py              # Helper functions
├── ui_components.py      # What you see on screen
├── document_processor.py # How documents get processed
├── rag_workflow.py       # RAG workflow
├── document_loader.py    # Reads different file types
├── state.py              # Keeps track
├── requirements.txt      # List of needed packages
├── chains/               # LangGraph pieces
│   ├── __init__.py
│   ├── document_relevance.py
│   ├── evaluate.py
│   ├── generate_answer.py
│   └── question_relevance.py
└── screenshots/*.png     # Pictures for this README

---

## Settings

### Your API Keys

Make a file called `.env` in the main folder with your keys:

```env
# You definitely need this one
OPENAI_API_KEY=your_openai_api_key_here

# This one is really useful too
TAVILY_API_KEY=your_tavily_api_key_here

# Only if you want to see what is happening behind the scenes
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Advanced-RAG-LangGraph
```

---

## License

This project uses the MIT License - check out the [LICENSE](LICENSE) file for the details.






