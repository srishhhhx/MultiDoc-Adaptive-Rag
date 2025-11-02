"""
Query Analysis Router for Multi-Tool RAG Agent - GROQ-POWERED VERSION

This module implements an intelligent query analysis router using Groq's llama-3.3-70b-versatile
model for 10-20x faster query analysis compared to Gemini. The 70B model is required for
the complex reasoning needed to reliably decompose queries and generate perfect JSON execution plans.

Performance: ~1-2 seconds vs ~3-12 seconds with Gemini
Fallback: Automatically falls back to Gemini if Groq fails
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Import Groq client
from backend.inference_clients.groq_client import GroqModelClient

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Enhanced Query Analysis Router Prompt with Metadata Extraction
# (EXACT SAME PROMPT AS GEMINI VERSION)
query_analysis_prompt_text = """You are an expert query analyst for an advanced RAG system. Your role is to analyze user queries, create structured execution plans, and extract structured metadata for efficient retrieval.

AVAILABLE DOCUMENTS:
{available_documents}

AVAILABLE TOOLS:
1. **vectorstore_retrieval**: For questions about uploaded documents, document summaries, content analysis, or any information that could be found in the provided documents.
2. **web_search**: For real-time information, current events, external facts, or information not likely to be in the documents.

YOUR DUAL TASK:
1. **Task Planning**: Decompose the query into one or more independent sub-tasks with appropriate tool selection
2. **Metadata Extraction**: Extract structured metadata including document references, entities, and temporal information

ANALYSIS GUIDELINES:

**Use vectorstore_retrieval when:**
- Query asks about "the document", "this document", "the file", "the text"
- Requests summaries, overviews, or analysis of provided content
- Asks for specific information that could be in uploaded documents
- Uses phrases like "according to the document", "based on the text"
- Asks about people, concepts, or data mentioned in documents
- References specific document names (e.g., "Resume.v23.pdf", "report.docx")

**Use web_search when:**
- Query asks for current/real-time information (weather, news, stock prices)
- Requests external facts not likely in documents (definitions, general knowledge)
- Asks about recent events or developments
- Needs information about current status of companies, people, or events

**Use BOTH tools for comparative or relational queries when:**
- The query asks to compare, contrast, or relate a concept from the document to an external, modern, or general knowledge concept.

**METADATA EXTRACTION GUIDELINES:**
- **source_document**: Extract any specific document filename mentioned (e.g., "Resume.v23.pdf", "quarterly_report.xlsx")
- **mentioned_people**: Extract person names mentioned in the query
- **mentioned_companies**: Extract company/organization names mentioned
- **date_range**: Extract temporal references ("today", "2025", "last quarter", "recent")
- When the user's query contains a generic reference like "this resume," "the paper," or "the document," you must analyze the `available_documents` list and map the reference to the most logical filename for the `source_document` field.

**CRITICAL OUTPUT FORMAT:**
You MUST respond with a valid JSON object containing two top-level keys:

```json
{{
  "tasks": [
    {{
      "tool": "vectorstore_retrieval" | "web_search",
      "query": "specific sub-query for this tool",
      "source_document": "filename for vectorstore_retrieval tasks (e.g., 'Resume.v23.pdf') or null for web_search tasks"
    }}
  ],
  "metadata": {{
    "source_document": "primary document filename if mentioned (e.g., 'Resume.v23.pdf') or null",
    "mentioned_people": ["array of person names"],
    "mentioned_companies": ["array of company names"],
    "date_range": "temporal reference if mentioned or null"
  }}
}}
```

**IMPORTANT:** For vectorstore_retrieval tasks, always include the `source_document` field in each task to enable precise document filtering during retrieval.

**EXAMPLES:**

Query: "What is the weather like today and give me a summary of the document provided?"
Response:
```json
{{
  "tasks": [
    {{
      "tool": "web_search",
      "query": "current weather today",
      "source_document": null
    }},
    {{
      "tool": "vectorstore_retrieval", 
      "query": "summary of the provided document",
      "source_document": null
    }}
  ],
  "metadata": {{
    "source_document": null,
    "mentioned_people": [],
    "mentioned_companies": [],
    "date_range": "today"
  }}
}}
```

Query: "Who is the CEO of Apple and what does Resume.v23.pdf say about John's leadership experience?"
Response:
```json
{{
  "tasks": [
    {{
      "tool": "web_search",
      "query": "current CEO of Apple 2024",
      "source_document": null
    }},
    {{
      "tool": "vectorstore_retrieval",
      "query": "John's leadership experience from Resume.v23.pdf",
      "source_document": "Resume.v23.pdf"
    }}
  ],
  "metadata": {{
    "source_document": "Resume.v23.pdf",
    "mentioned_people": ["John"],
    "mentioned_companies": ["Apple"],
    "date_range": null
  }}
}}
```

Query: "Summarize the main findings in the research paper"
Response:
```json
{{
  "tasks": [
    {{
      "tool": "vectorstore_retrieval",
      "query": "summarize the main findings in the research paper",
      "source_document": null
    }}
  ],
  "metadata": {{
    "source_document": null,
    "mentioned_people": [],
    "mentioned_companies": [],
    "date_range": null
  }}
}}
```

Query: "What did Microsoft report in quarterly_report.xlsx about recent AI developments?"
Response:
```json
{{
  "tasks": [
    {{
      "tool": "vectorstore_retrieval",
      "query": "Microsoft AI developments from quarterly_report.xlsx",
      "source_document": "quarterly_report.xlsx"
    }}
  ],
  "metadata": {{
    "source_document": "quarterly_report.xlsx",
    "mentioned_people": [],
    "mentioned_companies": ["Microsoft"],
    "date_range": "recent"
  }}
}}
```

Query: "What are the projects mentioned in this resume?"
Available Documents: ["Resume.v23.pdf", "ADV_QB.docx"]
Response:
```json
{{
  "tasks": [
    {{
      "tool": "vectorstore_retrieval",
      "query": "list the projects",
      "source_document": "Resume.v23.pdf"
    }}
  ],
  "metadata": {{
    "source_document": "Resume.v23.pdf",
    "mentioned_people": [],
    "mentioned_companies": [],
    "date_range": null
  }}
}}
```

Respond with ONLY the JSON object containing both 'tasks' and 'metadata', no additional text or explanation."""


class QueryAnalysisClient:
    """Client for performing query analysis with Groq primary and Gemini fallback"""
    
    def __init__(self):
        """Initialize both Groq and Gemini clients"""
        # Primary: Groq with 70B model for complex reasoning
        # NOTE: Using llama-3.3-70b-versatile for this critical reasoning task
        # The 70B model is required to handle sophisticated query decomposition
        # and reliable JSON generation with metadata extraction
        try:
            self.groq_client = GroqModelClient(model_name="llama-3.3-70b-versatile")
            logger.info("✅ Groq client initialized for query analysis (llama-3.3-70b-versatile)")
        except Exception as e:
            logger.warning(f"⚠️ Groq client initialization failed: {e}")
            self.groq_client = None
        
        # Fallback: Gemini for reliability
        try:
            self.gemini_client = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=os.environ["GOOGLE_API_KEY"],
            )
            logger.info("✅ Gemini fallback initialized for query analysis")
        except Exception as e:
            logger.warning(f"⚠️ Gemini fallback initialization failed: {e}")
            self.gemini_client = None
    
    def analyze(self, question: str, available_documents: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze query using Groq (primary) or Gemini (fallback)
        
        Args:
            question: User's query
            available_documents: Optional list of document filenames in session
            
        Returns:
            Dict with 'tasks' and 'metadata' keys
        """
        start_time = time.time()
        
        # Format available documents for prompt
        if available_documents:
            documents_text = str(available_documents)
        else:
            documents_text = "No documents available in current session"
        
        # Build full prompt
        full_prompt = query_analysis_prompt_text.format(
            available_documents=documents_text
        ) + f"\n\nNow analyze this query: {question}"
        
        # Try Groq first
        if self.groq_client:
            try:
                logger.info(f"🚀 GROQ QUERY ANALYSIS: Analyzing with llama-3.3-70b-versatile")
                
                # Use Groq for fast inference with JSON mode
                response_text, metrics = self.groq_client.infer(
                    prompt=full_prompt,
                    temperature=0.1,
                    max_tokens=1024,
                    json_mode=True
                )
                
                latency_ms = (time.time() - start_time) * 1000
                logger.info(f"✅ Groq query analysis complete ({latency_ms:.0f}ms, {metrics.total_tokens} tokens)")
                
                # Parse JSON response
                analysis_result = json.loads(response_text)
                
                # Validate structure
                if not isinstance(analysis_result, dict):
                    raise ValueError("Response must be a JSON object")
                
                if "tasks" not in analysis_result or "metadata" not in analysis_result:
                    raise ValueError("Response must contain 'tasks' and 'metadata' keys")
                
                # Validate and enhance tasks
                tasks = analysis_result["tasks"]
                if not isinstance(tasks, list):
                    raise ValueError("'tasks' must be a JSON array")
                
                for task in tasks:
                    if not isinstance(task, dict):
                        raise ValueError("Each task must be a JSON object")
                    if "tool" not in task or "query" not in task:
                        raise ValueError("Each task must have 'tool' and 'query' fields")
                    if task["tool"] not in ["vectorstore_retrieval", "web_search"]:
                        raise ValueError(f"Invalid tool: {task['tool']}")
                    
                    # Ensure source_document field exists
                    if "source_document" not in task:
                        task["source_document"] = None
                
                # Validate and enhance metadata
                metadata = analysis_result["metadata"]
                if not isinstance(metadata, dict):
                    raise ValueError("'metadata' must be a JSON object")
                
                metadata.setdefault("source_document", None)
                metadata.setdefault("mentioned_people", [])
                metadata.setdefault("mentioned_companies", [])
                metadata.setdefault("date_range", None)
                
                logger.info(f"Successfully parsed Groq analysis: {len(tasks)} tasks, metadata: {metadata}")
                return analysis_result
                
            except Exception as e:
                logger.error(f"❌ Groq query analysis failed: {e}")
                logger.warning("🔄 Falling back to Gemini...")
        
        # Fallback to Gemini
        if self.gemini_client:
            try:
                logger.info("🔄 GEMINI FALLBACK: Analyzing with gemini-2.5-flash")
                
                # Use Gemini with LangChain
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.output_parsers import StrOutputParser
                
                analysis_prompt = ChatPromptTemplate.from_messages([
                    ("system", query_analysis_prompt_text),
                    ("user", "Now analyze this query: {question}")
                ])
                
                chain = analysis_prompt | self.gemini_client | StrOutputParser()
                
                result = chain.invoke({
                    "question": question,
                    "available_documents": documents_text
                })
                
                latency_ms = (time.time() - start_time) * 1000
                logger.info(f"✅ Gemini fallback complete ({latency_ms:.0f}ms)")
                
                # Clean and parse JSON
                cleaned_result = result.strip()
                if cleaned_result.startswith("```json"):
                    cleaned_result = cleaned_result[7:]
                if cleaned_result.endswith("```"):
                    cleaned_result = cleaned_result[:-3]
                cleaned_result = cleaned_result.strip()
                
                analysis_result = json.loads(cleaned_result)
                
                # Same validation as Groq path
                if not isinstance(analysis_result, dict):
                    raise ValueError("Response must be a JSON object")
                
                if "tasks" not in analysis_result or "metadata" not in analysis_result:
                    raise ValueError("Response must contain 'tasks' and 'metadata' keys")
                
                tasks = analysis_result["tasks"]
                metadata = analysis_result["metadata"]
                
                # Enhance with defaults
                for task in tasks:
                    task.setdefault("source_document", None)
                
                metadata.setdefault("source_document", None)
                metadata.setdefault("mentioned_people", [])
                metadata.setdefault("mentioned_companies", [])
                metadata.setdefault("date_range", None)
                
                logger.info(f"Successfully parsed Gemini fallback: {len(tasks)} tasks")
                return analysis_result
                
            except Exception as e:
                logger.error(f"❌ Gemini fallback also failed: {e}")
        
        # Both failed - use pattern-based fallback
        logger.error("❌ Both Groq and Gemini failed - using pattern-based fallback")
        return _create_fallback_analysis(question, available_documents)


# Initialize global client
_query_analysis_client = QueryAnalysisClient()


def analyze_query(question: str, available_documents: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze a user query and create a structured execution plan with metadata extraction
    
    Args:
        question: The user's question to analyze
        available_documents: Optional list of document filenames available in the session
        
    Returns:
        Dict[str, Any]: Dictionary containing 'tasks' and 'metadata'
        
    Example:
        {
            "tasks": [
                {"tool": "vectorstore_retrieval", "query": "document summary"},
                {"tool": "web_search", "query": "current weather"}
            ],
            "metadata": {
                "source_document": "Resume.v23.pdf",
                "mentioned_people": ["John"],
                "mentioned_companies": ["Apple"],
                "date_range": "today"
            }
        }
    """
    return _query_analysis_client.analyze(question, available_documents)


def _create_fallback_analysis(question: str, available_documents: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create a fallback analysis result when both Groq and Gemini fail
    
    Args:
        question: The user's question
        available_documents: Optional list of document filenames available in the session
        
    Returns:
        Dict[str, Any]: Simple analysis result with tasks and empty metadata
    """
    logger.info("Creating pattern-based fallback execution plan")
    
    question_lower = question.lower()
    
    # Document-specific patterns
    document_patterns = [
        "summary", "summarize", "overview", "main points", "key findings",
        "this document", "the document", "the file", "the text", "the paper",
        "according to", "based on", "analyze this", "explain the content",
        "education", "college", "university", "degree", "graduated", "studied",
        "complete his", "complete her", "background", "experience", "career"
    ]
    
    # Web search patterns  
    web_patterns = [
        "weather", "current", "latest", "recent", "today", "now",
        "stock price", "news", "what is", "who is", "when was"
    ]
    
    # Check for hybrid query
    has_document_patterns = any(pattern in question_lower for pattern in document_patterns)
    has_web_patterns = any(pattern in question_lower for pattern in web_patterns)
    
    # Try to detect generic document references
    source_document = None
    if available_documents:
        generic_patterns = ["this resume", "the resume", "this document", "the document", "this paper", "the paper", "this file", "the file"]
        for pattern in generic_patterns:
            if pattern in question_lower:
                if "resume" in pattern:
                    for doc in available_documents:
                        if any(keyword in doc.lower() for keyword in ["resume", "cv"]):
                            source_document = doc
                            break
                elif "paper" in pattern:
                    for doc in available_documents:
                        if any(keyword in doc.lower() for keyword in ["paper", "research", "study", "article"]):
                            source_document = doc
                            break
                elif available_documents:
                    source_document = available_documents[0]
                break
    
    fallback_metadata = {
        "source_document": source_document,
        "mentioned_people": [],
        "mentioned_companies": [],
        "date_range": None
    }
    
    if has_document_patterns and has_web_patterns:
        logger.info("Detected hybrid query in fallback")
        return {
            "tasks": [
                {"tool": "vectorstore_retrieval", "query": question, "source_document": source_document},
                {"tool": "web_search", "query": question, "source_document": None}
            ],
            "metadata": fallback_metadata
        }
    elif has_document_patterns:
        return {
            "tasks": [{"tool": "vectorstore_retrieval", "query": question, "source_document": source_document}],
            "metadata": fallback_metadata
        }
    elif has_web_patterns:
        return {
            "tasks": [{"tool": "web_search", "query": question, "source_document": None}],
            "metadata": fallback_metadata
        }
    
    # Default: try documents first
    return {
        "tasks": [{"tool": "vectorstore_retrieval", "query": question, "source_document": source_document}],
        "metadata": fallback_metadata
    }


def get_execution_summary(analysis_result: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the analysis result
    
    Args:
        analysis_result: The analysis result from analyze_query()
        
    Returns:
        str: Human-readable summary
    """
    if not analysis_result or "tasks" not in analysis_result:
        return "No execution plan generated"
    
    tasks = analysis_result["tasks"]
    metadata = analysis_result.get("metadata", {})
    
    if len(tasks) == 1:
        task = tasks[0]
        tool_name = "document search" if task["tool"] == "vectorstore_retrieval" else "web search"
        summary = f"Single task: {tool_name}"
    else:
        tool_counts = {}
        for task in tasks:
            tool = task["tool"]
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        summary_parts = []
        if tool_counts.get("vectorstore_retrieval", 0) > 0:
            summary_parts.append(f"{tool_counts['vectorstore_retrieval']} document search(es)")
        if tool_counts.get("web_search", 0) > 0:
            summary_parts.append(f"{tool_counts['web_search']} web search(es)")
        
        summary = f"Multi-task plan: {', '.join(summary_parts)}"
    
    # Add metadata info
    metadata_info = []
    if metadata.get("source_document"):
        metadata_info.append(f"Document: {metadata['source_document']}")
    if metadata.get("mentioned_people"):
        metadata_info.append(f"People: {', '.join(metadata['mentioned_people'])}")
    if metadata.get("mentioned_companies"):
        metadata_info.append(f"Companies: {', '.join(metadata['mentioned_companies'])}")
    
    if metadata_info:
        summary += f" | {' | '.join(metadata_info)}"
    
    return summary
