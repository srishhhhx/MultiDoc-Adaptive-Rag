"""
Query Analysis Router for Multi-Tool RAG Agent

This module implements an intelligent query analysis router that acts as the entry point
for the RAG pipeline. It analyzes user queries, decomposes complex questions into
sub-tasks, and creates structured execution plans that route tasks to appropriate tools.

The router can handle:
- Document-specific queries (vectorstore_retrieval)
- Web search queries (web_search) 
- Hybrid queries requiring both tools
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the LLM for query analysis
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.environ["GOOGLE_API_KEY"], 
    temperature=0.1
)

# Enhanced Query Analysis Router Prompt with Metadata Extraction
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

# Create the analysis prompt
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", query_analysis_prompt_text),
    ("user", "Now analyze this query: {question}")
])

# Create the analysis chain
query_analysis_chain = analysis_prompt | llm | StrOutputParser()


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
    try:
        logger.info(f"Analyzing query: {question}")
        
        # Format available documents for the prompt
        if available_documents:
            documents_text = str(available_documents)
            logger.info(f"Available documents: {documents_text}")
        else:
            documents_text = "No documents available in current session"
            logger.info("No available documents provided")
        
        # Prepare input for the chain
        chain_input = {
            "question": question,
            "available_documents": documents_text
        }
        logger.info(f"Chain input: {chain_input}")
        
        # Get LLM analysis
        result = query_analysis_chain.invoke(chain_input)
        logger.info(f"Raw LLM response: {result}")
        
        # Parse JSON response
        try:
            # Clean the response - remove any markdown formatting
            cleaned_result = result.strip()
            if cleaned_result.startswith("```json"):
                cleaned_result = cleaned_result[7:]  # Remove ```json
            if cleaned_result.endswith("```"):
                cleaned_result = cleaned_result[:-3]  # Remove ```
            cleaned_result = cleaned_result.strip()
            
            analysis_result = json.loads(cleaned_result)
            
            # Validate the structure
            if not isinstance(analysis_result, dict):
                raise ValueError("Response must be a JSON object")
            
            if "tasks" not in analysis_result or "metadata" not in analysis_result:
                raise ValueError("Response must contain 'tasks' and 'metadata' keys")
            
            tasks = analysis_result["tasks"]
            metadata = analysis_result["metadata"]
            
            # Validate tasks structure
            if not isinstance(tasks, list):
                raise ValueError("'tasks' must be a JSON array")
            
            for task in tasks:
                if not isinstance(task, dict):
                    raise ValueError("Each task must be a JSON object")
                if "tool" not in task or "query" not in task:
                    raise ValueError("Each task must have 'tool' and 'query' fields")
                if task["tool"] not in ["vectorstore_retrieval", "web_search"]:
                    raise ValueError(f"Invalid tool: {task['tool']}")
                
                # Ensure source_document field exists (add default if missing)
                if "source_document" not in task:
                    task["source_document"] = None
            
            # Validate metadata structure
            if not isinstance(metadata, dict):
                raise ValueError("'metadata' must be a JSON object")
            
            # Ensure metadata has required fields (with defaults)
            metadata.setdefault("source_document", None)
            metadata.setdefault("mentioned_people", [])
            metadata.setdefault("mentioned_companies", [])
            metadata.setdefault("date_range", None)
            
            logger.info(f"Successfully parsed analysis result: {len(tasks)} tasks, metadata: {metadata}")
            return analysis_result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {result}")
            
            # Fallback: Create a simple plan with empty metadata
            return _create_fallback_analysis(question, available_documents)
            
    except Exception as e:
        logger.error(f"Error in query analysis: {e}")
        return _create_fallback_analysis(question, available_documents)


def _create_fallback_analysis(question: str, available_documents: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create a fallback analysis result when LLM analysis fails
    
    Args:
        question: The user's question
        available_documents: Optional list of document filenames available in the session
        
    Returns:
        Dict[str, Any]: Simple analysis result with tasks and empty metadata
    """
    logger.info("Creating fallback execution plan")
    
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
    
    # Check for hybrid query (contains both document and web patterns)
    has_document_patterns = any(pattern in question_lower for pattern in document_patterns)
    has_web_patterns = any(pattern in question_lower for pattern in web_patterns)
    
    # Try to detect generic document references and map to available documents
    source_document = None
    if available_documents:
        generic_patterns = ["this resume", "the resume", "this document", "the document", "this paper", "the paper", "this file", "the file"]
        for pattern in generic_patterns:
            if pattern in question_lower:
                # Try to map to most logical document
                if "resume" in pattern and available_documents:
                    # Look for resume-like files
                    for doc in available_documents:
                        if any(keyword in doc.lower() for keyword in ["resume", "cv"]):
                            source_document = doc
                            break
                elif "paper" in pattern and available_documents:
                    # Look for paper-like files
                    for doc in available_documents:
                        if any(keyword in doc.lower() for keyword in ["paper", "research", "study", "article"]):
                            source_document = doc
                            break
                elif available_documents:
                    # Default to first document if no specific match
                    source_document = available_documents[0]
                break
    
    # Create fallback metadata with potential source document
    fallback_metadata = {
        "source_document": source_document,
        "mentioned_people": [],
        "mentioned_companies": [],
        "date_range": None
    }
    
    if has_document_patterns and has_web_patterns:
        # Hybrid query - try to split into components
        logger.info("Detected hybrid query in fallback - creating multi-tool plan")
        return {
            "tasks": [
                {"tool": "vectorstore_retrieval", "query": question, "source_document": source_document},
                {"tool": "web_search", "query": question, "source_document": None}
            ],
            "metadata": fallback_metadata
        }
    elif has_document_patterns:
        # Document-specific query
        return {
            "tasks": [{"tool": "vectorstore_retrieval", "query": question, "source_document": source_document}],
            "metadata": fallback_metadata
        }
    elif has_web_patterns:
        # Web search query
        return {
            "tasks": [{"tool": "web_search", "query": question, "source_document": None}],
            "metadata": fallback_metadata
        }
    
    # Default: try documents first (most queries can benefit from document context)
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
    
    # Add metadata info if present
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


# Test function for development
def test_query_analysis():
    """Test the query analysis with sample queries"""
    test_queries = [
        "What is the weather like today and give me a summary of the document?",
        "Who is the CEO of Apple and what does the document say about leadership?", 
        "Summarize the main findings in the research paper",
        "What are the latest AI developments?",
        "Explain the methodology described in this document"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = analyze_query(query)
        print(f"Tasks: {result['tasks']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Summary: {get_execution_summary(result)}")


if __name__ == "__main__":
    test_query_analysis()
