"""
Query Classification Chain for Document-First Evaluation

This module classifies queries to determine if they are document-specific
and should strongly prefer document content over online search.
"""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1
)

# Query classification prompt
classification_prompt_text = """You are a query classifier for a RAG system. Your job is to determine if a query is document-specific and should prioritize uploaded document content over online search.

DOCUMENT-SPECIFIC QUERY INDICATORS:
1. **Direct Document References**: "this document", "the document", "the file", "the text", "the paper"
2. **Summary Requests**: "summary", "summarize", "overview", "main points", "key findings"
3. **Content Extraction**: "who is mentioned", "what does it say about", "extract information", "find details"
4. **Document Analysis**: "analyze this", "explain the content", "what are the conclusions"
5. **Specific Content Queries**: "according to the document", "based on the text", "from the material"
6. **Ambiguous but Contextual**: Questions that could refer to document content when documents are available

ONLINE-SEARCH QUERIES:
1. **Current Events**: "latest news", "recent developments", "current status"
2. **External Facts**: "what is the capital of", "when was X invented", "who is the CEO of"
3. **Real-time Data**: "stock price", "weather", "current date"
4. **Broad Knowledge**: "explain quantum physics", "history of", "how does X work" (without document context)

CLASSIFICATION RULES:
- If query contains strong document-specific indicators → DOCUMENT_FIRST
- If query explicitly asks for external/current information → ONLINE_SEARCH
- If query could benefit from both, or is ambiguous and documents are available → HYBRID
- When in doubt, lean towards HYBRID if documents exist, otherwise ONLINE_SEARCH

Query: "{question}"

Classify this query as either:
- DOCUMENT_FIRST: Should strongly prefer document content
- ONLINE_SEARCH: Should prefer online search
- HYBRID: Could use both sources

Respond with only the classification (DOCUMENT_FIRST, ONLINE_SEARCH, or HYBRID)."""

classify_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", classification_prompt_text),
    ]
)

# Create the classification chain
query_classifier = classify_prompt | llm | StrOutputParser()


def classify_query(question: str) -> str:
    """
    Classify a query to determine search strategy preference

    Args:
        question: The user's question

    Returns:
        str: Classification result (DOCUMENT_FIRST, ONLINE_SEARCH, or HYBRID)
    """
    try:
        result = query_classifier.invoke({"question": question})
        classification = result.strip().upper()

        # Ensure valid classification
        if classification not in ["DOCUMENT_FIRST", "ONLINE_SEARCH", "HYBRID"]:
            # Default to HYBRID for a more balanced approach on error
            return "HYBRID"

        return classification
    except Exception as e:
        print(f"Error in query classification: {e}")
        # Default to HYBRID on error for a more balanced approach
        return "HYBRID"


def is_document_specific_query(question: str) -> bool:
    """
    Simple boolean check if query should prefer documents

    Args:
        question: The user's question

    Returns:
        bool: True if query should prefer document content
    """
    # Fast pattern-based check for common document queries
    question_lower = question.lower()

    document_patterns = [
        "summary",
        "summarize",
        "overview",
        "main points",
        "key findings",
        "this document",
        "the document",
        "the file",
        "the text",
        "the paper",
        "who is mentioned",
        "what does it say",
        "according to",
        "based on",
        "analyze this",
        "explain the content",
        "what are the conclusions",
        "extract information",
        "find details",
        "from the material",
    ]

    # If any pattern matches, it's definitely document-first
    if any(pattern in question_lower for pattern in document_patterns):
        return True

    # Otherwise use LLM classification
    classification = classify_query(question)
    # Only consider it document-specific if explicitly classified as DOCUMENT_FIRST
    return classification == "DOCUMENT_FIRST"
