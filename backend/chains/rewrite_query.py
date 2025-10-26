"""
Query Rewriting Chain for Self-Correcting RAG Pipeline

This module implements query rewriting functionality that generates improved
search queries when the initial retrieval fails to find sufficient context.
It's part of the self-correcting loop that makes the RAG pipeline resilient
to "needle-in-a-haystack" queries.

The rewriter analyzes the original query and failed context to generate
more specific, targeted search queries.
"""

import os
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the LLM for query rewriting
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=os.environ["GOOGLE_API_KEY"], 
        temperature=0.3  # Slightly higher temperature for creative rewriting
    )
except Exception as e:
    logger.warning(f"Failed to initialize gemini-2.5-flash: {e}")
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=os.environ["GOOGLE_API_KEY"], 
        temperature=0.3
    )

# Query Rewriting Prompt
query_rewrite_prompt_text = """You are an expert at refining search queries. The user's original query failed to retrieve relevant documents. Your task is to rewrite the query to be more specific and targeted, based on the original intent and the failed context.

REWRITING STRATEGIES:
1. **Add Specificity**: If the original query was vague, add specific terms, keywords, or context
2. **Use Synonyms**: Try alternative terms that might match document content better
3. **Break Down Complex Queries**: Split multi-part questions into focused components
4. **Add Context Clues**: Include related terms that might appear alongside the target information
5. **Use Exact Phrases**: For specific items (like "Table VII"), try variations and related terms
6. **Consider Document Structure**: Think about how information might be organized or labeled

EXAMPLES OF GOOD REWRITES:
- "What is the performance?" → "performance metrics results accuracy evaluation"
- "Tell me about the method" → "methodology approach algorithm technique implementation"
- "Table VII" → "Table VII results data findings seventh table performance metrics"

IMPORTANT GUIDELINES:
- Do not answer the question - only generate a new search query
- Make the query more specific and targeted than the original
- Consider what terms are likely to appear in relevant documents
- If the failed context gives clues about document structure, use them
- Keep the core intent of the original question

ORIGINAL QUERY:
{question}

FAILED CONTEXT:
{documents}

REFINED QUERY:"""

# Create the rewriting prompt
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", query_rewrite_prompt_text),
    ("user", "Please generate a refined search query.")
])

# Create the rewriting chain
query_rewrite_chain = rewrite_prompt | llm | StrOutputParser()


def rewrite_query(original_question: str, failed_documents: list, attempt_number: int = 1) -> str:
    """
    Rewrite a query to improve retrieval results based on failed context.
    
    Args:
        original_question (str): The user's original question
        failed_documents (list): List of documents that were insufficient
        attempt_number (int): Which rewrite attempt this is (for logging)
        
    Returns:
        str: A rewritten, more targeted query
    """
    try:
        # Format failed documents for analysis
        if not failed_documents:
            doc_text = "No documents were retrieved."
        else:
            doc_texts = []
            for i, doc in enumerate(failed_documents, 1):
                if hasattr(doc, 'page_content'):
                    # Truncate long documents for context
                    content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    doc_texts.append(f"Document {i}:\n{content}")
                else:
                    content = str(doc)[:500] + "..." if len(str(doc)) > 500 else str(doc)
                    doc_texts.append(f"Document {i}:\n{content}")
            
            doc_text = "\n\n".join(doc_texts)
        
        logger.info(f"Rewriting query (attempt #{attempt_number}): '{original_question[:100]}...'")
        logger.info(f"Analyzing {len(failed_documents)} failed documents for rewrite clues")
        
        # Invoke the rewriting chain
        chain_input = {
            "question": original_question,
            "documents": doc_text
        }
        
        result = query_rewrite_chain.invoke(chain_input)
        
        # Clean the result
        rewritten_query = result.strip()
        
        # Ensure we have a meaningful rewrite
        if not rewritten_query or len(rewritten_query) < 3:
            # Fallback: add generic improvement terms
            rewritten_query = f"{original_question} details information data results"
            logger.warning(f"Generated fallback rewrite: '{rewritten_query}'")
        else:
            logger.info(f"Generated rewritten query: '{rewritten_query}'")
        
        return rewritten_query
        
    except Exception as e:
        logger.error(f"Error during query rewriting: {e}")
        # Fallback: return original with additional terms
        fallback_query = f"{original_question} information details data"
        logger.warning(f"Using fallback rewrite due to error: '{fallback_query}'")
        return fallback_query


def generate_progressive_rewrite(original_question: str, failed_documents: list, attempt_number: int) -> str:
    """
    Generate progressively more aggressive rewrites based on attempt number.
    
    Args:
        original_question (str): The user's original question
        failed_documents (list): List of documents that were insufficient
        attempt_number (int): Which attempt this is (1, 2, etc.)
        
    Returns:
        str: A rewritten query with strategy based on attempt number
    """
    if attempt_number == 1:
        # First rewrite: Add specificity and synonyms
        return rewrite_query(original_question, failed_documents, attempt_number)
    elif attempt_number == 2:
        # Second rewrite: More aggressive, break down the query
        try:
            # Add more aggressive rewriting for second attempt
            aggressive_prompt = f"""Break down this query into key search terms and add related concepts:
            
            Original: {original_question}
            
            Generate a comprehensive search query with multiple related terms:"""
            
            result = query_rewrite_chain.invoke({
                "question": aggressive_prompt,
                "documents": "Second attempt - need more comprehensive search terms"
            })
            
            return result.strip() if result.strip() else f"{original_question} comprehensive detailed analysis"
            
        except Exception as e:
            logger.error(f"Error in progressive rewrite: {e}")
            return f"{original_question} comprehensive analysis detailed information"
    else:
        # Fallback for higher attempts
        return f"{original_question} complete comprehensive detailed analysis information"
