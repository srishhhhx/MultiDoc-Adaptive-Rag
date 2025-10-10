"""
Context Assessment Chain for Query Rewriting Loop

This module implements a context assessment mechanism that evaluates whether
retrieved documents are sufficient to answer the user's question. It's part
of the self-correcting query rewriting loop that makes the RAG pipeline
resilient to failed document retrievals.

The assessment uses an LLM to make a binary decision: "sufficient" or "insufficient"
based on the quality and relevance of the retrieved context.
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

# Initialize the LLM for context assessment
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=os.environ["GOOGLE_API_KEY"], 
        temperature=0.1
    )
except Exception as e:
    logger.warning(f"Failed to initialize gemini-2.5-flash: {e}")
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=os.environ["GOOGLE_API_KEY"], 
        temperature=0.1
    )

# Context Assessment Prompt
context_assessment_prompt_text = """You are an expert document analyst. Your task is to determine if the provided context documents are sufficient to answer the given user question.

ASSESSMENT CRITERIA:
- Do the documents contain the specific information needed to answer the question?
- Is there enough detail and context to provide a comprehensive answer?
- Are key facts, data, or concepts present that directly relate to the question?
- For specific queries (like "Table VII" or named entities), is that exact information present?

IMPORTANT: Be strict in your assessment. If the documents are vague, incomplete, or don't directly address the question, respond with "insufficient".

Respond with only one of two words: "sufficient" or "insufficient".

USER QUESTION:
{original_question}

CONTEXT DOCUMENTS:
{documents}

ASSESSMENT:"""

# Create the assessment prompt
assessment_prompt = ChatPromptTemplate.from_messages([
    ("system", context_assessment_prompt_text),
    ("user", "Please assess the context sufficiency.")
])

# Create the assessment chain
context_assessment_chain = assessment_prompt | llm | StrOutputParser()


def assess_context_sufficiency(original_question: str, documents: list) -> str:
    """
    Assess whether the retrieved documents are sufficient to answer the original question.
    
    Args:
        original_question (str): The user's original question (not rewritten)
        documents (list): List of retrieved documents
        
    Returns:
        str: Either "sufficient" or "insufficient"
    """
    try:
        # Format documents for assessment
        if not documents:
            logger.info("No documents provided - returning insufficient")
            return "insufficient"
        
        # Convert documents to text format
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            if hasattr(doc, 'page_content'):
                doc_texts.append(f"Document {i}:\n{doc.page_content}")
            else:
                doc_texts.append(f"Document {i}:\n{str(doc)}")
        
        documents_text = "\n\n".join(doc_texts)
        
        logger.info(f"Assessing context sufficiency for original question: '{original_question[:100]}...'")
        logger.info(f"Number of documents to assess: {len(documents)}")
        
        # Invoke the assessment chain
        chain_input = {
            "original_question": original_question,
            "documents": documents_text
        }
        
        result = context_assessment_chain.invoke(chain_input)
        
        # Enhanced debugging
        logger.info(f"Raw LLM assessment result: '{result}' (type: {type(result)})")
        
        # Clean and validate the result
        assessment = result.strip().lower()
        logger.info(f"Cleaned assessment: '{assessment}'")
        
        # More precise parsing - check for exact matches first
        if assessment == "sufficient":
            final_assessment = "sufficient"
            logger.info("✅ Exact match: 'sufficient'")
        elif assessment == "insufficient":
            final_assessment = "insufficient"
            logger.info("❌ Exact match: 'insufficient'")
        elif "sufficient" in assessment and "insufficient" not in assessment:
            final_assessment = "sufficient"
            logger.info("✅ Contains 'sufficient' (no 'insufficient')")
        elif "insufficient" in assessment:
            final_assessment = "insufficient"
            logger.info("❌ Contains 'insufficient'")
        else:
            # Default to insufficient if unclear
            logger.warning(f"Unclear assessment result: '{result}' - defaulting to insufficient")
            final_assessment = "insufficient"
        
        logger.info(f"Final context assessment result: {final_assessment}")
        return final_assessment
        
    except Exception as e:
        logger.error(f"Error during context assessment: {e}")
        # Default to insufficient on error to trigger rewrite
        return "insufficient"
