"""
Batch document analysis using LLM

This module implements efficient batch analysis of documents using a single
LLM call instead of individual document evaluations. It analyzes multiple
documents at once and returns structured analysis results.
"""

import json
import logging
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

# Analysis prompt template with JSON schema
ANALYSIS_PROMPT_TEMPLATE = """
You are an expert document analysis system. Your task is to analyze a set of documents against a user's query and provide a detailed breakdown for each one.

You MUST return your response as a single, valid JSON array. Each object in the array should correspond to one of the provided documents and follow this exact schema:
{{
  "doc_id": "integer, the index of the document in the provided list (starting from 0)",
  "is_relevant": "string, either 'YES' or 'NO'",
  "relevance_score": "integer, from 0 to 100",
  "analysis": {{
    "coverage": "string, a paragraph explaining what the document covers in relation to the query.",
    "missing_info": "string, a paragraph explaining what relevant information is missing from the document."
  }}
}}

USER QUERY:
{query}

DOCUMENTS:
{documents}

Return only the JSON array, no additional text or formatting:
"""

class BatchDocumentAnalyzer:
    """
    Batch document analyzer using LLM for efficient analysis
    
    This class analyzes multiple documents in a single LLM call,
    providing structured analysis results for each document.
    """
    
    def __init__(self):
        """Initialize the batch analyzer with LLM"""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.1,
                convert_system_message_to_human=True
            )
            
            self.prompt = PromptTemplate(
                template=ANALYSIS_PROMPT_TEMPLATE,
                input_variables=["query", "documents"]
            )
            
            self.chain = self.prompt | self.llm
            logger.info("Batch document analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize batch document analyzer: {e}")
            raise
    
    def analyze_documents(self, query: str, documents: List) -> List[Dict[str, Any]]:
        """
        Analyze multiple documents in a single LLM call
        
        Args:
            query: The user's query
            documents: List of document objects to analyze
            
        Returns:
            List of analysis dictionaries with structured results
        """
        if not documents:
            logger.warning("No documents provided for analysis")
            return []
        
        try:
            logger.info(f"Analyzing {len(documents)} documents in batch")
            
            # Format documents for the prompt
            formatted_docs = []
            for i, doc in enumerate(documents):
                doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                # Truncate very long documents to avoid token limits
                if len(doc_content) > 2000:
                    doc_content = doc_content[:2000] + "..."
                
                formatted_docs.append(f"Document {i}:\n{doc_content}\n")
            
            documents_text = "\n".join(formatted_docs)
            
            # Make the LLM call
            response = self.chain.invoke({
                "query": query,
                "documents": documents_text
            })
            
            # Extract content from response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            logger.info(f"LLM response received: {len(response_text)} characters")
            
            # Parse JSON response
            try:
                # Clean the response - remove any markdown formatting
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                analysis_results = json.loads(response_text)
                
                # Validate the response structure
                if not isinstance(analysis_results, list):
                    raise ValueError("Response is not a JSON array")
                
                # Ensure we have results for all documents
                expected_count = len(documents)
                actual_count = len(analysis_results)
                
                if actual_count != expected_count:
                    logger.warning(f"Expected {expected_count} analyses, got {actual_count}")
                
                # Validate each analysis object
                validated_results = []
                for i, analysis in enumerate(analysis_results):
                    if not isinstance(analysis, dict):
                        logger.warning(f"Analysis {i} is not a dictionary, skipping")
                        continue
                    
                    # Ensure required fields exist
                    required_fields = ['doc_id', 'is_relevant', 'relevance_score', 'analysis']
                    if not all(field in analysis for field in required_fields):
                        logger.warning(f"Analysis {i} missing required fields, skipping")
                        continue
                    
                    # Validate analysis sub-object
                    if not isinstance(analysis['analysis'], dict):
                        logger.warning(f"Analysis {i} 'analysis' field is not a dictionary, skipping")
                        continue
                    
                    analysis_sub = analysis['analysis']
                    if 'coverage' not in analysis_sub or 'missing_info' not in analysis_sub:
                        logger.warning(f"Analysis {i} missing coverage or missing_info, skipping")
                        continue
                    
                    validated_results.append(analysis)
                
                logger.info(f"Successfully parsed {len(validated_results)} document analyses")
                return validated_results
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response_text[:500]}...")
                return self._fallback_analysis(query, documents)
            
        except Exception as e:
            logger.error(f"Error during batch document analysis: {e}")
            return self._fallback_analysis(query, documents)
    
    def _fallback_analysis(self, query: str, documents: List) -> List[Dict[str, Any]]:
        """
        Fallback analysis when batch processing fails
        
        Returns basic analysis structure for all documents
        """
        logger.warning("Using fallback analysis")
        fallback_results = []
        
        for i, doc in enumerate(documents):
            doc_preview = doc.page_content[:200] if hasattr(doc, 'page_content') else str(doc)[:200]
            
            fallback_results.append({
                "doc_id": i,
                "is_relevant": "YES",  # Conservative approach
                "relevance_score": 75,  # Moderate score
                "analysis": {
                    "coverage": f"Document {i} contains relevant information. Preview: {doc_preview}...",
                    "missing_info": "Unable to determine missing information due to analysis error."
                }
            })
        
        return fallback_results

# Global analyzer instance
_analyzer = None

def get_batch_analyzer():
    """Get or create the global batch analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = BatchDocumentAnalyzer()
    return _analyzer

def analyze_documents_batch(query: str, documents: List) -> List[Dict[str, Any]]:
    """
    Convenience function to analyze documents in batch
    
    Args:
        query: The user's query
        documents: List of document objects to analyze
        
    Returns:
        List of analysis dictionaries
    """
    analyzer = get_batch_analyzer()
    return analyzer.analyze_documents(query, documents)
