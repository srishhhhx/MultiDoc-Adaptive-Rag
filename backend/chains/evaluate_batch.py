"""
Batched Document Evaluation Chain for Performance Optimization

This module handles batch evaluation of multiple documents in a single API call,
reducing the number of LLM requests and improving performance while maintaining
the same evaluation quality as individual document evaluation.
"""

import os
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"],
)


class DocumentEvaluation(BaseModel):
    """Single document evaluation result"""
    document_index: int = Field(description="Index of the document (0-based)")
    score: str = Field(description="'yes' if relevant, 'no' if not relevant")
    relevance_score: float = Field(
        default=0.5,
        description="Relevance score between 0.0 and 1.0",
        ge=0.0,
        le=1.0,
    )
    coverage_assessment: str = Field(
        default="",
        description="Assessment of how well this document covers the query",
    )
    missing_information: str = Field(
        default="",
        description="Key information missing from this document",
    )


class BatchDocumentEvaluation(BaseModel):
    """Batch evaluation results for multiple documents"""
    evaluations: List[DocumentEvaluation] = Field(
        description="List of individual document evaluations"
    )
    overall_assessment: str = Field(
        description="Overall assessment of the document set quality"
    )


structured_output = llm.with_structured_output(BatchDocumentEvaluation)

system = """You are an expert document relevance evaluator for a RAG system. Your role is to evaluate MULTIPLE documents at once to determine which ones contain sufficient information to answer a user's query.

EVALUATION FRAMEWORK (same as individual evaluation):

1. TOPICAL RELEVANCE:
   - Do the documents directly address the main subject of the query?
   - Are the key concepts and themes aligned with what the user is asking?

2. INFORMATION SUFFICIENCY:
   - Is there enough detail to provide a comprehensive answer?
   - Are specific facts, data, or examples present when needed?

3. INFORMATION QUALITY:
   - Is the information accurate and credible?
   - Are there conflicting statements within the documents?

4. COMPLETENESS ASSESSMENT:
   - Does each document contribute meaningfully to answering the query?
   - Are there obvious gaps that prevent complete answers?

BATCH EVALUATION INSTRUCTIONS:
- Evaluate each document individually using the same criteria as single document evaluation
- Provide a score ('yes'/'no') for each document based on its individual merit
- Consider how documents complement each other but score each one independently
- Maintain consistency with single document evaluation standards

SCORING CRITERIA (BALANCED APPROACH):
- **For DOCUMENT_FIRST queries**: Prioritize document content, score 'yes' if relevant
- **For HYBRID queries**: Evaluate carefully, score 'yes' if substantially relevant
- **For ONLINE_SEARCH queries**: Score 'yes' only if highly specific and applicable

Be thorough but efficient. Focus on practical utility for answer generation."""

batch_evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            """Please evaluate ALL the following documents for their relevance to the user's query. Provide individual evaluations for each document.

USER QUERY:
{question}

QUERY CLASSIFICATION:
{query_classification}

DOCUMENTS TO EVALUATE:
{documents_text}

EVALUATION REQUIRED FOR EACH DOCUMENT:
1. Document Index: The position of the document (0, 1, 2, etc.)
2. Primary Score: 'yes' if document is sufficient, 'no' if insufficient
3. Relevance Score: 0.0-1.0 rating of how well the document matches the query
4. Coverage Assessment: How well does this document address the query requirements?
5. Missing Information: What key information is missing from this document?

Also provide an overall assessment of the document set quality.

Evaluate each document independently but consider the overall context.""",
        ),
    ]
)

batch_evaluate_docs = batch_evaluate_prompt | structured_output


def evaluate_documents_batch(question: str, documents: List, query_classification: str) -> List[Dict[str, Any]]:
    """
    Evaluate multiple documents in a single batch API call
    
    Args:
        question: User's query
        documents: List of document objects with page_content
        query_classification: Query classification (DOCUMENT_FIRST, HYBRID, ONLINE_SEARCH)
    
    Returns:
        List of evaluation dictionaries compatible with existing frontend format
    """
    if not documents:
        return []
    
    # Prepare documents text for batch evaluation
    documents_text_parts = []
    for i, doc in enumerate(documents):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        # Truncate very long documents to prevent token limits
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"
        documents_text_parts.append(f"Document {i}:\n{content}")
    
    documents_text = "\n\n" + "\n\n".join(documents_text_parts) + "\n\n"
    
    try:
        # Single API call for all documents
        batch_result = batch_evaluate_docs.invoke({
            "question": question,
            "query_classification": query_classification,
            "documents_text": documents_text
        })
        
        # Convert to format compatible with existing code
        evaluations = []
        for eval_result in batch_result.evaluations:
            # Create object that matches the structure expected by existing code
            class EvalResponse:
                def __init__(self, score, relevance_score, coverage_assessment, missing_information):
                    self.score = score
                    self.relevance_score = relevance_score
                    self.coverage_assessment = coverage_assessment
                    self.missing_information = missing_information
            
            evaluations.append(EvalResponse(
                score=eval_result.score,
                relevance_score=eval_result.relevance_score,
                coverage_assessment=eval_result.coverage_assessment,
                missing_information=eval_result.missing_information
            ))
        
        # Ensure we have evaluations for all documents (fallback for missing ones)
        while len(evaluations) < len(documents):
            evaluations.append(EvalResponse(
                score="no",
                relevance_score=0.0,
                coverage_assessment="Evaluation failed",
                missing_information="Could not evaluate this document"
            ))
        
        return evaluations[:len(documents)]  # Ensure exact match with input documents
        
    except Exception as e:
        print(f"Batch evaluation failed: {e}")
        # Fallback to individual evaluation format
        fallback_evaluations = []
        for _ in documents:
            class FallbackEval:
                score = "no"
                relevance_score = 0.0
                coverage_assessment = "Batch evaluation failed"
                missing_information = "Fallback evaluation"
            fallback_evaluations.append(FallbackEval())
        return fallback_evaluations
