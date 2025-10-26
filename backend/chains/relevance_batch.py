"""
Batched Relevance Evaluation Chain for Performance Optimization

This module combines document relevance and question relevance evaluations
into a single API call, reducing the number of LLM requests while maintaining
the same evaluation quality as separate calls.
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


class BatchRelevanceEvaluation(BaseModel):
    """Combined document and question relevance evaluation"""
    
    # Document relevance (answer grounding)
    document_grounding: bool = Field(
        description="True if the answer is grounded in the provided documents, False if hallucinated"
    )
    document_grounding_score: float = Field(
        default=0.5,
        description="Confidence score for document grounding (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    document_grounding_explanation: str = Field(
        description="Explanation of why the answer is or isn't grounded in documents"
    )
    
    # Question relevance (answer quality)
    question_relevance: bool = Field(
        description="True if the answer directly addresses the question, False if off-topic"
    )
    question_relevance_score: float = Field(
        default=0.5,
        description="Confidence score for question relevance (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    question_relevance_explanation: str = Field(
        description="Explanation of how well the answer addresses the question"
    )
    
    # Overall assessment
    overall_quality: str = Field(
        description="Overall quality assessment of the answer"
    )


structured_output = llm.with_structured_output(BatchRelevanceEvaluation)

system = """You are an expert answer quality evaluator for a RAG system. Your role is to evaluate BOTH the document grounding AND question relevance of a generated answer in a single comprehensive assessment.

EVALUATION FRAMEWORK:

1. DOCUMENT GROUNDING EVALUATION:
   - Is the answer factually supported by the provided documents?
   - Are there any claims in the answer that cannot be found in the documents?
   - Does the answer stay within the bounds of the provided information?
   - Are there any hallucinations or fabricated details?

2. QUESTION RELEVANCE EVALUATION:
   - Does the answer directly address what the user asked?
   - Is the answer complete and comprehensive for the question?
   - Does the answer stay on-topic and avoid irrelevant information?
   - Would this answer satisfy the user's information need?

SCORING CRITERIA:

**Document Grounding (True/False):**
- True: Answer is fully supported by documents, no hallucinations
- False: Answer contains information not found in documents or contradicts them

**Question Relevance (True/False):**
- True: Answer directly and comprehensively addresses the question
- False: Answer is off-topic, incomplete, or doesn't address the core question

**Confidence Scores (0.0-1.0):**
- 0.9-1.0: Very confident in the assessment
- 0.7-0.8: Confident with minor uncertainty
- 0.5-0.6: Moderate confidence, some ambiguity
- 0.3-0.4: Low confidence, significant uncertainty
- 0.0-0.2: Very uncertain or contradictory evidence

EVALUATION INSTRUCTIONS:
- Evaluate both aspects independently but consider their interaction
- Provide clear explanations for your assessments
- Be strict about document grounding to prevent hallucinations
- Be fair about question relevance while maintaining quality standards
- Consider the overall utility of the answer for the user

Be thorough and precise in your dual evaluation."""

batch_relevance_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            """Please evaluate the following answer for BOTH document grounding and question relevance.

USER QUESTION:
{question}

PROVIDED DOCUMENTS:
{documents_text}

GENERATED ANSWER:
{solution}

EVALUATION REQUIRED:

1. DOCUMENT GROUNDING:
   - Is the answer grounded in the provided documents? (True/False)
   - Confidence score (0.0-1.0)
   - Explanation of grounding assessment

2. QUESTION RELEVANCE:
   - Does the answer address the user's question? (True/False)
   - Confidence score (0.0-1.0)
   - Explanation of relevance assessment

3. OVERALL QUALITY:
   - Summary assessment of answer quality

Provide a comprehensive evaluation covering both aspects.""",
        ),
    ]
)

batch_relevance_evaluator = batch_relevance_prompt | structured_output


def evaluate_relevance_batch(question: str, documents: List, solution: str) -> Dict[str, Any]:
    """
    Evaluate both document relevance and question relevance in a single API call
    
    Args:
        question: User's query
        documents: List of document objects
        solution: Generated answer to evaluate
    
    Returns:
        Dictionary with both document_relevance_score and question_relevance_score
        compatible with existing caching system
    """
    if not documents:
        # Handle case with no documents
        documents_text = "No documents provided."
    else:
        # Prepare documents text
        documents_text_parts = []
        for i, doc in enumerate(documents):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            # Truncate very long documents
            if len(content) > 1500:
                content = content[:1500] + "... [truncated]"
            documents_text_parts.append(f"Document {i+1}:\n{content}")
        documents_text = "\n\n".join(documents_text_parts)
    
    try:
        # Single API call for both evaluations
        batch_result = batch_relevance_evaluator.invoke({
            "question": question,
            "documents_text": documents_text,
            "solution": solution
        })
        
        # Create objects compatible with existing code structure
        class DocumentRelevanceScore:
            def __init__(self, binary_score, confidence_score, explanation):
                self.binary_score = binary_score
                self.confidence_score = confidence_score
                self.explanation = explanation
        
        class QuestionRelevanceScore:
            def __init__(self, binary_score, confidence_score, explanation):
                self.binary_score = binary_score
                self.confidence_score = confidence_score
                self.explanation = explanation
        
        document_relevance_score = DocumentRelevanceScore(
            binary_score=batch_result.document_grounding,
            confidence_score=batch_result.document_grounding_score,
            explanation=batch_result.document_grounding_explanation
        )
        
        question_relevance_score = QuestionRelevanceScore(
            binary_score=batch_result.question_relevance,
            confidence_score=batch_result.question_relevance_score,
            explanation=batch_result.question_relevance_explanation
        )
        
        return {
            "document_relevance_score": document_relevance_score,
            "question_relevance_score": question_relevance_score,
            "batch_evaluation": True,  # Flag to indicate this was a batch evaluation
            "overall_quality": batch_result.overall_quality
        }
        
    except Exception as e:
        print(f"Batch relevance evaluation failed: {e}")
        
        # Fallback to mock scores
        class FallbackScore:
            binary_score = False
            confidence_score = 0.0
            explanation = f"Batch evaluation failed: {str(e)}"
        
        return {
            "document_relevance_score": FallbackScore(),
            "question_relevance_score": FallbackScore(),
            "batch_evaluation": False,
            "overall_quality": "Evaluation failed"
        }
