"""
Document Evaluation Chain with Groq Integration

This module handles document relevance evaluation using Groq's fast inference
with qwen3-32b. Falls back to Gemini if Groq fails.

Key improvements:
- 3-5x faster evaluation with Groq
- Structured JSON output with Pydantic validation
- Automatic fallback to Gemini
- Comprehensive metrics tracking
"""

import os
import logging
import json
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Import Groq client
try:
    from backend.inference_clients.groq_client import GroqModelClient

    GROQ_AVAILABLE = True
    logger.info("✅ Groq SDK imported successfully for document evaluation")
except ImportError as e:
    logger.warning(f"❌ Groq client not available: {e}. Will use Gemini only")
    GROQ_AVAILABLE = False

# Fallback to Gemini
from langchain_google_genai import ChatGoogleGenerativeAI


class EvaluateDocs(BaseModel):
    """
    Document evaluation results for LangGraph RAG workflows
    """

    score: str = Field(
        description="Whether documents are relevant to the question - 'yes' if sufficient, 'no' if insufficient"
    )
    relevance_score: float = Field(
        default=0.5,
        description="Relevance score between 0.0 and 1.0 indicating how well documents match the query",
        ge=0.0,
        le=1.0,
    )
    coverage_assessment: str = Field(
        default="",
        description="Assessment of how well the documents cover the query requirements",
    )
    missing_information: Optional[str] = Field(
        default="None",
        description="Description of key information missing from documents (if any). Use 'None' if no information is missing.",
    )

    @field_validator("missing_information", mode="before")
    @classmethod
    def convert_none_to_string(cls, v):
        """Convert None values to 'None' string"""
        if v is None:
            return "None"
        return v


class DocumentEvaluation(BaseModel):
    """
    Single document evaluation result for batch processing
    """
    document_index: int = Field(
        description="Index of the document in the provided list (0-based)"
    )
    is_relevant: str = Field(
        description="Whether this document is relevant: 'yes', 'no', or 'partial'"
    )
    relevance_score: float = Field(
        description="Relevance score between 0.0 and 1.0",
        ge=0.0,
        le=1.0,
    )
    coverage_assessment: str = Field(
        description="Assessment of what this document covers"
    )
    missing_information: str = Field(
        default="None",
        description="What information is missing from this document"
    )


class BatchEvaluationResult(BaseModel):
    """
    Batch evaluation results for multiple documents
    """
    evaluations: list[DocumentEvaluation] = Field(
        description="List of individual document evaluations"
    )


EVALUATION_PROMPT = """You are an expert document relevance evaluator for a RAG (Retrieval-Augmented Generation) system. Your role is to assess whether retrieved documents contain sufficient information to answer a user's query effectively.

EVALUATION FRAMEWORK:

1. TOPICAL RELEVANCE:
   - Do the documents directly address the main subject of the query?
   - Are the key concepts and themes aligned with what the user is asking?

2. INFORMATION SUFFICIENCY:
   - Is there enough detail to provide a comprehensive answer?
   - Are specific facts, data, or examples present when needed?
   - Can the query be answered without requiring external knowledge?

3. INFORMATION QUALITY:
   - Is the information accurate and credible?
   - Are there conflicting statements within the documents?
   - Is the information current and relevant to the query context?

4. COMPLETENESS ASSESSMENT:
   - Does the document set cover all aspects of the query?
   - Are there obvious gaps in information that would prevent a complete answer?

SCORING CRITERIA (BALANCED APPROACH):
- **For DOCUMENT_FIRST queries**:
  - Prioritize document content. Score 'yes' if documents contain relevant information that can contribute to the answer, even if partial.
  - Only score 'no' if documents are completely irrelevant or explicitly contradict the query.
- **For HYBRID queries**:
  - Evaluate relevance carefully. Score 'yes' if documents provide substantial, directly relevant information.
  - Score 'no' if documents are only marginally relevant, incomplete, or require significant external context.
- **For ONLINE_SEARCH queries**:
  - Documents are less likely to be relevant. Score 'yes' only if they provide highly specific and directly applicable information.
  - Lean towards 'no' if the query clearly seeks external, real-time, or broad knowledge.

GENERAL GUIDELINES:
- **Score 'yes'** if documents contain sufficient, relevant information to answer the query.
- **Score 'no'** if documents are completely off-topic, contain no useful information, or the query explicitly requires external information not present in the documents.
- Consider the user's likely intent: if they uploaded documents, they probably want to use them, but not at the expense of accuracy for general queries.

ADDITIONAL REQUIREMENTS:
- Provide a relevance score (0.0-1.0) indicating match quality
- Assess coverage of query requirements
- Identify any missing critical information

Be thorough but efficient in your evaluation. Focus on practical utility for answer generation.

USER QUERY:
{question}

QUERY CLASSIFICATION:
{query_classification}

RETRIEVED DOCUMENTS:
{document}

EVALUATION REQUIRED:
Please provide your evaluation as a JSON object with the following structure:

{{
  "score": "yes" or "no",
  "relevance_score": 0.0 to 1.0,
  "coverage_assessment": "description of coverage",
  "missing_information": "what's missing if any, or 'None' if nothing is missing"
}}

IMPORTANT: Use the string "None" (not null) for missing_information if no information is missing.

Provide your comprehensive evaluation based on the framework above."""


BATCH_EVALUATION_PROMPT = """You are an expert document relevance evaluator for a RAG (Retrieval-Augmented Generation) system. Your role is to assess whether retrieved documents contain sufficient information to answer a user's query effectively.

**CRITICAL: BATCH EVALUATION MODE**
You will be given a list of multiple documents. You must evaluate EACH document INDIVIDUALLY and return a JSON array with one evaluation per document.

EVALUATION FRAMEWORK:

1. TOPICAL RELEVANCE:
   - Does this specific document directly address the main subject of the query?
   - Are the key concepts and themes aligned with what the user is asking?

2. INFORMATION SUFFICIENCY:
   - Does this document contain enough detail to contribute to the answer?
   - Are specific facts, data, or examples present when needed?

3. INFORMATION QUALITY:
   - Is the information in this document accurate and credible?
   - Does this document contain conflicting statements?

4. COMPLETENESS ASSESSMENT:
   - What aspects of the query does this document cover?
   - What information is missing from this document?

SCORING CRITERIA:
- **Score 'yes'** if the document contains substantial, directly relevant information
- **Score 'partial'** if the document has some relevant information but is incomplete
- **Score 'no'** if the document is off-topic or contains no useful information

USER QUERY:
{question}

DOCUMENT LIST:
{document_list}

EVALUATION REQUIRED:
You MUST provide your evaluation as a JSON object with an "evaluations" array. Each element in the array corresponds to one document in the order provided above.

{{
  "evaluations": [
    {{
      "document_index": 0,
      "is_relevant": "yes" or "no" or "partial",
      "relevance_score": 0.0 to 1.0,
      "coverage_assessment": "what this document covers",
      "missing_information": "what's missing from this document, or 'None'"
    }},
    {{
      "document_index": 1,
      "is_relevant": "yes" or "no" or "partial",
      "relevance_score": 0.0 to 1.0,
      "coverage_assessment": "what this document covers",
      "missing_information": "what's missing from this document, or 'None'"
    }}
  ]
}}

**CRITICAL REQUIREMENTS:**
1. You MUST return exactly {num_documents} evaluations (one per document)
2. The document_index must match the document number (0-based)
3. Use the string "None" (not null) for missing_information if nothing is missing
4. Evaluate each document independently - don't compare documents to each other

Provide your comprehensive batch evaluation now."""


class DocumentEvaluationClient:
    """
    Document evaluation client with Groq primary and Gemini fallback.
    """

    def __init__(self):
        """Initialize evaluation client"""
        self.groq_client = None
        self.gemini_client = None
        self.metrics = []

        # Initialize Groq
        if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
            try:
                self.groq_client = GroqModelClient(
                    model_name="qwen/qwen3-32b", enable_fallback=False
                )
                logger.info(
                    "✅ Document Evaluation: Groq client initialized (qwen/qwen3-32b)"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")

        # Initialize Gemini fallback
        try:
            self.gemini_client = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0,
                google_api_key=os.environ["GOOGLE_API_KEY"],
            )
            logger.info("✅ Document Evaluation: Gemini fallback initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    def evaluate(
        self, question: str, document: str, query_classification: str = "DOCUMENT_FIRST"
    ) -> tuple[EvaluateDocs, Dict[str, Any]]:
        """
        Evaluate document relevance using Groq (primary) or Gemini (fallback).

        Args:
            question: User's query
            document: Retrieved document content
            query_classification: Query type classification

        Returns:
            Tuple of (evaluation_result, metrics_dict)
        """
        start_time = time.time()

        # Create prompt
        prompt = EVALUATION_PROMPT.format(
            question=question,
            query_classification=query_classification,
            document=document,
        )

        logger.info(f"Evaluating documents for query: '{question[:100]}...'")

        # Try Groq first
        if self.groq_client:
            try:
                logger.info("🚀 Using Groq for document evaluation...")
                response_text, groq_metrics = self.groq_client.infer(
                    prompt=prompt, temperature=0.0, max_tokens=512, json_mode=True
                )

                # Parse JSON response
                response_dict = json.loads(response_text)

                # Handle None values in missing_information
                if response_dict.get("missing_information") is None:
                    response_dict["missing_information"] = "None"

                evaluation = EvaluateDocs(**response_dict)

                metrics = {
                    "model": "groq-qwen3-32b",
                    "latency_ms": groq_metrics.latency_ms,
                    "tokens": groq_metrics.total_tokens,
                    "success": True,
                    "fallback_used": False,
                }

                logger.info(
                    f"✅ Groq evaluation complete: {evaluation.score} "
                    f"(relevance: {evaluation.relevance_score:.2f}, "
                    f"{groq_metrics.latency_ms:.0f}ms)"
                )

                self.metrics.append(metrics)
                return evaluation, metrics

            except Exception as e:
                logger.warning(f"Groq evaluation failed: {e}, falling back to Gemini")

        # Fallback to Gemini
        try:
            logger.info("🔄 Using Gemini fallback for document evaluation...")
            gemini_start = time.time()

            # Use Gemini's structured output
            structured_llm = self.gemini_client.with_structured_output(EvaluateDocs)
            evaluation = structured_llm.invoke(prompt)

            gemini_latency = (time.time() - gemini_start) * 1000

            metrics = {
                "model": "gemini-2.5-flash",
                "latency_ms": gemini_latency,
                "tokens": 0,
                "success": True,
                "fallback_used": True,
            }

            logger.info(
                f"✅ Gemini evaluation complete: {evaluation.score} "
                f"(relevance: {evaluation.relevance_score:.2f}, "
                f"{gemini_latency:.0f}ms)"
            )

            self.metrics.append(metrics)
            return evaluation, metrics

        except Exception as e:
            logger.error(f"Both Groq and Gemini failed: {e}")
            total_latency = (time.time() - start_time) * 1000

            # Return default evaluation
            evaluation = EvaluateDocs(
                score="no",
                relevance_score=0.0,
                coverage_assessment="Evaluation failed",
                missing_information="Unable to evaluate due to error",
            )

            metrics = {
                "model": "none",
                "latency_ms": total_latency,
                "tokens": 0,
                "success": False,
                "fallback_used": True,
                "error": str(e),
            }

            self.metrics.append(metrics)
            return evaluation, metrics

    def evaluate_batch(
        self, question: str, documents: list, query_classification: str = "DOCUMENT_FIRST"
    ) -> tuple[BatchEvaluationResult, Dict[str, Any]]:
        """
        Evaluate multiple documents in a single batch API call.
        
        This eliminates the N+1 query problem by processing all documents at once.
        
        Args:
            question: User's query
            documents: List of document objects or strings
            query_classification: Query type classification
            
        Returns:
            Tuple of (batch_evaluation_result, metrics_dict)
        """
        start_time = time.time()
        
        # Format documents for batch evaluation
        document_list = []
        for i, doc in enumerate(documents):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            # Truncate very long documents to prevent token overflow
            if len(content) > 1500:
                content = content[:1500] + "... [truncated]"
            document_list.append(f"--- Document [{i}] ---\n{content}")
        
        document_list_text = "\n\n".join(document_list)
        
        # Create batch prompt
        prompt = BATCH_EVALUATION_PROMPT.format(
            question=question,
            document_list=document_list_text,
            num_documents=len(documents)
        )
        
        logger.info(f"🚀 BATCH EVALUATION: Processing {len(documents)} documents in single API call")
        
        # Try Groq first
        if self.groq_client:
            try:
                logger.info("🚀 Using Groq for batch document evaluation...")
                response_text, groq_metrics = self.groq_client.infer(
                    prompt=prompt, 
                    temperature=0.0, 
                    max_tokens=2048,  # Increased for batch results
                    json_mode=True
                )
                
                # Parse JSON response
                response_dict = json.loads(response_text)
                
                # Validate we got the right number of evaluations
                if len(response_dict.get("evaluations", [])) != len(documents):
                    logger.warning(
                        f"Expected {len(documents)} evaluations, got {len(response_dict.get('evaluations', []))}. "
                        "Falling back to individual evaluation."
                    )
                    raise ValueError("Incorrect number of evaluations returned")
                
                # Handle None values in missing_information
                for eval_dict in response_dict["evaluations"]:
                    if eval_dict.get("missing_information") is None:
                        eval_dict["missing_information"] = "None"
                
                batch_result = BatchEvaluationResult(**response_dict)
                
                metrics = {
                    "model": "groq-qwen3-32b",
                    "latency_ms": groq_metrics.latency_ms,
                    "tokens": groq_metrics.total_tokens,
                    "success": True,
                    "fallback_used": False,
                    "documents_evaluated": len(documents),
                    "batch_mode": True
                }
                
                logger.info(
                    f"✅ Groq batch evaluation complete: {len(batch_result.evaluations)} documents "
                    f"({groq_metrics.latency_ms:.0f}ms total, "
                    f"{groq_metrics.latency_ms/len(documents):.0f}ms per doc)"
                )
                
                self.metrics.append(metrics)
                return batch_result, metrics
                
            except Exception as e:
                logger.warning(f"Groq batch evaluation failed: {e}, falling back to Gemini")
        
        # Fallback to Gemini
        try:
            logger.info("🔄 Using Gemini fallback for batch document evaluation...")
            gemini_start = time.time()
            
            # Use Gemini's structured output
            structured_llm = self.gemini_client.with_structured_output(BatchEvaluationResult)
            batch_result = structured_llm.invoke(prompt)
            
            gemini_latency = (time.time() - gemini_start) * 1000
            
            metrics = {
                "model": "gemini-2.0-flash",
                "latency_ms": gemini_latency,
                "tokens": 0,
                "success": True,
                "fallback_used": True,
                "documents_evaluated": len(documents),
                "batch_mode": True
            }
            
            logger.info(
                f"✅ Gemini batch evaluation complete: {len(batch_result.evaluations)} documents "
                f"({gemini_latency:.0f}ms total, "
                f"{gemini_latency/len(documents):.0f}ms per doc)"
            )
            
            self.metrics.append(metrics)
            return batch_result, metrics
            
        except Exception as e:
            logger.error(f"Both Groq and Gemini batch evaluation failed: {e}")
            total_latency = (time.time() - start_time) * 1000
            
            # Return default evaluations for all documents
            default_evaluations = [
                DocumentEvaluation(
                    document_index=i,
                    is_relevant="yes",
                    relevance_score=0.5,
                    coverage_assessment="Evaluation failed",
                    missing_information="Unable to evaluate due to error"
                )
                for i in range(len(documents))
            ]
            
            batch_result = BatchEvaluationResult(evaluations=default_evaluations)
            
            metrics = {
                "model": "none",
                "latency_ms": total_latency,
                "tokens": 0,
                "success": False,
                "fallback_used": True,
                "documents_evaluated": len(documents),
                "batch_mode": True,
                "error": str(e)
            }
            
            self.metrics.append(metrics)
            return batch_result, metrics

    def get_metrics(self) -> list:
        """Get all collected metrics"""
        return self.metrics


# Global client instance
_evaluation_client = None


def get_evaluation_client() -> DocumentEvaluationClient:
    """Get or create global evaluation client"""
    global _evaluation_client
    if _evaluation_client is None:
        _evaluation_client = DocumentEvaluationClient()
    return _evaluation_client


def evaluate_documents(
    question: str, document: str, query_classification: str = "DOCUMENT_FIRST"
) -> EvaluateDocs:
    """
    Evaluate document relevance for a query.

    Uses Groq qwen3-32b for fast evaluation with Gemini fallback.

    Args:
        question: User's query
        document: Retrieved document content
        query_classification: Query type classification

    Returns:
        EvaluateDocs object with evaluation results
    """
    client = get_evaluation_client()
    evaluation, metrics = client.evaluate(question, document, query_classification)
    return evaluation


def evaluate_documents_batch(
    question: str, documents: list, query_classification: str = "DOCUMENT_FIRST"
) -> BatchEvaluationResult:
    """
    Evaluate multiple documents in a single batch API call.
    
    This is the recommended method for evaluating multiple documents as it
    eliminates the N+1 query problem and significantly reduces latency.
    
    Uses Groq qwen3-32b for fast evaluation with Gemini fallback.
    
    Args:
        question: User's query
        documents: List of document objects or strings
        query_classification: Query type classification
        
    Returns:
        BatchEvaluationResult with list of individual document evaluations
    """
    client = get_evaluation_client()
    batch_result, metrics = client.evaluate_batch(question, documents, query_classification)
    return batch_result
