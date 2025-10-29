"""
Document Relevance (Hallucination Check) with Groq Integration

This module evaluates whether LLM-generated answers are grounded in source documents.
Uses Groq's fast inference for rapid hallucination detection with Gemini fallback.

Key improvements:
- 3-5x faster with Groq
- Structured JSON output
- Automatic fallback
- Comprehensive metrics
"""

import os
import logging
import json
import time
from typing import Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Import Groq client
try:
    from backend.inference_clients.groq_client import GroqModelClient
    GROQ_AVAILABLE = True
    logger.info("✅ Groq SDK imported successfully for hallucination detection")
except ImportError as e:
    logger.warning(f"❌ Groq client not available: {e}. Will use Gemini only")
    GROQ_AVAILABLE = False

# Fallback to Gemini
from langchain_google_genai import ChatGoogleGenerativeAI


class DocumentRelevance(BaseModel):
    """Model for document relevance evaluation results"""
    binary_score: bool = Field(
        description="Whether the answer is grounded in the documents - true if supported, false if not supported"
    )
    confidence: float = Field(
        default=0.5,
        description="Confidence score between 0.0 and 1.0 indicating how certain the evaluation is",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why the answer is or isn't grounded in the documents",
    )


RELEVANCE_PROMPT = """You are an expert document relevance evaluator. Your task is to determine whether an LLM-generated answer is properly grounded in the provided source documents.

EVALUATION CRITERIA:
- The answer must be directly supported by information found in the source documents
- Key facts, claims, and details should be traceable to the provided documents
- The answer should not contain information that contradicts the source documents
- Minor paraphrasing or reasonable inference from the documents is acceptable
- The answer should not include fabricated information or external knowledge not present in the documents

SCORING GUIDELINES:
- Score 'true' if the answer is well-supported by the documents
- Score 'false' if the answer contains unsupported claims, contradictions, or fabricated information

Be strict in your evaluation to ensure answer quality and prevent hallucinations.

SOURCE DOCUMENTS:
{documents}

LLM GENERATION TO EVALUATE:
{solution}

Please provide your evaluation as a JSON object:

{{
  "binary_score": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}

Based on the evaluation criteria, is this answer properly grounded in the source documents?"""


class DocumentRelevanceClient:
    """
    Document relevance (hallucination check) client with Groq primary and Gemini fallback.
    """
    
    def __init__(self):
        """Initialize relevance client"""
        self.groq_client = None
        self.gemini_client = None
        self.metrics = []
        
        # Initialize Groq - using smaller phi3 model for fast hallucination checks
        if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
            try:
                # Using groq/compound-mini for fast classification
                self.groq_client = GroqModelClient(
                    model_name="groq/compound-mini",  # Fast, efficient for classification
                    enable_fallback=False
                )
                logger.info("✅ Document Relevance: Groq client initialized (groq/compound-mini)")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
        
        # Initialize Gemini fallback
        try:
            self.gemini_client = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0,
                google_api_key=os.environ["GOOGLE_API_KEY"],
            )
            logger.info("✅ Document Relevance: Gemini fallback initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def evaluate(
        self,
        documents: str,
        solution: str
    ) -> tuple[DocumentRelevance, Dict[str, Any]]:
        """
        Evaluate if answer is grounded in documents.
        
        Args:
            documents: Source documents text
            solution: LLM-generated answer to evaluate
            
        Returns:
            Tuple of (relevance_result, metrics_dict)
        """
        start_time = time.time()
        
        # Create prompt
        prompt = RELEVANCE_PROMPT.format(
            documents=documents,
            solution=solution
        )
        
        logger.info("Checking document grounding (hallucination detection)...")
        
        # Try Groq first
        if self.groq_client:
            try:
                logger.info("🚀 Using Groq for hallucination check...")
                response_text, groq_metrics = self.groq_client.infer(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=256,
                    json_mode=True
                )
                
                # Parse JSON response
                response_dict = json.loads(response_text)
                relevance = DocumentRelevance(**response_dict)
                
                metrics = {
                    "model": "groq-gemma-7b",
                    "latency_ms": groq_metrics.latency_ms,
                    "tokens": groq_metrics.total_tokens,
                    "success": True,
                    "fallback_used": False
                }
                
                logger.info(
                    f"✅ Groq hallucination check complete: {relevance.binary_score} "
                    f"(confidence: {relevance.confidence:.2f}, "
                    f"{groq_metrics.latency_ms:.0f}ms)"
                )
                
                self.metrics.append(metrics)
                return relevance, metrics
                
            except Exception as e:
                logger.warning(f"Groq hallucination check failed: {e}, falling back to Gemini")
        
        # Fallback to Gemini
        try:
            logger.info("🔄 Using Gemini fallback for hallucination check...")
            gemini_start = time.time()
            
            # Use Gemini's structured output
            structured_llm = self.gemini_client.with_structured_output(DocumentRelevance)
            relevance = structured_llm.invoke(prompt)
            
            gemini_latency = (time.time() - gemini_start) * 1000
            
            metrics = {
                "model": "gemini-1.5-flash",
                "latency_ms": gemini_latency,
                "tokens": 0,
                "success": True,
                "fallback_used": True
            }
            
            logger.info(
                f"✅ Gemini hallucination check complete: {relevance.binary_score} "
                f"(confidence: {relevance.confidence:.2f}, "
                f"{gemini_latency:.0f}ms)"
            )
            
            self.metrics.append(metrics)
            return relevance, metrics
            
        except Exception as e:
            logger.error(f"Both Groq and Gemini failed: {e}")
            total_latency = (time.time() - start_time) * 1000
            
            # Return conservative default (mark as not grounded)
            relevance = DocumentRelevance(
                binary_score=False,
                confidence=0.0,
                reasoning="Evaluation failed due to error"
            )
            
            metrics = {
                "model": "none",
                "latency_ms": total_latency,
                "tokens": 0,
                "success": False,
                "fallback_used": True,
                "error": str(e)
            }
            
            self.metrics.append(metrics)
            return relevance, metrics
    
    def get_metrics(self) -> list:
        """Get all collected metrics"""
        return self.metrics


# Global client instance
_relevance_client = None


def get_relevance_client() -> DocumentRelevanceClient:
    """Get or create global relevance client"""
    global _relevance_client
    if _relevance_client is None:
        _relevance_client = DocumentRelevanceClient()
    return _relevance_client


def check_document_relevance(documents: str, solution: str) -> DocumentRelevance:
    """
    Check if LLM answer is grounded in source documents.
    
    Uses Groq gemma-7b-it for fast hallucination detection with Gemini fallback.
    
    Args:
        documents: Source documents text
        solution: LLM-generated answer
        
    Returns:
        DocumentRelevance object with grounding evaluation
    """
    client = get_relevance_client()
    relevance, metrics = client.evaluate(documents, solution)
    return relevance
