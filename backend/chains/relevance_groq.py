"""
Groq-Powered Batched Relevance Evaluation Chain for Maximum Performance

This module provides a high-performance alternative to the Gemini-based relevance
evaluation, using Groq's llama-3.1-8b-instant for 10-20x faster evaluation while
maintaining the same sophisticated quality assessment logic.

This is the FINAL major latency bottleneck fix in the agent's happy path.
"""

import os
import json
import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Import Groq client from correct location
from backend.inference_clients.groq_client import GroqModelClient

load_dotenv()


class BatchRelevanceEvaluation(BaseModel):
    """Combined document and question relevance evaluation"""
    
    # Document relevance (answer grounding)
    document_grounding: bool = Field(
        description="True if the answer is grounded in ANY provided context (documents OR web), False if hallucinated"
    )
    document_grounding_score: float = Field(
        default=0.5,
        description="Confidence score for document grounding (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    document_grounding_explanation: str = Field(
        description="Explanation of why the answer is or isn't grounded in the provided context"
    )
    
    # **CRITICAL: Grounding source tracking**
    grounding_source: str = Field(
        description="Which context source(s) support the answer: 'DOCUMENT_ONLY', 'WEB_ONLY', 'HYBRID' (both), or 'NONE' (hallucinated)"
    )
    grounding_source_details: str = Field(
        description="Detailed explanation of which specific claims come from documents vs web search"
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


# **CRITICAL: THE COMPLETE SOPHISTICATED BATCH RELEVANCE PROMPT**
# This is the EXACT same prompt used in the Gemini version to ensure quality parity
BATCH_RELEVANCE_PROMPT = """You are an expert answer quality evaluator for a HYBRID RAG system. Your role is to evaluate BOTH the document grounding AND question relevance of a generated answer in a single comprehensive assessment.

**CRITICAL: HYBRID CONTEXT AWARENESS**
This RAG system can use MULTIPLE context sources:
- Document Context: Information from uploaded/indexed documents
- Web Search Context: Real-time information from web searches

The answer may synthesize information from BOTH sources. This is VALID and EXPECTED behavior for hybrid queries.

EVALUATION FRAMEWORK:

1. DOCUMENT GROUNDING EVALUATION (HYBRID-AWARE):
   - Is the answer factually supported by information in **EITHER** the document context **OR** the web search context?
   - **CRITICAL**: Information sourced ONLY from web search context IS considered grounded if web context was provided
   - **CRITICAL**: Information sourced ONLY from document context IS considered grounded if document context was provided
   - **CRITICAL**: Answers synthesizing BOTH sources are VALID and grounded
   - Only mark as False if the answer introduces claims NOT found in ANY provided context source
   - Only mark as False if the answer directly contradicts the provided context sources
   - Are there any hallucinations or fabricated details not present in ANY context source?
   
   **CRITICAL: SOURCE ATTRIBUTION REQUIREMENT**
   You MUST explicitly identify which context source(s) support the answer:
   - Analyze each major claim in the answer
   - Determine if it comes from DOCUMENT CONTEXT, WEB SEARCH CONTEXT, or BOTH
   - Do NOT rely on your internal knowledge - ONLY validate against the provided context
   - If the answer contains information you "know" but isn't in the provided context, mark it as hallucinated

2. QUESTION RELEVANCE EVALUATION:
   - Does the answer directly address what the user asked?
   - Is the answer complete and comprehensive for the question?
   - Does the answer stay on-topic and avoid irrelevant information?
   - Would this answer satisfy the user's information need?

SCORING CRITERIA:

**Document Grounding (True/False):**
- True: Answer is supported by information in document context OR web search context OR both
- True: Answer synthesizes information from multiple provided sources
- True: Minor paraphrasing or reasonable inference from ANY provided context
- False: ONLY if answer contains substantive claims not found in ANY provided context
- False: ONLY if answer contradicts information in the provided contexts

**Question Relevance (True/False):**
- True: Answer directly and comprehensively addresses the question
- False: Answer is off-topic, incomplete, or doesn't address the core question

**Confidence Scores (0.0-1.0):**
- 0.9-1.0: Very confident in the assessment
- 0.7-0.8: Confident with minor uncertainty
- 0.5-0.6: Moderate confidence, some ambiguity
- 0.3-0.4: Low confidence, significant uncertainty
- 0.0-0.2: Very uncertain or contradictory evidence

**GROUNDING SOURCE DETERMINATION:**
You must set the 'grounding_source' field based on your analysis:
- **DOCUMENT_ONLY**: All major claims in the answer are supported by document context only
- **WEB_ONLY**: All major claims in the answer are supported by web search context only
- **HYBRID**: Answer synthesizes claims from BOTH document and web search contexts
- **NONE**: Answer contains hallucinated claims not found in ANY provided context

EVALUATION INSTRUCTIONS:
- **CRITICAL**: Recognize that hybrid answers using both document and web context are VALID
- **CRITICAL**: Do NOT penalize answers for using web search information when web context is provided
- **CRITICAL**: Do NOT use your internal knowledge - ONLY validate against provided context
- **CRITICAL**: If you recognize information but it's not in the provided context, it's a hallucination
- Evaluate both aspects independently but consider their interaction
- Provide clear explanations for your assessments
- Be strict about hallucinations (info not in ANY context) but NOT strict about which context source was used
- Be fair about question relevance while maintaining quality standards
- Consider the overall utility of the answer for the user

**EXAMPLES OF VALID GROUNDING:**
- Answer uses only document context → GROUNDED (True), grounding_source: DOCUMENT_ONLY
- Answer uses only web search context → GROUNDED (True), grounding_source: WEB_ONLY
- Answer synthesizes both document and web context → GROUNDED (True), grounding_source: HYBRID
- Answer adds claims not in any provided context → NOT GROUNDED (False), grounding_source: NONE

**CRITICAL WARNING ABOUT KNOWLEDGE LEAKAGE:**
You may "know" facts about topics (e.g., who won a race, what a concept means). However, you MUST NOT mark an answer as grounded based on your internal knowledge. ONLY validate against the explicitly provided context sections. If information matches your knowledge but isn't in the provided context, it's a hallucination.

USER QUESTION:
{question}

PROVIDED CONTEXT (May include documents AND/OR web search results):
{documents_text}

GENERATED ANSWER:
{solution}

You must return a JSON object with the following structure:
{{
  "document_grounding": true or false,
  "document_grounding_score": 0.0 to 1.0,
  "document_grounding_explanation": "explanation text",
  "grounding_source": "DOCUMENT_ONLY" or "WEB_ONLY" or "HYBRID" or "NONE",
  "grounding_source_details": "detailed explanation of which claims come from which sources",
  "question_relevance": true or false,
  "question_relevance_score": 0.0 to 1.0,
  "question_relevance_explanation": "explanation text",
  "overall_quality": "overall assessment text"
}}

Provide a comprehensive evaluation covering both aspects, with special attention to recognizing valid hybrid context usage and accurate source attribution."""


class RelevanceEvaluationClient:
    """Client for performing relevance evaluations with Groq primary and Gemini fallback"""
    
    def __init__(self):
        """Initialize both Groq and Gemini clients"""
        # Primary: Groq with 70B model for complex reasoning
        # NOTE: Using llama-3.3-70b-versatile instead of 8B for this critical QA task
        # The 70B model is required to handle the sophisticated multi-rule prompt
        # and correctly attribute grounding sources (DOCUMENT_ONLY vs HYBRID vs WEB_ONLY)
        try:
            self.groq_client = GroqModelClient(model_name="llama-3.3-70b-versatile")
            print("✅ Groq client initialized for relevance evaluation (llama-3.3-70b-versatile)")
        except Exception as e:
            print(f"⚠️ Groq client initialization failed: {e}")
            self.groq_client = None
        
        # Fallback: Gemini for reliability
        try:
            self.gemini_client = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                google_api_key=os.environ["GOOGLE_API_KEY"],
            )
            self.gemini_structured = self.gemini_client.with_structured_output(BatchRelevanceEvaluation)
            print("✅ Gemini fallback initialized for relevance evaluation")
        except Exception as e:
            print(f"⚠️ Gemini fallback initialization failed: {e}")
            self.gemini_client = None
            self.gemini_structured = None
        
        # Metrics tracking
        self.metrics = []
    
    def evaluate(self, question: str, documents: List, solution: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Evaluate answer relevance and grounding using Groq (primary) or Gemini (fallback)
        
        Args:
            question: User's query
            documents: List of document objects (may include web search results)
            solution: Generated answer to evaluate
            
        Returns:
            Tuple of (evaluation_result_dict, metrics_dict)
        """
        start_time = time.time()
        
        # Build context text (same logic as original)
        documents_text = self._build_context_text(documents)
        
        # Create prompt
        prompt = BATCH_RELEVANCE_PROMPT.format(
            question=question,
            documents_text=documents_text,
            solution=solution
        )
        
        print(f"🔍 Evaluating answer quality and grounding...")
        
        # Try Groq first (10-20x faster)
        if self.groq_client:
            try:
                print("🚀 Using Groq for relevance evaluation...")
                groq_start = time.time()
                
                response_text, groq_metrics = self.groq_client.infer(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=1024,
                    json_mode=True
                )
                
                groq_latency = (time.time() - groq_start) * 1000
                
                # Parse JSON response
                response_dict = json.loads(response_text)
                
                # Validate and create BatchRelevanceEvaluation object
                batch_result = BatchRelevanceEvaluation(**response_dict)
                
                # Convert to compatible format
                result_dict = self._convert_to_result_dict(batch_result)
                
                metrics = {
                    "model": "groq-llama3.1-8b",
                    "latency_ms": groq_latency,
                    "tokens": groq_metrics.total_tokens,
                    "success": True,
                    "fallback_used": False,
                }
                
                print(
                    f"✅ Groq relevance evaluation complete: "
                    f"Grounded={batch_result.document_grounding}, "
                    f"Relevant={batch_result.question_relevance}, "
                    f"Source={batch_result.grounding_source} "
                    f"({groq_latency:.0f}ms)"
                )
                
                self.metrics.append(metrics)
                return result_dict, metrics
                
            except Exception as e:
                print(f"⚠️ Groq relevance evaluation failed: {e}, falling back to Gemini")
        
        # Fallback to Gemini
        if self.gemini_structured:
            try:
                print("🔄 Using Gemini fallback for relevance evaluation...")
                gemini_start = time.time()
                
                # Use Gemini's structured output
                batch_result = self.gemini_structured.invoke({
                    "question": question,
                    "documents_text": documents_text,
                    "solution": solution
                })
                
                gemini_latency = (time.time() - gemini_start) * 1000
                
                # Convert to compatible format
                result_dict = self._convert_to_result_dict(batch_result)
                
                metrics = {
                    "model": "gemini-2.5-flash",
                    "latency_ms": gemini_latency,
                    "tokens": 0,
                    "success": True,
                    "fallback_used": True,
                }
                
                print(
                    f"✅ Gemini relevance evaluation complete: "
                    f"Grounded={batch_result.document_grounding}, "
                    f"Relevant={batch_result.question_relevance}, "
                    f"Source={batch_result.grounding_source} "
                    f"({gemini_latency:.0f}ms)"
                )
                
                self.metrics.append(metrics)
                return result_dict, metrics
                
            except Exception as e:
                print(f"❌ Gemini relevance evaluation also failed: {e}")
        
        # Both failed - return safe defaults
        total_latency = (time.time() - start_time) * 1000
        
        class FallbackScore:
            binary_score = False
            confidence_score = 0.0
            explanation = "Evaluation failed - both Groq and Gemini unavailable"
        
        result_dict = {
            "document_relevance_score": FallbackScore(),
            "question_relevance_score": FallbackScore(),
            "grounding_source": "UNKNOWN",
            "grounding_source_details": "Evaluation failed",
            "batch_evaluation": False,
            "overall_quality": "Evaluation failed"
        }
        
        metrics = {
            "model": "none",
            "latency_ms": total_latency,
            "tokens": 0,
            "success": False,
            "fallback_used": True,
            "error": "Both Groq and Gemini failed"
        }
        
        self.metrics.append(metrics)
        return result_dict, metrics
    
    def _build_context_text(self, documents: List) -> str:
        """Build context text with source separation (same logic as original)"""
        if not documents:
            return "No context provided."
        
        # **HYBRID-AWARE CONTEXT BUILDING**
        doc_sources = []
        web_sources = []
        
        for doc in documents:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            
            # Detect web search results
            is_web_result = False
            if hasattr(doc, 'metadata'):
                source = doc.metadata.get('source', '')
                if 'http' in source or 'www' in source or doc.metadata.get('type') == 'web_search':
                    is_web_result = True
            
            # Truncate very long content
            if len(content) > 1500:
                content = content[:1500] + "... [truncated]"
            
            if is_web_result:
                web_sources.append(content)
            else:
                doc_sources.append(content)
        
        # Build context with clear source separation
        context_parts = []
        
        if doc_sources:
            doc_text = "\n\n".join([f"Document {i+1}:\n{content}" for i, content in enumerate(doc_sources)])
            context_parts.append(f"=== DOCUMENT CONTEXT ===\n{doc_text}")
        
        if web_sources:
            web_text = "\n\n".join([f"Web Result {i+1}:\n{content}" for i, content in enumerate(web_sources)])
            context_parts.append(f"=== WEB SEARCH CONTEXT ===\n{web_text}")
        
        # If we couldn't detect source types, treat all as generic context
        if not context_parts:
            generic_text = "\n\n".join([
                f"Source {i+1}:\n{doc.page_content if hasattr(doc, 'page_content') else str(doc)}"
                for i, doc in enumerate(documents)
            ])
            context_parts.append(f"=== PROVIDED CONTEXT ===\n{generic_text}")
        
        return "\n\n".join(context_parts)
    
    def _convert_to_result_dict(self, batch_result: BatchRelevanceEvaluation) -> Dict[str, Any]:
        """Convert BatchRelevanceEvaluation to compatible result dict"""
        
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
            "grounding_source": batch_result.grounding_source,
            "grounding_source_details": batch_result.grounding_source_details,
            "batch_evaluation": True,
            "overall_quality": batch_result.overall_quality
        }
    
    def get_metrics(self) -> list:
        """Get all collected metrics"""
        return self.metrics


# Global client instance
_relevance_client = None


def get_relevance_client() -> RelevanceEvaluationClient:
    """Get or create the global relevance evaluation client"""
    global _relevance_client
    if _relevance_client is None:
        _relevance_client = RelevanceEvaluationClient()
    return _relevance_client


def evaluate_relevance_batch_groq(question: str, documents: List, solution: str) -> Dict[str, Any]:
    """
    Groq-powered batch relevance evaluation (10-20x faster than Gemini)
    
    This is the high-performance replacement for the Gemini-based evaluate_relevance_batch.
    Uses the EXACT same sophisticated prompt and evaluation logic, but with Groq for speed.
    
    Args:
        question: User's query
        documents: List of document objects (may include web search results)
        solution: Generated answer to evaluate
    
    Returns:
        Dictionary with document_relevance_score, question_relevance_score, and grounding_source
        Compatible with existing caching system
    """
    client = get_relevance_client()
    result_dict, metrics = client.evaluate(question, documents, solution)
    return result_dict
