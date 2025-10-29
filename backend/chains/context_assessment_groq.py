"""
Enhanced Gap Analysis Context Assessment Chain with Groq Integration

This module implements context assessment using Groq's fast inference API
with llama3-8b-instruct for rapid evaluation tasks. Falls back to Gemini
if Groq fails.

Key improvements:
- 3-5x faster inference with Groq
- Automatic fallback to Gemini
- Comprehensive latency tracking
- Structured JSON output parsing
"""

import os
import logging
import json
import re
import time
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Import Groq client
try:
    from backend.inference_clients.groq_client import GroqModelClient
    GROQ_AVAILABLE = True
    logger.info("✅ Groq SDK imported successfully for context assessment")
except ImportError as e:
    logger.warning(f"❌ Groq client not available: {e}. Will use Gemini only")
    GROQ_AVAILABLE = False

# Fallback to Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Enhanced Gap Analysis Prompt for Context Assessment
CONTEXT_ASSESSMENT_PROMPT = """You are an expert document analyst performing a structured Gap Analysis to determine context sufficiency. Your task is to systematically evaluate whether the retrieved documents can adequately answer the user's question.

ANALYSIS FRAMEWORK:

1. QUESTION DECOMPOSITION:
   - Break down the user's question into key components or sub-questions
   - Identify what specific information types are needed (facts, procedures, examples, etc.)
   - Note any implicit requirements or context needed for a complete answer

2. CONTENT COVERAGE ANALYSIS:
   - For each question component, assess what percentage is covered by the documents
   - Identify which documents contain relevant information for each component
   - Note the quality and specificity of the information found

3. GAP IDENTIFICATION:
   - List any critical information gaps that would prevent a complete answer
   - Distinguish between minor gaps (can be reasonably inferred) vs major gaps (require additional sources)
   - Consider whether partial information is sufficient for a useful response

4. SUFFICIENCY DECISION CRITERIA:
   - SUFFICIENT: Documents cover ≥70% of question components with adequate detail, OR provide enough context for a meaningful partial answer
   - INSUFFICIENT: Documents cover <50% of question components, OR missing critical foundational information that cannot be reasonably inferred

IMPORTANT GUIDELINES:
- Favor "sufficient" when documents provide substantial relevant information, even if not 100% complete
- Consider that users often benefit from partial but accurate answers rather than no answer
- Only mark as "insufficient" when the gap is so significant that any answer would be misleading or unhelpful
- For multi-part questions, assess overall utility rather than requiring every sub-question to be fully answered

USER QUESTION:
{original_question}

CONTEXT DOCUMENTS:
{documents}

STRUCTURED ANALYSIS:

1. Question Components:
[List the key components/sub-questions]

2. Coverage Assessment:
[For each component, note coverage level and source documents]

3. Gap Analysis:
[Identify any significant gaps and their impact]

4. Final Decision:
Based on the analysis above, provide your assessment as a JSON object:

```json
{{
  "question_components": ["component1", "component2", "..."],
  "coverage_assessment": {{
    "component1": {{"coverage_percentage": 80, "quality": "good", "source_documents": [1, 2]}},
    "component2": {{"coverage_percentage": 30, "quality": "poor", "source_documents": []}}
  }},
  "identified_gaps": ["gap1", "gap2"],
  "overall_coverage_percentage": 65,
  "reasoning": "Detailed explanation of the decision",
  "final_decision": "sufficient"
}}
```

The final_decision field must be exactly "sufficient" or "insufficient".

ASSESSMENT:"""


class ContextAssessmentClient:
    """
    Context assessment client with Groq primary and Gemini fallback.
    
    Uses Groq's llama3-8b-instruct for fast evaluation, with automatic
    fallback to Gemini Flash if Groq fails.
    """
    
    def __init__(self):
        """Initialize assessment client with Groq and Gemini"""
        self.groq_client = None
        self.gemini_client = None
        self.metrics = []
        
        # Initialize Groq client
        if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
            try:
                self.groq_client = GroqModelClient(
                    model_name="llama-3.1-8b-instant",
                    enable_fallback=False  # We handle fallback manually
                )
                logger.info("✅ Context Assessment: Groq client initialized (llama-3.1-8b-instant)")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
        
        # Initialize Gemini fallback
        try:
            self.gemini_client = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=os.environ["GOOGLE_API_KEY"],
                temperature=0.1
            )
            logger.info("✅ Context Assessment: Gemini fallback initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def assess(self, original_question: str, documents: list) -> tuple[str, Dict[str, Any]]:
        """
        Assess context sufficiency using Groq (primary) or Gemini (fallback).
        
        Args:
            original_question: User's original question
            documents: List of retrieved documents
            
        Returns:
            Tuple of (assessment_result, metrics_dict)
            assessment_result is "sufficient" or "insufficient"
        """
        start_time = time.time()
        
        # Format documents
        if not documents:
            logger.info("No documents provided - returning insufficient")
            return "insufficient", {
                "model": "none",
                "latency_ms": 0,
                "success": True,
                "fallback_used": False
            }
        
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            if hasattr(doc, 'page_content'):
                doc_texts.append(f"Document {i}:\n{doc.page_content}")
            else:
                doc_texts.append(f"Document {i}:\n{str(doc)}")
        
        documents_text = "\n\n".join(doc_texts)
        
        # Create prompt
        prompt = CONTEXT_ASSESSMENT_PROMPT.format(
            original_question=original_question,
            documents=documents_text
        )
        
        logger.info(f"Assessing context sufficiency for: '{original_question[:100]}...'")
        logger.info(f"Number of documents: {len(documents)}")
        
        # Try Groq first
        if self.groq_client:
            try:
                logger.info("🚀 Using Groq for context assessment...")
                response_text, groq_metrics = self.groq_client.infer(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=1024
                )
                
                # Parse response
                assessment = self._parse_assessment(response_text)
                
                metrics = {
                    "model": "groq-llama3-8b",
                    "latency_ms": groq_metrics.latency_ms,
                    "tokens": groq_metrics.total_tokens,
                    "success": True,
                    "fallback_used": False
                }
                
                logger.info(
                    f"✅ Groq assessment complete: {assessment} "
                    f"({groq_metrics.latency_ms:.0f}ms)"
                )
                
                self.metrics.append(metrics)
                return assessment, metrics
                
            except Exception as e:
                logger.warning(f"Groq assessment failed: {e}, falling back to Gemini")
        
        # Fallback to Gemini
        try:
            logger.info("🔄 Using Gemini fallback for context assessment...")
            gemini_start = time.time()
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", CONTEXT_ASSESSMENT_PROMPT),
                ("user", "Please assess the context sufficiency.")
            ])
            
            chain = prompt_template | self.gemini_client | StrOutputParser()
            response_text = chain.invoke({
                "original_question": original_question,
                "documents": documents_text
            })
            
            gemini_latency = (time.time() - gemini_start) * 1000
            
            # Parse response
            assessment = self._parse_assessment(response_text)
            
            metrics = {
                "model": "gemini-1.5-flash",
                "latency_ms": gemini_latency,
                "tokens": 0,
                "success": True,
                "fallback_used": True
            }
            
            logger.info(
                f"✅ Gemini assessment complete: {assessment} "
                f"({gemini_latency:.0f}ms)"
            )
            
            self.metrics.append(metrics)
            return assessment, metrics
            
        except Exception as e:
            logger.error(f"Both Groq and Gemini failed: {e}")
            total_latency = (time.time() - start_time) * 1000
            
            metrics = {
                "model": "none",
                "latency_ms": total_latency,
                "tokens": 0,
                "success": False,
                "fallback_used": True,
                "error": str(e)
            }
            
            self.metrics.append(metrics)
            return "insufficient", metrics
    
    def _parse_assessment(self, response_text: str) -> str:
        """
        Parse assessment result from LLM response.
        
        Tries multiple parsing strategies:
        1. JSON extraction and parsing
        2. String pattern matching
        3. Conservative fallback
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            "sufficient" or "insufficient"
        """
        logger.debug(f"Parsing assessment from {len(response_text)} chars")
        
        try:
            # Method 1: Extract JSON from markdown code fences
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, response_text, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                json_str = json_match.group(1)
                logger.debug("✅ Found JSON in markdown fence")
            else:
                # Method 2: Look for JSON object with final_decision
                json_pattern = r'\{[^{}]*"final_decision"[^{}]*\}'
                json_match = re.search(json_pattern, response_text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(0)
                    logger.debug("✅ Found JSON object pattern")
                else:
                    # Method 3: Extract largest JSON-like block
                    brace_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
                    json_matches = re.findall(brace_pattern, response_text, re.DOTALL)
                    
                    if json_matches:
                        for match in json_matches:
                            if "final_decision" in match:
                                json_str = match
                                logger.debug("✅ Found JSON in brace pattern")
                                break
                        else:
                            json_str = max(json_matches, key=len)
                            logger.debug("✅ Using largest JSON block")
                    else:
                        raise ValueError("No JSON pattern found")
            
            # Parse JSON
            gap_analysis = json.loads(json_str)
            final_decision = gap_analysis.get("final_decision", "").lower().strip()
            
            if final_decision in ["sufficient", "insufficient"]:
                logger.info(
                    f"✅ Parsed JSON assessment: {final_decision} "
                    f"(coverage: {gap_analysis.get('overall_coverage_percentage', 'N/A')}%)"
                )
                return final_decision
            else:
                raise ValueError(f"Invalid final_decision: {final_decision}")
                
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.debug(f"JSON parsing failed: {e}, trying string matching")
            
            # Fallback: String-based parsing
            assessment_lower = response_text.strip().lower()
            
            if "final_decision" in assessment_lower:
                if '"sufficient"' in assessment_lower or "'sufficient'" in assessment_lower:
                    logger.info("✅ String match: sufficient")
                    return "sufficient"
                elif '"insufficient"' in assessment_lower or "'insufficient'" in assessment_lower:
                    logger.info("❌ String match: insufficient")
                    return "insufficient"
            
            # Secondary fallback
            if "sufficient" in assessment_lower and "insufficient" not in assessment_lower:
                logger.info("✅ Simple match: sufficient")
                return "sufficient"
            elif "insufficient" in assessment_lower:
                logger.info("❌ Simple match: insufficient")
                return "insufficient"
            else:
                logger.warning("⚠️ Unclear result, defaulting to sufficient")
                return "sufficient"
    
    def get_metrics(self) -> list:
        """Get all collected metrics"""
        return self.metrics


# Global client instance
_assessment_client = None


def get_assessment_client() -> ContextAssessmentClient:
    """Get or create global assessment client"""
    global _assessment_client
    if _assessment_client is None:
        _assessment_client = ContextAssessmentClient()
    return _assessment_client


def assess_context_sufficiency(original_question: str, documents: list) -> str:
    """
    Assess whether retrieved documents are sufficient to answer the question.
    
    Uses Groq llama3-8b-instruct for fast evaluation with Gemini fallback.
    
    Args:
        original_question: User's original question
        documents: List of retrieved documents
        
    Returns:
        "sufficient" or "insufficient"
    """
    client = get_assessment_client()
    assessment, metrics = client.assess(original_question, documents)
    return assessment
