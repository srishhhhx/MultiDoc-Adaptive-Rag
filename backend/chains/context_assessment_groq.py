"""
Enhanced Gap Analysis Context Assessment Chain with Groq Integration

This module implements context assessment using Groq's llama-3.3-70b-versatile
model for reliable Gap Analysis on complex multi-part queries. Falls back to 
Gemini if Groq fails.

CRITICAL: Upgraded from 8B to 70B model to fix false negative bug where perfect
context was incorrectly flagged as insufficient, triggering unnecessary rewrite loops.

Key improvements:
- Reliable Gap Analysis with 70B reasoning power
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


# Enhanced Tool-Aware Gap Analysis Prompt for Context Assessment
CONTEXT_ASSESSMENT_PROMPT = """You are a "Gap Analysis" expert for an advanced multi-tool RAG agent. Your critical job is to determine if the provided context is sufficient to answer the user's *entire* question.

You will be given the user's original question and the context retrieved from TWO different tools:
1. **Document Retriever** (local files/vectorstore)
2. **Web Search** (real-time internet search)

### Your Task
1. **Deconstruct the Query:** Break down the `<original_question>` into all its individual sub-questions or required pieces of information.
2. **Analyze Combined Context:** Review *both* the `<document_context>` and the `<web_context>` to see if they *together* provide enough information to answer *all* parts of the user's query.
3. **Identify Gaps:** Explicitly state which parts of the query *cannot* be answered by the combined context.
4. **Make a Decision:** Based on your gap analysis, decide if the overall context is "sufficient" or "insufficient".
   - **sufficient**: ALL parts of the query are adequately covered by the combined context (documents + web results)
   - **insufficient**: *Any* significant part of the query is missing or poorly covered

### CRITICAL INSTRUCTION
You MUST analyze the COMBINED context from BOTH sources. Do NOT assess documents in isolation.
- If documents cover some components and web results cover others, that is SUFFICIENT
- Only mark as "insufficient" if the combined context has critical gaps

### JSON Output Format
You MUST provide your response *only* in the following JSON format:
```json
{{
  "question_components": ["Analysis of component 1", "Information needed for component 2", "Winner of X event"],
  "coverage_assessment": {{
    "Component 1": {{ "coverage_percentage": 90, "quality": "excellent", "source_tool": "document_retriever" }},
    "Component 2": {{ "coverage_percentage": 0, "quality": "missing", "source_tool": "none" }},
    "Component 3": {{ "coverage_percentage": 100, "quality": "good", "source_tool": "web_search" }}
  }},
  "identified_gaps": ["Information about Component 2 is completely missing from all context."],
  "overall_coverage_percentage": 63.33,
  "reasoning": "The combined context successfully covers Component 1 (from documents) and Component 3 (from web search), but Component 2 is not addressed at all, making the total context insufficient.",
  "final_decision": "insufficient"
}}
```

The final_decision field must be exactly "sufficient" or "insufficient".

<original_question>
{original_question}
</original_question>

<document_context>
{documents}
</document_context>

<web_context>
{web_results}
</web_context>

Provide your JSON assessment of the combined context:"""


class ContextAssessmentClient:
    """
    Context assessment client with Groq primary and Gemini fallback.
    
    Uses Groq's llama-3.3-70b-versatile for reliable Gap Analysis on complex
    multi-part queries, with automatic fallback to Gemini Flash if Groq fails.
    
    CRITICAL: Upgraded from 8B to 70B to fix false negative bug (55% coverage
    on perfect context), preventing unnecessary rewrite loop triggers.
    """
    
    def __init__(self):
        """Initialize assessment client with Groq and Gemini"""
        self.groq_client = None
        self.gemini_client = None
        self.metrics = []
        
        # Initialize Groq client
        # CRITICAL: Upgraded to 70B model to fix false negative bug
        # The 8B model was incorrectly flagging perfect context as insufficient (55% coverage)
        # This triggered the entire buggy rewrite loop unnecessarily
        # The 70B model provides reliable Gap Analysis for complex multi-part queries
        if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
            try:
                self.groq_client = GroqModelClient(
                    model_name="llama-3.3-70b-versatile",  # Upgraded from 8B for reliability
                    enable_fallback=False  # We handle fallback manually
                )
                logger.info("✅ Context Assessment: Groq client initialized (llama-3.3-70b-versatile)")
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
    
    def assess(self, original_question: str, documents: list, web_search_results: list = None) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Assess context sufficiency using Groq (primary) or Gemini (fallback).
        
        CRITICAL: Now TOOL-AWARE - analyzes COMBINED context from both documents and web results.
        
        Args:
            original_question: User's original question
            documents: List of retrieved documents from vectorstore
            web_search_results: List of web search results (optional)
            
        Returns:
            Tuple of (assessment_result, gap_analysis_json, metrics_dict)
            assessment_result is "sufficient" or "insufficient"
            gap_analysis_json is the full JSON report from the LLM
        """
        start_time = time.time()
        
        # Handle empty context
        if not documents and not web_search_results:
            logger.info("No documents or web results provided - returning insufficient")
            return "insufficient", {"final_decision": "insufficient", "reasoning": "No context provided"}, {
                "model": "none",
                "latency_ms": 0,
                "success": True,
                "fallback_used": False
            }
        
        # Format documents
        doc_texts = []
        if documents:
            for i, doc in enumerate(documents, 1):
                if hasattr(doc, 'page_content'):
                    doc_texts.append(f"--- Document {i} (Source: {doc.metadata.get('source', 'N/A')}) ---\n{doc.page_content}")
                else:
                    doc_texts.append(f"--- Document {i} ---\n{str(doc)}")
        
        documents_text = "\n\n".join(doc_texts) if doc_texts else "No documents were retrieved."
        
        # Format web results
        web_texts = []
        if web_search_results:
            for i, result in enumerate(web_search_results, 1):
                if hasattr(result, 'page_content'):
                    web_texts.append(f"--- Web Result {i} (Source: {result.metadata.get('source', 'N/A')}) ---\n{result.page_content}")
                elif isinstance(result, dict):
                    content = result.get('content', str(result))
                    source = result.get('url', result.get('source', 'N/A'))
                    web_texts.append(f"--- Web Result {i} (Source: {source}) ---\n{content}")
                else:
                    web_texts.append(f"--- Web Result {i} ---\n{str(result)}")
        
        web_results_text = "\n\n".join(web_texts) if web_texts else "No web search results were retrieved."
        
        # Create prompt with BOTH sources
        prompt = CONTEXT_ASSESSMENT_PROMPT.format(
            original_question=original_question,
            documents=documents_text,
            web_results=web_results_text
        )
        
        logger.info(f"Assessing context sufficiency for: '{original_question[:100]}...'")
        logger.info(f"Number of documents: {len(documents) if documents else 0}")
        logger.info(f"Number of web results: {len(web_search_results) if web_search_results else 0}")
        
        # Try Groq first
        if self.groq_client:
            try:
                logger.info("🚀 Using Groq for context assessment...")
                response_text, groq_metrics = self.groq_client.infer(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=1024
                )
                
                # Parse response - now returns (decision, json_dict)
                assessment, gap_analysis_json = self._parse_assessment(response_text)
                
                metrics = {
                    "model": "groq-llama3.3-70b",
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
                return assessment, gap_analysis_json, metrics
                
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
                "documents": documents_text,
                "web_results": web_results_text
            })
            
            gemini_latency = (time.time() - gemini_start) * 1000
            
            # Parse response - now returns (decision, json_dict)
            assessment, gap_analysis_json = self._parse_assessment(response_text)
            
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
            return assessment, gap_analysis_json, metrics
            
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
            return "insufficient", {"final_decision": "insufficient", "reasoning": "Both Groq and Gemini failed"}, metrics
    
    def _parse_assessment(self, response_text: str) -> tuple[str, Dict[str, Any]]:
        """
        Parse assessment result from LLM response.
        
        Tries multiple parsing strategies:
        1. JSON extraction and parsing
        2. String pattern matching
        3. Conservative fallback
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Tuple of ("sufficient" or "insufficient", full_json_dict)
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
                return final_decision, gap_analysis  # Return both decision and full JSON
            else:
                raise ValueError(f"Invalid final_decision: {final_decision}")
                
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.debug(f"JSON parsing failed: {e}, trying string matching")
            
            # Fallback: String-based parsing
            assessment_lower = response_text.strip().lower()
            
            if "final_decision" in assessment_lower:
                if '"sufficient"' in assessment_lower or "'sufficient'" in assessment_lower:
                    logger.info("✅ String match: sufficient")
                    return "sufficient", {"final_decision": "sufficient", "reasoning": "Parsed from string match"}
                elif '"insufficient"' in assessment_lower or "'insufficient'" in assessment_lower:
                    logger.info("❌ String match: insufficient")
                    return "insufficient", {"final_decision": "insufficient", "reasoning": "Parsed from string match"}
            
            # Secondary fallback
            if "sufficient" in assessment_lower and "insufficient" not in assessment_lower:
                logger.info("✅ Simple match: sufficient")
                return "sufficient", {"final_decision": "sufficient", "reasoning": "Simple string match"}
            elif "insufficient" in assessment_lower:
                logger.info("❌ Simple match: insufficient")
                return "insufficient", {"final_decision": "insufficient", "reasoning": "Simple string match"}
            else:
                logger.warning("⚠️ Unclear result, defaulting to sufficient")
                return "sufficient", {"final_decision": "sufficient", "reasoning": "Default fallback"}
    
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


def _fast_heuristic_check(original_question: str, documents: list, web_search_results: list = None) -> str:
    """
    **PERFORMANCE OPTIMIZATION**: Fast heuristic pre-check before expensive LLM assessment.

    Quickly determines if context is obviously sufficient based on simple metrics.
    Saves 2-3 seconds by avoiding LLM call when context is clearly good.

    Returns:
        "sufficient" if heuristics suggest good context, "uncertain" if LLM assessment needed
    """
    # Calculate total context length
    total_docs = len(documents) + (len(web_search_results) if web_search_results else 0)

    if total_docs == 0:
        return "uncertain"  # No context at all - need LLM to confirm

    # Calculate total tokens (rough estimate: chars / 4)
    total_chars = sum(len(doc.page_content) for doc in documents)
    if web_search_results:
        total_chars += sum(len(result.page_content) for result in web_search_results)

    total_tokens = total_chars // 4

    # Check if question keywords appear in context
    question_words = set(original_question.lower().split())
    # Remove common words
    stop_words = {'what', 'is', 'the', 'how', 'why', 'when', 'where', 'who', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    question_keywords = question_words - stop_words

    context_text = " ".join([doc.page_content.lower() for doc in documents])
    if web_search_results:
        context_text += " " + " ".join([result.page_content.lower() for result in web_search_results])

    keyword_matches = sum(1 for keyword in question_keywords if keyword in context_text)
    keyword_ratio = keyword_matches / max(len(question_keywords), 1)

    # **Heuristic Rules** (conservative - only skip LLM if we're confident)
    # Rule 1: Rich context with good keyword coverage
    if total_tokens >= 500 and keyword_ratio >= 0.7 and total_docs >= 3:
        logger.info(f"⚡ FAST HEURISTIC: Context looks sufficient (tokens={total_tokens}, keywords={keyword_ratio:.1%}, docs={total_docs}) - skipping LLM assessment")
        return "sufficient"

    # Rule 2: Very rich context (even with lower keyword match)
    if total_tokens >= 1000 and keyword_ratio >= 0.5 and total_docs >= 5:
        logger.info(f"⚡ FAST HEURISTIC: Rich context detected (tokens={total_tokens}, docs={total_docs}) - skipping LLM assessment")
        return "sufficient"

    # Otherwise, need LLM assessment
    logger.info(f"🤔 Heuristic uncertain (tokens={total_tokens}, keywords={keyword_ratio:.1%}, docs={total_docs}) - proceeding to LLM assessment")
    return "uncertain"


def assess_context_sufficiency(original_question: str, documents: list, web_search_results: list = None) -> tuple[str, Dict[str, Any]]:
    """
    Assess whether retrieved context is sufficient to answer the question.

    CRITICAL: Now TOOL-AWARE - analyzes COMBINED context from both documents and web results.

    **OPTIMIZED**: Includes fast heuristic pre-check to avoid expensive LLM calls when context is obviously good.

    Uses Groq llama-3.3-70b-versatile for reliable Gap Analysis with Gemini fallback.

    Previous Bug: Only assessed documents, ignored web results → false "insufficient" on hybrid queries
    Fix: Now accepts and analyzes both documents AND web_search_results

    Args:
        original_question: User's original question
        documents: List of retrieved documents from vectorstore
        web_search_results: List of web search results (optional)

    Returns:
        Tuple of ("sufficient" or "insufficient", full_gap_analysis_json)
    """
    # **PERFORMANCE OPTIMIZATION**: Try fast heuristic check first (saves 2-3s)
    heuristic_result = _fast_heuristic_check(original_question, documents, web_search_results)

    if heuristic_result == "sufficient":
        # Context is obviously good - skip expensive LLM call
        return "sufficient", {
            "assessment": "sufficient",
            "method": "fast_heuristic",
            "reason": "Context passed fast heuristic checks (sufficient length, keyword coverage, and document count)"
        }

    # Heuristic uncertain - proceed with full LLM assessment
    client = get_assessment_client()
    assessment, gap_analysis_json, metrics = client.assess(original_question, documents, web_search_results)
    return assessment, gap_analysis_json
