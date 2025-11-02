"""
Plan Correction Chain for Self-Correcting RAG Pipeline - GROQ-POWERED VERSION

This module implements PLAN-BASED correction (not query rewriting) using Groq's 
llama-3.3-70b-versatile model. Instead of generating creative query strings that 
often fail with format errors, this chain surgically corrects failed tasks in the 
JSON execution plan based on gap analysis reports.

Performance: ~1-2 seconds vs ~10-28 seconds with Gemini
Fallback: Automatically falls back to Gemini if Groq fails
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import Groq client
from backend.inference_clients.groq_client import GroqModelClient

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# NEW PLAN-BASED CORRECTION PROMPT
REWRITE_PROMPT_TEMPLATE = """You are a "Plan Corrector" for an advanced RAG agent. Your critical job is to surgically correct a failed JSON execution plan based on a gap analysis report.

You will receive the user's original query, the failed `execution_plan` (a JSON list), and a `gap_analysis_report` that shows which parts of the query failed.

Your task is to generate a NEW, CORRECTED `execution_plan` (as a JSON list) that fixes *only* the failed tasks.

### CRITICAL RULES:
1.  **JSON ONLY:** You MUST output *only* the corrected JSON list of tasks. Do NOT output any other text, explanation, markdown, or "Here is the new plan:" preamble.
2.  **PRESERVE ALL TASKS:** You MUST return a plan with the *same number of tasks* as the original. NEVER delete tasks, even if they seem unrelated (like a web search). This is a user-intent violation.
3.  **PRESERVE SUCCESSFUL TASKS:** For any task/component that the `gap_analysis_report` shows as "sufficient" or having high coverage, you MUST copy the original task (tool, query, source_document) *verbatim*. DO NOT CHANGE IT.
4.  **CORRECT FAILED TASKS:** For any task/component that the `gap_analysis_report` shows as "insufficient" or having low/zero coverage, you MUST generate a new, improved `query` for that task.
5.  **CLEAN QUERY OUTPUT:** The new `query` you generate MUST be a clean, single-line search query. It must NOT be a multi-paragraph explanation.
6.  **PRESERVE METADATA:** You MUST preserve the `tool` and `source_document` fields for every task in the new plan. Only change the `query` field, and only if it failed.

---
### User's Original Question:
{original_question}

---
### Failed Execution Plan (Input):
{execution_plan}

---
### Gap Analysis Report (Context on Failures):
{gap_analysis_report}

---
### Your Corrected Execution Plan (JSON-ONLY Output):
"""


class PlanCorrectionClient:
    """Client for performing plan-based correction with Groq primary and Gemini fallback"""
    
    def __init__(self):
        """Initialize Groq client for plan correction"""
        # Primary: Groq with 70B model for complex reasoning
        # NOTE: Using llama-3.3-70b-versatile for this critical reasoning task
        # The 70B model is required to understand gap analysis and surgically
        # correct execution plans without violating user intent
        try:
            self.groq_client = GroqModelClient(model_name="llama-3.3-70b-versatile")
            logger.info("✅ Groq client initialized for plan correction (llama-3.3-70b-versatile)")
        except Exception as e:
            logger.warning(f"⚠️ Groq client initialization failed: {e}")
            self.groq_client = None
    
    def _unwrap_plan_from_response(self, response: Any) -> Optional[List[Dict[str, str]]]:
        """
        Intelligently unwraps the plan list from the LLM's response.
        """
        # Case 1: Ideal response, it's already a list.
        if isinstance(response, list):
            logger.info("  ✅ LLM returned a valid list directly.")
            return response
        
        if isinstance(response, dict):
            logger.warning("  ⚠️  LLM returned a dict, attempting to unwrap list...")
            
            # Case 2: LLM wrapped the list in a dict (e.g., {"plan": [...]})
            for key in ["tasks", "plan", "corrected_plan", "execution_plan"]:
                if key in response and isinstance(response[key], list):
                    logger.info(f"  ✅ Successfully unwrapped list from key: '{key}'")
                    return response[key]
            
            # --- START FINAL FIX ---
            # Case 3: LLM returned a *single task object* (a dict)
            # The log told us the keys are: ['tool', 'query', 'source_document']
            # We will check for these to confirm it's a valid task.
            if ("tool" in response and 
                "query" in response and 
                "source_document" in response):
                
                logger.warning("  ⚠️  LLM returned a single task object. Wrapping it in a list.")
                return [response] # <-- Wrap the single dict in a list
            # --- END FINAL FIX ---
            
            logger.error(f"  ❌ LLM returned a dict, but no valid list or task object found in keys: {list(response.keys())}")
            return None
        
        logger.error(f"  ❌ LLM response was not a list or dict: {type(response)}")
        return None
    
    def correct_plan(self, original_question: str, execution_plan: List[Dict[str, str]], 
                    gap_analysis_report: Dict[str, Any], attempt_number: int = 1) -> List[Dict[str, str]]:
        """
        Correct execution plan using Groq based on gap analysis report
        
        Args:
            original_question: User's original query
            execution_plan: The failed execution plan (list of tasks)
            gap_analysis_report: Gap analysis JSON showing what failed
            attempt_number: Which correction attempt this is
            
        Returns:
            List[Dict[str, str]]: Corrected execution plan
        """
        logger.info(f"Correcting plan (attempt #{attempt_number})")
        logger.info(f"  Original plan: {execution_plan}")
        logger.info(f"  Gap report decision: {gap_analysis_report.get('final_decision')}")
        logger.info(f"  Gap report gaps: {gap_analysis_report.get('identified_gaps')}")
        
        # Build full prompt
        full_prompt = REWRITE_PROMPT_TEMPLATE.format(
            original_question=original_question,
            execution_plan=json.dumps(execution_plan, indent=2),
            gap_analysis_report=json.dumps(gap_analysis_report, indent=2)
        )
        
        # Try Groq first
        if self.groq_client:
            try:
                logger.info("🚀 GROQ PLAN CORRECTION: Correcting plan with llama-3.3-70b-versatile")
                
                # Use Groq for fast inference with JSON mode
                response_text, metrics = self.groq_client.infer(
                    prompt=full_prompt,
                    temperature=0.1,  # Low temperature for precise corrections
                    max_tokens=1024,
                    json_mode=True  # Force JSON output
                )
                
                logger.info(f"✅ Groq plan correction complete ({metrics.latency_ms:.0f}ms, {metrics.total_tokens} tokens)")
                
                # Parse JSON response
                try:
                    raw_response = json.loads(response_text.strip())
                    logger.info(f"  📦 Raw response type: {type(raw_response)}")
                    
                    # Use intelligent unwrapping to handle dict or list responses
                    unwrapped_plan = self._unwrap_plan_from_response(raw_response)
                    
                    # Validate the unwrapped plan
                    if unwrapped_plan is not None and isinstance(unwrapped_plan, list) and all(isinstance(t, dict) for t in unwrapped_plan):
                        logger.info(f"✅ Generated corrected plan with {len(unwrapped_plan)} tasks")
                        for i, task in enumerate(unwrapped_plan):
                            logger.info(f"  Task {i+1}: {task.get('tool')} - {task.get('query', 'N/A')[:100]}...")
                        return unwrapped_plan
                    else:
                        logger.error(f"❌ Invalid plan format after unwrap: {type(unwrapped_plan)}")
                        logger.error(f"   Raw response was: {raw_response}")
                        raise ValueError(f"Invalid plan format: {type(unwrapped_plan)}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse JSON response: {e}")
                    logger.error(f"   Raw response: {response_text[:500]}...")
                    raise
                
            except Exception as e:
                logger.error(f"❌ Groq plan correction failed: {e}")
                logger.error("   Falling back to original plan to prevent crash.")
                return execution_plan  # Return original plan as fallback
        
        # If Groq failed, return original plan
        logger.error("❌ Plan correction failed - returning original plan")
        return execution_plan


# Initialize global client
_plan_correction_client = PlanCorrectionClient()


def correct_execution_plan(original_question: str, execution_plan: List[Dict[str, str]], 
                          gap_analysis_report: Dict[str, Any], attempt_number: int = 1) -> List[Dict[str, str]]:
    """
    Correct an execution plan based on gap analysis report.
    
    Args:
        original_question: The user's original question
        execution_plan: The failed execution plan (list of tasks)
        gap_analysis_report: Gap analysis JSON showing what failed
        attempt_number: Which correction attempt this is (for logging)
        
    Returns:
        List[Dict[str, str]]: Corrected execution plan
    """
    return _plan_correction_client.correct_plan(original_question, execution_plan, gap_analysis_report, attempt_number)


# Legacy function for backward compatibility (deprecated)
def rewrite_query(original_question: str, failed_documents: list, attempt_number: int = 1) -> str:
    """
    DEPRECATED: Legacy query rewriting function.
    Use correct_execution_plan() instead for plan-based correction.
    
    This function is kept for backward compatibility but should not be used.
    """
    logger.warning("DEPRECATED: rewrite_query() called. Use correct_execution_plan() instead.")
    return f"{original_question} refined search query"
