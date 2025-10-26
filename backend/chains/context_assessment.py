"""
Enhanced Gap Analysis Context Assessment Chain

This module implements a sophisticated context assessment mechanism using structured
Gap Analysis to evaluate whether retrieved documents are sufficient to answer the
user's question. It's part of the self-correcting query rewriting loop that makes
the RAG pipeline resilient to failed document retrievals.

The assessment uses a multi-step analytical framework:
1. Question Decomposition - Breaking down complex queries into components
2. Content Coverage Analysis - Assessing what percentage of each component is covered
3. Gap Identification - Distinguishing between minor and major information gaps
4. Sufficiency Decision - Using a 70%/50% threshold with nuanced criteria

This approach reduces unnecessary query rewrites by being more tolerant of partial
but useful information, while still triggering rewrites when gaps are truly critical.
"""

import os
import logging
import json
import re
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

# Enhanced Gap Analysis Prompt for Context Assessment
context_assessment_prompt_text = """You are an expert document analyst performing a structured Gap Analysis to determine context sufficiency. Your task is to systematically evaluate whether the retrieved documents can adequately answer the user's question.

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
        
        # Enhanced debugging for structured response
        logger.info(f"Raw LLM Gap Analysis result length: {len(result)} characters")
        logger.info(f"Gap Analysis preview: '{result[:200]}...'")
        
        # ROBUST JSON EXTRACTION AND PARSING
        final_assessment = None
        gap_analysis_data = None
        
        try:
            # Method 1: Extract JSON from markdown code fences
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, result, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                json_str = json_match.group(1)
                logger.info("✅ Found JSON in markdown code fence")
            else:
                # Method 2: Look for JSON object pattern without code fences
                json_pattern = r'\{[^{}]*"final_decision"[^{}]*\}'
                json_match = re.search(json_pattern, result, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(0)
                    logger.info("✅ Found JSON object pattern")
                else:
                    # Method 3: Extract the largest JSON-like block
                    brace_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
                    json_matches = re.findall(brace_pattern, result, re.DOTALL)
                    
                    if json_matches:
                        # Find the match that contains "final_decision"
                        for match in json_matches:
                            if "final_decision" in match:
                                json_str = match
                                logger.info("✅ Found JSON in brace pattern match")
                                break
                        else:
                            # Use the largest JSON-like block
                            json_str = max(json_matches, key=len)
                            logger.info("✅ Using largest JSON-like block")
                    else:
                        raise ValueError("No JSON pattern found in response")
            
            # Parse the extracted JSON
            gap_analysis_data = json.loads(json_str)
            final_assessment = gap_analysis_data.get("final_decision", "").lower().strip()
            
            if final_assessment in ["sufficient", "insufficient"]:
                logger.info(f"✅ Successfully parsed JSON Gap Analysis: '{final_assessment}'")
                logger.info(f"   Overall coverage: {gap_analysis_data.get('overall_coverage_percentage', 'N/A')}%")
                logger.info(f"   Components analyzed: {len(gap_analysis_data.get('question_components', []))}")
                logger.info(f"   Gaps identified: {len(gap_analysis_data.get('identified_gaps', []))}")
            else:
                raise ValueError(f"Invalid final_decision value: '{final_assessment}'")
                
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.warning(f"JSON parsing failed: {e}")
            logger.info("Falling back to string-based parsing...")
            
            # FALLBACK: String-based parsing (original logic as backup)
            assessment_lower = result.strip().lower()
            
            # Look for structured decision patterns first
            if "final_decision" in assessment_lower:
                if '"sufficient"' in assessment_lower or "'sufficient'" in assessment_lower:
                    final_assessment = "sufficient"
                    logger.info("✅ Fallback: Found 'sufficient' in final_decision field")
                elif '"insufficient"' in assessment_lower or "'insufficient'" in assessment_lower:
                    final_assessment = "insufficient"
                    logger.info("❌ Fallback: Found 'insufficient' in final_decision field")
            
            # Secondary fallback: simple string matching
            if not final_assessment:
                if "sufficient" in assessment_lower and "insufficient" not in assessment_lower:
                    final_assessment = "sufficient"
                    logger.info("✅ Fallback: Contains 'sufficient' only")
                elif "insufficient" in assessment_lower:
                    final_assessment = "insufficient"
                    logger.info("❌ Fallback: Contains 'insufficient'")
                else:
                    # Conservative default to reduce unnecessary rewrites
                    final_assessment = "sufficient"
                    logger.warning("⚠️ Fallback: Unclear result - defaulting to 'sufficient'")
        
        # Ensure we have a valid result
        if final_assessment not in ["sufficient", "insufficient"]:
            logger.error(f"Invalid final assessment: '{final_assessment}' - defaulting to 'sufficient'")
            final_assessment = "sufficient"
        
        logger.info(f"Final context assessment result: {final_assessment}")
        return final_assessment
        
    except Exception as e:
        logger.error(f"Error during context assessment: {e}")
        # Default to insufficient on error to trigger rewrite
        return "insufficient"
