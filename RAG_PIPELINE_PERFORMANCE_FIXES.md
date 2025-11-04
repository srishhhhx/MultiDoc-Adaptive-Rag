# RAG Pipeline Performance Optimization Guide

## Current State
- **Simple queries**: 30 seconds (Target: 3-8s) ❌
- **Complex queries**: 45-60 seconds (Target: 8-15s) ❌
- **Maximum acceptable**: 20 seconds

## Root Cause: Pipeline Bottlenecks

### Latency Breakdown (Typical Complex Query)

| Stage | Time | Percentage | Bottleneck? |
|-------|------|------------|-------------|
| 1. Query Analysis (Gemini) | 3-5s | 8% | Minor |
| 2. Hybrid Search (FAISS+BM25) | 1-3s | 5% | ✅ Fast |
| 3. **Reranking (Cross-encoder)** | 3-7s | 15% | ⚠️ **HIGH** |
| 4. Document Evaluation (Groq) | 2-4s | 8% | Minor |
| 5. Context Assessment (Groq) | 2-3s | 6% | Minor |
| 6. **Rewrite Loop (if triggered)** | 10-15s | 30% | ⚠️ **HIGH** |
| 7. Answer Generation (Gemini) | 4-8s | 15% | Moderate |
| 8. Hallucination Check (Groq) | 2-3s | 6% | Minor |
| 9. **Web Search (if triggered)** | 5-10s | 15% | ⚠️ **HIGH** |
| **Total (with loops)** | **32-58s** | | ❌ |

**Top 3 Bottlenecks:**
1. **Self-correction loop** (rewrite attempts) - adds 20-30s
2. **Cross-encoder reranking** - 3-7s per rerank
3. **Web search fallback** - adds 5-10s

---

## Solution 1: Disable/Optimize Self-Correction Loop (HIGHEST IMPACT)

The self-correction loop can retry queries up to 2 times, adding 10-15s per retry.

### Option A: Disable Self-Correction (Fastest)

**Impact**: Reduces latency by 20-30s for queries that would trigger rewrites

**File**: `backend/rag_workflow.py`

Change the conditional routing to skip context assessment:

```python
# Around line 200 - Find this block
workflow.add_edge("Evaluate Documents", "Context Assessment")

# REPLACE WITH (skip assessment, go straight to generation):
workflow.add_edge("Evaluate Documents", "Generate Answer")

# COMMENT OUT the conditional routing from Context Assessment:
# workflow.add_conditional_edges(
#     "Context Assessment",
#     self._route_after_assessment,
#     {...},
# )
```

**Trade-off**: May get slightly lower quality answers when context is insufficient

### Option B: Reduce Max Rewrite Attempts

**Impact**: Reduces latency by 10-15s (allows only 1 rewrite instead of 2)

**File**: `backend/state.py` or `backend/rag_workflow.py`

Find the max attempts check and change from 2 to 1:

```python
# Look for: MAX_REWRITE_ATTEMPTS = 2
MAX_REWRITE_ATTEMPTS = 1  # Change to 1
```

### Option C: Make Context Assessment Faster

**Impact**: Saves 1-2s per assessment

**File**: `backend/chains/context_assessment_groq.py`

Use a simpler prompt or increase temperature for faster responses.

---

## Solution 2: Optimize Reranking (HIGH IMPACT)

Cross-encoder reranking is slow (3-7s). Multiple options:

### Option A: Disable Reranking for Small Result Sets

**File**: `backend/chains/multi_tool_executor.py` (around line 159-165)

```python
def _process_task(self, task: Dict[str, str], metadata: Optional[Dict[str, Any]] = None):
    # ... existing code ...

    # Step 2: Rerank ONLY if we have many documents
    RERANK_THRESHOLD = 10  # Only rerank if we have 10+ docs

    if len(initial_docs) >= RERANK_THRESHOLD:
        logger.info(f"   🔄 Reranking {len(initial_docs)} docs (top_k={self.RERANK_TOP_K})")
        reranked_docs = rerank_documents(
            query=query,
            documents=initial_docs,
            top_k=self.RERANK_TOP_K
        )
    else:
        logger.info(f"   ⚡ Skipping rerank (only {len(initial_docs)} docs)")
        reranked_docs = initial_docs[:self.RERANK_TOP_K]  # Just take top-k

    return {"type": "docs", "data": reranked_docs}
```

**Impact**: Saves 3-7s when retrieval returns <10 documents

### Option B: Use Lighter Reranking Model

**File**: `backend/chains/rerank_documents.py`

Replace the reranking model with a faster one:

```python
# Current (line ~30):
model = CrossEncoder("BAAI/bge-reranker-base", max_length=512, device=device)

# REPLACE WITH faster model:
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device=device)
```

**Impact**: 2-3x faster reranking (2-3s instead of 3-7s)
**Trade-off**: Slightly lower reranking quality (~3-5%)

### Option C: Disable Reranking Entirely

**File**: `backend/chains/multi_tool_executor.py` (around line 159)

```python
# Comment out reranking entirely
# reranked_docs = rerank_documents(query=query, documents=initial_docs, top_k=self.RERANK_TOP_K)

# Just return top-k from hybrid search
reranked_docs = initial_docs[:self.RERANK_TOP_K]
```

**Impact**: Saves 3-7s
**Trade-off**: Lower precision (but hybrid search already helps)

---

## Solution 3: Optimize LLM Calls (MODERATE IMPACT)

### Option A: Use Streaming for Answer Generation

**File**: `backend/chains/generate_answer.py`

Enable streaming to reduce perceived latency:

```python
# Change LLM initialization to support streaming
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3,
    streaming=True  # Add this
)
```

**Impact**: User sees tokens immediately, perceived latency drops to <2s

### Option B: Reduce Temperature for Faster Responses

**File**: All chain files with LLM calls

```python
# Lower temperature = faster generation
temperature=0  # Instead of 0.3 or 0.5
```

**Impact**: Saves 0.5-1s per LLM call (5-10% faster)

### Option C: Use Smaller/Faster Models

**Current**: Gemini Pro, Groq Llama3-8B

**Option**: Switch Gemini to Gemini Flash (3-5x faster)

```python
# In all files using ChatGoogleGenerativeAI
model="gemini-1.5-flash"  # Instead of "gemini-pro"
```

**Impact**: Saves 2-4s per Gemini call (40-60% faster)
**Trade-off**: Slightly lower reasoning quality

---

## Solution 4: Reduce Retrieval Size (MINOR IMPACT)

### Option: Retrieve Fewer Documents

**File**: `backend/chains/multi_tool_executor.py` and `backend/document_processor.py`

```python
# Change k from 20 to 10
k=10  # Instead of 20

# In retriever creation:
search_kwargs={"k": 10}  # Instead of 20
```

**Impact**: Saves 1-2s (faster retrieval + reranking)
**Trade-off**: May miss relevant docs in large collections

---

## Solution 5: Optimize Query Analysis (MINOR IMPACT)

### Option: Simplify Query Analysis

**File**: `backend/chains/query_analysis_router.py`

Simplify the prompt to reduce tokens and processing time:

```python
# Use a shorter, more direct prompt
# Reduce examples in few-shot prompting
# Remove verbose instructions
```

**Impact**: Saves 1-2s

---

## Solution 6: Cache Expensive Operations (MODERATE IMPACT)

### Option A: Cache Reranking Results

Add simple caching for identical queries:

```python
# In rerank_documents.py
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def _cached_rerank(query_hash, doc_contents_hash):
    # Actual reranking logic
    pass

def rerank_documents(query, documents, top_k):
    # Create cache keys
    query_hash = hashlib.md5(query.encode()).hexdigest()
    doc_hash = hashlib.md5(str([d.page_content for d in documents]).encode()).hexdigest()

    # Check cache
    return _cached_rerank(query_hash, doc_hash)
```

**Impact**: Instant response for repeated queries

### Option B: Cache Context Signatures

The system already has context caching - ensure it's working:

```python
# Verify cache hits in logs
logger.info(f"Cache hit rate: {hits}/{total}")
```

---

## Solution 7: Parallel Execution (MODERATE IMPACT)

### Option: Run Independent Steps in Parallel

Some steps can run concurrently:

```python
import asyncio

async def parallel_workflow():
    # Run document evaluation and context assessment in parallel
    results = await asyncio.gather(
        evaluate_documents_async(docs),
        assess_context_async(context),
    )
```

**Impact**: Saves 2-4s by parallelizing independent operations

---

## Recommended Solution Stack (Prioritized)

### **Immediate Fixes** (Implement These First)

#### 1. **Disable Self-Correction Loop** ⚡ HIGH IMPACT
```python
# backend/rag_workflow.py (line ~200)
# Change from:
workflow.add_edge("Evaluate Documents", "Context Assessment")

# To:
workflow.add_edge("Evaluate Documents", "Generate Answer")
```

**Expected Savings**: **20-30 seconds**
**New Total**: 10-30 seconds ✅

#### 2. **Skip Reranking for Small Result Sets** ⚡ HIGH IMPACT
```python
# backend/chains/multi_tool_executor.py (line ~159)
if len(initial_docs) >= 10:  # Only rerank if 10+ docs
    reranked_docs = rerank_documents(query, initial_docs, self.RERANK_TOP_K)
else:
    reranked_docs = initial_docs[:self.RERANK_TOP_K]
```

**Expected Savings**: **3-7 seconds** (conditional)
**New Total**: 7-23 seconds ✅

#### 3. **Switch to Gemini Flash** ⚡ MODERATE IMPACT
```python
# All files using ChatGoogleGenerativeAI
model="gemini-1.5-flash"  # Faster than gemini-pro
```

**Expected Savings**: **4-8 seconds**
**New Total**: 3-15 seconds ✅✅

### **Additional Optimizations** (If Still Needed)

#### 4. Use Lighter Reranking Model
```python
# backend/chains/rerank_documents.py
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", ...)
```

**Expected Savings**: **2-4 seconds**

#### 5. Reduce Max Rewrite Attempts (if keeping self-correction)
```python
MAX_REWRITE_ATTEMPTS = 1  # Instead of 2
```

**Expected Savings**: **10-15 seconds** (if triggered)

#### 6. Enable Streaming
```python
# backend/chains/generate_answer.py
streaming=True
```

**Expected Savings**: Perceived latency reduced to <2s

---

## Final Performance Targets

| Implementation | Simple Query | Complex Query | Max |
|----------------|--------------|---------------|-----|
| **Current** | 30s ❌ | 45-60s ❌ | 60s+ |
| **After Fix 1+2** | 7-10s ✅ | 12-23s ⚠️ | 30s |
| **After Fix 1+2+3** | 3-6s ✅✅ | 8-15s ✅ | 20s ✅ |
| **With All Optimizations** | 2-5s ✅✅ | 5-12s ✅✅ | 15s ✅✅ |

---

## Trade-offs Summary

| Fix | Latency Reduction | Accuracy Impact | Recommended? |
|-----|-------------------|-----------------|--------------|
| Disable self-correction | -20-30s | -5-10% | ✅ YES |
| Skip reranking (conditional) | -3-7s | -2-3% | ✅ YES |
| Gemini Flash | -4-8s | -3-5% | ✅ YES |
| Lighter reranking model | -2-4s | -3-5% | ⚠️ Optional |
| Reduce retrieval size | -1-2s | -2-4% | ⚠️ Optional |
| Streaming | Perceived -6s | 0% | ✅ YES |

---

## Implementation Order

1. **Start here**: Disable self-correction (biggest impact)
2. **Next**: Switch to Gemini Flash (easy win)
3. **Then**: Skip reranking for small sets (conditional optimization)
4. **Optional**: Lighter reranking model (if still slow)
5. **Polish**: Add streaming (better UX)

---

## Monitoring & Validation

After implementing fixes, add timing logs:

```python
import time

def timed_execution(func_name):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"⏱️  {func_name} took {elapsed:.2f}s")
        return result
    return wrapper

# Apply to each pipeline stage
@timed_execution("query_analysis")
def analyze_query(state):
    # ...
```

Track metrics:
- P50 (median) latency
- P95 latency
- Timeout rate
- Accuracy (via user feedback)

---

## My Recommendation

**Start with the "Big 3" fixes:**

1. ✅ **Disable self-correction** (saves 20-30s)
2. ✅ **Use Gemini Flash** (saves 4-8s)
3. ✅ **Skip reranking conditionally** (saves 3-7s)

This will bring your latency from **30-60s down to 3-15s** with minimal accuracy loss.

Then monitor and add more optimizations if needed!
