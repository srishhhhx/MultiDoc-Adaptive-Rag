# Performance Optimization Fixes

## Problem Statement
After implementing retrieval optimizations, latency has significantly increased:
- **Document ingestion**: ~35 seconds (was ~5-10 seconds)
- **Query response**: 40-45 seconds (was ~5-10 seconds)

## Root Cause
The primary culprit is the **embedding model upgrade** from `all-mpnet-base-v2` to `BAAI/bge-large-en-v1.5`:
- Model size: 420MB → 1.34GB (3x larger)
- Encoding speed on CPU: ~100 docs/sec → ~15-20 docs/sec (5-6x slower)

Secondary factors:
- HNSW index creation is ~20-30% slower than flat index
- BM25 index creation adds small overhead
- Hybrid search runs both FAISS + BM25 (though this is minor ~50-100ms)

---

## Solution 1: Use Smaller BGE Model (RECOMMENDED)

### Implementation
Use `BAAI/bge-base-en-v1.5` or `BAAI/bge-small-en-v1.5` instead of the large variant.

**Option A: bge-base-en-v1.5** (Best balance)
- Size: ~420MB (same as old model)
- Speed: ~50-70 docs/sec (2x faster than large)
- Accuracy: 95% of large model's performance
- **Best for most use cases**

**Option B: bge-small-en-v1.5** (Fastest)
- Size: ~130MB
- Speed: ~150 docs/sec (faster than original)
- Accuracy: ~90% of large model
- **Best if speed is critical**

### Changes Needed
File: `backend/document_processor.py` (line 36)

**For bge-base:**
```python
self.embedding_function = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",  # Changed from bge-large
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

**For bge-small:**
```python
self.embedding_function = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # Changed from bge-large
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

### Expected Results
- **Ingestion**: 5-10 seconds (back to baseline)
- **Query**: 5-10 seconds (back to baseline)
- **Accuracy**: Minimal loss (3-5% vs large model)

---

## Solution 2: Revert to Original Model (SAFEST)

If you need immediate stability, revert to the original embedding model.

### Changes Needed
File: `backend/document_processor.py` (line 36)

```python
self.embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Reverted
    model_kwargs={"device": "cpu"},
)
```

### Expected Results
- **Ingestion**: Back to original speed
- **Query**: Back to original speed
- **Trade-off**: Lose some accuracy improvement from BGE model

---

## Solution 3: GPU Acceleration (HIGH IMPACT)

If you have a GPU available (NVIDIA, Apple Silicon), use it for embeddings.

### Implementation

**For NVIDIA GPU:**
```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

self.embedding_function = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
```

**For Apple Silicon (MPS):**
```python
import torch

# Check for MPS (Metal Performance Shaders) availability
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

self.embedding_function = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
```

### Expected Results
- **Speedup**: 10-50x faster than CPU (GPU dependent)
- **Ingestion**: 2-5 seconds
- **Query**: 2-5 seconds
- **Keeps**: All accuracy benefits

---

## Solution 4: Optimize HNSW Index Parameters (MINOR IMPACT)

Reduce HNSW construction complexity for faster indexing.

### Changes Needed
File: `backend/document_processor.py` (line 531-532)

**Current:**
```python
index = faiss.IndexHNSWFlat(dimension, 32)
index.hnsw.efConstruction = 40
```

**Optimized:**
```python
index = faiss.IndexHNSWFlat(dimension, 16)  # Reduced from 32
index.hnsw.efConstruction = 20  # Reduced from 40
```

### Expected Results
- **Ingestion**: ~20% faster index build
- **Query**: Minimal impact on search quality
- **Trade-off**: Slightly lower recall (~97% vs 99%)

---

## Solution 5: Make Hybrid Search Optional (MINOR IMPACT)

Skip BM25 for small document sets to reduce overhead.

### Implementation
File: `backend/document_processor.py`

Add conditional BM25 creation:

```python
# In process_file_for_session() around line 743
# Only create BM25 if document count exceeds threshold
HYBRID_SEARCH_THRESHOLD = 10  # Only use hybrid for 10+ documents

if len(all_chunks) >= HYBRID_SEARCH_THRESHOLD:
    logger.info("🔄 Creating/updating BM25 index for hybrid search...")
    bm25_index, tokenized_docs = self._create_bm25_index(all_chunks)
    if bm25_index:
        self._save_bm25_index(collection_name, bm25_index, tokenized_docs)
        logger.info("✅ BM25 index created/updated successfully")
else:
    logger.info(f"⚠️  Skipping BM25 (only {len(all_chunks)} chunks, threshold: {HYBRID_SEARCH_THRESHOLD})")
```

### Expected Results
- **Ingestion**: Saves ~1-2 seconds for small documents
- **Query**: Minimal impact (BM25 search is already fast)

---

## Solution 6: Batch Processing Optimization (MODERATE IMPACT)

Process embeddings in larger batches.

### Implementation
File: `backend/document_processor.py`

Modify embedding creation to use batching:

```python
# When creating embeddings, use batch encoding
from langchain_huggingface import HuggingFaceEmbeddings

self.embedding_function = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={
        "device": "cpu",
        "batch_size": 32  # Process 32 chunks at once
    },
    encode_kwargs={"normalize_embeddings": True}
)
```

### Expected Results
- **Ingestion**: 10-20% faster
- **Query**: No change (single query at a time)

---

## Solution 7: Model Caching (ONE-TIME COST)

Ensure model is cached and doesn't re-download.

### Implementation
Models are cached by default in `~/.cache/huggingface/`, but you can verify:

```python
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/persistent/cache'
```

### Expected Results
- **First run**: Download time (one-time)
- **Subsequent runs**: No download overhead

---

## Solution 8: Use Flat Index for Small Datasets (CONDITIONAL)

Only use HNSW for large document collections.

### Implementation
File: `backend/document_processor.py`

Add conditional index selection:

```python
def _create_vector_database(self, doc_splits: List, collection_name: str):
    # ... existing code ...

    HNSW_THRESHOLD = 1000  # Use HNSW only if >1000 chunks

    if len(doc_splits) >= HNSW_THRESHOLD:
        # Use HNSW for large datasets
        vectorstore = self._create_hnsw_index(doc_splits)
        logger.info(f"✅ Created HNSW index (large dataset: {len(doc_splits)} chunks)")
    else:
        # Use flat index for small datasets
        vectorstore = FAISS.from_documents(doc_splits, self.embedding_function)
        logger.info(f"✅ Created flat index (small dataset: {len(doc_splits)} chunks)")
```

### Expected Results
- **Ingestion**: 20-30% faster for small documents
- **Query**: Negligible difference for small datasets

---

## Recommended Solution Stack

### For Immediate Relief (Choose ONE):

**Option A: Production Balance** (RECOMMENDED)
```python
# Use bge-base instead of bge-large
model_name="BAAI/bge-base-en-v1.5"
```
- **Ingestion**: ~8-12 seconds ✅
- **Query**: ~8-12 seconds ✅
- **Accuracy**: 95% of large model ✅

**Option B: Maximum Speed**
```python
# Revert to original model
model_name="sentence-transformers/all-mpnet-base-v2"
```
- **Ingestion**: ~5-10 seconds ✅
- **Query**: ~5-10 seconds ✅
- **Accuracy**: Baseline (still good) ✅

**Option C: Best Quality (if you have GPU)**
```python
# Keep bge-large but use GPU
model_name="BAAI/bge-large-en-v1.5"
model_kwargs={"device": "cuda"}  # or "mps" for Apple Silicon
```
- **Ingestion**: ~2-5 seconds ✅✅
- **Query**: ~2-5 seconds ✅✅
- **Accuracy**: Best possible ✅✅

### For Long-term Optimization (Combine):

1. **Use bge-base model** (primary fix)
2. **Add GPU support if available** (10-50x speedup)
3. **Conditional HNSW** (saves time for small docs)
4. **Batch processing** (10-20% improvement)

---

## Performance Comparison Table

| Configuration | Ingestion Time | Query Time | Accuracy | Recommendation |
|---------------|----------------|------------|----------|----------------|
| **Current (bge-large CPU)** | 35s ❌ | 40-45s ❌ | 100% ✅ | Too slow |
| **bge-base CPU** | 8-12s ✅ | 8-12s ✅ | 95% ✅ | **BEST** |
| **bge-small CPU** | 5-8s ✅✅ | 5-8s ✅✅ | 90% ⚠️ | Good if speed critical |
| **Original (mpnet)** | 5-10s ✅✅ | 5-10s ✅✅ | 92% ✅ | Safe fallback |
| **bge-large GPU** | 2-5s ✅✅✅ | 2-5s ✅✅✅ | 100% ✅ | **IDEAL** |
| **bge-base GPU** | 1-3s ✅✅✅ | 1-3s ✅✅✅ | 95% ✅ | Excellent |

---

## Testing the Fix

After applying any solution, test with:

```bash
# 1. Upload a test document and check logs
# Look for embedding time in logs

# 2. Ask a test question and measure response time
# Should be back to <10 seconds

# 3. Compare results quality
# Ensure answers are still accurate
```

---

## My Recommendation

**For your immediate use case**, I recommend **Solution 1 (bge-base)** because:

1. ✅ **Solves the latency problem** - back to ~8-12 seconds
2. ✅ **Minimal accuracy loss** - only 3-5% vs large model
3. ✅ **Simple change** - just change one line
4. ✅ **No infrastructure changes** - works on existing CPU setup
5. ✅ **Keeps all other optimizations** - HNSW, BM25, hybrid search still work

**If you have GPU access**, use **Solution 3** with bge-large for best of both worlds.

---

## Implementation Priority

1. **Immediate** (do now): Switch to `bge-base-en-v1.5`
2. **Next** (if available): Add GPU support
3. **Later** (optimization): Add conditional HNSW/BM25
4. **Future** (nice-to-have): Batch processing optimization

Let me know which solution you'd like to implement and I can make the code changes!
