# Caching Evaluation for Adaptive RAG System

## 🎯 Executive Summary

**LMCache Verdict**: ❌ **NOT APPLICABLE** for your use case

**Why?** LMCache is designed for **self-hosted model inference** (vLLM/SGLang), but your system uses **API-based LLM services** (Gemini Flash, Groq API).

**Better Options**: ✅ **Semantic caching**, **Prompt caching**, and **Result caching** are more suitable and will provide real benefits.

---

## 📊 LMCache Analysis

### What is LMCache?

LMCache is a **KV (Key-Value) cache sharing system** for transformer models during inference. It:
- Caches the internal attention key-value tensors from transformer layers
- Shares these caches across multiple inference requests
- Reduces redundant computation when processing similar or repeated text

### Why It Doesn't Work for Your System

| Requirement | Your System | LMCache Needs |
|-------------|-------------|---------------|
| **Model Hosting** | API-based (Gemini, Groq) | Self-hosted (vLLM, SGLang) |
| **Infrastructure** | Serverless APIs | GPU servers with vLLM |
| **Control Level** | High-level API calls | Low-level model inference |
| **Cost Model** | Pay-per-token | GPU compute costs |

**Verdict**: LMCache operates at the **model inference layer**, which you don't control when using APIs.

---

## ✅ Caching Strategies That WILL Work

I've identified **4 caching strategies** that are applicable and beneficial for your RAG pipeline:

---

## Strategy 1: Semantic Caching for LLM Responses ⚡ HIGH IMPACT

### What It Is
Cache LLM responses based on **semantic similarity** of queries, not exact matches.

**Example**:
```
Query 1: "What are the main findings in the report?"
Query 2: "What did the report find?" ← Similar enough to use cached response
```

### Implementation Options

#### Option A: GPTCache (Recommended)
**Library**: https://github.com/zilliztech/GPTCache

**Features**:
- Semantic similarity matching using embeddings
- Multiple storage backends (Redis, SQLite, etc.)
- TTL (time-to-live) support
- Works with any LLM API

**Integration**:
```python
from gptcache import Cache
from gptcache.adapter.langchain_models import LangChainChat

# Initialize cache
cache = Cache()
cache.init(
    embedding_func=your_embedding_function,
    data_manager=your_storage,
    similarity_threshold=0.85  # 85% similarity = cache hit
)

# Wrap your LLM calls
cached_llm = LangChainChat(chat=your_gemini_llm, cache=cache)
```

**Expected Savings**:
- **Hit Rate**: 20-40% for common queries
- **Time Saved**: Full LLM call time (2-4s per call)
- **Cost Saved**: Significant (no API charges for cache hits)

#### Option B: Custom Semantic Cache
Build your own using:
- Redis for storage
- Your existing `bge-base-en-v1.5` embeddings for similarity
- Cosine similarity threshold

### Where to Apply in Your Pipeline

| Stage | Cache Key | Potential Hit Rate | Savings |
|-------|-----------|-------------------|---------|
| **Query Analysis** | Original query | 15-25% | 1-2s |
| **Context Assessment** | Query + context hash | 10-20% | 2-3s |
| **Answer Generation** | Query + context hash | 5-15% | 2-4s |
| **Query Rewriting** | Query + gap analysis | 5-10% | 1-2s |

**Total Potential**: Save 2-8 seconds on 15-30% of queries

---

## Strategy 2: Prompt Caching (Native Provider Support) ⚡ MODERATE IMPACT

### What It Is
Some LLM providers cache the **prompt prefix** (system prompts, long contexts) to reduce compute.

### Gemini Context Caching

**Status**: ✅ Available for Gemini 1.5 Flash/Pro

**How It Works**:
- Cache large, static prompt components (system prompts, documents)
- Subsequent requests with same prefix are cheaper/faster
- TTL: Up to 1 hour

**Documentation**: https://ai.google.dev/gemini-api/docs/caching

**Integration**:
```python
from google.generativeai import caching

# Cache long document context
cached_content = caching.CachedContent.create(
    model='models/gemini-1.5-flash',
    system_instruction="You are a helpful assistant...",
    contents=[large_document_context],
    ttl=3600  # 1 hour
)

# Use cached content
response = model.generate_content(
    user_query,
    cached_content=cached_content
)
```

**Expected Savings**:
- **Cost**: 75% reduction on cached tokens
- **Speed**: 10-30% faster (reduced processing)
- **Hit Rate**: Depends on query patterns

**Best For**:
- Long document contexts (>2K tokens)
- Repeated queries on same documents
- System prompts across multiple queries

### Where to Apply

| Use Case | Cacheable Component | Benefit |
|----------|-------------------|---------|
| **Answer Generation** | Retrieved document context | 75% cost reduction |
| **Context Assessment** | Document context | 75% cost reduction |
| **Evaluation** | System prompt + docs | Faster evaluation |

---

## Strategy 3: Retrieval & Embedding Caching ⚡ HIGH IMPACT

### What It Is
Cache the results of expensive retrieval operations.

### Cache Layers

#### Layer 1: Embedding Cache
**Cache**: Query embeddings to avoid re-encoding

```python
import hashlib
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_embedding(query: str):
    # Generate embedding only if not cached
    return embedding_function.embed_query(query)
```

**Expected Savings**:
- **Time**: 100-500ms per query (embedding generation)
- **Hit Rate**: 10-20% (repeated or similar queries)

#### Layer 2: Retrieval Results Cache
**Cache**: FAISS similarity search results + BM25 results

```python
import redis
import hashlib
import pickle

redis_client = redis.Redis(host='localhost', port=6379)

def get_retrieval_cache_key(query, collection_name, k):
    content = f"{query}:{collection_name}:{k}"
    return hashlib.md5(content.encode()).hexdigest()

def cached_hybrid_search(query, collection_name, k):
    cache_key = get_retrieval_cache_key(query, collection_name, k)

    # Try cache first
    cached = redis_client.get(cache_key)
    if cached:
        logger.info(f"⚡ CACHE HIT: Retrieval results for '{query[:50]}'")
        return pickle.loads(cached)

    # Cache miss - do retrieval
    results = document_processor.hybrid_search(query, collection_name, k)

    # Cache for 1 hour
    redis_client.setex(cache_key, 3600, pickle.dumps(results))

    return results
```

**Expected Savings**:
- **Time**: 1-3s per query (hybrid search)
- **Hit Rate**: 15-30% (similar queries, browsing behavior)

#### Layer 3: Reranking Cache
**Cache**: Cross-encoder reranking results

```python
def cached_rerank(query, documents, top_k):
    cache_key = hashlib.md5(
        (query + str([d.page_content for d in documents])).encode()
    ).hexdigest()

    cached = redis_client.get(f"rerank:{cache_key}")
    if cached:
        logger.info("⚡ CACHE HIT: Reranking results")
        return pickle.loads(cached)

    results = rerank_documents(query, documents, top_k)
    redis_client.setex(f"rerank:{cache_key}", 3600, pickle.dumps(results))

    return results
```

**Expected Savings**:
- **Time**: 3-7s per query (expensive reranking)
- **Hit Rate**: 5-15% (exact same retrieval results)

---

## Strategy 4: Web Search Caching ⚡ MODERATE IMPACT

### What It Is
Cache Tavily web search results for repeated queries.

### Implementation

```python
def cached_web_search(query):
    cache_key = f"websearch:{hashlib.md5(query.encode()).hexdigest()}"

    cached = redis_client.get(cache_key)
    if cached:
        logger.info(f"⚡ CACHE HIT: Web search for '{query[:50]}'")
        return pickle.loads(cached)

    results = tavily_search_tool.invoke(query)

    # Cache for 6 hours (web content doesn't change rapidly)
    redis_client.setex(cache_key, 21600, pickle.dumps(results))

    return results
```

**Expected Savings**:
- **Time**: 3-8s per query (web search API latency)
- **Cost**: Tavily API calls reduced
- **Hit Rate**: 10-25% (common queries, trending topics)

---

## 📊 Overall Performance Impact Estimation

### Without Caching (Current)

| Query Type | Avg Latency | Breakdown |
|------------|-------------|-----------|
| Simple | 8-16s | No cache benefit |
| Complex | 15-28s | No cache benefit |
| Repeated | 15-28s | **Same as first time** |

### With Comprehensive Caching

| Query Type | First Request | Cached Request | Improvement |
|------------|---------------|----------------|-------------|
| **Simple** | 8-16s | **2-5s** | **60-70% faster** ⚡ |
| **Complex** | 15-28s | **5-12s** | **50-60% faster** ⚡ |
| **Repeated exact** | 15-28s | **0.5-2s** | **90-95% faster** ⚡⚡ |

### Expected Cache Hit Rates

| Cache Type | Hit Rate | Latency Saved | Cost Saved |
|------------|----------|---------------|------------|
| Semantic (LLM responses) | 20-40% | 2-4s | High (API $) |
| Prompt caching (Gemini) | 30-50% | 0.5-1s | 75% token cost |
| Retrieval results | 15-30% | 1-3s | Moderate |
| Reranking | 5-15% | 3-7s | Low |
| Web search | 10-25% | 3-8s | High (API $) |
| **Combined Effect** | **40-60%** | **5-15s** | **Significant** |

---

## 💡 Recommended Implementation Plan

### Phase 1: Quick Wins (Week 1)

1. **Semantic Caching with GPTCache**
   - Apply to answer generation (highest value)
   - Use your existing embeddings
   - Redis backend

2. **Retrieval Results Cache**
   - Simple Redis cache for hybrid search
   - 1-hour TTL

**Expected Impact**: 20-30% queries become 40-60% faster

### Phase 2: Native Optimizations (Week 2)

3. **Gemini Context Caching**
   - Cache long document contexts
   - Apply to answer generation & assessment

4. **Web Search Cache**
   - Redis-based Tavily cache
   - 6-hour TTL

**Expected Impact**: Additional 10-20% improvement

### Phase 3: Advanced Optimizations (Week 3)

5. **Reranking Cache**
   - Cache expensive cross-encoder results

6. **Embedding Cache**
   - LRU cache for query embeddings

**Expected Impact**: Final 5-10% improvement

---

## 🛠️ Implementation Example: Semantic Caching

### Option 1: Using GPTCache

```python
# Install
# pip install gptcache

from gptcache import Cache
from gptcache.manager import get_data_manager, CacheBase
from gptcache.similarity_evaluation.simple import SearchDistanceEvaluation
from gptcache.adapter import openai  # Works with other LLMs too

# Initialize cache
cache_base = CacheBase('sqlite')
data_manager = get_data_manager(cache_base, max_size=1000)

cache = Cache()
cache.init(
    embedding_func=lambda x: your_embedding_function.embed_query(x),
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(threshold=0.85),
)

# Wrap LLM calls
def cached_llm_call(prompt):
    cache_key = cache.get_cache_key(prompt)
    cached_response = cache.get(cache_key)

    if cached_response:
        logger.info("⚡ SEMANTIC CACHE HIT")
        return cached_response

    # Cache miss - call LLM
    response = llm.invoke(prompt)
    cache.set(cache_key, response)

    return response
```

### Option 2: Custom Redis-Based Semantic Cache

```python
import redis
import numpy as np
import pickle
from typing import Optional

class SemanticCache:
    def __init__(self, embedding_function, similarity_threshold=0.85):
        self.redis_client = redis.Redis(host='localhost', port=6379)
        self.embedding_function = embedding_function
        self.similarity_threshold = similarity_threshold

    def _get_cache_key(self, embedding):
        # Store as hash of embedding
        return hashlib.md5(embedding.tobytes()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        # Get query embedding
        query_embedding = np.array(self.embedding_function.embed_query(query))

        # Search all cached embeddings for similar ones
        # In production, use a vector database (FAISS, Milvus) for efficiency
        for key in self.redis_client.scan_iter("cache:*"):
            cached_data = pickle.loads(self.redis_client.get(key))
            cached_embedding = cached_data['embedding']

            # Calculate cosine similarity
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )

            if similarity >= self.similarity_threshold:
                logger.info(f"⚡ SEMANTIC CACHE HIT (similarity: {similarity:.2f})")
                return cached_data['response']

        return None

    def set(self, query: str, response: str, ttl: int = 3600):
        query_embedding = np.array(self.embedding_function.embed_query(query))
        cache_key = f"cache:{self._get_cache_key(query_embedding)}"

        self.redis_client.setex(
            cache_key,
            ttl,
            pickle.dumps({
                'embedding': query_embedding,
                'query': query,
                'response': response
            })
        )

# Usage
semantic_cache = SemanticCache(embedding_function)

def cached_answer_generation(query, context):
    cache_key = f"{query}:{hash(context)}"

    cached = semantic_cache.get(cache_key)
    if cached:
        return cached

    answer = generate_answer(query, context)
    semantic_cache.set(cache_key, answer)

    return answer
```

---

## 📈 Cost-Benefit Analysis

### Development Effort

| Strategy | Complexity | Dev Time | Maintenance |
|----------|-----------|----------|-------------|
| GPTCache semantic | Low | 4-8 hours | Low |
| Gemini prompt caching | Very Low | 2-4 hours | Very Low |
| Retrieval cache (Redis) | Low | 4-6 hours | Low |
| Web search cache | Very Low | 2-3 hours | Very Low |
| Reranking cache | Low | 3-5 hours | Low |
| **Total** | **Low-Medium** | **15-26 hours** | **Low** |

### Infrastructure Costs

| Component | Monthly Cost | Notes |
|-----------|--------------|-------|
| Redis (managed) | $10-50 | Depends on size (ElastiCache, Redis Cloud) |
| Redis (self-hosted) | $0-10 | RAM + server costs |
| Storage for cache | $1-5 | Minimal |
| **Total** | **$11-65/month** | Very low |

### ROI Calculation

**Assumptions**:
- 1000 queries/day
- 40% cache hit rate
- Average 5s saved per cache hit
- API cost savings: $0.002 per cached query

**Savings**:
- **Time saved**: 400 queries × 5s = 2000s/day (33 min)
- **Cost saved**: 400 × $0.002 = $0.80/day ≈ **$24/month**
- **User experience**: 40% of queries are much faster

**ROI**: Positive within first month (saves more than infrastructure cost)

---

## ⚠️ Considerations & Limitations

### Cache Invalidation

**Challenge**: When to invalidate cached results?

**Solutions**:
1. **TTL-based**: Set reasonable expiration times
   - LLM responses: 1-6 hours
   - Retrieval results: 1 hour
   - Web search: 6-24 hours

2. **Event-based**: Invalidate when documents change
   ```python
   def on_document_upload(session_id):
       # Clear retrieval cache for this session
       redis_client.delete(f"retrieval:{session_id}:*")
   ```

3. **Hybrid**: Combine TTL + manual invalidation

### Cache Warming

**Strategy**: Pre-populate cache with common queries

```python
common_queries = [
    "What is the main topic?",
    "Summarize the key findings",
    "What are the conclusions?"
]

for query in common_queries:
    # Generate and cache responses
    response = rag_pipeline.process(query)
```

### Memory Management

**Monitor**:
- Cache hit rate
- Cache size
- Memory usage
- Eviction rate

**Tools**:
- Redis INFO command
- Monitoring dashboards (Redis Insight, Grafana)

---

## 🎯 Final Recommendations

### For Your Adaptive RAG System

1. **Start with Semantic Caching** (Highest ROI)
   - Use GPTCache for answer generation
   - Expected: 20-40% queries become 60-70% faster
   - Low complexity, high impact

2. **Add Retrieval Caching** (Quick Win)
   - Simple Redis cache for hybrid search results
   - Expected: 15-30% additional cache hits
   - Very easy to implement

3. **Enable Gemini Context Caching** (Free Performance)
   - Use native Gemini feature for long contexts
   - Expected: 10-30% faster + 75% cost reduction
   - Minimal code changes

4. **Optional: Web Search Caching** (Cost Savings)
   - Cache Tavily results for common queries
   - Expected: Reduce Tavily API costs by 10-25%
   - Easy to add

### Expected Combined Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Avg latency (cached)** | 15-28s | **5-12s** | **60-70% faster** ⚡ |
| **P50 latency** | 18s | **10s** | **45% faster** |
| **P95 latency** | 35s | **20s** | **43% faster** |
| **API costs** | Baseline | **-40-60%** | Significant savings 💰 |
| **Cache hit rate** | 0% | **40-60%** | Major impact |

---

## 🚀 Next Steps

1. **Set up Redis** (if not already)
   ```bash
   docker run -d -p 6379:6379 redis:latest
   ```

2. **Implement Semantic Caching** (Phase 1)
   - Install GPTCache: `pip install gptcache`
   - Wrap answer generation

3. **Add Retrieval Cache** (Phase 1)
   - Simple Redis cache with 1-hour TTL

4. **Enable Gemini Context Caching** (Phase 2)
   - Update Gemini API calls

5. **Monitor & Tune** (Ongoing)
   - Track hit rates
   - Adjust thresholds
   - Optimize TTLs

---

## ✅ Conclusion

**LMCache**: ❌ Not applicable (requires self-hosted models)

**Better Options**: ✅ Semantic caching, prompt caching, retrieval caching

**Expected Impact**:
- **60-70% faster** for 40-60% of queries
- **40-60% cost reduction** on API calls
- **Low complexity** (~20 hours dev time)
- **Positive ROI** within first month

**Recommendation**: Implement semantic + retrieval caching first (highest ROI), then add others incrementally.

Your adaptive RAG system will benefit significantly from caching! 🎉
