# Optional: Even Faster Model (bge-small)

If 8-12 seconds is still too slow, you can use the smallest BGE model.

## Change Needed

File: `backend/document_processor.py` (line 62)

**Change from**:
```python
model_name="BAAI/bge-base-en-v1.5",
```

**To**:
```python
model_name="BAAI/bge-small-en-v1.5",
```

## Performance

- **Ingestion**: ~5-8 seconds (CPU) or 1-2 seconds (GPU)
- **Query**: ~5-8 seconds (CPU) or 1-2 seconds (GPU)
- **Accuracy**: ~90% of large model (still good for most use cases)

## When to Use

- You need the absolute fastest response times
- Slight accuracy trade-off is acceptable
- You're working with straightforward queries

## Comparison Table

| Model | CPU Speed | GPU Speed | Accuracy | Recommendation |
|-------|-----------|-----------|----------|----------------|
| bge-large | ❌ Very Slow | Fast | 100% | Don't use on CPU |
| **bge-base** | ✅ Good | Very Fast | 95% | **RECOMMENDED** |
| bge-small | ✅✅ Fast | Very Fast | 90% | If speed critical |
| all-mpnet-base-v2 | ✅ Good | N/A | 92% | Safe fallback |

Choose bge-base for the best balance!
