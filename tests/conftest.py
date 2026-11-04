"""Shared test setup.

Tests must run offline: no API keys, no model downloads. We therefore
- run inside a temp working directory (the FAISS dir is a relative path),
- replace HuggingFaceEmbeddings with a deterministic hashing embedder
  *before* any backend module is imported.
"""
import hashlib
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_workdir = tempfile.mkdtemp(prefix="rag-tests-")
os.chdir(_workdir)

for var in ("CEREBRAS_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY", "ADMIN_API_KEY", "LLM_PROVIDER"):
    os.environ.pop(var, None)
os.environ["TAVILY_API_KEY"] = "test"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["PRELOAD_MODELS"] = "0"

from langchain_core.embeddings import Embeddings  # noqa: E402


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words embedding (64 dims)."""

    dim = 64
    model_name = "hash-test-embeddings"

    def __init__(self, *args, **kwargs):
        pass

    def _vec(self, text):
        v = [0.0] * self.dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


import langchain_huggingface  # noqa: E402

langchain_huggingface.HuggingFaceEmbeddings = HashEmbeddings


@pytest.fixture
def api_module():
    import backend.api as api

    return api


@pytest.fixture
def client(api_module):
    from fastapi.testclient import TestClient

    return TestClient(api_module.app)
