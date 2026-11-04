"""
Inference Clients Module

This module provides unified interfaces for different LLM inference providers:
- Groq API (fast inference for evaluation tasks)
- Cerebras (high-quality generation, see backend/llm.py)
- Local models (privacy-sensitive tasks)
"""

from .groq_client import GroqModelClient

__all__ = ['GroqModelClient']
