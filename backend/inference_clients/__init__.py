"""
Inference Clients Module

This module provides unified interfaces for different LLM inference providers:
- Groq API (fast inference for evaluation tasks)
- Gemini/Vertex AI (high-quality generation)
- Local models (privacy-sensitive tasks)
"""

from .groq_client import GroqModelClient

__all__ = ['GroqModelClient']
