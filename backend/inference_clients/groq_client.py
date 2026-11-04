"""
Groq API Client for Fast LLM Inference

This module provides a robust client for Groq's API, optimized for evaluation
and classification tasks in the RAG pipeline. Groq offers extremely fast inference
with models like Llama3 and Phi3, making it ideal for latency-sensitive operations.

Key Features:
- Async and sync inference support
- Automatic retry with exponential backoff
- Latency tracking and metrics
- Structured JSON output parsing
- Fallback to fallback LLM on failures
- Type-safe responses with Pydantic

Usage:
    from inference_clients.groq_client import GroqModelClient
    
    client = GroqModelClient("llama3-8b-8192")
    response = await client.infer_async("Evaluate this document...")
    
    # Or synchronous
    response = client.infer("Evaluate this document...")
"""

import os
import time
import json
import logging
import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass
from groq import Groq, AsyncGroq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Metrics for a single inference call"""
    model_name: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    success: bool
    error: Optional[str] = None
    fallback_used: bool = False


class GroqAPIError(Exception):
    """Custom exception for Groq API errors"""
    pass


class GroqModelClient:
    """
    Client for Groq API inference with robust error handling and metrics.
    
    Supports both synchronous and asynchronous inference with automatic retries,
    fallback mechanisms, and comprehensive latency tracking.
    
    Attributes:
        model_name: Groq model identifier (e.g., "llama3-8b-8192")
        api_key: Groq API key from environment
        base_url: Groq API endpoint
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts on failure
    """
    
    # Supported Groq models
    SUPPORTED_MODELS = {
        "llama-3.1-8b-instant": {
            "name": "llama-3.1-8b-instant",
            "context_window": 131072,
            "description": "Meta Llama 3.1 8B - Fast, balanced performance"
        },
        "llama-3.3-70b-versatile": {
            "name": "llama-3.3-70b-versatile",
            "context_window": 131072,
            "description": "Meta Llama 3.3 70B - High quality, slower"
        },
        "meta-llama/llama-4-scout-17b-16e-instruct": {
            "name": "meta-llama/llama-4-scout-17b-16e-instruct",
            "context_window": 131072,
            "description": "Meta Llama 4 Scout 17B - Latest model"
        },
        "groq/compound-mini": {
            "name": "groq/compound-mini",
            "context_window": 131072,
            "description": "Groq Compound Mini - Fast classification"
        },
        "qwen/qwen3-32b": {
            "name": "qwen/qwen3-32b",
            "context_window": 32768,
            "description": "Qwen 3 32B - Advanced reasoning and analysis"
        }
    }
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 2,
        enable_fallback: bool = True
    ):
        """
        Initialize Groq client.
        
        Args:
            model_name: Groq model identifier
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            enable_fallback: Whether to fallback to fallback LLM on failures
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback
        
        # Validate API key
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Validate model
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model '{model_name}' not in supported list. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        # Initialize Groq SDK clients
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)
        
        logger.info(f"✅ Groq client initialized: {model_name}")
    
    def _create_messages(self, prompt: str) -> list:
        """Create messages list for Groq API"""
        return [{"role": "user", "content": prompt}]
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def _make_request(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]] = None
    ):
        """Make request to Groq API with retries"""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        return self.client.chat.completions.create(**kwargs)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    async def _make_request_async(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]] = None
    ):
        """Make async request to Groq API with retries"""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        return await self.async_client.chat.completions.create(**kwargs)
    
    def infer(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
        **kwargs
    ) -> tuple[str, InferenceMetrics]:
        """
        Synchronous inference with Groq API.
        
        Args:
            prompt: Input prompt text
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            json_mode: Whether to request JSON output format
            **kwargs: Additional parameters for Groq API
            
        Returns:
            Tuple of (response_text, metrics)
            
        Raises:
            GroqAPIError: If inference fails after retries
        """
        start_time = time.time()
        
        try:
            # Create messages
            messages = self._create_messages(prompt)
            response_format = {"type": "json_object"} if json_mode else None
            
            logger.debug(f"🚀 Groq inference starting: {self.model_name}")
            
            # Make request using Groq SDK
            completion = self._make_request(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )
            
            # Extract response from SDK object
            response_text = completion.choices[0].message.content
            usage = completion.usage
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            metrics = InferenceMetrics(
                model_name=self.model_name,
                latency_ms=latency_ms,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                success=True
            )
            
            logger.info(
                f"✅ Groq inference complete: {self.model_name} "
                f"({latency_ms:.0f}ms, {metrics.total_tokens} tokens)"
            )
            
            return response_text, metrics
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Groq inference failed: {e}")
            
            # Try fallback if enabled
            if self.enable_fallback:
                logger.warning("🔄 Attempting fallback to fallback LLM...")
                try:
                    response_text, fallback_metrics = self._fallback_to_llm(
                        prompt, temperature, max_tokens
                    )
                    fallback_metrics.fallback_used = True
                    return response_text, fallback_metrics
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback also failed: {fallback_error}")
            
            # Create error metrics
            metrics = InferenceMetrics(
                model_name=self.model_name,
                latency_ms=latency_ms,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                success=False,
                error=str(e)
            )
            
            raise GroqAPIError(f"Groq inference failed: {e}") from e
    
    async def infer_async(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
        **kwargs
    ) -> tuple[str, InferenceMetrics]:
        """
        Asynchronous inference with Groq API.
        
        Args:
            prompt: Input prompt text
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            json_mode: Whether to request JSON output format
            **kwargs: Additional parameters for Groq API
            
        Returns:
            Tuple of (response_text, metrics)
            
        Raises:
            GroqAPIError: If inference fails after retries
        """
        start_time = time.time()
        
        try:
            # Create messages
            messages = self._create_messages(prompt)
            response_format = {"type": "json_object"} if json_mode else None
            
            logger.debug(f"🚀 Groq async inference starting: {self.model_name}")
            
            # Make async request using Groq SDK
            completion = await self._make_request_async(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )
            
            # Extract response from SDK object
            response_text = completion.choices[0].message.content
            usage = completion.usage
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            metrics = InferenceMetrics(
                model_name=self.model_name,
                latency_ms=latency_ms,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                success=True
            )
            
            logger.info(
                f"✅ Groq async inference complete: {self.model_name} "
                f"({latency_ms:.0f}ms, {metrics.total_tokens} tokens)"
            )
            
            return response_text, metrics
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Groq async inference failed: {e}")
            
            # Try fallback if enabled
            if self.enable_fallback:
                logger.warning("🔄 Attempting fallback to fallback LLM...")
                try:
                    response_text, fallback_metrics = self._fallback_to_llm(
                        prompt, temperature, max_tokens
                    )
                    fallback_metrics.fallback_used = True
                    return response_text, fallback_metrics
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback also failed: {fallback_error}")
            
            # Create error metrics
            metrics = InferenceMetrics(
                model_name=self.model_name,
                latency_ms=latency_ms,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                success=False,
                error=str(e)
            )
            
            raise GroqAPIError(f"Groq async inference failed: {e}") from e
    
    def infer_json(
        self,
        prompt: str,
        response_model: type[BaseModel],
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs
    ) -> tuple[BaseModel, InferenceMetrics]:
        """
        Inference with structured JSON output parsed into Pydantic model.
        
        Args:
            prompt: Input prompt text
            response_model: Pydantic model class for response parsing
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (parsed_model, metrics)
            
        Raises:
            GroqAPIError: If inference or parsing fails
        """
        # Request JSON mode
        response_text, metrics = self.infer(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            **kwargs
        )
        
        try:
            # Parse JSON response
            response_dict = json.loads(response_text)
            parsed_model = response_model(**response_dict)
            logger.debug(f"✅ Successfully parsed response into {response_model.__name__}")
            return parsed_model, metrics
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"❌ Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response_text[:200]}...")
            raise GroqAPIError(f"Failed to parse JSON response: {e}") from e
    
    async def infer_json_async(
        self,
        prompt: str,
        response_model: type[BaseModel],
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs
    ) -> tuple[BaseModel, InferenceMetrics]:
        """
        Async inference with structured JSON output parsed into Pydantic model.
        
        Args:
            prompt: Input prompt text
            response_model: Pydantic model class for response parsing
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (parsed_model, metrics)
            
        Raises:
            GroqAPIError: If inference or parsing fails
        """
        # Request JSON mode
        response_text, metrics = await self.infer_async(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            **kwargs
        )
        
        try:
            # Parse JSON response
            response_dict = json.loads(response_text)
            parsed_model = response_model(**response_dict)
            logger.debug(f"✅ Successfully parsed response into {response_model.__name__}")
            return parsed_model, metrics
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"❌ Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response_text[:200]}...")
            raise GroqAPIError(f"Failed to parse JSON response: {e}") from e
    
    def _fallback_to_llm(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> tuple[str, InferenceMetrics]:
        """
        Fallback to fallback LLM Flash when Groq fails.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Tuple of (response_text, metrics)
        """
        from backend.llm import get_llm
        from langchain_core.output_parsers import StrOutputParser
        
        start_time = time.time()
        
        try:
            llm = get_llm("fallback").bind(temperature=temperature, max_tokens=max_tokens)
            
            chain = llm | StrOutputParser()
            response = chain.invoke(prompt)
            
            latency_ms = (time.time() - start_time) * 1000
            
            metrics = InferenceMetrics(
                model_name="fallback-llm",
                latency_ms=latency_ms,
                prompt_tokens=0,  # token counts not tracked for fallback
                completion_tokens=0,
                total_tokens=0,
                success=True,
                fallback_used=True
            )
            
            logger.info(f"✅ fallback LLM fallback successful ({latency_ms:.0f}ms)")
            return response, metrics
            
        except Exception as e:
            logger.error(f"❌ fallback LLM fallback failed: {e}")
            raise
    
    def close(self):
        """Close HTTP clients"""
        self.client.close()
        asyncio.run(self.async_client.aclose())
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all supported Groq models"""
        return cls.SUPPORTED_MODELS
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        return cls.SUPPORTED_MODELS.get(model_name)
