"""
Unified LLM client factory.

Every chain obtains its model through ``get_llm(role)`` so that the provider,
model name, sampling temperature and reasoning effort are configured in one
place (and overridable through environment variables) instead of being
hard-coded in each chain.

Providers (both OpenAI-compatible):
- ``cerebras`` (default): https://api.cerebras.ai/v1
- ``groq``: https://api.groq.com/openai/v1

Environment variables:
    LLM_PROVIDER              cerebras | groq            (default: cerebras)
    CEREBRAS_API_KEY / GROQ_API_KEY
    CEREBRAS_MODEL            default: qwen-3.8-27b
    GROQ_MODEL                default: llama-3.3-70b-versatile
    CEREBRAS_BASE_URL         default: https://api.cerebras.ai/v1
    CEREBRAS_REQUESTS_PER_MIN client-side throttle (default: 5, the free tier)
    LLM_REASONING_<ROLE>      override reasoning effort for one role,
                              e.g. LLM_REASONING_PLAN=low
    LLM_MAX_RETRIES           default: 4 (429s/5xx are retried with backoff)
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

PROVIDERS = {
    "cerebras": {
        "base_url_env": "CEREBRAS_BASE_URL",
        "base_url": "https://api.cerebras.ai/v1",
        "key_env": "CEREBRAS_API_KEY",
        "model_env": "CEREBRAS_MODEL",
        "model": "qwen-3.8-27b",
    },
    "groq": {
        "base_url_env": "GROQ_BASE_URL",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "model": "llama-3.3-70b-versatile",
    },
}


@dataclass(frozen=True)
class RoleConfig:
    temperature: float
    # Reasoning effort sent to the provider. "none" disables thinking, which
    # keeps latency low and prevents reasoning text leaking into JSON output.
    reasoning_effort: Optional[str] = "none"
    max_tokens: Optional[int] = None


# Roles map to pipeline stages. Defaults favour speed: only planning thinks.
ROLES: Dict[str, RoleConfig] = {
    "generate": RoleConfig(temperature=0.0, reasoning_effort="none"),
    "plan": RoleConfig(temperature=0.1, reasoning_effort="low"),
    "classify": RoleConfig(temperature=0.1),
    "rewrite": RoleConfig(temperature=0.3),
    "assess": RoleConfig(temperature=0.1),
    "evaluate": RoleConfig(temperature=0.0),
    "metadata": RoleConfig(temperature=0.0, max_tokens=500),
    "fallback": RoleConfig(temperature=0.0),
}

_rate_limiters: Dict[str, InMemoryRateLimiter] = {}


def _provider_name() -> str:
    name = os.getenv("LLM_PROVIDER", "cerebras").lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Use one of {sorted(PROVIDERS)}")
    return name


def _rate_limiter(provider: str) -> Optional[InMemoryRateLimiter]:
    """One shared limiter per provider so parallel chains don't stampede it."""
    if provider != "cerebras":
        return None
    if provider not in _rate_limiters:
        rpm = float(os.getenv("CEREBRAS_REQUESTS_PER_MIN", "5"))
        _rate_limiters[provider] = InMemoryRateLimiter(
            requests_per_second=rpm / 60.0,
            check_every_n_seconds=0.1,
            max_bucket_size=max(1, int(rpm // 5) or 1),
        )
    return _rate_limiters[provider]


def get_llm(role: str, *, streaming: bool = False, provider: Optional[str] = None) -> ChatOpenAI:
    """Build a chat model for a pipeline role."""
    if role not in ROLES:
        raise ValueError(f"Unknown LLM role '{role}'. Use one of {sorted(ROLES)}")
    cfg = ROLES[role]
    provider = provider or _provider_name()
    p = PROVIDERS[provider]

    api_key = os.getenv(p["key_env"])
    if not api_key:
        # Keep imports working (tests, tooling); calls will fail with a 401.
        logger.warning("%s is not set; LLM calls for role '%s' will fail", p["key_env"], role)
        api_key = "missing-api-key"

    kwargs = dict(
        model=os.getenv(p["model_env"], p["model"]),
        api_key=api_key,
        base_url=os.getenv(p["base_url_env"], p["base_url"]),
        temperature=cfg.temperature,
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "4")),
        streaming=streaming,
        rate_limiter=_rate_limiter(provider),
    )
    if cfg.max_tokens:
        kwargs["max_tokens"] = cfg.max_tokens

    # Only Cerebras' Qwen models understand reasoning_effort.
    effort = os.getenv(f"LLM_REASONING_{role.upper()}", cfg.reasoning_effort)
    if provider == "cerebras" and effort:
        kwargs["extra_body"] = {"reasoning_effort": effort}

    return ChatOpenAI(**kwargs)


def structured(llm: ChatOpenAI, schema):
    """Wrap ``llm`` so it returns ``schema`` instances.

    Uses the provider's JSON-schema mode (``response_format``). Override with
    LLM_STRUCTURED_METHOD=function_calling|json_mode if a model rejects it.
    """
    method = os.getenv("LLM_STRUCTURED_METHOD", "json_schema")
    return llm.with_structured_output(schema, method=method)
