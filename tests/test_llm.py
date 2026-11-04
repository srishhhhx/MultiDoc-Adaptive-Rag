import pytest

from backend import llm


def test_default_provider_is_cerebras_with_reasoning_off(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("CEREBRAS_API_KEY", "k")
    m = llm.get_llm("generate")
    assert m.model_name == "qwen-3.8-27b"
    assert str(m.openai_api_base).startswith("https://api.cerebras.ai")
    assert m.extra_body == {"reasoning_effort": "none"}


def test_plan_role_thinks_and_env_override(monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "k")
    assert llm.get_llm("plan").extra_body == {"reasoning_effort": "low"}
    monkeypatch.setenv("LLM_REASONING_PLAN", "high")
    assert llm.get_llm("plan").extra_body == {"reasoning_effort": "high"}


def test_streaming_flag(monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "k")
    assert llm.get_llm("generate", streaming=True).streaming is True


def test_groq_provider_has_no_reasoning_param(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    m = llm.get_llm("evaluate")
    assert "groq.com" in str(m.openai_api_base)
    assert not m.extra_body


def test_metadata_role_caps_tokens(monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "k")
    assert llm.get_llm("metadata").max_tokens == 500


def test_unknown_role_and_provider_rejected(monkeypatch):
    with pytest.raises(ValueError):
        llm.get_llm("nope")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError):
        llm.get_llm("generate")


def test_missing_key_does_not_break_import_time_construction(monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    assert llm.get_llm("classify") is not None
