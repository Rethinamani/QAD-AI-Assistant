# chatbot/llm_providers.py
"""
Multi-provider LLM dispatch.

Ollama, Groq, and OpenAI all accept the same [{"role", "content"}, ...]
chat-message shape and return a single completion, so one function fans
out to whichever client the caller (a request-level provider/model choice
from the UI) asks for. Clients are created lazily so a missing API key for
a provider the user never selects doesn't break startup.
"""

import logging

from ollama import Client as OllamaClient

from config import (
    OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
    DEFAULT_LLM_PROVIDER,
)

logger = logging.getLogger(__name__)

# A handful of current, popular models per hosted provider. Ollama's list is
# whatever the local server actually has pulled (see list_models below).
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-3.5-turbo",
]

# Providers the UI can offer. "available" reflects whether the credentials
# needed to actually call it are configured.
PROVIDERS = {
    "ollama": {
        "label":         "Ollama (local)",
        "default_model": OLLAMA_LLM_MODEL,
        "available":     True,
    },
    "groq": {
        "label":         "Groq",
        "default_model": GROQ_MODEL,
        "available":     bool(GROQ_API_KEY),
    },
    "openai": {
        "label":         "ChatGPT (OpenAI)",
        "default_model": OPENAI_MODEL,
        "available":     bool(OPENAI_API_KEY),
    },
}

_ollama_client = OllamaClient(host=OLLAMA_BASE_URL)
_groq_client = None
_openai_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "Groq is not configured — set GROQ_API_KEY in .env."
            )
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "ChatGPT is not configured — set OPENAI_API_KEY in .env."
            )
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _friendly_api_error(provider_label: str, e: Exception) -> RuntimeError:
    """Turn a hosted-provider SDK exception into a short, user-facing message."""
    status = getattr(e, "status_code", None)
    if status == 401:
        detail = "invalid or expired API key"
    elif status == 429:
        detail = "rate limited or out of quota — check your account's billing/credits"
    elif status == 404:
        detail = "model not found or not accessible with this API key"
    elif status is not None:
        detail = f"API error (HTTP {status})"
    else:
        detail = str(e)
    return RuntimeError(f"{provider_label} request failed: {detail}")


def list_models(provider: str) -> list[str]:
    """Models to offer in the UI for a given provider."""
    provider = (provider or DEFAULT_LLM_PROVIDER).lower()

    if provider == "ollama":
        try:
            data  = _ollama_client.list()
            names = [
                m.get("model") for m in data.get("models", [])
                if m.get("model") and "embed" not in m.get("model", "").lower()
                and m.get("model") != OLLAMA_EMBED_MODEL
            ]
            return names or [OLLAMA_LLM_MODEL]
        except Exception as e:
            logger.warning(f"Could not list local Ollama models: {e}")
            return [OLLAMA_LLM_MODEL]

    if provider == "groq":
        return GROQ_MODELS
    if provider == "openai":
        return OPENAI_MODELS
    return []


def call_llm(
    messages: list[dict],
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 600,
    temperature: float = 0.1,
) -> str:
    """Run a chat completion against the chosen provider and return the text."""
    provider = (provider or DEFAULT_LLM_PROVIDER).lower()
    if provider not in PROVIDERS:
        logger.warning(
            f"Unknown LLM provider '{provider}', falling back to "
            f"{DEFAULT_LLM_PROVIDER}."
        )
        provider = DEFAULT_LLM_PROVIDER

    model = model or PROVIDERS[provider]["default_model"]

    if provider == "ollama":
        response = _ollama_client.chat(
            model=model,
            messages=messages,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p":       0.9,
            },
        )
        return response["message"]["content"]

    if provider == "groq":
        client = _get_groq_client()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            raise _friendly_api_error("Groq", e) from e
        return response.choices[0].message.content

    if provider == "openai":
        client = _get_openai_client()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            raise _friendly_api_error("ChatGPT (OpenAI)", e) from e
        return response.choices[0].message.content

    raise ValueError(f"Unsupported LLM provider: {provider}")
