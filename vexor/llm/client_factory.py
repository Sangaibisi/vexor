"""LLM client and provider factories — replaces GenAIBuilder + GenAIProviderBuilder + registries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vexor._helpers import require_package

if TYPE_CHECKING:
    from vexor.config.llm import LLMSpec


# ---------------------------------------------------------------------------
# GenAI client  (raw API client — OpenAI / Gemini)
# ---------------------------------------------------------------------------


def create_llm_client(spec: "LLMSpec") -> Any:
    """Instantiate the raw API client for the chosen platform."""
    _CLIENTS = {
        "OpenAI": _openai_client,
        "Gemini": _gemini_client,
    }
    factory = _CLIENTS.get(spec.platform)
    if factory is None:
        raise ValueError(f"Unsupported LLM platform: {spec.platform}")
    return factory(spec.api_key)


def _openai_client(api_key: str) -> Any:
    import openai

    return openai.Client(api_key=api_key)


def _gemini_client(api_key: str) -> Any:
    from google import genai

    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# LLM provider  (CrewAI / LangChain wrapper for agents)
# ---------------------------------------------------------------------------


def create_llm_provider(spec: "LLMSpec") -> Any:
    """Create an LLM instance for the chosen agent framework."""
    _PROVIDERS = {
        "CrewAI": _crewai_provider,
        "Langchain": _langchain_provider,
    }
    factory = _PROVIDERS.get(spec.provider)
    if factory is None:
        raise ValueError(f"Unsupported LLM provider: {spec.provider}")
    return factory(spec)


def _crewai_provider(spec: "LLMSpec") -> Any:
    require_package("crewai", group_name="crewai")
    from crewai import LLM

    return LLM(
        model=spec.model,
        temperature=spec.temperature,
    )


def _langchain_provider(spec: "LLMSpec") -> Any:
    require_package("langchain", group_name="langchain")

    _LC_CREATORS = {
        "OpenAI": _lc_openai,
        "Gemini": _lc_gemini,
    }
    creator = _LC_CREATORS.get(spec.platform)
    if creator is None:
        raise ValueError(f"Langchain does not support platform: {spec.platform}")
    return creator(spec)


def _lc_openai(spec: "LLMSpec") -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(api_key=spec.api_key, model=spec.model, temperature=spec.temperature)


def _lc_gemini(spec: "LLMSpec") -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        google_api_key=spec.api_key, model=spec.model, temperature=spec.temperature,
    )
