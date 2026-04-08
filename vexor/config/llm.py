"""LLM (large language model) configuration spec."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator


class LLMSpec(BaseModel, extra="forbid"):
    """Everything needed to instantiate an LLM client and provider."""

    platform: Literal["OpenAI", "Gemini"]
    api_key: SecretStr = Field(default="")
    model: str = Field(default="gpt-4o")
    temperature: Optional[float] = Field(default=0.7)
    provider: Literal["CrewAI", "Langchain"] = Field(default="CrewAI")

    @field_validator("api_key")
    @classmethod
    def _unwrap_key(cls, v: SecretStr) -> str:
        return v.get_secret_value() if isinstance(v, SecretStr) else v
