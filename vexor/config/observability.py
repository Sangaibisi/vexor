"""Logging and tracing configuration specs."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, SecretStr, StrictBool, field_validator


class LogSpec(BaseModel, extra="forbid"):
    """Controls vexor's logging behaviour."""

    enabled: Optional[StrictBool] = Field(default=True)
    production_mode: Optional[StrictBool] = Field(default=False)
    log_to_file: Optional[StrictBool] = Field(default=False)


class TracingSpec(BaseModel, extra="forbid"):
    """Configuration for an observability platform (AgentOps, Langfuse, Opik)."""

    platform: str
    project_name: str
    api_key: SecretStr
    langfuse_public_key: Optional[SecretStr] = Field(default=None)
    langfuse_host: Optional[str] = Field(default=None)

    @field_validator("api_key")
    @classmethod
    def _unwrap_api_key(cls, v: SecretStr) -> str:
        return v.get_secret_value() if isinstance(v, SecretStr) else v

    @field_validator("langfuse_public_key")
    @classmethod
    def _unwrap_langfuse_key(cls, v: Optional[SecretStr]) -> Optional[str]:
        if v is None:
            return None
        return v.get_secret_value() if isinstance(v, SecretStr) else v
