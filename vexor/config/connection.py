"""Connection and storage configuration specs."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, SecretStr, StrictBool, StrictInt, StrictStr, field_validator


class ServerConnectionSpec(BaseModel, extra="forbid"):
    """How to reach the Qdrant instance."""

    api_key: Optional[SecretStr] = Field(default="")
    port: Optional[StrictInt] = Field(default=6333)
    https: Optional[StrictBool] = Field(default=False)
    prefer_grpc: Optional[StrictBool] = Field(default=True)
    grpc_port: Optional[StrictInt] = Field(default=6334)
    host: Optional[StrictStr] = Field(default="localhost")
    timeout: Optional[StrictInt] = Field(default=1000)

    @field_validator("api_key")
    @classmethod
    def _unwrap_api_key(cls, v: SecretStr) -> str:
        return v.get_secret_value() if isinstance(v, SecretStr) else v


class S3Credentials(BaseModel, extra="forbid"):
    """AWS / MinIO S3 credentials and bucket info."""

    access_key_id: StrictStr
    secret_access_key: SecretStr
    bucket_name: StrictStr
    file_name: StrictStr
    document_name: StrictStr
    endpoint_url: Optional[StrictStr] = Field(default=None)
    region: Optional[StrictStr] = Field(default=None)

    @field_validator("secret_access_key")
    @classmethod
    def _unwrap_secret(cls, v: SecretStr) -> str:
        return v.get_secret_value() if isinstance(v, SecretStr) else v

    @field_validator("endpoint_url")
    @classmethod
    def _check_endpoint_reachable(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(v)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=3):
                return v
        except OSError:
            raise ValueError(f"S3 endpoint '{v}' is not reachable.")


class RemoteStorageSpec(BaseModel, extra="forbid"):
    """Remote data-store: a DuckDB backend plus S3 credentials."""

    db_backend: StrictStr = Field(default="duckdb")
    s3: S3Credentials
