"""Environment-based settings loader for examples."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class EnvConfig(BaseSettings):
    """Reads settings from environment variables or .env-dev file."""

    model_config = {"env_file": ".env-dev", "extra": "ignore"}

    # Collection
    COLLECTION_NAME: str = "test"

    # Server
    HOST: str = "localhost"
    PORT: int = 6333
    API_KEY: str = ""
    HTTPS: bool = False
    PREFER_GRPC: bool = True
    GRPC_PORT: int = 6334
    TIMEOUT: int = 1000

    # Embedding
    DENSE_EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    SPARSE_EMBEDDING_MODEL_NAME: str = "prithivida/Splade_PP_en_v1"
    EMBEDDING_CACHE_DIR: str = ".cache"

    # Data
    DATA_DIR: str = "datas"
    DATA_TYPE: str = "tabular"
    BATCH_SIZE: int = 256

    # Logging
    IS_LOG_ACTIVE: bool = True
    IS_LOG_PROD: bool = False
    IS_LOG_TO_FILE: bool = False
