"""Custom exception hierarchy for the vexor package."""

from __future__ import annotations

from loguru import logger


class VexorError(Exception):
    """Base exception for all vexor errors."""

    def __init__(self, message: str, *, log_level: str = "error") -> None:
        self.message = message
        super().__init__(self.message)
        getattr(logger, log_level)(self.message)

    def __str__(self) -> str:
        return self.message


# -- Connection --


class ConnectionError(VexorError):
    """Failed to establish a connection to the Qdrant server."""

    def __init__(self) -> None:
        super().__init__("Unable to connect to the Qdrant server.")


# -- Collection --


class CollectionError(VexorError):
    """Base for collection-related errors."""


class CollectionCreateError(CollectionError):
    """Could not create the specified collection."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Failed to create collection '{name}'.")


class CollectionDeleteError(CollectionError):
    """Could not delete the specified collection."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Failed to delete collection '{name}'.")


# -- Shard --


class ShardError(VexorError):
    """Could not create a shard key."""

    def __init__(self, collection_name: str) -> None:
        super().__init__(f"Failed to create shard key for collection '{collection_name}'.")


# -- Schema / Validation --


class SchemaError(VexorError):
    """Base for configuration and schema validation errors."""


class MissingEmbeddingError(SchemaError):
    """No embedding model was provided."""

    def __init__(self) -> None:
        super().__init__("At least one embedding model (dense, sparse, or late-interaction) must be specified.")


class InsufficientEmbeddingModelsError(SchemaError):
    """Hybrid search requires two or more embedding models."""

    def __init__(self) -> None:
        super().__init__("Hybrid search requires at least two embedding models (e.g. dense + sparse).")


class EmptyColumnsError(SchemaError):
    """No columns remained after filtering."""

    def __init__(self) -> None:
        super().__init__("No columns found to include after filtering.")


class PayloadNotFoundError(SchemaError):
    """A requested payload field is missing from the data columns."""

    def __init__(self, field: str) -> None:
        super().__init__(f"Payload field '{field}' not found in data columns.")


class ShardKeyNotFoundError(SchemaError):
    """A requested shard key is missing from the data columns."""

    def __init__(self, key: str) -> None:
        super().__init__(f"Shard key '{key}' not found in data columns.")


# -- Data --


class DataError(VexorError):
    """Base for data-related errors."""


class NoRecordsFoundWarning(DataError):
    """No records were found for the given identifier."""

    def __init__(self, identifier: int | str) -> None:
        super().__init__(f"No records found for '{identifier}'.", log_level="warning")


# -- Runtime --


class InvalidRecommendRequest(VexorError):
    """The recommendation request parameters are invalid."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Invalid recommendation request: {reason}")


class UnsupportedChunkerError(VexorError):
    """The requested chunking method is not supported."""

    def __init__(self, method: str, available: list[str]) -> None:
        super().__init__(f"Unsupported chunking method '{method}'. Available: {available}")


class UploadError(VexorError):
    """Vector upload did not complete successfully."""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"Upload mismatch: expected {expected} vectors, but collection has {actual}.")
