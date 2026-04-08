"""Embedder protocol — the contract every embedding adapter must satisfy."""

from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable

from fastembed import SparseEmbedding
from qdrant_client.models import SparseVector


@runtime_checkable
class Embedder(Protocol):
    """Structural interface for an embedding model adapter."""

    # Field names used in collection vector configs
    dense_field_name: str
    sparse_field_name: str

    # Capability flags
    has_dense: bool
    has_sparse: bool
    has_late_interaction: bool

    # Batch embedding
    def embed_passages(self, texts: List[str]) -> Dict[str, List[List[float]]]: ...

    def embed_sparse_passages(self, texts: List[str]) -> Dict[str, List[SparseVector]]: ...

    # Single-query embedding
    def embed_query(self, text: str) -> List[float]: ...

    def embed_sparse_query(self, text: str) -> SparseEmbedding: ...
