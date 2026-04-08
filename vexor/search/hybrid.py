"""Hybrid search — combines dense + sparse (+ late-interaction) via RRF fusion."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from qdrant_client import models
from qdrant_client.models import QueryResponse, SparseVector

from vexor.errors import InsufficientEmbeddingModelsError

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from vexor.config.request import BatchQuery, SingleQuery
    from vexor.config.search import SearchParams
    from vexor.embedding.protocol import Embedder


def _build_prefetch(
    embedder: "Embedder",
    query_text: str,
    limit: int,
) -> List[models.Prefetch]:
    """Build prefetch entries for each available embedding type."""
    prefetch: List[models.Prefetch] = []

    if embedder.has_dense:
        prefetch.append(
            models.Prefetch(
                query=embedder.embed_query(query_text),
                using=embedder.dense_field_name,
                limit=limit,
            )
        )

    if embedder.has_sparse:
        sparse = embedder.embed_sparse_query(query_text)
        prefetch.append(
            models.Prefetch(
                query=SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
                using=embedder.sparse_field_name,
                limit=limit,
            )
        )

    if len(prefetch) < 2:
        raise InsufficientEmbeddingModelsError()

    return prefetch


def hybrid_search(
    client: "QdrantClient",
    embedder: "Embedder",
    query: "SingleQuery",
    params: "SearchParams",
) -> QueryResponse:
    """Single hybrid search using reciprocal rank fusion."""
    prefetch = _build_prefetch(embedder, query.text, params.limit)
    return client.query_points(
        collection_name=params.collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=params.resolved_filter(),
        limit=params.limit,
        shard_key_selector=params.shard_key,
    )


def hybrid_search_batch(
    client: "QdrantClient",
    embedder: "Embedder",
    query: "BatchQuery",
    params: "SearchParams",
) -> list[QueryResponse]:
    """Batch hybrid search."""
    requests = [
        models.QueryRequest(
            prefetch=_build_prefetch(embedder, text, params.limit),
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            filter=params.resolved_filter(),
            limit=params.limit,
            shard_key_selector=params.shard_key,
        )
        for text in query.texts
    ]
    return client.query_batch_points(
        collection_name=params.collection,
        requests=requests,
    )
