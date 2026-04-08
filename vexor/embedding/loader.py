"""Embedding model loader — tries FastEmbed first, falls back to SentenceBERT."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from loguru import logger

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from vexor.config.collection import CollectionSpec
    from vexor.config.embedding import EmbeddingSpec
    from vexor.config.ingestion import IngestionSpec
    from vexor.embedding.fastembed_adapter import FastEmbedAdapter
    from vexor.embedding.sbert_adapter import SentenceBERTAdapter


def load_embedder(
    client: "QdrantClient",
    collection: "CollectionSpec",
    embedding: "EmbeddingSpec",
    ingestion: "IngestionSpec",
) -> Union["FastEmbedAdapter", "SentenceBERTAdapter"]:
    """Load an embedding adapter — FastEmbed preferred, SentenceBERT as fallback."""
    try:
        from vexor.embedding.fastembed_adapter import FastEmbedAdapter

        adapter = FastEmbedAdapter(client, collection, embedding, ingestion)
        if adapter.has_dense or adapter.has_sparse:
            return adapter
        raise RuntimeError("FastEmbed did not initialise any models.")
    except Exception as exc:
        logger.warning(f"FastEmbed unavailable ({exc}), falling back to SentenceBERT.")
        from vexor.embedding.sbert_adapter import SentenceBERTAdapter

        return SentenceBERTAdapter(collection, embedding, ingestion)
