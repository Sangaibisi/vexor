"""FastEmbed adapter — uses the qdrant-client FastEmbed integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client.models import SparseVector

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from vexor.config.collection import CollectionSpec
    from vexor.config.embedding import EmbeddingSpec
    from vexor.config.ingestion import IngestionSpec


class FastEmbedAdapter:
    """Wraps FastEmbed dense + sparse models set up through the Qdrant client."""

    def __init__(
        self,
        client: "QdrantClient",
        collection: "CollectionSpec",
        embedding: "EmbeddingSpec",
        ingestion: "IngestionSpec",
    ) -> None:
        self._client = client
        self._collection = collection
        self._embedding = embedding
        self._batch_size = ingestion.batch_size
        self._threads = ingestion.embedding_threads

        self.dense_field_name: str = ""
        self.sparse_field_name: str = ""
        self.has_dense: bool = False
        self.has_sparse: bool = False
        self.has_late_interaction: bool = False

        self._dense_model: TextEmbedding | None = None
        self._sparse_model: SparseTextEmbedding | None = None

        self._setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        cfg = self._embedding

        if cfg.dense is not None:
            supported = {d.get("model") for d in TextEmbedding.list_supported_models()}
            if cfg.dense.model_name in supported:
                self._init_dense(cfg.dense.model_name, cfg.dense.cache_dir)

        if cfg.sparse is not None:
            supported = {d.get("model") for d in SparseTextEmbedding.list_supported_models()}
            if cfg.sparse.model_name in supported:
                self._init_sparse(cfg.sparse.model_name, cfg.sparse.cache_dir)

    def _init_dense(self, model_name: str, cache_dir: str) -> None:
        self._client.set_model(
            embedding_model_name=model_name,
            cache_dir=cache_dir,
            local_files_only=False,
            threads=self._threads,
        )
        self._dense_model = self._client._get_or_init_model(
            model_name=self._client.embedding_model_name,
        )
        self.dense_field_name = self._client.get_vector_field_name()
        self._collection.vectors_config.update(
            self._client.get_fastembed_vector_params(
                on_disk=False, quantization_config=None, hnsw_config=None,
            )
        )
        self.has_dense = True

    def _init_sparse(self, model_name: str, cache_dir: str) -> None:
        self._client.set_sparse_model(
            embedding_model_name=model_name,
            cache_dir=cache_dir,
            local_files_only=False,
        )
        self._sparse_model = self._client._get_or_init_sparse_model(
            model_name=self._client.sparse_embedding_model_name,
        )
        self.sparse_field_name = self._client.get_sparse_vector_field_name()
        self._collection.sparse_vectors_config.update(
            self._client.get_fastembed_sparse_vector_params(on_disk=False, modifier=None)
        )
        self.has_sparse = True

    # ------------------------------------------------------------------
    # Batch embedding
    # ------------------------------------------------------------------

    def embed_passages(self, texts: List[str]) -> Dict[str, List[List[float]]]:
        return {
            self.dense_field_name: [
                arr.tolist()
                for arr in self._dense_model.passage_embed(texts=texts, batch_size=self._batch_size)
            ]
        }

    def embed_sparse_passages(self, texts: List[str]) -> Dict[str, List[SparseVector]]:
        return {
            self.sparse_field_name: [
                SparseVector(indices=arr.indices.tolist(), values=arr.values.tolist())
                for arr in self._sparse_model.passage_embed(texts=texts, batch_size=self._batch_size)
            ]
        }

    # ------------------------------------------------------------------
    # Single-query embedding
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> List[float]:
        return list(self._dense_model.query_embed(text))[0]

    def embed_sparse_query(self, text: str) -> SparseEmbedding:
        return list(self._sparse_model.query_embed(text))[0]
