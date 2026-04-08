"""SentenceBERT adapter — fallback when FastEmbed does not support a model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from qdrant_client import models
from qdrant_client.models import SparseVector

from vexor._helpers import require_package

if TYPE_CHECKING:
    from fastembed import SparseEmbedding

    from vexor.config.collection import CollectionSpec
    from vexor.config.embedding import EmbeddingSpec
    from vexor.config.ingestion import IngestionSpec


class SentenceBERTAdapter:
    """Uses ``sentence-transformers`` to provide dense embeddings."""

    def __init__(
        self,
        collection: "CollectionSpec",
        embedding: "EmbeddingSpec",
        ingestion: "IngestionSpec",
    ) -> None:
        require_package("sentence_transformers", group_name="sentence_transformers_cpu")
        from sentence_transformers import SentenceTransformer

        cfg = embedding.dense
        self._model: SentenceTransformer = SentenceTransformer(
            model_name_or_path=cfg.model_name,
            device=cfg.device,
            cache_folder=cfg.cache_dir,
        )
        self._normalize = cfg.normalize
        self._show_progress = cfg.show_progress
        self._device = cfg.device
        self._batch_size = ingestion.batch_size

        self.dense_field_name: str = cfg.model_name
        self.sparse_field_name: str = ""
        self.has_dense: bool = True
        self.has_sparse: bool = False
        self.has_late_interaction: bool = False

        # Register vector params on the collection spec
        collection.vectors_config.update(
            {
                self.dense_field_name: models.VectorParams(
                    size=self._model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE,
                )
            }
        )

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def embed_passages(self, texts: List[str]) -> Dict[str, List[List[float]]]:
        return {
            self.dense_field_name: [
                arr.tolist()
                for arr in self._model.encode(
                    sentences=texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=self._show_progress,
                    device=self._device,
                )
            ]
        }

    def embed_sparse_passages(self, texts: List[str]) -> Dict[str, List[SparseVector]]:
        raise NotImplementedError("SentenceBERT adapter does not support sparse embeddings.")

    # ------------------------------------------------------------------
    # Single query
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> List[float]:
        return list(self._model.encode(text))

    def embed_sparse_query(self, text: str) -> "SparseEmbedding":
        raise NotImplementedError("SentenceBERT adapter does not support sparse embeddings.")
