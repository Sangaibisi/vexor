"""Embedding model configuration specs."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, StrictBool, StrictStr, model_validator

from vexor.errors import MissingEmbeddingError


class DenseModelSpec(BaseModel, extra="forbid"):
    """Configuration for a dense embedding model."""

    model_name: Optional[StrictStr] = Field(default="BAAI/bge-small-en-v1.5")
    cache_dir: Optional[StrictStr] = Field(default=".cache")
    normalize: Optional[StrictBool] = Field(default=True)
    device: Optional[Literal["cpu"]] = Field(default="cpu")
    show_progress: Optional[StrictBool] = Field(default=False)


class SparseModelSpec(BaseModel, extra="forbid"):
    """Configuration for a sparse embedding model (e.g. SPLADE)."""

    model_name: Optional[StrictStr] = Field(default="prithivida/Splade_PP_en_v1")
    cache_dir: Optional[StrictStr] = Field(default=".cache")


class LateInteractionModelSpec(BaseModel, extra="forbid"):
    """Configuration for a late-interaction model (e.g. ColBERT)."""

    model_name: Optional[StrictStr] = Field(default="colbert-ir/colbertv2.0")
    cache_dir: Optional[StrictStr] = Field(default=".cache")


class EmbeddingSpec(BaseModel, extra="forbid"):
    """At least one embedding model must be specified."""

    dense: Optional[DenseModelSpec] = None
    sparse: Optional[SparseModelSpec] = None
    late_interaction: Optional[LateInteractionModelSpec] = None

    @model_validator(mode="after")
    def _require_at_least_one(self) -> "EmbeddingSpec":
        fields: Dict[str, Any] = self.model_dump()
        if not any(v is not None for v in fields.values()):
            raise MissingEmbeddingError()
        return self
