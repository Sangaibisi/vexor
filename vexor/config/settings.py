"""Top-level settings aggregator."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from vexor.config.collection import CollectionSpec
from vexor.config.connection import RemoteStorageSpec, ServerConnectionSpec
from vexor.config.embedding import EmbeddingSpec
from vexor.config.ingestion import DataFormat, IngestionSpec
from vexor.config.llm import LLMSpec
from vexor.config.observability import LogSpec, TracingSpec
from vexor.config.segmentation import SegmentationSpec


class VexorSettings(BaseModel, extra="forbid"):
    """Single object that carries every configuration a vexor session needs."""

    collection: CollectionSpec
    server: ServerConnectionSpec
    embedding: EmbeddingSpec
    ingestion: IngestionSpec = Field(default_factory=IngestionSpec)
    remote_storage: Optional[RemoteStorageSpec] = Field(default=None)
    segmentation: Optional[SegmentationSpec] = Field(default=None)
    llm: Optional[LLMSpec] = Field(default=None)
    tracing: Optional[TracingSpec] = Field(default=None)
    log: LogSpec = Field(default_factory=LogSpec)

    @model_validator(mode="after")
    def _cross_validate(self) -> "VexorSettings":
        has_remote = self.remote_storage is not None
        has_local = self.ingestion.data_dir is not None

        if not has_remote and not has_local:
            raise ValueError("Either 'remote_storage' or 'ingestion.data_dir' must be provided.")
        if has_remote and has_local:
            raise ValueError("Only one of 'remote_storage' or 'ingestion.data_dir' can be set.")

        is_pdf = self.ingestion.data_format == DataFormat.PDF
        if is_pdf and self.segmentation is None:
            raise ValueError("'segmentation' must be provided when data_format is 'pdf'.")
        if not is_pdf and self.segmentation is not None:
            raise ValueError("'segmentation' should only be set when data_format is 'pdf'.")

        return self
