"""Ingestion (data loading) configuration specs."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr, model_validator


class DataFormat(str, Enum):
    """Supported input data formats."""

    TABULAR = "tabular"
    VECTORIZED_TABULAR = "vectorized_tabular"
    PDF = "pdf"


class IngestionSpec(BaseModel, extra="forbid"):
    """Controls how data is loaded and uploaded to a collection."""

    data_dir: Optional[StrictStr] = None
    data_format: DataFormat = Field(default=DataFormat.TABULAR)
    batch_size: Optional[StrictInt] = Field(default=256)
    recreate_collection: Optional[StrictBool] = Field(default=False)
    upload_method: Optional[Literal["add", "upsert", "upload"]] = Field(default="upsert")
    embedding_threads: Optional[StrictInt] = Field(default=1)
    parallel_threads: Optional[StrictInt] = Field(default=1)
    shard_count_per_key: Optional[StrictInt] = Field(default=1)

    @model_validator(mode="after")
    def _validate_format_vs_dir(self) -> "IngestionSpec":
        if self.data_dir is None and self.data_format != DataFormat.TABULAR:
            raise ValueError(f"data_format must be 'tabular' when data_dir is not set (got '{self.data_format.value}').")
        return self
