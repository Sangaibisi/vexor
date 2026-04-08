"""Collection and index configuration specs."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr
from qdrant_client import models


# ---------------------------------------------------------------------------
# Index parameter specs  (one per Qdrant index type)
# ---------------------------------------------------------------------------


class KeywordIndex(models.KeywordIndexParams):
    type: Literal[models.KeywordIndexType.KEYWORD] = Field(default=models.KeywordIndexType.KEYWORD)
    is_tenant: Optional[StrictBool] = Field(default=False)
    on_disk: Optional[StrictBool] = Field(default=False)


class IntegerIndex(models.IntegerIndexParams):
    type: Literal[models.IntegerIndexType.INTEGER] = Field(default=models.IntegerIndexType.INTEGER)
    lookup: Optional[StrictBool] = Field(default=True)
    range: Optional[StrictBool] = Field(default=True)
    is_principal: Optional[StrictBool] = Field(default=False)
    on_disk: Optional[StrictBool] = Field(default=False)


class FloatIndex(models.FloatIndexParams):
    type: Literal[models.FloatIndexType.FLOAT] = Field(default=models.FloatIndexType.FLOAT)
    is_principal: Optional[StrictBool] = Field(default=False)
    on_disk: Optional[StrictBool] = Field(default=False)


class BoolIndex(models.BoolIndexParams):
    type: Literal[models.BoolIndexType.BOOL] = Field(default=models.BoolIndexType.BOOL)


class DatetimeIndex(models.DatetimeIndexParams):
    type: Literal[models.DatetimeIndexType.DATETIME] = Field(default=models.DatetimeIndexType.DATETIME)
    is_principal: Optional[StrictBool] = Field(default=True)
    on_disk: Optional[StrictBool] = Field(default=False)


class TextIndex(models.TextIndexParams):
    type: Literal[models.TextIndexType.TEXT] = Field(default=models.TextIndexType.TEXT)
    tokenizer: Optional[models.TokenizerType] = Field(default=models.TokenizerType.WORD)
    min_token_len: Optional[StrictInt] = Field(default=2)
    max_token_len: Optional[StrictInt] = Field(default=16)
    lowercase: Optional[StrictBool] = Field(default=True)
    on_disk: Optional[StrictBool] = Field(default=False)


class UuidIndex(models.UuidIndexParams):
    type: Literal[models.UuidIndexType.UUID] = Field(default=models.UuidIndexType.UUID)


# ---------------------------------------------------------------------------
# HNSW / optimizer / quantization
# ---------------------------------------------------------------------------


class HnswSpec(models.HnswConfigDiff):
    m: Optional[StrictInt] = Field(default=16)
    on_disk: Optional[StrictBool] = Field(default=False)
    ef_construct: Optional[StrictInt] = Field(default=100)


class OptimizerSpec(models.OptimizersConfigDiff):
    memmap_threshold: Optional[StrictInt] = Field(default=None)
    indexing_threshold: StrictInt = Field(default=20000)


class ScalarQuantizationSpec(models.ScalarQuantization):
    """INT8 scalar quantization with sensible defaults."""

    scalar: models.ScalarQuantizationConfig = models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8,
        quantile=0.99,
        always_ram=True,
    )


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

PayloadSchemaParams = Union[
    KeywordIndex,
    IntegerIndex,
    FloatIndex,
    BoolIndex,
    DatetimeIndex,
    TextIndex,
    UuidIndex,
]


class CollectionSpec(BaseModel, extra="forbid"):
    """Declares a Qdrant collection and its physical parameters."""

    name: StrictStr
    vectors_config: Union[Dict[str, models.VectorParams]] = Field(default_factory=dict)
    sparse_vectors_config: Optional[Dict[str, models.SparseVectorParams]] = Field(default_factory=dict)
    sharding_method: models.ShardingMethod = Field(default=models.ShardingMethod.AUTO)
    shard_number: Optional[StrictInt] = Field(default=1)
    on_disk_payload: Optional[StrictBool] = Field(default=True)
    quantization_config: Optional[ScalarQuantizationSpec] = Field(default_factory=ScalarQuantizationSpec)
    optimizer_config: Optional[OptimizerSpec] = Field(default_factory=OptimizerSpec)
    hnsw_config: Optional[HnswSpec] = Field(default_factory=HnswSpec)
