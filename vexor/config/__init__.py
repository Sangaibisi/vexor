"""Public configuration types for the vexor package."""

from vexor.config.collection import (
    BoolIndex,
    CollectionSpec,
    DatetimeIndex,
    FloatIndex,
    HnswSpec,
    IntegerIndex,
    KeywordIndex,
    OptimizerSpec,
    ScalarQuantizationSpec,
    TextIndex,
    UuidIndex,
)
from vexor.config.connection import (
    RemoteStorageSpec,
    S3Credentials,
    ServerConnectionSpec,
)
from vexor.config.embedding import (
    DenseModelSpec,
    EmbeddingSpec,
    LateInteractionModelSpec,
    SparseModelSpec,
)
from vexor.config.filtering import ConditionBuilder, FilterSpec
from vexor.config.ingestion import DataFormat, IngestionSpec
from vexor.config.llm import LLMSpec
from vexor.config.observability import LogSpec, TracingSpec
from vexor.config.request import (
    AgenticQuery,
    BatchQuery,
    PersonalizedRecommendQuery,
    RecommendQuery,
    SingleQuery,
)
from vexor.config.search import (
    FacetParams,
    RecommendParams,
    SearchParams,
)
from vexor.config.segmentation import SegmentationMethod, SegmentationSpec
from vexor.config.settings import VexorSettings

__all__ = [
    # settings
    "VexorSettings",
    # connection
    "ServerConnectionSpec",
    "S3Credentials",
    "RemoteStorageSpec",
    # collection
    "CollectionSpec",
    "HnswSpec",
    "OptimizerSpec",
    "ScalarQuantizationSpec",
    "KeywordIndex",
    "IntegerIndex",
    "FloatIndex",
    "BoolIndex",
    "DatetimeIndex",
    "TextIndex",
    "UuidIndex",
    # embedding
    "DenseModelSpec",
    "SparseModelSpec",
    "LateInteractionModelSpec",
    "EmbeddingSpec",
    # ingestion
    "IngestionSpec",
    "DataFormat",
    # segmentation
    "SegmentationSpec",
    "SegmentationMethod",
    # filtering
    "FilterSpec",
    "ConditionBuilder",
    # llm
    "LLMSpec",
    # observability
    "TracingSpec",
    "LogSpec",
    # search
    "SearchParams",
    "RecommendParams",
    "FacetParams",
    # request
    "SingleQuery",
    "BatchQuery",
    "RecommendQuery",
    "PersonalizedRecommendQuery",
    "AgenticQuery",
]
