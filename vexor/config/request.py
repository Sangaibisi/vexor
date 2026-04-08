"""Request configuration specs (what to search for)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import qdrant_client.conversions.common_types as types
from pydantic import BaseModel, Field, StrictInt, StrictStr
from qdrant_client import models

from vexor.config.search import FacetParams


class SingleQuery(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    """A single search query (text or point-ID for similarity)."""

    text: Union[StrictStr, "types.PointId"]


class BatchQuery(BaseModel, extra="forbid"):
    """Multiple search queries in one call."""

    texts: List[StrictStr]


class AgenticQuery(SingleQuery, extra="forbid"):
    """A search query with field-mapping info for agent tools."""

    field_mapping: Dict[str, Tuple[str, Any]]


class RecommendQuery(BaseModel, extra="forbid"):
    """Input for a recommendation: positive / negative example IDs."""

    positive: Optional[Sequence[models.RecommendExample]] = Field(default_factory=list)
    negative: Optional[Sequence[models.RecommendExample]] = Field(default_factory=list)


class PersonalizedRecommendQuery(RecommendQuery, extra="forbid"):
    """Recommendation anchored to a customer or entity ID."""

    entity_id: Union[StrictInt, StrictStr, List[Union[StrictInt, StrictStr]]]
    feedback: Optional[Dict[StrictStr, List[StrictStr]]] = Field(default_factory=dict)


class RecommendBatchQuery(BaseModel):
    """Wraps a facet config for the digital-twin recommend flow."""

    facet_params: FacetParams


class UpsellOptions(BaseModel, extra="forbid"):
    """Optional upsell target for personalised recommendations."""

    upsell_target: Optional[StrictStr] = Field(default=None)


class RecommendExtraOptions(BaseModel, extra="forbid"):
    """Field-mapping and pre-filter for personalised recommendations."""

    field_map: Dict[StrictStr, StrictStr] = Field(default_factory=dict)
    entity_id_filter: Optional[Union[models.FieldCondition, List[models.FieldCondition]]] = Field(default=None)
