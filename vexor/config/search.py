"""Search, recommend, and facet parameter specs."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import AliasChoices, BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr
from qdrant_client import models

from vexor.config.filtering import ConditionBuilder, FilterSpec


class SearchParams(BaseModel, extra="forbid"):
    """Parameters shared by all search / scroll operations."""

    collection: StrictStr
    filter: Optional[Union[FilterSpec, models.Filter, None]] = Field(
        default_factory=models.Filter,
        validation_alias=AliasChoices("filter", "scroll_filter"),
    )
    limit: Optional[StrictInt] = Field(default=5)
    scroll_limit: Optional[StrictInt] = Field(default=500_000)
    shard_key: Optional[Union[StrictStr, List[StrictStr], None]] = Field(default=None)
    score_threshold: Optional[Union[StrictFloat, None]] = Field(default=None)

    def resolved_filter(self) -> models.Filter:
        """Return a Qdrant ``models.Filter`` ready for the client."""
        if isinstance(self.filter, FilterSpec):
            return ConditionBuilder().build(self.filter)
        return self.filter if self.filter is not None else models.Filter()


class RecommendParams(SearchParams, extra="forbid"):
    """Extra knobs for recommendation calls."""

    strategy: Optional[models.RecommendStrategy] = Field(default=models.RecommendStrategy.AVERAGE_VECTOR)


class GenericRecommendBatchFilterSpec(BaseModel):
    """Per-request filters for batch recommendation."""

    request_filters: Dict[str, Union[FilterSpec, models.Filter]]


class GenericRecommendBatchParams(RecommendParams, extra="forbid"):
    """Params for batch personalised recommendation."""

    filter: Optional[Union[FilterSpec, GenericRecommendBatchFilterSpec, models.Filter, None]] = Field(
        default_factory=models.Filter,
        validation_alias=AliasChoices("filter", "scroll_filter"),
    )


class FacetParams(BaseModel):
    """Parameters for a facet (unique-value count) query."""

    collection: StrictStr
    key: StrictStr
    filter: Optional[Union[FilterSpec, models.Filter, None]] = Field(default_factory=models.Filter)
    limit: Optional[StrictInt] = Field(default=50)
    exact: Optional[StrictBool] = Field(default=True)
    shard_key: Optional[Union[StrictStr, List[StrictStr], None]] = Field(default=None)

    def resolved_filter(self) -> models.Filter:
        if isinstance(self.filter, FilterSpec):
            return ConditionBuilder().build(self.filter)
        return self.filter if self.filter is not None else models.Filter()
