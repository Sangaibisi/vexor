"""Guard functions for search and recommendation validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vexor.errors import InvalidRecommendRequest

if TYPE_CHECKING:
    from vexor.config.request import PersonalizedRecommendQuery, RecommendQuery, UpsellOptions
    from vexor.config.search import RecommendParams


def check_recommend_strategy(query: "RecommendQuery", params: "RecommendParams") -> None:
    """Validate that the recommendation strategy has the required inputs."""
    from qdrant_client import models

    if params.strategy == models.RecommendStrategy.AVERAGE_VECTOR:
        if not query.positive:
            raise InvalidRecommendRequest("AVERAGE_VECTOR strategy requires at least one positive example.")
    elif params.strategy == models.RecommendStrategy.BEST_SCORE:
        if not query.positive and not query.negative:
            raise InvalidRecommendRequest("BEST_SCORE strategy requires at least one positive or negative example.")


def check_personalized_inputs(
    query: "PersonalizedRecommendQuery",
    params: "RecommendParams",
    upsell: "UpsellOptions | None" = None,
) -> None:
    """Validate inputs for personalised recommendation."""
    check_recommend_strategy(query, params)
