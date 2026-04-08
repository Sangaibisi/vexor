"""DigitalTwinRecommender — facet-driven batch recommendation."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List

from qdrant_client import models

from vexor._helpers import CUSTOMER_ID, SCROLL_ALL_LIMIT, TARGET

if TYPE_CHECKING:
    from vexor.config.request import RecommendBatchQuery, RecommendExtraOptions, RecommendQuery
    from vexor.config.search import RecommendParams
    from vexor.search.recommender import Recommender


class DigitalTwinRecommender:
    """Recommend for every unique entity found via a facet query."""

    def __init__(self, recommender: "Recommender") -> None:
        self._recommender = recommender
        self._client = recommender._client
        self._log = recommender._log

    def compute(
        self,
        batch_query: "RecommendBatchQuery",
        extra: "RecommendExtraOptions",
    ) -> List[Dict[str, Any]]:
        """Run recommendations for each unique entity in the facet results."""
        from vexor.config.request import RecommendQuery
        from vexor.config.search import RecommendParams

        fp = batch_query.facet_params
        customer_field = extra.field_map.get(CUSTOMER_ID, CUSTOMER_ID)
        target_field = extra.field_map.get(TARGET, TARGET)

        # Get unique entity values
        facet_hits = self._client.facet(
            collection_name=fp.collection,
            key=customer_field,
            facet_filter=fp.resolved_filter(),
            limit=fp.limit,
            exact=fp.exact,
            shard_key_selector=fp.shard_key,
        ).hits

        queries: List[RecommendQuery] = []
        params_list: List[RecommendParams] = []

        for hit in facet_hits:
            entity_id = hit.value
            records, _ = self._client.scroll(
                collection_name=fp.collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key=customer_field, match=models.MatchValue(value=entity_id))],
                ),
                limit=SCROLL_ALL_LIMIT,
            )

            if not records:
                continue

            existing = [r.payload.get(target_field) for r in records if r.payload.get(target_field)]
            must_not = [
                models.FieldCondition(key=target_field, match=models.MatchValue(value=t))
                for t in existing
            ]

            q = RecommendQuery(positive=[r.id for r in records])
            p = RecommendParams(
                collection=fp.collection,
                filter=models.Filter(must_not=must_not),
                limit=fp.limit,
            )
            queries.append(q)
            params_list.append(p)

        if not queries:
            return []

        responses = self._recommender.find_similar_batch_filtered(queries, params_list)

        results: List[Dict[str, Any]] = []
        for resp in responses:
            counts = Counter(
                p.payload.get(target_field) for p in resp.points if p.payload and p.payload.get(target_field)
            )
            results.append(dict(counts))

        return results
