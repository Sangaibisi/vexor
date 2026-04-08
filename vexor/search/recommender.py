"""Recommender — ID-based similarity recommendations."""

from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from qdrant_client import models
from qdrant_client.models import QueryResponse, ScoredPoint

from vexor._helpers import (
    CUSTOMER_ID,
    NEGATIVE_FEEDBACK,
    POSITIVE_FEEDBACK,
    SCROLL_ALL_LIMIT,
    TARGET,
    format_kv_sentence,
)
from vexor.config.filtering import ConditionBuilder, FilterSpec
from vexor.errors import NoRecordsFoundWarning
from vexor.search.validators import check_personalized_inputs, check_recommend_strategy

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from vexor.config.request import (
        PersonalizedRecommendQuery,
        RecommendBatchQuery,
        RecommendExtraOptions,
        RecommendQuery,
        UpsellOptions,
    )
    from vexor.config.search import GenericRecommendBatchParams, RecommendParams, SearchParams
    from vexor.core.session import VexorSession
    from vexor.embedding.protocol import Embedder


class Recommender:
    """Recommendation engine — find similar items by positive / negative examples."""

    def __init__(self, session: "VexorSession", embedder: "Embedder") -> None:
        self._session = session
        self._client = session.client
        self._log = session.log
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Basic recommend
    # ------------------------------------------------------------------

    def find_similar(
        self,
        query: "RecommendQuery",
        params: "RecommendParams",
    ) -> List[ScoredPoint]:
        """Find items similar to positive examples, dissimilar to negative."""
        check_recommend_strategy(query, params)
        resp = self._client.query_points(
            collection_name=params.collection,
            query=models.RecommendQuery(
                positive=query.positive,
                negative=query.negative,
                strategy=params.strategy,
            ),
            query_filter=params.resolved_filter(),
            limit=params.limit,
            shard_key_selector=params.shard_key,
        )
        return resp.points

    def find_similar_batch(
        self,
        queries: List["RecommendQuery"],
        params: "RecommendParams",
    ) -> List[List[ScoredPoint]]:
        """Batch recommend — one set of params, multiple query sets."""
        for q in queries:
            check_recommend_strategy(q, params)

        requests = [
            models.QueryRequest(
                query=models.RecommendQuery(
                    positive=q.positive,
                    negative=q.negative,
                    strategy=params.strategy,
                ),
                filter=params.resolved_filter(),
                limit=params.limit,
                shard_key_selector=params.shard_key,
            )
            for q in queries
        ]
        responses = self._client.query_batch_points(
            collection_name=params.collection,
            requests=requests,
        )
        return [r.points for r in responses]

    def find_similar_batch_filtered(
        self,
        queries: List["RecommendQuery"],
        params_list: List["RecommendParams"],
    ) -> List[QueryResponse]:
        """Each query gets its own params (and thus its own filter)."""
        requests = [
            models.QueryRequest(
                query=models.RecommendQuery(
                    positive=q.positive,
                    negative=q.negative,
                    strategy=p.strategy,
                ),
                filter=p.resolved_filter(),
                limit=p.limit,
                shard_key_selector=p.shard_key,
            )
            for q, p in zip(queries, params_list)
        ]
        return self._client.query_batch_points(
            collection_name=params_list[0].collection,
            requests=requests,
        )

    # ------------------------------------------------------------------
    # Personalised recommend  (generic_recommend equivalent)
    # ------------------------------------------------------------------

    def personalized(
        self,
        query: "PersonalizedRecommendQuery",
        params: "RecommendParams",
        extra: "RecommendExtraOptions",
        upsell: Optional["UpsellOptions"] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Recommend based on a customer/entity ID with feedback filtering."""
        check_personalized_inputs(query, params, upsell)

        target_field = extra.field_map.get(TARGET, TARGET)
        customer_field = extra.field_map.get(CUSTOMER_ID, CUSTOMER_ID)

        # If shard_key and positive IDs are given, skip scroll
        if params.shard_key and query.positive:
            points = self.find_similar(query, params)
            return self._format_results(points, target_field)

        # Scroll for user records
        entity_filter = extra.entity_id_filter or models.FieldCondition(
            key=customer_field,
            match=models.MatchValue(value=query.entity_id),
        )
        records, _ = self._client.scroll(
            collection_name=params.collection,
            scroll_filter=models.Filter(must=[entity_filter]),
            limit=SCROLL_ALL_LIMIT,
            shard_key_selector=params.shard_key,
        )

        if not records:
            raise NoRecordsFoundWarning(query.entity_id)

        # Prepare positive/negative from records
        prepared_query, must_not_filter, shard_key = self._prepare_recommend_data(
            query, records, params.resolved_filter(), extra, upsell, target_field,
        )

        # Run recommend
        rec_params = params.model_copy(update={"shard_key": shard_key})
        # Merge must_not into filter
        base_filter = params.resolved_filter()
        if must_not_filter:
            base_filter.must_not = (base_filter.must_not or []) + must_not_filter

        points = self.find_similar(prepared_query, rec_params)
        return self._format_results(points, target_field)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_recommend_data(
        self,
        query: "PersonalizedRecommendQuery",
        records: list,
        base_filter: models.Filter,
        extra: "RecommendExtraOptions",
        upsell: Optional["UpsellOptions"],
        target_field: str,
    ) -> Tuple["PersonalizedRecommendQuery", list, Optional[str]]:
        """Extract existing targets from records and build must_not filter."""
        existing_targets = [r.payload.get(target_field) for r in records if r.payload.get(target_field)]
        must_not = [models.FieldCondition(key=target_field, match=models.MatchValue(value=t)) for t in existing_targets]

        # Handle feedback-based filtering
        if query.feedback:
            neg_feedback = query.feedback.get(NEGATIVE_FEEDBACK, [])
            must_not.extend(
                models.FieldCondition(key=target_field, match=models.MatchValue(value=t)) for t in neg_feedback
            )

        # Extract shard key if available
        shard_key = None
        if records and records[0].payload:
            for key in extra.field_map:
                if key in records[0].payload:
                    shard_key = records[0].payload[key]
                    break

        # Use record IDs as positive examples if none provided
        if not query.positive:
            from vexor.config.request import PersonalizedRecommendQuery

            query = query.model_copy(update={"positive": [r.id for r in records]})

        return query, must_not, shard_key

    @staticmethod
    def _format_results(points: list, target_field: str) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "predictCharType": [
                {"charName": p.payload.get(target_field, ""), "charValue": p.score}
                for p in points
                if p.payload
            ]
        }
