"""SearchEngine — text-based vector search operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from qdrant_client import models
from qdrant_client.models import PointId, QueryResponse

from vexor._helpers import SCROLL_ALL_LIMIT, quote_if_spaced
from vexor.config.filtering import ConditionBuilder, FilterSpec
from vexor.embedding.loader import load_embedder
from vexor.search.hybrid import hybrid_search, hybrid_search_batch

if TYPE_CHECKING:
    from vexor.config.request import AgenticQuery, BatchQuery, SingleQuery
    from vexor.config.search import FacetParams, SearchParams
    from vexor.core.session import VexorSession


class SearchEngine:
    """All text-based vector search, scroll, and facet operations."""

    def __init__(self, session: "VexorSession") -> None:
        self._session = session
        self._client = session.client
        self._log = session.log
        s = session.settings
        self._embedder = load_embedder(s.server, s.collection, s.embedding, s.ingestion)

    # ------------------------------------------------------------------
    # Dense search
    # ------------------------------------------------------------------

    def search(self, query: "SingleQuery", params: "SearchParams") -> QueryResponse:
        """Single-query dense vector search."""
        embedding = self._embedder.embed_query(query.text)
        return self._client.query_points(
            collection_name=params.collection,
            query=embedding,
            query_filter=params.resolved_filter(),
            limit=params.limit,
            shard_key_selector=params.shard_key,
            score_threshold=params.score_threshold,
        )

    def search_batch(self, query: "BatchQuery", params: "SearchParams") -> list[QueryResponse]:
        """Batch dense vector search."""
        requests = [
            models.QueryRequest(
                query=self._embedder.embed_query(text),
                filter=params.resolved_filter(),
                limit=params.limit,
                shard_key_selector=params.shard_key,
                score_threshold=params.score_threshold,
            )
            for text in query.texts
        ]
        return self._client.query_batch_points(
            collection_name=params.collection,
            requests=requests,
        )

    # ------------------------------------------------------------------
    # FastEmbed-native search
    # ------------------------------------------------------------------

    def search_fastembed(self, query: "SingleQuery", params: "SearchParams") -> List[QueryResponse]:
        """Search using the Qdrant client's built-in FastEmbed query method."""
        return self._client.query(
            collection_name=params.collection,
            query_text=query.text,
            query_filter=params.resolved_filter(),
            limit=params.limit,
        )

    def search_fastembed_batch(self, query: "BatchQuery", params: "SearchParams") -> List[List[QueryResponse]]:
        """Batch FastEmbed search."""
        return self._client.query_batch(
            collection_name=params.collection,
            query_texts=query.texts,
            query_filter=[params.resolved_filter()] * len(query.texts),
            limit=params.limit,
        )

    # ------------------------------------------------------------------
    # Hybrid search  (delegates to hybrid module)
    # ------------------------------------------------------------------

    def hybrid_search(self, query: "SingleQuery", params: "SearchParams") -> QueryResponse:
        return hybrid_search(self._client, self._embedder, query, params)

    def hybrid_search_batch(self, query: "BatchQuery", params: "SearchParams") -> list[QueryResponse]:
        return hybrid_search_batch(self._client, self._embedder, query, params)

    # ------------------------------------------------------------------
    # Scroll / browse
    # ------------------------------------------------------------------

    def browse(
        self,
        params: "SearchParams",
    ) -> Tuple[List[models.Record], Optional[PointId]]:
        """Scroll through all matching records."""
        return self._client.scroll(
            collection_name=params.collection,
            scroll_filter=params.resolved_filter(),
            limit=min(params.scroll_limit, SCROLL_ALL_LIMIT),
            shard_key_selector=params.shard_key,
        )

    # ------------------------------------------------------------------
    # Facet
    # ------------------------------------------------------------------

    def facet_counts(self, params: "FacetParams") -> list:
        """Return unique values and their counts for a payload field."""
        return self._client.facet(
            collection_name=params.collection,
            key=quote_if_spaced(params.key),
            facet_filter=params.resolved_filter(),
            limit=params.limit,
            exact=params.exact,
            shard_key_selector=params.shard_key,
        ).hits
