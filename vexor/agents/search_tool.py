"""VectorSearchTool — shared tool used by both CrewAI and ReAct agents."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    from vexor.config.request import AgenticQuery
    from vexor.config.search import SearchParams


def run_vector_search(
    query_text: str,
    search_fn: Callable,
    request_template: "AgenticQuery",
    params: "SearchParams",
) -> str:
    """Execute a vector search and return formatted JSON results.

    Shared logic used by both CrewAI tool and ReAct tool.
    """
    from vexor.config.request import SingleQuery

    results = search_fn(
        query=SingleQuery(text=query_text),
        params=params,
    )

    formatted: List[Dict[str, Any]] = []
    points = results.points if hasattr(results, "points") else results
    for point in points:
        entry: Dict[str, Any] = {}
        if point.payload:
            for result_key, (source_key, default) in request_template.field_mapping.items():
                entry[result_key] = point.payload.get(source_key, default)
        formatted.append(entry)

    return json.dumps(formatted, ensure_ascii=False, indent=2)
