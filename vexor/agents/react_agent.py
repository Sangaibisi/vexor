"""ReactSearchAgent — ReAct-pattern agent using LangGraph."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from vexor._helpers import require_package
from vexor.agents.search_tool import run_vector_search

if TYPE_CHECKING:
    from vexor.config.request import AgenticQuery
    from vexor.config.search import SearchParams
    from vexor.search.engine import SearchEngine

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class ReactSearchAgent:
    """Single ReAct agent that uses a vector-search tool to answer questions."""

    def __init__(self, search_engine: "SearchEngine", llm: Any) -> None:
        require_package("langchain", group_name="langchain")
        self._engine = search_engine
        self._llm = llm

    def run(
        self,
        query: "AgenticQuery",
        params: "SearchParams",
        tracer: Any = None,
    ) -> Any:
        """Stream the agent execution and return the final message."""
        from langchain_core.tools import tool as lc_tool
        from langgraph.prebuilt import create_react_agent

        engine = self._engine
        req_template = query
        search_params = params

        @lc_tool
        def agentic_search_tool(query_text: str) -> str:
            """Search the Qdrant vector database for relevant information."""
            return run_vector_search(query_text, engine.search, req_template, search_params)

        prompt = self._load_prompt()
        agent = create_react_agent(self._llm, [agentic_search_tool])

        config = {}
        if tracer:
            config["callbacks"] = [tracer]

        final_message = None
        for step in agent.stream(
            {"messages": [("user", prompt.format(query=query.text))]},
            config=config,
            stream_mode="values",
        ):
            final_message = step["messages"][-1]

        return final_message

    @staticmethod
    def _load_prompt() -> str:
        path = _PROMPTS_DIR / "react_template.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        template = data.get("template")
        if template is None:
            raise KeyError(f"'template' key not found in: {path}")
        return template
