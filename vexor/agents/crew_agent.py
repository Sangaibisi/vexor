"""CrewSearchAgent — multi-agent search using CrewAI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from vexor._helpers import require_package
from vexor.agents.search_tool import run_vector_search

if TYPE_CHECKING:
    from vexor.config.request import AgenticQuery
    from vexor.config.search import SearchParams
    from vexor.search.engine import SearchEngine

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class CrewSearchAgent:
    """Orchestrate a CrewAI crew that searches and answers questions."""

    def __init__(self, search_engine: "SearchEngine", llm: Any) -> None:
        require_package("crewai", group_name="crewai")
        self._engine = search_engine
        self._llm = llm

    def run(
        self,
        query: "AgenticQuery",
        params: "SearchParams",
    ) -> Any:
        """Kick off the crew and return the structured result."""
        from crewai import Agent, Crew, Process, Task
        from crewai.tools import BaseTool

        # Build tool
        engine = self._engine
        req_template = query
        search_params = params

        class _QdrantTool(BaseTool):
            name: str = "QdrantVectorSearchTool"
            description: str = "Search the Qdrant vector database for relevant information."

            def _run(self, query_text: str) -> str:
                return run_vector_search(query_text, engine.search, req_template, search_params)

        tool = _QdrantTool()

        # Load YAML configs
        import yaml

        with open(_PROMPTS_DIR / "crew_agents.yaml") as f:
            agents_cfg = yaml.safe_load(f)
        with open(_PROMPTS_DIR / "crew_tasks.yaml") as f:
            tasks_cfg = yaml.safe_load(f)

        # Build agents
        search_agent = Agent(
            role=agents_cfg["search_agent"]["role"],
            goal=agents_cfg["search_agent"]["goal"],
            backstory=agents_cfg["search_agent"]["backstory"],
            tools=[tool],
            llm=self._llm,
        )
        answer_agent = Agent(
            role=agents_cfg["answer_agent"]["role"],
            goal=agents_cfg["answer_agent"]["goal"],
            backstory=agents_cfg["answer_agent"]["backstory"],
            tools=[tool],
            llm=self._llm,
        )

        # Build tasks
        search_task = Task(
            description=tasks_cfg["search_task"]["description"],
            expected_output=tasks_cfg["search_task"]["expected_output"],
            agent=search_agent,
        )
        answer_task = Task(
            description=tasks_cfg["answer_task"]["description"],
            expected_output=tasks_cfg["answer_task"]["expected_output"],
            agent=answer_agent,
        )

        # Execute
        crew = Crew(
            agents=[search_agent, answer_agent],
            tasks=[search_task, answer_task],
            process=Process.sequential,
        )
        return crew.kickoff(inputs={"query": query.text})
