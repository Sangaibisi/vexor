"""Tracing factory — replaces TracingBuilder + TracingRegistry."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

from vexor._helpers import require_package

if TYPE_CHECKING:
    from vexor.config.observability import TracingSpec


def create_tracer(spec: "TracingSpec", provider_name: str) -> Optional[Any]:
    """Return a tracer/callback for the chosen observability platform."""
    _TRACERS = {
        "AgentOps": _agentops_tracer,
        "Langfuse": _langfuse_tracer,
        "Opik": _opik_tracer,
    }
    factory = _TRACERS.get(spec.platform)
    if factory is None:
        raise ValueError(f"Unsupported tracing platform: {spec.platform}")
    return factory(spec, provider_name)


# ---------------------------------------------------------------------------
# AgentOps
# ---------------------------------------------------------------------------


def _agentops_tracer(spec: "TracingSpec", provider_name: str) -> Optional[Any]:
    require_package("agentops", group_name="agentops")
    import agentops

    agentops.init(api_key=spec.api_key, default_tags=[spec.project_name])
    return None


# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------


def _langfuse_tracer(spec: "TracingSpec", provider_name: str) -> Optional[Any]:
    require_package("langfuse", group_name="langfuse")

    os.environ["LANGFUSE_SECRET_KEY"] = spec.api_key
    if spec.langfuse_public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = spec.langfuse_public_key
    if spec.langfuse_host:
        os.environ["LANGFUSE_HOST"] = spec.langfuse_host

    from langfuse import Langfuse

    lf = Langfuse()
    lf.auth_check()

    if provider_name == "CrewAI":
        from openinference.instrumentation.crewai import CrewAIInstrumentor

        CrewAIInstrumentor().instrument()
        return None
    else:
        from langfuse.callback import CallbackHandler

        return CallbackHandler()


# ---------------------------------------------------------------------------
# Opik
# ---------------------------------------------------------------------------


def _opik_tracer(spec: "TracingSpec", provider_name: str) -> Optional[Any]:
    require_package("opik", group_name="opik")
    import opik

    opik.configure(api_key=spec.api_key, use_local=False)

    if provider_name == "CrewAI":
        from opik.integrations.crewai import track_crewai

        track_crewai(project_name=spec.project_name)
        return None
    else:
        from opik.integrations.langchain import OpikTracer

        return OpikTracer(tags=["vexor"], project_name=spec.project_name)
