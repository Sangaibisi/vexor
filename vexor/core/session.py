"""VexorSession — the central connection/context object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import requests
from loguru import logger
from qdrant_client import QdrantClient

from vexor.errors import ConnectionError
from vexor.observability.log_setup import configure_logging

if TYPE_CHECKING:
    from vexor.config.settings import VexorSettings


class VexorSession:
    """Wraps a :class:`QdrantClient` plus shared resources (logger, DB, chunker, LLM).

    Every other vexor component receives a ``VexorSession`` via its constructor
    instead of inheriting from a deep class hierarchy.
    """

    def __init__(self, settings: "VexorSettings") -> None:
        self.settings = settings
        self.log = configure_logging(settings.log)

        # Qdrant client
        self.client: QdrantClient = self._connect(settings.server)

        # Lazy-initialised resources (populated when needed)
        self.db_connection: Optional[Any] = None
        self.chunker: Optional[Any] = None
        self.llm_client: Optional[Any] = None
        self.llm_provider: Optional[Any] = None
        self.tracer: Optional[Any] = None

        self._init_optional_resources()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self, server: Any) -> QdrantClient:
        if not self._is_reachable(server):
            raise ConnectionError()

        from vexor import __version__

        self.log.info(f"vexor v{__version__}")
        return QdrantClient(**server.model_dump())

    @staticmethod
    def _is_reachable(server: Any) -> bool:
        protocol = "https" if server.https else "http"
        try:
            requests.get(f"{protocol}://{server.host}:{server.port}/dashboard/", timeout=5)
            return True
        except requests.ConnectionError:
            logger.error(f"Cannot reach Qdrant at {server.host}:{server.port}")
            return False

    # ------------------------------------------------------------------
    # Optional resources
    # ------------------------------------------------------------------

    def _init_optional_resources(self) -> None:
        s = self.settings

        # Database (DuckDB for S3)
        if s.remote_storage is not None:
            from vexor.storage.duckdb_connector import DuckDBConnector

            self.db_connection = DuckDBConnector(s.remote_storage).connection

        # Text chunker
        if s.segmentation is not None:
            from vexor.segmentation.chunker_factory import create_chunker

            model_name = s.embedding.dense.model_name if s.embedding.dense else None
            self.chunker = create_chunker(s.segmentation, model_name)

        # LLM client + provider
        if s.llm is not None:
            from vexor.llm.client_factory import create_llm_client, create_llm_provider

            self.llm_client = create_llm_client(s.llm)
            self.llm_provider = create_llm_provider(s.llm)

            if s.tracing is not None:
                from vexor.llm.tracing_factory import create_tracer

                self.tracer = create_tracer(s.tracing, s.llm.provider)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying Qdrant connection."""
        self.client.close()
        self.log.info("Session closed.")
