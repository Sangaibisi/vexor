"""Vexor — vector similarity search engine built on Qdrant."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

__version__ = "3.0.0-dev1"

# Public API re-exports
from vexor.config.settings import VexorSettings
from vexor.core.session import VexorSession
from vexor.ingestion.pipeline import IngestionPipeline
from vexor.search.engine import SearchEngine
from vexor.search.recommender import Recommender

__all__ = [
    "VexorSettings",
    "VexorSession",
    "IngestionPipeline",
    "SearchEngine",
    "Recommender",
]
