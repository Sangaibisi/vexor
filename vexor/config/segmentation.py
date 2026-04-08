"""Text segmentation (chunking) configuration specs."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SegmentationMethod(str, Enum):
    """Supported chunking strategies."""

    FAST = "Fast"
    RECURSIVE = "Recursive"
    SEMANTIC = "Semantic"
    TOKEN = "Token"


class SegmentationSpec(BaseModel, extra="forbid"):
    """How to split text into chunks before embedding."""

    method: SegmentationMethod = Field(default=SegmentationMethod.FAST)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    min_sentences: Optional[int] = Field(default=2)
    min_characters: Optional[int] = Field(default=24)
