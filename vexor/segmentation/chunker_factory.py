"""Factory for text chunkers — replaces the old ChunkerBuilder + ChunkerRegistry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from vexor.errors import UnsupportedChunkerError

if TYPE_CHECKING:
    from vexor.config.segmentation import SegmentationSpec


def create_chunker(spec: "SegmentationSpec", embedding_model: Optional[str] = None) -> Any:
    """Instantiate a chunker based on the segmentation spec."""
    method = spec.method.value.lower()

    _FACTORIES = {
        "fast": _make_fast,
        "recursive": _make_recursive,
        "semantic": _make_semantic,
        "token": _make_token,
    }

    factory = _FACTORIES.get(method)
    if factory is None:
        raise UnsupportedChunkerError(method, list(_FACTORIES))

    return factory(spec, embedding_model)


def _make_fast(spec: "SegmentationSpec", _model: Optional[str]) -> Any:
    from chonkie import Chunker as FastChunker

    return FastChunker(chunk_size=spec.chunk_size)


def _make_recursive(spec: "SegmentationSpec", _model: Optional[str]) -> Any:
    from chonkie import RecursiveChunker

    return RecursiveChunker(chunk_size=spec.chunk_size, min_characters_per_chunk=spec.min_characters)


def _make_semantic(spec: "SegmentationSpec", model: Optional[str]) -> Any:
    from chonkie import SemanticChunker

    return SemanticChunker(
        chunk_size=spec.chunk_size,
        embedding_model=model,
        min_sentences_per_chunk=spec.min_sentences,
    )


def _make_token(spec: "SegmentationSpec", _model: Optional[str]) -> Any:
    from chonkie import TokenChunker

    return TokenChunker(chunk_size=spec.chunk_size, chunk_overlap=spec.chunk_overlap)
