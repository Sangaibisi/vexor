"""TextBuilder — convert a DataFrame row into a single embedding-ready string."""

from __future__ import annotations


class TextBuilder:
    """Joins column key-value pairs into a flat sentence."""

    @staticmethod
    def from_row(row: dict) -> str:
        return "  ".join(f"{k}: {v}" for k, v in row.items())
