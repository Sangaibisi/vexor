"""ColumnResolver — decide which DataFrame columns to embed vs. keep as payload."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from vexor._helpers import ensure_str_list

if TYPE_CHECKING:
    import pandas as pd


class ColumnResolver:
    """Resolve included / excluded columns from a DataFrame."""

    @staticmethod
    def resolve(
        data_columns: "pd.Index",
        columns: List[str],
        is_included: bool,
    ) -> Tuple[List[str], List[str]]:
        """Return ``(included_cols, excluded_cols)``."""
        included: List[str] = []
        excluded: List[str] = []

        if is_included:
            included = list(columns)
        else:
            excluded = list(columns)

        for col in excluded:
            if col not in data_columns:
                raise ValueError(f"'{col}' not found in data columns (excluded list).")

        for col in included:
            if col not in data_columns:
                raise ValueError(f"'{col}' not found in data columns (included list).")

        if included:
            included = ensure_str_list(included)
            excluded = list(data_columns.difference(included))
        elif excluded:
            excluded = ensure_str_list(excluded)
            included = list(data_columns.difference(excluded))
        else:
            included = list(data_columns)

        return included, excluded
