"""Filter configuration and condition builder."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from qdrant_client import models

from vexor._helpers import CREATED_AT


class FilterSpec(BaseModel, extra="forbid"):
    """Declarative filter that maps to a Qdrant ``models.Filter``."""

    must: Optional[Union[Dict[str, Any], List[models.Condition]]] = Field(default=None)
    must_not: Optional[Union[Dict[str, Any], List[models.Condition]]] = Field(default=None)
    should: Optional[Union[Dict[str, Any], List[models.Condition]]] = Field(default=None)
    min_should: Optional[Union[Dict[str, Any], models.MinShould]] = Field(default=None)


class ConditionBuilder:
    """Converts a :class:`FilterSpec` into a Qdrant ``models.Filter``."""

    def build(self, spec: FilterSpec) -> models.Filter:
        return models.Filter(
            must=self._parse_conditions(spec.must),
            must_not=self._parse_conditions(spec.must_not),
            should=self._parse_conditions(spec.should),
        )

    # ------------------------------------------------------------------

    def _parse_conditions(self, raw: Union[dict, list, None]) -> list[models.FieldCondition]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        return [self._make_condition(key, value) for key, value in raw.items()]

    @staticmethod
    def _make_condition(key: str, value: Any) -> models.FieldCondition:
        key = f'"{key}"' if " " in key else key
        if isinstance(value, dict):
            range_cls = models.DatetimeRange if key == CREATED_AT else models.Range
            return models.FieldCondition(key=key, range=range_cls(**value))
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))
