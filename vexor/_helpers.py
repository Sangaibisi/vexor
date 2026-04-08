"""Internal pure-utility functions shared across vexor modules."""

from __future__ import annotations

import uuid
from datetime import datetime as dt
from itertools import islice
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUSTOMER_ID = "customer_id"
CREATED_AT = "__created_at__"
UPDATED_AT = "__updated_at__"
TARGET = "target"
DATE_FMT = "%Y-%m-%d"
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
POSITIVE_FEEDBACK = "positive"
NEGATIVE_FEEDBACK = "negative"
LOG_FILE_NAME = "vexor.log"
SCROLL_ALL_LIMIT = 2_000_000
UPSERT = "upsert"


# ---------------------------------------------------------------------------
# Iterators
# ---------------------------------------------------------------------------


def iter_batches(iterable: Iterator, size: int = 256) -> Iterator[List[Any]]:
    """Yield successive *size*-element lists from *iterable*."""
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch


def make_id_iterator() -> Iterator[str]:
    """Return an infinite iterator of hex UUID4 strings."""
    return iter(lambda: uuid.uuid4().hex, None)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def normalize_to_list(value: Optional[Union[str, List[str]]]) -> List[str]:
    """Coerce *None*, a single string, or a list of strings into ``list[str]``."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    raise ValueError("Expected None, a string, or a list of strings.")


def ensure_str_list(values: list) -> list:
    """Cast every element to ``str`` if not already."""
    return list(map(str, values)) if not all(isinstance(v, str) for v in values) else values


def quote_if_spaced(name: str) -> str:
    """Wrap *name* in double-quotes when it contains whitespace."""
    return f'"{name}"' if " " in name else name


# ---------------------------------------------------------------------------
# Date / time
# ---------------------------------------------------------------------------


def utc_now_str() -> str:
    """Return the current local datetime as *YYYY-MM-DD HH:MM:SS*."""
    return dt.now().strftime(DATETIME_FMT)


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------


def format_kv_sentence(query: list) -> List[str]:
    """Build one sentence per sub-list from ``[{"charName": …, "charValue": …}, …]``."""
    return [", ".join(f"{item['charName']}: {item['charValue']}" for item in sublist) for sublist in query]


# ---------------------------------------------------------------------------
# Dependency checking
# ---------------------------------------------------------------------------


def require_package(*packages: str, auto_install: bool = False, group_name: Optional[str] = None) -> None:
    """Verify that optional packages are importable; raise a helpful error when not.

    Parameters
    ----------
    packages:
        One or more top-level module names to probe.
    auto_install:
        If ``True``, attempt ``pip install`` automatically.
    group_name:
        Optional extras-group name (e.g. ``"crewai"``) for the install hint.
    """
    import subprocess
    import sys
    from importlib import import_module

    from loguru import logger

    for pkg in packages:
        try:
            import_module(pkg)
        except ModuleNotFoundError as exc:
            logger.warning(str(exc))
            if auto_install:
                target = f"vexor[{group_name}]" if group_name else pkg
                subprocess.check_call([sys.executable, "-m", "pip", "install", target])  # noqa: S603
                logger.info(f"{target} installed successfully.")
            else:
                hint = f"pip install {pkg}"
                if "_" in pkg:
                    hint += f"  (or pip install {pkg.replace('_', '-')})"
                if group_name:
                    hint += f"\n  OR install the extras group: pip install vexor[{group_name}]"
                raise ModuleNotFoundError(
                    f"'{pkg}' is required but not installed.\n  {hint}"
                ) from exc
