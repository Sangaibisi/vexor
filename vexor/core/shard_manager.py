"""ShardManager — shard key creation and data routing."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

from loguru import logger

from vexor._helpers import normalize_to_list
from vexor.errors import ShardError, ShardKeyNotFoundError

if TYPE_CHECKING:
    import pandas as pd
    from qdrant_client import QdrantClient, models

    from vexor.core.cluster_info import ClusterInspector


class ShardManager:
    """Handles custom shard-key creation and per-shard data routing."""

    # Shared across instances for deduplication
    _known_shard_keys: Set[str] = set()

    def __init__(self, client: "QdrantClient", collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name
        self.shard_keys: List[str] = []

    def configure(self, keys: Optional[Union[str, List[str]]]) -> None:
        """Normalise and store shard keys."""
        self.shard_keys = normalize_to_list(keys)

    def validate_columns(self, data_columns: "pd.Index") -> None:
        """Ensure every requested shard key exists in the data."""
        for key in self.shard_keys:
            if key not in data_columns:
                raise ShardKeyNotFoundError(key)

    def standardize_keys(self, data: "pd.DataFrame") -> Set[Tuple[str, str]]:
        """Return ``{(original, stripped)}`` tuples for every unique shard-key combo."""
        if not self.shard_keys:
            return set()

        grouped = data.groupby(self.shard_keys).groups.keys()
        result: Set[Tuple[str, str]] = set()
        for group in grouped:
            items = group if isinstance(group, tuple) else (group,)
            combined = "_".join(str(v) for v in items)
            result.add((combined, combined.strip()))
        return result

    def ensure_shard_keys(
        self,
        data: "pd.DataFrame",
        inspector: "ClusterInspector",
    ) -> Set[str]:
        """Create any shard keys that do not yet exist on the cluster."""
        unique_keys = self.standardize_keys(data)
        if not unique_keys:
            return set()

        cluster_info = inspector.get_cluster_info()
        existing: Set[str] = set()
        for shard in cluster_info.get("local_shards", []) + cluster_info.get("remote_shards", []):
            key = shard.get("shard_key")
            if key:
                existing.add(key)
        existing.update(self._known_shard_keys)

        new_keys = {stripped for _, stripped in unique_keys if stripped not in existing}
        for key in new_keys:
            self._create_shard_key(key)
            self._known_shard_keys.add(key)

        return {stripped for _, stripped in unique_keys}

    def _create_shard_key(self, key: str, retries: int = 5) -> None:
        for attempt in range(1, retries + 1):
            try:
                self._client.create_shard_key(
                    collection_name=self._collection_name,
                    shard_key=key,
                )
                logger.info(f"Shard key '{key}' created.")
                return
            except Exception:
                if attempt == retries:
                    raise ShardError(self._collection_name)
                time.sleep(0.5 * attempt)
