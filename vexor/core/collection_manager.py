"""CollectionManager — create, delete, recreate, snapshot collections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

from vexor._helpers import quote_if_spaced
from vexor.config.collection import PayloadSchemaParams
from vexor.errors import CollectionCreateError, CollectionDeleteError

if TYPE_CHECKING:
    from vexor.config.collection import CollectionSpec
    from vexor.core.session import VexorSession


class CollectionManager:
    """Manages Qdrant collection lifecycle."""

    def __init__(self, session: "VexorSession") -> None:
        self._session = session
        self._client = session.client
        self._log = session.log

    def ensure_collection(self, spec: "CollectionSpec", *, recreate: bool = False) -> None:
        """Create the collection if it does not exist, optionally recreating it."""
        name = spec.name
        exists = self._client.collection_exists(name)

        if exists and not recreate:
            self._log.info(f"Collection '{name}' already exists — skipping creation.")
            return

        if exists and recreate:
            try:
                self._client.delete_collection(name)
                self._log.info(f"Deleted existing collection '{name}'.")
            except Exception as exc:
                raise CollectionDeleteError(name) from exc

        try:
            params = spec.model_dump(exclude={"name"}, exclude_none=True)
            # Rename keys to match Qdrant client kwargs
            params["collection_name"] = name
            if "optimizer_config" in params:
                params["optimizers_config"] = params.pop("optimizer_config")
            self._client.create_collection(**params)
            self._log.info(f"Collection '{name}' created.")
        except Exception as exc:
            raise CollectionCreateError(name) from exc

    def create_indexes(
        self,
        collection_name: str,
        payload_indexes: Optional[Dict[str, PayloadSchemaParams]] = None,
    ) -> None:
        """Create payload field indexes on the collection."""
        if not payload_indexes:
            return

        for field_name, schema in payload_indexes.items():
            quoted = quote_if_spaced(field_name)
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name=quoted,
                field_schema=schema,
            )
            self._log.info(f"Index created on '{field_name}'.")

    def clone_via_snapshot(self, source: str, target: str) -> None:
        """Snapshot *source* and recover it into *target*."""
        snapshot_info = self._client.create_snapshot(collection_name=source)
        protocol = "https" if self._session.settings.server.https else "http"
        host = self._session.settings.server.host
        port = self._session.settings.server.port
        snap_url = f"{protocol}://{host}:{port}/collections/{source}/snapshots/{snapshot_info.name}"
        self._client.recover_snapshot(collection_name=target, location=snap_url)
        self._log.info(f"Cloned '{source}' -> '{target}' via snapshot.")
