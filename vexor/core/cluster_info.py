"""ClusterInspector — REST-based cluster / collection introspection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import requests

if TYPE_CHECKING:
    from qdrant_client import QdrantClient, models

    from vexor.config.connection import ServerConnectionSpec


class ClusterInspector:
    """Lightweight wrapper for Qdrant collection and cluster REST queries."""

    def __init__(
        self,
        collection_name: str,
        server: "ServerConnectionSpec",
        client: "QdrantClient",
    ) -> None:
        self._collection_name = collection_name
        self._client = client
        self._protocol = "https" if server.https else "http"
        self._base_url = f"{self._protocol}://{server.host}:{server.port}"
        self._headers = {"api-key": server.api_key} if server.api_key else {}

    def get_collection_info(self) -> "models.CollectionInfo":
        return self._client.get_collection(collection_name=self._collection_name)

    def collection_exists(self) -> bool:
        return self._client.collection_exists(collection_name=self._collection_name)

    def list_collections(self) -> List["models.CollectionsResponse"]:
        return self._client.get_collections()

    def get_cluster_info(self) -> Dict[str, Any]:
        url = f"{self._base_url}/collections/{self._collection_name}/cluster"
        resp = requests.get(url, headers=self._headers, timeout=10)
        return resp.json()
