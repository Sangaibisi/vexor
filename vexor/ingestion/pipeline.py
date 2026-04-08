"""IngestionPipeline — orchestrates the full ingest flow (read, embed, upload)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

from loguru import logger
from qdrant_client import models

from vexor._helpers import CREATED_AT, normalize_to_list, quote_if_spaced, utc_now_str
from vexor.config.collection import PayloadSchemaParams
from vexor.config.ingestion import DataFormat
from vexor.core.cluster_info import ClusterInspector
from vexor.core.collection_manager import CollectionManager
from vexor.core.shard_manager import ShardManager
from vexor.embedding.loader import load_embedder
from vexor.errors import EmptyColumnsError, PayloadNotFoundError, UploadError
from vexor.ingestion.column_resolver import ColumnResolver
from vexor.ingestion.readers import create_reader
from vexor.ingestion.text_builder import TextBuilder

if TYPE_CHECKING:
    import pandas as pd

    from vexor.core.session import VexorSession


class IngestionPipeline:
    """One-shot pipeline: configure once, call :meth:`run` to ingest."""

    def __init__(
        self,
        session: "VexorSession",
        *,
        columns: Optional[Union[str, List[str]]] = None,
        is_columns_included: bool = False,
        payloads: Optional[Union[str, List[str]]] = None,
        shard_keys: Optional[Union[str, List[str]]] = None,
        payload_indexes: Optional[Dict[str, PayloadSchemaParams]] = None,
        add_datetime_payload: bool = False,
    ) -> None:
        self._session = session
        self._settings = session.settings
        self._log = session.log

        # Normalise inputs
        is_pdf = self._settings.ingestion.data_format == DataFormat.PDF
        if is_pdf:
            self._columns = ["text"]
            self._is_included = True
            self._payloads = ["text", "start_index", "end_index", "token_count"]
        else:
            self._columns = normalize_to_list(columns)
            self._is_included = is_columns_included
            self._payloads = normalize_to_list(payloads)

        self._shard_keys = normalize_to_list(shard_keys)
        self._payload_indexes = payload_indexes or {}
        self._add_datetime = add_datetime_payload

        if self._add_datetime and CREATED_AT not in self._payloads:
            self._payloads.append(CREATED_AT)

        # State
        self._included_cols: List[str] = []
        self._excluded_cols: List[str] = []
        self._is_vectorized: bool = False
        self._uploaded: int = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full ingest pipeline."""
        s = self._settings

        # 1. Load embedder
        embedder = load_embedder(
            self._session.client, s.collection, s.embedding, s.ingestion,
        )

        # 2. Ensure collection
        if self._shard_keys:
            s.collection.sharding_method = models.ShardingMethod.CUSTOM
        col_mgr = CollectionManager(self._session)
        col_mgr.ensure_collection(s.collection, recreate=s.ingestion.recreate_collection)

        # 3. Shard manager
        shard_mgr = ShardManager(self._session.client, s.collection.name)
        shard_mgr.configure(self._shard_keys)

        inspector = ClusterInspector(s.collection.name, s.server, self._session.client)

        # 4. Choose upload fn
        upload_fn: Callable = (
            self._upload_with_shards if self._shard_keys else self._upload_direct
        )

        # 5. Create reader
        remote = s.remote_storage
        reader = create_reader(
            data_dir=s.ingestion.data_dir,
            data_format=s.ingestion.data_format,
            batch_size=s.ingestion.batch_size,
            chunker=self._session.chunker,
            db_conn=self._session.db_connection,
            bucket=remote.s3.bucket_name if remote else None,
            file_name=remote.s3.file_name if remote else None,
            document=remote.s3.document_name if remote else None,
        )

        # 6. Iterate batches
        first_batch = True
        for id_iter, batch_df in reader:
            if self._add_datetime:
                batch_df[CREATED_AT] = utc_now_str()

            if first_batch:
                self._resolve_columns(batch_df)
                self._detect_vectorized(batch_df)
                shard_mgr.validate_columns(batch_df.columns)
                first_batch = False

            ids = [next(id_iter) for _ in range(len(batch_df))]
            upload_fn(ids, batch_df, embedder, shard_mgr, inspector)

        # 7. Wait for indexing
        self._wait_for_green(s.collection.name)

        # 8. Create indexes
        if self._payload_indexes:
            col_mgr.create_indexes(s.collection.name, self._payload_indexes)

        # 9. Close DB
        if self._session.db_connection is not None:
            self._session.db_connection.close()

        self._log.info(f"Ingestion complete — {self._uploaded} vectors uploaded.")

    # ------------------------------------------------------------------
    # Upload strategies
    # ------------------------------------------------------------------

    def _upload_direct(self, ids, data, embedder, shard_mgr, inspector, shard_key=None):
        self._do_upload(ids, data, embedder, shard_key)

    def _upload_with_shards(self, ids, data, embedder, shard_mgr, inspector):
        active_keys = shard_mgr.ensure_shard_keys(data, inspector)
        key_cols = shard_mgr.shard_keys

        for group_key, group_df in data.groupby(key_cols):
            sk = "_".join(str(v) for v in (group_key if isinstance(group_key, tuple) else (group_key,)))
            group_ids = [ids[i] for i in group_df.index]
            self._do_upload(group_ids, group_df, embedder, sk.strip())

    def _do_upload(self, ids, data: "pd.DataFrame", embedder, shard_key=None):
        s = self._settings
        method = s.ingestion.upload_method

        # Build payload metadata
        payload_data = data[self._payloads].to_dict(orient="records") if self._payloads else [{}] * len(data)

        if self._is_vectorized:
            vectors = data[self._included_cols].values.tolist()
        else:
            texts = [TextBuilder.from_row(row) for _, row in data[self._included_cols].iterrows()]
            vectors = None
            for fn_name in ["embed_passages", "embed_sparse_passages"]:
                if hasattr(embedder, fn_name) and getattr(embedder, f"has_{'dense' if 'sparse' not in fn_name else 'sparse'}"):
                    embed_fn = getattr(embedder, fn_name)
                    result = embed_fn(texts)
                    if vectors is None:
                        vectors = result
                    else:
                        vectors.update(result)

        collection_name = s.collection.name
        if method == "add":
            self._session.client.add(
                collection_name=collection_name,
                documents=texts if not self._is_vectorized else None,
                metadata=payload_data,
                ids=ids,
            )
        elif method == "upsert":
            points = [
                models.PointStruct(id=pid, vector=vectors if self._is_vectorized else vec, payload=pay)
                for pid, vec, pay in zip(ids, vectors if self._is_vectorized else [vectors] * len(ids), payload_data)
            ]
            self._session.client.upsert(collection_name=collection_name, points=points)
        elif method == "upload":
            self._session.client.upload_collection(
                collection_name=collection_name,
                vectors=vectors,
                payload=payload_data,
                ids=ids,
                parallel=s.ingestion.parallel_threads,
                shard_key_selector=shard_key,
            )

        self._uploaded += len(ids)

        # Verify
        actual = self._session.client.count(collection_name).count
        if actual != self._uploaded:
            raise UploadError(self._uploaded, actual)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_columns(self, data: "pd.DataFrame") -> None:
        for p in self._payloads:
            if p != CREATED_AT and p not in data.columns:
                raise PayloadNotFoundError(p)

        inc, exc = ColumnResolver.resolve(data.columns, self._columns, self._is_included)
        # Remove payload-only columns from embedding columns when not included
        if not self._is_included and self._payloads:
            for p in self._payloads:
                if p in inc and p != CREATED_AT:
                    inc.remove(p)

        if not inc:
            raise EmptyColumnsError()

        self._included_cols = inc
        self._excluded_cols = exc

    def _detect_vectorized(self, data: "pd.DataFrame") -> None:
        if self._settings.ingestion.data_format == DataFormat.VECTORIZED_TABULAR:
            self._is_vectorized = True

    def _wait_for_green(self, collection_name: str) -> None:
        import time

        for _ in range(120):
            info = self._session.client.get_collection(collection_name)
            if info.status == models.CollectionStatus.GREEN:
                return
            time.sleep(0.5)
        self._log.warning("Collection did not reach GREEN status within 60 seconds.")
