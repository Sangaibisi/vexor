"""DuckDB connector for S3-backed data access."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    from vexor.config.connection import RemoteStorageSpec


class DuckDBConnector:
    """Initialise an in-memory DuckDB connection with S3/MinIO secrets."""

    def __init__(self, storage: "RemoteStorageSpec") -> None:
        self.connection: duckdb.DuckDBPyConnection = duckdb.connect(database=":memory:")
        self._load_extensions()
        self._configure_s3(storage.s3)

    def _load_extensions(self) -> None:
        self.connection.install_extension("httpfs")
        self.connection.load_extension("httpfs")

    def _configure_s3(self, s3: "RemoteStorageSpec.s3") -> None:
        if s3.endpoint_url:
            self._create_minio_secret(s3)
        else:
            self._create_aws_secret(s3)

    def _create_minio_secret(self, s3) -> None:
        self.connection.execute(
            f"""
            CREATE SECRET minio_secret (
                TYPE S3,
                KEY_ID '{s3.access_key_id}',
                SECRET '{s3.secret_access_key}',
                ENDPOINT '{s3.endpoint_url}',
                USE_SSL false,
                URL_STYLE 'path'
            );
            """
        )

    def _create_aws_secret(self, s3) -> None:
        region = s3.region or "us-east-1"
        self.connection.execute(
            f"""
            CREATE SECRET aws_secret (
                TYPE S3,
                KEY_ID '{s3.access_key_id}',
                SECRET '{s3.secret_access_key}',
                REGION '{region}'
            );
            """
        )
