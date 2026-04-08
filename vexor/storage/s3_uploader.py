"""S3DataUploader — upload parquet or PDF data to S3 as Delta tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from vexor._helpers import require_package

if TYPE_CHECKING:
    import pandas as pd

    from vexor.config.connection import S3Credentials
    from vexor.config.observability import LogSpec


class S3DataUploader:
    """Upload local data files to an S3 bucket as DeltaLake tables."""

    def __init__(self, s3: "S3Credentials") -> None:
        self._s3 = s3

        if s3.endpoint_url:
            self._s3_url = f"s3://{s3.bucket_name}/{s3.file_name}/{s3.document_name}"
            self._storage_options = {
                "AWS_ACCESS_KEY_ID": s3.access_key_id,
                "AWS_SECRET_ACCESS_KEY": s3.secret_access_key,
                "AWS_ENDPOINT_URL": s3.endpoint_url,
                "AWS_ALLOW_HTTP": "true",
                "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
            }
        else:
            self._s3_url = f"s3://{s3.bucket_name}/{s3.file_name}/{s3.document_name}"
            self._storage_options = {
                "AWS_ACCESS_KEY_ID": s3.access_key_id,
                "AWS_SECRET_ACCESS_KEY": s3.secret_access_key,
                "AWS_REGION": s3.region or "us-east-1",
                "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
            }

    def upload_data(self, file_path: str) -> None:
        """Read a parquet file and write it to S3 as a DeltaLake table."""
        import pandas as pd

        df = pd.read_parquet(file_path)
        self._write_delta(df)
        logger.info(f"Uploaded {len(df)} rows from '{file_path}' to {self._s3_url}")

    def upload_pdf(self, file_path: str) -> None:
        """Extract text from a PDF and write it to S3 as a DeltaLake table."""
        import pandas as pd
        import pymupdf

        pages = []
        with pymupdf.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    pages.append({"page": page.number + 1, "text": text})

        df = pd.DataFrame(pages)
        self._write_delta(df)
        logger.info(f"Uploaded {len(df)} pages from '{file_path}' to {self._s3_url}")

    def _write_delta(self, df: "pd.DataFrame") -> None:
        from deltalake import DeltaTable
        from deltalake.writer import write_deltalake

        try:
            dt = DeltaTable(self._s3_url, storage_options=self._storage_options)
            write_deltalake(dt, df, mode="append", engine="rust", storage_options=self._storage_options)
        except Exception:
            write_deltalake(self._s3_url, df, mode="overwrite", engine="rust", storage_options=self._storage_options)
