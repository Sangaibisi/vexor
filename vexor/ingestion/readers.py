"""Data readers — each yields ``(id_iterator, DataFrame)`` batches."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Iterator, List, Tuple

if TYPE_CHECKING:
    import duckdb
    import pandas as pd

from vexor._helpers import make_id_iterator
from vexor.config.ingestion import DataFormat


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------


def read_parquet(data_dir: str, batch_size: int) -> Iterator[Tuple[Iterator[str], "pd.DataFrame"]]:
    """Yield batches from local ``.parquet`` files."""
    import glob
    import os

    import pyarrow.parquet as pq

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    files = glob.glob(f"{data_dir}/*.parquet")
    if not files:
        raise FileNotFoundError(f"No .parquet files in: {data_dir}")

    for path in files:
        pf = pq.ParquetFile(source=path)
        for batch in pf.iter_batches(batch_size=batch_size):
            yield make_id_iterator(), batch.to_pandas()


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def read_pdf(data_dir: str, batch_size: int, chunker: Any) -> Iterator[Tuple[Iterator[str], "pd.DataFrame"]]:
    """Extract text from PDFs, chunk it, and yield batches."""
    from pathlib import Path

    import pandas as pd
    import pymupdf

    directory = Path(data_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files in: {data_dir}")

    if chunker is None:
        raise ValueError("A chunker (segmentation config) is required for PDF ingestion.")

    pages: List[str] = []
    for pdf_path in pdf_files:
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    pages.append(text)

    chunked = chunker.chunk_batch(pages)
    all_chunks = [chunk for doc_chunks in chunked for chunk in doc_chunks]

    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        df = pd.DataFrame(
            [
                {
                    "text": c.text,
                    "start_index": c.start_index,
                    "end_index": c.end_index,
                    "token_count": c.token_count,
                }
                for c in batch
            ]
        )
        yield make_id_iterator(), df


# ---------------------------------------------------------------------------
# S3 (via DuckDB)
# ---------------------------------------------------------------------------


def read_s3(
    db_conn: "duckdb.DuckDBPyConnection",
    bucket: str,
    file_name: str,
    document: str,
    batch_size: int,
) -> Iterator[Tuple[Iterator[str], "pd.DataFrame"]]:
    """Read parquet data from S3 through DuckDB."""
    parquet_path = f"s3://{bucket}/{file_name}/{document}/*.parquet"

    try:
        total = db_conn.execute("SELECT COUNT(*) FROM read_parquet(?);", [parquet_path]).fetchone()[0]
    except Exception as exc:
        raise ValueError(f"Failed to query S3 data: {exc}") from exc

    for offset in range(0, total, batch_size):
        df = db_conn.execute(
            "SELECT * FROM read_parquet(?) LIMIT ? OFFSET ?;",
            [parquet_path, batch_size, offset],
        ).fetchdf()
        yield make_id_iterator(), df


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_reader(
    *,
    data_dir: str | None,
    data_format: DataFormat,
    batch_size: int,
    chunker: Any = None,
    db_conn: Any = None,
    bucket: str | None = None,
    file_name: str | None = None,
    document: str | None = None,
) -> Iterator[Tuple[Iterator[str], "pd.DataFrame"]]:
    """Return the appropriate reader iterator based on data source."""
    if db_conn is not None:
        if not all([bucket, file_name, document]):
            raise ValueError("bucket, file_name, and document are required for S3 data.")
        return read_s3(db_conn, bucket, file_name, document, batch_size)

    if data_format == DataFormat.PDF:
        return read_pdf(data_dir, batch_size, chunker)

    return read_parquet(data_dir, batch_size)
