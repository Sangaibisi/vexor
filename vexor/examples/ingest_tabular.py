"""Example: Ingest tabular (parquet) data."""

from vexor import IngestionPipeline, VexorSession
from vexor.config import (
    CollectionSpec,
    DenseModelSpec,
    EmbeddingSpec,
    IngestionSpec,
    KeywordIndex,
    LogSpec,
    SearchParams,
    ServerConnectionSpec,
    SparseModelSpec,
    TextIndex,
    VexorSettings,
)
from vexor.examples.env_config import EnvConfig

if __name__ == "__main__":
    env = EnvConfig()

    settings = VexorSettings(
        collection=CollectionSpec(name=env.COLLECTION_NAME),
        server=ServerConnectionSpec(
            api_key=env.API_KEY,
            host=env.HOST,
            port=env.PORT,
            https=env.HTTPS,
            prefer_grpc=env.PREFER_GRPC,
            grpc_port=env.GRPC_PORT,
            timeout=env.TIMEOUT,
        ),
        embedding=EmbeddingSpec(
            dense=DenseModelSpec(model_name=env.DENSE_EMBEDDING_MODEL_NAME, cache_dir=env.EMBEDDING_CACHE_DIR),
            sparse=SparseModelSpec(model_name=env.SPARSE_EMBEDDING_MODEL_NAME, cache_dir=env.EMBEDDING_CACHE_DIR),
        ),
        ingestion=IngestionSpec(data_dir=env.DATA_DIR, batch_size=env.BATCH_SIZE),
        log=LogSpec(enabled=env.IS_LOG_ACTIVE, production_mode=env.IS_LOG_PROD, log_to_file=env.IS_LOG_TO_FILE),
    )

    session = VexorSession(settings)

    pipeline = IngestionPipeline(
        session,
        columns=["Transaction Amount", "Product Family", "Contract Type", "Order Total Price", "Campaign Name", "Billing ID"],
        is_columns_included=True,
        payloads=["Transaction Amount", "Product Family", "Campaign Name", "Billing ID", "Income Level"],
        payload_indexes={
            "Transaction Amount": KeywordIndex(),
            "Billing ID": KeywordIndex(),
            "Product Family": TextIndex(),
            "Campaign Name": TextIndex(),
        },
        shard_keys=["Product Family"],
    )
    pipeline.run()
    session.close()
