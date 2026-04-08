"""Example: Dense vector search."""

from vexor import SearchEngine, VexorSession
from vexor.config import (
    BatchQuery,
    CollectionSpec,
    DenseModelSpec,
    EmbeddingSpec,
    FilterSpec,
    IngestionSpec,
    LogSpec,
    SearchParams,
    ServerConnectionSpec,
    SingleQuery,
    SparseModelSpec,
    VexorSettings,
)
from vexor.examples.env_config import EnvConfig

if __name__ == "__main__":
    env = EnvConfig()

    settings = VexorSettings(
        collection=CollectionSpec(name=env.COLLECTION_NAME),
        server=ServerConnectionSpec(
            api_key=env.API_KEY, host=env.HOST, port=env.PORT,
            https=env.HTTPS, prefer_grpc=env.PREFER_GRPC,
            grpc_port=env.GRPC_PORT, timeout=env.TIMEOUT,
        ),
        embedding=EmbeddingSpec(
            dense=DenseModelSpec(model_name=env.DENSE_EMBEDDING_MODEL_NAME, cache_dir=env.EMBEDDING_CACHE_DIR),
            sparse=SparseModelSpec(model_name=env.SPARSE_EMBEDDING_MODEL_NAME, cache_dir=env.EMBEDDING_CACHE_DIR),
        ),
        ingestion=IngestionSpec(data_dir=env.DATA_DIR),
        log=LogSpec(enabled=env.IS_LOG_ACTIVE),
    )

    session = VexorSession(settings)
    engine = SearchEngine(session)

    # Single query
    result = engine.search(
        query=SingleQuery(text="ship movie"),
        params=SearchParams(
            collection=env.COLLECTION_NAME,
            filter=FilterSpec(must_not={"director": "Guillermo del Toro"}),
            limit=5,
        ),
    )
    print(result)

    # Batch query
    results = engine.search_batch(
        query=BatchQuery(texts=["wizard", "Hogwarts"]),
        params=SearchParams(
            collection=env.COLLECTION_NAME,
            filter=FilterSpec(must={"director": "Chris Columbus"}),
            limit=3,
        ),
    )
    print(results)

    session.close()
