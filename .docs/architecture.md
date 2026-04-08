# Architecture Documentation

## Overview

**Vexor** is a vector similarity search engine built on Qdrant with support for multiple embedding models and AI-powered search capabilities. It uses composition over inheritance, factory functions instead of registry patterns, and domain-split configuration.

## Design Principles

1. **Composition over inheritance** — No deep class hierarchies. Components receive a `VexorSession` via constructor.
2. **Factory functions** — Simple dict-dispatch factories replace Builder/Registry patterns.
3. **Domain-split config** — Configuration split into ~11 focused Pydantic modules under `config/`.
4. **Guard functions** — Explicit validation functions called inline, not decorator-based validators.
5. **Protocol-based contracts** — `typing.Protocol` for the Embedder interface, not ABC.

## Architecture Layers

### 1. Configuration Layer (`vexor/config/`)
All Pydantic models that define how vexor behaves. Split by domain:
- `settings.py` — Top-level `VexorSettings` aggregator
- `connection.py` — `ServerConnectionSpec`, `S3Credentials`, `RemoteStorageSpec`
- `collection.py` — `CollectionSpec`, index types, HNSW/optimizer/quantization specs
- `embedding.py` — `DenseModelSpec`, `SparseModelSpec`, `EmbeddingSpec`
- `ingestion.py` — `IngestionSpec`, `DataFormat` enum
- `segmentation.py` — `SegmentationSpec`, `SegmentationMethod` enum
- `filtering.py` — `FilterSpec`, `ConditionBuilder`
- `llm.py` — `LLMSpec`
- `observability.py` — `TracingSpec`, `LogSpec`
- `search.py` — `SearchParams`, `RecommendParams`, `FacetParams`
- `request.py` — `SingleQuery`, `BatchQuery`, `RecommendQuery`, etc.

### 2. Core Layer (`vexor/core/`)
- `VexorSession` — Central context object wrapping QdrantClient + logger + optional resources
- `CollectionManager` — Create, delete, recreate, snapshot collections
- `ShardManager` — Custom shard key creation and data routing
- `ClusterInspector` — REST-based cluster/collection introspection

### 3. Embedding Layer (`vexor/embedding/`)
- `Embedder` Protocol — Structural interface for embedding adapters
- `FastEmbedAdapter` — FastEmbed dense + sparse models via Qdrant client
- `SentenceBERTAdapter` — Fallback using sentence-transformers
- `load_embedder()` — Factory: tries FastEmbed first, falls back to SentenceBERT

### 4. Ingestion Layer (`vexor/ingestion/`)
- `IngestionPipeline` — Orchestrates read -> embed -> upload flow
- `readers.py` — `read_parquet()`, `read_pdf()`, `read_s3()` generators
- `ColumnResolver` — Include/exclude column logic
- `TextBuilder` — Row-to-text conversion for embedding

### 5. Search Layer (`vexor/search/`)
- `SearchEngine` — Dense, FastEmbed, hybrid search, browse, facet
- `Recommender` — Basic and personalised recommendations
- `DigitalTwinRecommender` — Facet-driven batch recommendations
- `hybrid.py` — Shared `_build_prefetch()` for dense + sparse fusion
- `validators.py` — Guard functions for strategy validation

### 6. Agent Layer (`vexor/agents/`)
- `CrewSearchAgent` — Multi-agent CrewAI search and answer
- `ReactSearchAgent` — LangGraph ReAct pattern agent
- `VectorSearchTool` — Shared search tool used by both agent types

### 7. Integration Layer
- `vexor/storage/` — `DuckDBConnector`, `S3DataUploader`
- `vexor/llm/` — `create_llm_client()`, `create_llm_provider()`, `create_tracer()`
- `vexor/segmentation/` — `create_chunker()` factory
- `vexor/observability/` — `configure_logging()` function

## Data Flow

### Ingestion
```
VexorSettings -> VexorSession -> IngestionPipeline
  -> create_reader() yields (id_iter, DataFrame) batches
  -> load_embedder() creates adapter
  -> CollectionManager.ensure_collection()
  -> For each batch: embed -> upload to Qdrant
  -> CollectionManager.create_indexes()
```

### Search
```
VexorSettings -> VexorSession -> SearchEngine
  -> embedder.embed_query(text) -> vector
  -> client.query_points(vector) -> QueryResponse
```

### Agentic RAG
```
SearchEngine -> CrewSearchAgent/ReactSearchAgent
  -> VectorSearchTool wraps engine.search()
  -> Agent orchestrates search + answer generation
```
