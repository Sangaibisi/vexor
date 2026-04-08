# Domain Glossary

## Core Entities

### Configuration
- **VexorSettings** — Top-level configuration aggregating all sub-specs
- **CollectionSpec** — Qdrant collection definition
- **ServerConnectionSpec** — Qdrant server connection parameters
- **EmbeddingSpec** — Embedding model configuration
- **IngestionSpec** — Data loading parameters
- **FilterSpec** — Declarative filter conditions
- **SearchParams / RecommendParams** — Query parameters

### Request Types
- **SingleQuery** — Single text or point-ID search
- **BatchQuery** — Multiple queries
- **RecommendQuery** — Positive/negative examples
- **AgenticQuery** — Query with field mapping for agents

### Core Components
- **VexorSession** — Central connection object
- **CollectionManager** — Collection CRUD
- **ShardManager** — Shard key management
- **ClusterInspector** — Cluster info queries

### Embedding
- **Embedder** — Protocol contract
- **FastEmbedAdapter** / **SentenceBERTAdapter** — Implementations

### Search
- **SearchEngine** — Dense, hybrid, browse, facet
- **Recommender** — ID-based recommendations
- **DigitalTwinRecommender** — Batch recommendations

### Agents
- **CrewSearchAgent** — CrewAI multi-agent
- **ReactSearchAgent** — LangGraph ReAct
- **VectorSearchTool** — Shared tool

### Enums
- **DataFormat** — tabular, vectorized_tabular, pdf
- **SegmentationMethod** — Fast, Recursive, Semantic, Token
