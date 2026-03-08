# System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                       │
│            (Search, Analytics, Insights, History)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                         HTTP/REST
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    FastAPI REST Server                           │
│                    (Port 8000)                                   │
├─────────────────────────────────────────────────────────────────┤
│ POST /query          - Execute semantic search with caching     │
│ GET  /cache/stats    - Cache performance metrics                │
│ DELETE /cache        - Clear cache                              │
│ GET  /status         - System status                            │
│ GET  /health         - Health check                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐  ┌────────────────────┐  ┌────────────────────┐
│  Embeddings  │  │    Clustering      │  │ Semantic Cache     │
│  (SentTrans) │  │   (GMM + PCA)      │  │ (Dict-based)       │
└──────────────┘  └────────────────────┘  └────────────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Pinecone Index │
                    │  (Vector DB)    │
                    └─────────────────┘
```

## Data Flow

### 1. Indexing Pipeline (One-time Setup)

```
Raw Dataset (20 Newsgroups)
    │
    ▼
Text Preprocessing
(cleaning, length filtering)
    │
    ▼
Embedding Generation
(Sentence Transformers, 384-dim)
    │
    ▼
Clustering (GMM)
(20 components, soft assignments)
    │
    ▼
Pinecone Indexing
(Serverless storage)
```

### 2. Query Execution Flow

```
User Query (via Streamlit)
    │
    ▼
1. Generate Query Embedding
(Sentence Transformers)
    │
    ▼
2. Predict Cluster
(GMM soft assignment)
    │
    ▼
3. Check Semantic Cache
(local dict lookup)
    │
    ├─── Cache HIT ──────┐
    │                    ▼
    │              Return cached result
    │              + similarity score
    │
    └─── Cache MISS ─────┐
                         ▼
                  Query Pinecone
                  (approximate KNN)
                         │
                         ▼
                  Retrieve top-k results
                         │
                         ▼
                  Cache the result
                         │
                         ▼
                  Return results
                  + metadata

```

## Component Details

### 1. Embedding Layer (src/embeddings.py)

**Purpose**: Convert text to fixed-size vectors

**Implementation**:
- Model: `all-MiniLM-L6-v2` (Sentence Transformers)
- Dimensions: 384
- Speed: ~2000 docs/sec on CPU
- Similarity: Cosine distance (L2 normalized)

**Key Methods**:
```python
EmbeddingManager
├── embed_text(text: str) -> ndarray[384]
├── embed_batch(texts: List[str]) -> ndarray[N, 384]
└── get_dimension() -> int
```

### 2. Clustering Layer (src/clustering.py)

**Purpose**: Group similar documents probabilistically

**Implementation**:
- Algorithm: Gaussian Mixture Model (GMM)
- Components: 20 (configurable)
- Dimensionality Reduction: PCA (20 components)
- Assignment Type: Soft (probabilistic)

**Key Methods**:
```python
GMMClusterer
├── fit(embeddings: ndarray) -> None
├── predict(embedding: ndarray) -> int (hard assignment)
├── predict_proba(embedding: ndarray) -> ndarray (soft)
├── get_cluster_keywords(cluster_id: int) -> List[str]
└── get_cluster_size(cluster_id: int) -> int
```

**Benefits**:
- Soft assignments allow documents in multiple clusters
- Probabilistic confidence scores for clustering
- Efficient inference with PCA preprocessing
- Interpretable cluster keywords via TF-IDF

### 3. Semantic Cache (src/semantic_cache.py)

**Purpose**: Cache frequent query results based on semantic similarity

**Implementation**:
- Storage: Dictionary with cluster-based partitioning
- Structure: `cluster_id -> List[CacheEntry]`
- Similarity Metric: Cosine distance
- Threshold: 0.85 (configurable)

**Key Methods**:
```python
SemanticCache
├── get(query_embedding, cluster_id, threshold) -> Optional[CachedResult]
├── put(query_embedding, cluster_id, result) -> None
├── clear() -> None
├── get_stats() -> CacheStats
└── update_stats(cache_hit: bool) -> None
```

**Data Structure**:
```python
class CacheEntry:
    query_embedding: ndarray[384]
    cluster_id: int
    result: QueryResult
    timestamp: datetime
    hit_count: int
```

### 4. Vector Database (src/pinecone_client.py)

**Purpose**: Fast approximate nearest neighbor search at scale

**Pinecone Configuration**:
- Deployment: Serverless (AWS)
- Index Type: Approximate KNN (HNSW)
- Dimensionality: 384 (from Sentence Transformers)
- Metric: Cosine similarity
- Operations: Upsert, query, delete

**Key Methods**:
```python
PineconeClient
├── create_or_connect_index() -> None
├── upsert_vectors(vectors, ids, metadata) -> None
├── query(vector, top_k) -> List[Match]
├── delete_vectors(ids) -> None
└── get_index_stats() -> IndexStats
```

### 5. FastAPI Service (src/app.py)

**Purpose**: REST API for query execution and monitoring

**Endpoints**:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check |
| GET | `/status` | System status |
| POST | `/query` | Execute search with caching |
| GET | `/cache/stats` | Cache metrics |
| DELETE | `/cache` | Clear cache |

**Request/Response Models**:
```python
class QueryRequest:
    text: str                              # Query text
    top_k: int = 5                        # Number of results
    use_cache: bool = True                # Enable caching
    cache_threshold: float = 0.85         # Cache hit threshold

class QueryResponse:
    cache_hit: bool
    similarity_score: float               # If cached
    dominant_cluster: int
    cluster_confidence: float
    results: List[SearchResult]
    execution_time_ms: float
```

### 6. Streamlit Frontend (streamlit_app.py)

**Purpose**: Interactive web UI for the semantic search system

**Tabs**:

1. **Search Tab**
   - Query input text area
   - Real-time search results
   - Cache status indicators
   - Result expansion with content preview
   - Cache clear button

2. **Cache Analytics Tab**
   - Hit rate metrics (percentage)
   - Cache hit/miss pie chart
   - Queries per cluster bar chart
   - Cache entries by cluster table
   - Performance statistics

3. **Clustering Insights Tab**
   - Soft clustering explanation
   - GMM and optimization metrics
   - Sample clustering results table
   - Cluster keywords visualization

4. **Query History Tab**
   - Chronological query list
   - Query metadata display
   - Export as JSON
   - Clear history option
   - Expandable query details

## Data Pipeline

### Setup Phase (scripts/load_data.py)

```
1. Download Dataset
   └─ 20 Newsgroups (19,997 documents)

2. Preprocessing
   ├─ Remove headers/footers/quotes
   ├─ Filter by length (min 50 chars, max 5000 chars)
   ├─ Lowercase normalization
   └─ Total: ~15,000 documents

3. Embedding Generation
   ├─ Batch size: 128
   ├─ Speed: ~2000 docs/sec
   ├─ Output: (15000, 384) matrix
   └─ Time: ~7 seconds

4. Clustering
   ├─ PCA reduction: 384 -> 20 dimensions
   ├─ GMM fitting: 20 components
   ├─ Soft assignments per doc
   └─ Time: ~30 seconds

5. Indexing
   ├─ Prepare vectors with metadata
   ├─ Upsert to Pinecone serverless
   ├─ Create index if not exists
   └─ Time: ~5-10 minutes
```

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Embed text | ~5ms | Single document |
| Embed batch (128 docs) | ~50ms | Batch processing |
| GMM prediction | ~1ms | Cluster assignment |
| Cache lookup | <1ms | Dictionary access |
| Pinecone query | <100ms | Approximate KNN |
| Total query (cache hit) | ~10ms | Minimal latency |
| Total query (cache miss) | ~150ms | Includes Pinecone |

## Scalability Considerations

### Dataset Size
- Current: 15,000 documents
- Scalable to: 1M+ documents
- Limitation: Pinecone serverless scales automatically

### Query Throughput
- Single server: ~100-500 QPS
- Bottleneck: Pinecone serverless rate limits
- Solution: Upgrade Pinecone plan for higher throughput

### Embedding Model
- Latency vs Quality trade-off
- Current: `all-MiniLM-L6-v2` (fast)
- Optional: `all-mpnet-base-v2` (higher quality, slower)

### Clustering
- PCA reduces 384 -> 20 dims (95% variance retention)
- GMM fitting is offline (no impact on query latency)
- Soft assignments add <1ms per query

## Deployment Strategies

### 1. Local Development
```bash
python -m uvicorn src.app:app --reload
streamlit run streamlit_app.py
```

### 2. Docker Container
```bash
docker build -t semantic-search:latest .
docker run -e PINECONE_API_KEY=xxx -p 8000:8000 semantic-search:latest
```

### 3. Docker Compose (Full Stack)
```bash
docker-compose up
# API: localhost:8000
# UI: localhost:8501
```

### 4. Cloud Deployment
- Deploy FastAPI to: Cloud Run, App Engine, EC2
- Deploy Streamlit to: Streamlit Cloud, Heroku
- Store vector data in: Pinecone (serverless)

## Configuration

All settings in `.env`:
```env
PINECONE_API_KEY=xxx                # Required
CACHE_SIMILARITY_THRESHOLD=0.85     # Cache hit threshold
N_CLUSTERS=20                       # GMM components
PINECONE_INDEX_NAME=semantic-search # Index name
```

Tunable parameters in code:
- Embedding model: `src/embeddings.py`
- PCA dimensions: `src/clustering.py`
- Top-k results: FastAPI endpoint
- Batch size: `scripts/load_data.py`

## Monitoring & Debugging

### Logs
- API logs: Console output from FastAPI
- Streamlit logs: Browser developer console
- Data loading: `scripts/load_data.py` output

### Metrics
- Cache hit rate: `/cache/stats` endpoint
- Query latency: Response execution_time_ms
- Cluster distribution: Cache stats endpoint
- Index stats: Pinecone dashboard

### Debugging
1. **API Health**: `curl http://localhost:8000/health`
2. **Cache Stats**: `curl http://localhost:8000/cache/stats | jq`
3. **Query Details**: Check Streamlit UI for cache/similarity info
4. **Pinecone**: Verify index status in Pinecone console
