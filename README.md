# Semantic Search System

A production-ready semantic search system combining Pinecone vector database, Gaussian Mixture Model clustering, and intelligent semantic caching.

<img width="1919" height="962" alt="Screenshot 2026-03-09 012905" src="https://github.com/user-attachments/assets/f2360d7e-c15e-4bef-bbba-fbe26d7c0e81" />
<img width="1919" height="954" alt="Screenshot 2026-03-09 012809" src="https://github.com/user-attachments/assets/ebd08239-7f8d-44fd-a2d5-bc31da51cc9a" />


## Architecture Overview

### Components

1. **Embeddings Layer** (`backend/ml/embeddings.py`)
   - Uses Sentence Transformers (all-MiniLM-L6-v2)
   - 384-dimensional embeddings for fast and efficient search
   - Cosine similarity matching

2. **Clustering Layer** (`backend/ml/clustering.py`)
   - Gaussian Mixture Model (GMM) with 20 components
   - Soft clustering with probabilistic assignments
   - Automatic PCA reduction for high-dimensional data
   - Cluster interpretation via keywords

3. **Semantic Cache** (`backend/ml/semantic_cache.py`)
   - Cluster-aware caching mechanism
   - Dictionary-based storage with cluster partitioning
   - Configurable similarity threshold (default: 0.85)
   - Hit/miss statistics tracking

4. **Vector Database** (`backend/ml/vector_db.py`)
   - Cloud-based Pinecone integration
   - Serverless deployment on AWS
   - Fast approximate nearest neighbor search
   - Automatic index creation and management

5. **API Service** (`backend/main.py`)
   - FastAPI-based REST API
   - Query endpoint with cache tracking
   - Statistics and monitoring endpoints
   - Health checks and status reporting

### Data Pipeline

- **Dataset**: 20 Newsgroups (UCI Archive)
- **Processing**: Text cleaning, length filtering, PCA dimensionality reduction
- **Workflow**: Load → Preprocess → Embed → Cluster → Index → Serve

## Setup & Installation

### Prerequisites
- Python 3.10+
- Pinecone API Key (free tier available)
- Docker (optional, for containerization)

### Local Installation

1. Clone the repository
```bash
git clone <repository-url>
cd semantic-search-system
```

2. Install dependencies using uv
```bash
uv pip install -r requirements.txt
```

Or manually:
```bash
pip install sentence-transformers pinecone-client scikit-learn fastapi uvicorn streamlit plotly requests
```

3. Set environment variables
```bash
cp .env.example .env
# Edit .env and add your PINECONE_API_KEY
```

4. Load data into Pinecone
```bash
python scripts/load_data.py
```

This will:
- Download 20 Newsgroups dataset
- Generate embeddings for all documents
- Fit GMM clustering
- Index vectors in Pinecone
- Print cluster insights

5. Start the API server (in one terminal)
```bash
python -m uvicorn backend.main:app --reload
```

Server will be available at `http://localhost:8000`

6. Start the Streamlit frontend (in another terminal)
```bash
streamlit run frontend/streamlit_app.py
```

The web interface will open at `http://localhost:8501`

## API Endpoints

### GET `/health`
Health check endpoint
```bash
curl http://localhost:8000/health
```

### GET `/status`
System status and component information
```bash
curl http://localhost:8000/api/status
```

### POST `/query`
Execute semantic search with caching
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "machine learning algorithms",
    "top_k": 5,
    "use_cache": true
  }'
```

Response includes:
- `cache_hit`: Whether result was cached
- `similarity_score`: Similarity to cached query
- `dominant_cluster`: Predicted cluster ID
- `cluster_confidence`: Confidence of cluster assignment
- `results`: List of nearest neighbors from Pinecone

### GET `/cache/stats`
Cache performance statistics
```bash
curl http://localhost:8000/api/cache/stats
```

Response includes:
- `total_queries`: Total queries processed
- `cache_hits`: Number of cache hits
- `cache_misses`: Number of cache misses
- `hit_rate`: Cache hit rate percentage
- `avg_similarity`: Average similarity of cached matches

### DELETE `/cache`
Clear all cached entries
```bash
curl -X DELETE http://localhost:8000/api/cache
```

## Docker Deployment

Build the Docker image:
```bash
docker build -t semantic-search:latest .
```

Run the container:
```bash
docker run -e PINECONE_API_KEY=<your-key> -p 8000:8000 semantic-search:latest
```

## Web Interface (Streamlit)

The system includes a modern web interface built with Streamlit for interactive exploration:

### Features

**Search Tab**
- Text input for semantic queries
- Real-time search results with similarity scores
- Cache hit/miss indicators
- Cluster assignment display
- Result content preview with expandable items

**Cache Analytics Tab**
- Real-time cache performance metrics
- Hit/miss distribution pie charts
- Queries per cluster bar charts
- Cache entries by cluster table
- Hit rate percentage tracking

**Clustering Insights Tab**
- Soft clustering explanation
- GMM and optimization metrics reference
- Sample clustering results with probabilities
- Cluster-specific keywords analysis

**Query History Tab**
- Complete query history with timestamps
- Export queries as JSON
- Search metadata display
- Clear history option

### Running the Web Interface

```bash
streamlit run frontend/streamlit_app.py
```

Then navigate to `http://localhost:8501` in your browser.

## Analysis & Experimentation

Run the Jupyter notebook for in-depth analysis:
```bash
jupyter notebook analysis.ipynb
```

This notebook demonstrates:
- Data loading and preprocessing
- Embedding generation
- GMM clustering analysis
- Cluster visualization with PCA
- Semantic cache testing
- Query similarity analysis

## Configuration

### Tunable Parameters

In `.env`:
```env
# Clustering
N_CLUSTERS=20                          # Number of GMM components
CACHE_SIMILARITY_THRESHOLD=0.85        # Cache hit threshold (0.0-1.0)
PINECONE_INDEX_NAME=semantic-search    # Pinecone index name
```

In code:
- **EmbeddingManager**: Model selection, batch size
- **GMMClusterer**: n_components, PCA dimensions
- **SemanticCache**: similarity_threshold
- **API**: top_k results, batch processing

## Performance Characteristics

- **Embedding Generation**: ~2000 docs/sec (all-MiniLM-L6-v2)
- **Vector Search**: <100ms per query (Pinecone serverless)
- **Cache Lookup**: <1ms per query (local dictionary)
- **GMM Inference**: <10ms per document (with PCA)

## Design Decisions

### Why Gaussian Mixture Model?
- Soft clustering allows documents to belong to multiple clusters
- Probabilistic framework provides confidence scores
- Efficient inference for cache routing
- Interpretable cluster assignments

### Why Semantic Cache?
- Cluster-aware partitioning reduces search space
- Similarity threshold prevents false matches
- Statistics tracking for monitoring
- Fast local lookups reduce API calls

### Why Pinecone?
- Serverless deployment (no infrastructure management)
- Fast approximate nearest neighbors (HNSW)
- Scalable to millions of vectors
- Free tier suitable for development

### Why all-MiniLM-L6-v2?
- Fast inference (384 dimensions)
- Excellent performance on general semantic search
- Lightweight (~90MB)
- Good trade-off between speed and accuracy

## Monitoring & Debugging

### Cache Performance
```bash
curl http://localhost:8000/api/cache/stats | jq
```

### Query Execution with Caching
1. Check if query embedding is in cache for predicted cluster
2. If cache hit, return cached result with similarity score
3. If cache miss, query Pinecone and cache result
4. Return query with cache metadata

### Cluster Analysis
Use the Jupyter notebook to:
- Visualize clusters with PCA
- Inspect cluster keywords
- Analyze cluster sizes
- Test cache behavior

## Troubleshooting

### Pinecone Connection Error
```
ValueError: PINECONE_API_KEY not provided
```
Solution: Add your API key to `.env` file

### Index Creation Timeout
```
Timeout waiting for index creation
```
Solution: Check Pinecone dashboard for index status. Serverless indexes take ~1-2 minutes to create.

### Memory Issues with Large Datasets
- Reduce `n_samples` in `scripts/load_data.py`
- Increase `batch_size` in embedding generation
- Use PCA dimensionality reduction

## API Documentation

Full interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Future Enhancements

- [ ] Multi-stage reranking with cross-encoders
- [ ] Hierarchical clustering for faster search
- [ ] Real-time index updates
- [ ] Query expansion with keyword synonyms
- [ ] A/B testing framework for cache thresholds
- [ ] Distributed caching with Redis
- [ ] Fine-tuned embeddings on custom datasets

## License

MIT License

