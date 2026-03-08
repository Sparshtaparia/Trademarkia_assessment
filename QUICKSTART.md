# Quick Start Guide

Get the semantic search system up and running in minutes.

## Option 1: Local Development (Recommended)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment
```bash
cp .env.example .env
```
Edit `.env` and add your Pinecone API key:
```
PINECONE_API_KEY=pcsk_5Xavyn_BwE2HEcGHhvytr7drAYKLof9woLZjX25z8LFPPgn9X6p3EFNubfNyZELMeFo3Qn
```

### Step 3: Load Data (First Time Only)
```bash
python scripts/load_data.py
```
This downloads the 20 Newsgroups dataset, generates embeddings, and indexes them in Pinecone. Takes ~5-10 minutes.

### Step 4: Start API Server
Open a terminal and run:
```bash
python -m uvicorn backend.main:app --reload
```
✓ API running at `http://localhost:8000`

### Step 5: Start Web Interface
Open a new terminal and run:
```bash
streamlit run frontend/streamlit_app.py
```
✓ Web UI running at `http://localhost:8501`

## Option 2: Docker Compose (All-in-One)

### Prerequisites
- Docker and Docker Compose installed

### Step 1: Configure Environment
```bash
cp .env.example .env
# Edit .env with your Pinecone API key
```

### Step 2: Load Data
```bash
docker-compose run api python scripts/load_data.py
```

### Step 3: Start Services
```bash
docker-compose up
```
✓ API running at `http://localhost:8000`
✓ Web UI running at `http://localhost:8501`

## API Usage

### Search Query
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "machine learning",
    "top_k": 5,
    "use_cache": true
  }'
```

### Cache Statistics
```bash
curl http://localhost:8000/api/cache/stats | jq
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Web Interface Features

### 🔍 Search Tab
- Enter any search query
- See real-time results with similarity scores
- Cache hit/miss indicators
- Cluster assignments

### 📊 Cache Analytics
- Hit rate and performance metrics
- Distribution charts
- Cluster statistics

### 📈 Clustering Insights
- Soft clustering explanation
- Cluster keywords
- Sample results

### 📋 Query History
- All previous queries
- Execution timestamps
- Export as JSON

## Troubleshooting

### Pinecone Connection Error
```
ValueError: PINECONE_API_KEY not provided
```
**Fix:** Add your API key to `.env` file

### API Not Responding
```
requests.exceptions.ConnectionError: Cannot reach API
```
**Fix:** Ensure API server is running with `python -m uvicorn backend.main:app --reload`

### Streamlit Can't Find API
```
Error: Cannot reach API at http://localhost:8000
```
**Fix:** Check that both API and Streamlit are running. In Streamlit UI, update API URL in left sidebar if needed.

### Data Loading Timeout
```
Timeout waiting for index creation
```
**Fix:** Check Pinecone dashboard. Serverless indexes take 1-2 minutes to create. Retry after waiting.

## Next Steps

1. **Explore the System**
   - Try different search queries in the web UI
   - Check cache analytics to see hit rates
   - Review clustering insights

2. **Analyze Results**
   - Open `analysis.ipynb` for detailed clustering analysis
   - Experiment with different parameters
   - Visualize cluster distributions

3. **Customize**
   - Edit `.env` to adjust cache threshold, cluster count, etc.
   - Modify `backend/ml/embeddings.py` to use different models
   - Update `backend/ml/clustering.py` for different clustering algorithms

## API Documentation

Full interactive documentation available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Environment Variables

See `.env.example` for all available configuration options:
- `PINECONE_API_KEY` - Your Pinecone API key (required)
- `CACHE_SIMILARITY_THRESHOLD` - Cache hit threshold (default: 0.85)
- `N_CLUSTERS` - Number of GMM clusters (default: 20)
- `PINECONE_INDEX_NAME` - Index name in Pinecone (default: semantic-search)

## Performance Tips

- Use cache threshold of 0.85-0.95 for stricter matches
- Reduce top_k for faster responses
- Use PCA dimensionality reduction for large datasets
- Enable Streamlit caching in config for faster UI loads

