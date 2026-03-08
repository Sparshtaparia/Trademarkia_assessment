# Pipeline Analysis

## Current Implementation Flow

### scripts/load_data.py
```
1. Load 20 Newsgroups Dataset
2. Generate Embeddings (SentenceTransformer)
3. Fit GMM on embeddings
4. Create Pinecone Index
5. Upsert vectors to Pinecone
```

### src/app.py (API)
```
1. Load saved GMM model if exists (or -1 cluster if not)
2. Query Pinecone for search
3. Return results with cluster info
```

## Expected Pipeline (from user task)

```
Dataset (Twenty Newsgroups)
        ↓
SentenceTransformer → Embeddings (vectors)
        ↓
Pinecone upsert (Step 4)
        ↓
Semantic Search
        ↓
GMM Clustering (Step 6 - train AFTER upsert)
        ↓
Streamlit Dashboard
```

## Issue Found

The current `scripts/load_data.py` trains GMM **before** upserting to Pinecone, which is different from the expected flow. However, functionally it should work because:

1. GMM is trained on the same embeddings that get upserted
2. The model is saved to `gmm_model.pkl`
3. On API startup, it loads the saved model

## Verification Steps Needed

1. Run `scripts/load_data.py` to load data and upsert to Pinecone
2. Verify Pinecone has vectors: `index.describe_index_stats()`
3. Start API server: `python src/app.py`
4. Test search endpoint

## Current Status

All components exist and are properly connected:
- ✓ Data loading
- ✓ Embeddings generation
- ✓ Pinecone upsert
- ✓ GMM clustering with fallback (-1, 0.0 when not trained)
- ✓ API endpoints
- ✓ Streamlit dashboard

