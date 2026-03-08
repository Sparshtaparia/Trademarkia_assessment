# Project Restructuring TODO

## ✅ Completed Structure

```
project-root/
├── backend/                  # Python Backend
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point
│   ├── run_server.py        # Server runner
│   ├── streamlit_app.py     # Streamlit UI
│   ├── .streamlit/          # Streamlit config
│   ├── api/                # API Layer
│   │   ├── __init__.py
│   │   ├── routes.py       # API endpoints
│   │   └── schemas.py      # Pydantic models
│   ├── ml/                 # ML Services
│   │   ├── __init__.py
│   │   ├── embeddings.py   # Sentence Transformers
│   │   ├── clustering.py   # GMM Clustering
│   │   ├── semantic_cache.py  # Cache
│   │   └── vector_db.py   # Pinecone Client
│   ├── pipeline/           # Data Processing
│   │   ├── __init__.py
│   │   └── data_pipeline.py
│   └── docker/              # Docker Configs
│       ├── Dockerfile
│       ├── Dockerfile.streamlit
│       └── docker-compose.yml
├── frontend/               # Frontend (Next.js + Streamlit configs)
│   ├── app/
│   ├── components/
│   ├── hooks/
│   ├── lib/
│   ├── public/
│   ├── styles/
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.mjs
│   ├── postcss.config.mjs
│   └── components.json
├── scripts/                # Utility Scripts
│   ├── ingest_data.py
│   └── load_data.py
├── requirements.txt        # Python dependencies
├── pyproject.toml
└── gmm_model.pkl          # Trained model
```

## Completed Tasks ✅
- [x] Created `backend/` directory structure
- [x] Created `backend/ml/` for ML services (embeddings, clustering, vector_db, semantic_cache)
- [x] Created `backend/api/` for FastAPI routes
- [x] Created `backend/pipeline/` for data processing
- [x] Created `backend/docker/` for Docker configs
- [x] Created `frontend/` for frontend files
- [x] Moved all config files (package.json, tsconfig.json, etc.) to frontend/
- [x] Updated import paths in scripts/
- [x] Removed old `services/` directory

## Running the Application

### FastAPI Backend
```bash
cd backend
python -m uvicorn main:app --reload

# Or use the runner
python backend/run_server.py
```

### Streamlit Frontend
```bash
streamlit run backend/streamlit_app.py
```

### Docker
```bash
cd backend/docker
docker-compose up
```

