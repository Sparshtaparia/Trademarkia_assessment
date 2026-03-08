# Docker Setup Guide

## Prerequisites

- Docker
- Docker Compose
- Pinecone API Key

## Quick Start

### 1. Configure Environment

Copy the example environment file and add your Pinecone API key:

```bash
cp .env.example .env
# Edit .env and add your PINECONE_API_KEY
```

### 2. Build and Run All Services

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 3. Access the Application

- **FastAPI Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Streamlit Frontend**: http://localhost:8501

## Service Commands

### Start Only Backend (FastAPI)
```bash
docker build -t semantic-search-api -f Dockerfile .
docker run -p 8000:8000 -e PINECONE_API_KEY=your_key semantic-search-api
```

### Start Only Frontend (Streamlit)
```bash
docker build -t semantic-search-frontend -f Dockerfile.streamlit .
docker run -p 8501:8501 -e PINECONE_API_KEY=your_key semantic-search-frontend
```

### Run Data Ingestion
```bash
# Using docker-compose (one-time)
docker-compose run --rm ingest

# Or directly
docker build -t semantic-search-ingest -f Dockerfile .
docker run -v /path/to/20_newsgroups:/data \
    -e PINECONE_API_KEY=your_key \
    semantic-search-ingest \
    python ingest_local_dataset.py
```

## Volume Mounts

The following volumes are mounted:

| Container | Volume | Description |
|-----------|--------|-------------|
| api | `./gmm_model.pkl` | GMM clustering model |
| frontend | `./gmm_model.pkl` | GMM clustering model |
| ingest | `./gmm_model.pkl` | Saves trained model |
| ingest | `${DATASET_PATH}:/data` | Dataset (default: ./data/20_newsgroups) |

## Troubleshooting

### Check Container Logs
```bash
# API logs
docker-compose logs api

# Frontend logs
docker-compose logs frontend

# Ingestion logs
docker-compose logs ingest
```

### Rebuild Containers
```bash
docker-compose build --no-cache
```

### Stop All Services
```bash
docker-compose down
```

### Remove Volumes
```bash
docker-compose down -v
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PINECONE_API_KEY` | Pinecone API key | Yes |
| `PYTHONUNBUFFERED` | Enable unbuffered output | No |
| `PYTHONPATH` | Python module path | No |
| `DATASET_PATH` | Path to 20 newsgroups dataset | No |

