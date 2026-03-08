"""
FastAPI application for semantic search with caching
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import os
from dotenv import load_dotenv

from src.embeddings import EmbeddingManager
from src.clustering import GMMClusterer
from src.semantic_cache import SemanticCache
from src.pinecone_client import PineconeVectorDB
from src.data_pipeline import DataPipeline

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Semantic Search System",
    description="Search with Pinecone, GMM clustering, and semantic caching",
    version="0.1.0"
)

# Global components
embedding_manager: Optional[EmbeddingManager] = None
gmm_clusterer: Optional[GMMClusterer] = None
semantic_cache: Optional[SemanticCache] = None
pinecone_db: Optional[PineconeVectorDB] = None
is_initialized = False


# Pydantic models
class QueryRequest(BaseModel):
    text: str
    top_k: int = 5
    use_cache: bool = True


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    similarity_score: float
    dominant_cluster: int
    cluster_confidence: float
    results: List[Dict]
    timestamp: str


class CacheStatsResponse(BaseModel):
    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    total_entries: int
    clusters_with_entries: int
    avg_similarity: float


class StatusResponse(BaseModel):
    initialized: bool
    components: Dict
    cache_stats: Optional[Dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global embedding_manager, gmm_clusterer, semantic_cache, pinecone_db, is_initialized
    
    try:
        print("[App] Starting up system...")
        
        # Check for API keys
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        # Initialize components that don't require API keys
        embedding_manager = EmbeddingManager()
        gmm_clusterer = GMMClusterer(n_components=20)
        semantic_cache = SemanticCache(similarity_threshold=0.85)
        
        # Only initialize Pinecone if API key is available
        if pinecone_api_key:
            pinecone_db = PineconeVectorDB()
            pinecone_db.create_index(dimension=embedding_manager.embedding_dim)
            print("[App] Pinecone initialized")
        else:
            pinecone_db = None
            print("[App] WARNING: PINECONE_API_KEY not set, Pinecone disabled")
        
        is_initialized = True
        print("[App] System initialized successfully")
    except Exception as e:
        print(f"[App] Initialization error: {e}")
        import traceback
        traceback.print_exc()
        is_initialized = False


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    return StatusResponse(
        initialized=is_initialized,
        components={
            "embeddings": embedding_manager is not None,
            "clustering": gmm_clusterer is not None,
            "cache": semantic_cache is not None,
            "pinecone": pinecone_db is not None
        },
        cache_stats=semantic_cache.get_stats() if semantic_cache else None
    )


@app.post("/query", response_model=QueryResponse)
async def query_search(request: QueryRequest):
    """
    Search with semantic cache and clustering
    
    Args:
        text: Query text
        top_k: Number of results
        use_cache: Whether to use cache
        
    Returns:
        Query results with cache info
    """
    if not is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        from datetime import datetime
        
        # Generate embedding
        query_embedding = embedding_manager.encode_single(request.text)
        
        # Get dominant cluster
        cluster_id, cluster_confidence = gmm_clusterer.get_dominant_cluster(query_embedding)
        
        # Try cache lookup
        cache_hit = False
        similarity_score = 0.0
        cached_result = None
        
        if request.use_cache:
            cached_result, similarity_score = semantic_cache.lookup(query_embedding, cluster_id)
            if cached_result is not None:
                cache_hit = True
                print(f"[API] Cache hit for cluster {cluster_id}")
        
        # If cache miss, query Pinecone
        if not cache_hit:
            results = pinecone_db.query(query_embedding, top_k=request.top_k)
            
            # Format results
            formatted_results = [
                {
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match.get("metadata", {})
                }
                for match in results
            ]
            
            # Cache the result
            if request.use_cache:
                semantic_cache.add(
                    query=request.text,
                    embedding=query_embedding,
                    result=formatted_results,
                    cluster_id=cluster_id
                )
        else:
            formatted_results = cached_result
        
        return QueryResponse(
            query=request.text,
            cache_hit=cache_hit,
            similarity_score=float(similarity_score),
            dominant_cluster=int(cluster_id),
            cluster_confidence=float(cluster_confidence),
            results=formatted_results,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get cache statistics"""
    if not is_initialized or semantic_cache is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    stats = semantic_cache.get_stats()
    return CacheStatsResponse(**stats)


@app.delete("/cache")
async def clear_cache():
    """Clear cache"""
    if not is_initialized or semantic_cache is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    semantic_cache.clear()
    return {"message": "Cache cleared"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
