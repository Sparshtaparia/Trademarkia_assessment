"""
FastAPI routes for semantic search API
"""
from fastapi import APIRouter, HTTPException
from typing import Dict
from datetime import datetime

from backend.api.schemas import (
    QueryRequest,
    QueryResponse,
    CacheStatsResponse,
    StatusResponse,
    TrainingResponse,
    ClusteringStatusResponse,
    HealthResponse
)

# Create router
router = APIRouter()

# Global components references (set by main.py)
embedding_manager = None
gmm_clusterer = None
semantic_cache = None
pinecone_db = None
is_initialized = False


def set_components(embeddings, clustering, cache, pinecone, initialized):
    """Set global component references"""
    global embedding_manager, gmm_clusterer, semantic_cache, pinecone_db, is_initialized
    embedding_manager = embeddings
    gmm_clusterer = clustering
    semantic_cache = cache
    pinecone_db = pinecone
    is_initialized = initialized


@router.get("/status", response_model=StatusResponse)
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


@router.post("/query", response_model=QueryResponse)
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
        
        # If cache miss, query Pinecone
        if not cache_hit:
            results = pinecone_db.query(query_embedding, top_k=request.top_k, include_metadata=True)
            
            # Format results
            formatted_results = []
            for match in results:
                metadata = match.get("metadata", {})
                content = metadata.get("content", "")
                if not content and "text" in metadata:
                    content = metadata.get("text", "")[:500]
                
                formatted_results.append({
                    "id": match["id"],
                    "score": match["score"],
                    "title": f"Document {match['id']}",
                    "content": content,
                    "cluster_id": metadata.get("cluster_id", -1),
                    "cluster_prob": metadata.get("cluster_prob", 0.0),
                    "metadata": metadata
                })
            
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


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get cache statistics"""
    if not is_initialized or semantic_cache is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    stats = semantic_cache.get_stats()
    return CacheStatsResponse(**stats)


@router.delete("/cache")
async def clear_cache():
    """Clear cache"""
    if not is_initialized or semantic_cache is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    semantic_cache.clear()
    return {"message": "Cache cleared"}


@router.post("/clustering/train", response_model=TrainingResponse)
async def train_clustering():
    """
    Train GMM clustering model from Pinecone vectors
    
    This endpoint fetches vectors from Pinecone and trains the GMM model.
    """
    if not is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if pinecone_db is None:
        raise HTTPException(status_code=503, detail="Pinecone not initialized")
    
    try:
        import numpy as np
        
        # Get vector IDs from Pinecone
        print("[API] Fetching vector IDs from Pinecone...")
        vector_ids = pinecone_db.get_all_vector_ids(limit=10000)
        
        if not vector_ids:
            raise HTTPException(status_code=400, detail="No vectors found in Pinecone index")
        
        print(f"[API] Found {len(vector_ids)} vectors in Pinecone")
        
        # Fetch vectors
        vectors_dict = pinecone_db.fetch_vectors(vector_ids)
        
        if not vectors_dict:
            raise HTTPException(status_code=400, detail="Could not fetch vectors from Pinecone")
        
        # Convert to numpy array
        embeddings = np.array(list(vectors_dict.values()))
        print(f"[API] Training GMM on {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
        
        # Train GMM
        metrics = gmm_clusterer.fit(embeddings)
        
        # Save the trained model
        model_path = gmm_clusterer.save("gmm_model.pkl")
        
        return TrainingResponse(
            message="GMM trained successfully",
            n_vectors=len(vectors_dict),
            n_clusters=gmm_clusterer.n_components,
            metrics=metrics,
            model_saved=model_path
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@router.get("/clustering/status", response_model=ClusteringStatusResponse)
async def get_clustering_status():
    """Get clustering model status"""
    if not is_initialized or gmm_clusterer is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    is_trained = gmm_clusterer.gmm is not None
    
    return ClusteringStatusResponse(
        is_trained=is_trained,
        n_components=gmm_clusterer.n_components,
        n_features=gmm_clusterer.n_features
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy")

