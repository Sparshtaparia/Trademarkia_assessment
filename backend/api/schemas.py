"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel
from typing import List, Dict, Optional


class QueryRequest(BaseModel):
    """Request model for search queries"""
    text: str
    top_k: int = 5
    use_cache: bool = True


class QueryResponse(BaseModel):
    """Response model for search queries"""
    query: str
    cache_hit: bool
    similarity_score: float
    dominant_cluster: int
    cluster_confidence: float
    results: List[Dict]
    timestamp: str


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics"""
    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    total_entries: int
    clusters_with_entries: int
    avg_similarity: float


class StatusResponse(BaseModel):
    """Response model for system status"""
    initialized: bool
    components: Dict
    cache_stats: Optional[Dict] = None


class TrainingResponse(BaseModel):
    """Response model for clustering training"""
    message: str
    n_vectors: int
    n_clusters: int
    metrics: Dict
    model_saved: str


class ClusteringStatusResponse(BaseModel):
    """Response model for clustering status"""
    is_trained: bool
    n_components: int
    n_features: Optional[int] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str

