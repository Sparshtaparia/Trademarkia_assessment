"""
Semantic Cache module - cluster-aware caching with similarity matching
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    query: str
    embedding: List[float]
    result: Any
    cluster_id: int
    timestamp: str
    similarity: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SemanticCache:
    """Cluster-aware semantic cache with similarity matching"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize semantic cache
        
        Args:
            similarity_threshold: Threshold for cache hit (0.0 to 1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[int, List[CacheEntry]] = {}  # cluster_id -> list of entries
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_similarity": 0.0
        }
        print(f"[Cache] Initialized with similarity threshold: {similarity_threshold}")
    
    def add(self, query: str, embedding: np.ndarray, result: Any, cluster_id: int) -> None:
        """
        Add entry to cache
        
        Args:
            query: Query text
            embedding: Query embedding
            result: Query result
            cluster_id: Dominant cluster ID
        """
        if cluster_id not in self.cache:
            self.cache[cluster_id] = []
        
        entry = CacheEntry(
            query=query,
            embedding=embedding.tolist(),
            result=result,
            cluster_id=cluster_id,
            timestamp=datetime.now().isoformat()
        )
        self.cache[cluster_id].append(entry)
        print(f"[Cache] Added entry to cluster {cluster_id}. Cache size: {self.get_total_entries()}")
    
    def lookup(self, query_embedding: np.ndarray, cluster_id: int) -> Tuple[Optional[Any], float]:
        """
        Lookup query in cache
        
        Args:
            query_embedding: Query embedding
            cluster_id: Predicted cluster ID
            
        Returns:
            Tuple of (cached_result or None, max_similarity)
        """
        self.stats["total_queries"] += 1
        
        # Search in predicted cluster first
        if cluster_id in self.cache and len(self.cache[cluster_id]) > 0:
            result, similarity = self._search_cluster(query_embedding, cluster_id)
            if result is not None:
                self.stats["cache_hits"] += 1
                self.stats["avg_similarity"] = (
                    self.stats["avg_similarity"] * (self.stats["cache_hits"] - 1) + similarity
                ) / self.stats["cache_hits"]
                return result, similarity
        
        self.stats["cache_misses"] += 1
        return None, 0.0
    
    def _search_cluster(self, query_embedding: np.ndarray, cluster_id: int) -> Tuple[Optional[Any], float]:
        """
        Search for similar query in specific cluster
        
        Args:
            query_embedding: Query embedding
            cluster_id: Cluster to search in
            
        Returns:
            Tuple of (cached_result or None, similarity_score)
        """
        max_similarity = 0.0
        best_result = None
        
        entries = self.cache[cluster_id]
        for entry in entries:
            embedding = np.array(entry.embedding)
            similarity = self._cosine_similarity(query_embedding, embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_result = entry.result if similarity >= self.similarity_threshold else None
        
        return best_result, max_similarity
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm1 = vec1 / np.linalg.norm(vec1)
        norm2 = vec2 / np.linalg.norm(vec2)
        return float(np.dot(norm1, norm2))
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        hit_rate = (
            self.stats["cache_hits"] / self.stats["total_queries"]
            if self.stats["total_queries"] > 0
            else 0.0
        )
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "total_entries": self.get_total_entries(),
            "clusters_with_entries": len(self.cache)
        }
    
    def get_total_entries(self) -> int:
        """Get total number of cached entries"""
        return sum(len(entries) for entries in self.cache.values())
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_similarity": 0.0
        }
        print("[Cache] Cache cleared")
    
    def get_entries_by_cluster(self, cluster_id: int) -> List[Dict]:
        """Get all entries in a cluster"""
        if cluster_id not in self.cache:
            return []
        return [entry.to_dict() for entry in self.cache[cluster_id]]
