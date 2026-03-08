"""
Pinecone client module - cloud vector database integration
"""
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from typing import List, Tuple, Dict, Optional
import os


class PineconeVectorDB:
    """Wrapper for Pinecone vector database"""
    
    def __init__(self, api_key: Optional[str] = None, index_name: str = "semantic-search"):
        """
        Initialize Pinecone client
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Name of the index to use/create
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not provided and not in environment")
        
        self.index_name = index_name
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        
        print(f"[Pinecone] Initialized with API key")
    
    def create_index(self, dimension: int = 384, metric: str = "cosine") -> None:
        """
        Create or connect to Pinecone index
        
        Args:
            dimension: Embedding dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        # Check if index exists
        existing_indexes = self.pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes]
        
        if self.index_name not in index_names:
            print(f"[Pinecone] Creating index '{self.index_name}' with dimension {dimension}")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            print(f"[Pinecone] Using existing index '{self.index_name}'")
        
        self.index = self.pc.Index(self.index_name)
        print(f"[Pinecone] Connected to index '{self.index_name}'")
    
    def upsert_vectors(self, vectors: List[Tuple[str, np.ndarray, Dict]], batch_size: int = 100) -> int:
        """
        Upsert vectors to Pinecone
        
        Args:
            vectors: List of (id, embedding, metadata) tuples
            batch_size: Batch size for upserting
            
        Returns:
            Number of vectors upserted
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        # Convert to format expected by Pinecone
        # Handle both tuples with metadata (id, embedding, metadata) and without (id, embedding)
        vectors_to_upsert = []
        for v in vectors:
            if len(v) == 3:
                vec_id, embedding, metadata = v
                vectors_to_upsert.append((str(vec_id), embedding.tolist(), metadata))
            else:
                vec_id, embedding = v
                vectors_to_upsert.append((str(vec_id), embedding.tolist(), {}))
        
        upserted_count = 0
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
            upserted_count += len(batch)
            if (i // batch_size + 1) % 5 == 0:
                print(f"[Pinecone] Upserted {upserted_count}/{len(vectors_to_upsert)} vectors")
        
        print(f"[Pinecone] Total vectors upserted: {upserted_count}")
        return upserted_count
    
    def query(self, query_embedding: np.ndarray, top_k: int = 5, include_metadata: bool = False) -> List[Dict]:
        """
        Query Pinecone for nearest neighbors
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            include_metadata: Whether to include metadata
            
        Returns:
            List of results with id, score, and optional metadata
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=include_metadata
        )
        
        return results.get("matches", [])
    
    def fetch_vectors(self, ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Fetch vectors by their IDs
        
        Args:
            ids: List of vector IDs to fetch
            
        Returns:
            Dictionary mapping id to embedding vector
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        if not ids:
            return {}
        
        # Fetch vectors from Pinecone
        fetch_response = self.index.fetch(ids=ids)
        
        vectors = {}
        for vector_id, vector_data in fetch_response.get("vectors", {}).items():
            vectors[vector_id] = np.array(vector_data.get("values", []))
        
        print(f"[Pinecone] Fetched {len(vectors)} vectors")
        return vectors
    
    def get_all_vector_ids(self, limit: int = 10000) -> List[str]:
        """
        Get all vector IDs from the index (up to limit)
        
        Args:
            limit: Maximum number of IDs to return
            
        Returns:
            List of vector IDs
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        # Get index stats to see total vectors
        stats = self.index.describe_index_stats()
        total_vectors = stats.get("total_vector_count", 0)
        
        all_ids = []
        
        # Use query to get IDs - this is approximate
        # For exact listing, consider maintaining a separate ID tracking
        sample_query = np.zeros(self.index.describe_index_stats().get("dimension", 384))
        
        # Fetch in batches using query
        for i in range(0, min(total_vectors, limit), 1000):
            results = self.index.query(
                vector=sample_query.tolist(),
                top_k=min(1000, total_vectors - i),
                include_metadata=False
            )
            batch_ids = [match["id"] for match in results.get("matches", [])]
            all_ids.extend(batch_ids)
        
        # Remove duplicates
        unique_ids = list(set(all_ids))
        print(f"[Pinecone] Retrieved {len(unique_ids)} unique vector IDs")
        return unique_ids[:limit]
    
    def delete_index(self) -> None:
        """Delete the index"""
        if self.index_name:
            try:
                self.pc.delete_index(self.index_name)
                print(f"[Pinecone] Deleted index '{self.index_name}'")
            except Exception as e:
                print(f"[Pinecone] Error deleting index: {e}")
    
    def get_index_stats(self) -> Dict:
        """Get index statistics"""
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
            "index_fullness": stats.get("index_fullness", 0.0)
        }

