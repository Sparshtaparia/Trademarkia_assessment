"""
Clustering module - Gaussian Mixture Model for soft clustering
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict
import json
import joblib
import os


class GMMClusterer:
    """Gaussian Mixture Model for probabilistic soft clustering"""
    
    def __init__(self, n_components: int = 20, random_state: int = 42):
        """
        Initialize GMM clusterer
        
        Args:
            n_components: Number of clusters
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = None
        self.pca = None
        self.n_features = None
        print(f"[Clustering] Initialized GMM with {n_components} components")
    
    def fit(self, embeddings: np.ndarray, n_pca_components: int = 50) -> Dict:
        """
        Fit GMM to embeddings
        
        Args:
            embeddings: Array of embeddings (N, D)
            n_pca_components: Number of PCA components for dimensionality reduction
            
        Returns:
            Dictionary with fitting metrics
        """
        self.n_features = embeddings.shape[1]
        
        # Optional PCA for large dimensions
        if embeddings.shape[1] > n_pca_components:
            self.pca = PCA(n_components=n_pca_components, random_state=self.random_state)
            embeddings_reduced = self.pca.fit_transform(embeddings)
            print(f"[Clustering] Applied PCA: {embeddings.shape[1]} -> {n_pca_components}")
        else:
            embeddings_reduced = embeddings
        
        # Fit GMM       
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            n_init=10
        )
        self.gmm.fit(embeddings_reduced)
        
        metrics = {
            "bic": float(self.gmm.bic(embeddings_reduced)),
            "aic": float(self.gmm.aic(embeddings_reduced)),
            "converged": bool(self.gmm.converged_),
            "n_iter": int(self.gmm.n_iter_)
        }
        print(f"[Clustering] Fitted GMM - BIC: {metrics['bic']:.2f}, Converged: {metrics['converged']}")
        return metrics
    
    def save(self, filepath: str = "gmm_model.pkl") -> str:
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if self.gmm is None:
            print("[Clustering] WARNING: No trained model to save")
            return ""
        
        model_data = {
            "gmm": self.gmm,
            "pca": self.pca,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "n_features": self.n_features
        }
        
        joblib.dump(model_data, filepath)
        print(f"[Clustering] Model saved to {filepath}")
        return filepath
    
    def load(self, filepath: str = "gmm_model.pkl") -> bool:
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"[Clustering] Model file not found: {filepath}")
            return False
        
        try:
            model_data = joblib.load(filepath)
            self.gmm = model_data["gmm"]
            self.pca = model_data["pca"]
            self.n_components = model_data["n_components"]
            self.random_state = model_data["random_state"]
            self.n_features = model_data["n_features"]
            print(f"[Clustering] Model loaded from {filepath}")
            print(f"[Clustering] Loaded model with {self.n_components} components, {self.n_features} features")
            return True
        except Exception as e:
            print(f"[Clustering] Error loading model: {e}")
            return False
    
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get cluster probability distribution for embeddings
        
        Args:
            embeddings: Array of embeddings (N, D)
            
        Returns:
            Probability matrix (N, n_components)
        """
        if self.gmm is None:
            # Return uniform probability distribution (safe fallback)
            print("[Clustering] WARNING: GMM not fitted, returning uniform probabilities")
            n_samples = embeddings.shape[0]
            return np.ones((n_samples, self.n_components)) / self.n_components
        
        if self.pca is not None:
            embeddings_reduced = self.pca.transform(embeddings)
        else:
            embeddings_reduced = embeddings
        
        return self.gmm.predict_proba(embeddings_reduced)
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get hard cluster assignments
        
        Args:
            embeddings: Array of embeddings (N, D)
            
        Returns:
            Cluster labels (N,)
        """
        if self.gmm is None:
            # Return -1 for all samples (safe fallback)
            print("[Clustering] WARNING: GMM not fitted, returning -1 labels")
            return np.full(embeddings.shape[0], -1, dtype=int)
        
        if self.pca is not None:
            embeddings_reduced = self.pca.transform(embeddings)
        else:
            embeddings_reduced = embeddings
        
        return self.gmm.predict(embeddings_reduced)
    
    def get_dominant_cluster(self, embedding: np.ndarray) -> Tuple[int, float]:
        """
        Get dominant cluster and confidence for a single embedding
        
        Args:
            embedding: Single embedding vector
            
        Returns:
            Tuple of (cluster_id, confidence)
        """
        if self.gmm is None:
            # Return -1 cluster with 0 confidence (safe fallback)
            print("[Clustering] WARNING: GMM not fitted, returning cluster -1 with 0.0 confidence")
            return -1, 0.0
        
        proba = self.predict_proba(embedding.reshape(1, -1))[0]
        cluster_id = int(np.argmax(proba))
        confidence = float(proba[cluster_id])
        return cluster_id, confidence
    
    def get_cluster_keywords(self, embeddings: np.ndarray, texts: List[str], n_keywords: int = 5) -> Dict:
        """
        Get representative keywords for each cluster using top documents
        
        Args:
            embeddings: Array of embeddings
            texts: List of text documents
            n_keywords: Number of keywords per cluster
            
        Returns:
            Dictionary mapping cluster_id to keywords
        """
        labels = self.predict(embeddings)
        cluster_keywords = {}
        
        for cluster_id in range(self.n_components):
            cluster_mask = labels == cluster_id
            if cluster_mask.sum() == 0:
                cluster_keywords[cluster_id] = []
                continue
            
            # Get top documents in cluster
            cluster_texts = [texts[i] for i in np.where(cluster_mask)[0]]
            
            # Extract keywords from first few documents
            all_words = []
            for text in cluster_texts[:5]:
                words = text.lower().split()
                # Filter out common words
                words = [w for w in words if len(w) > 3]
                all_words.extend(words)
            
            # Get most common words
            from collections import Counter
            word_counts = Counter(all_words)
            keywords = [word for word, _ in word_counts.most_common(n_keywords)]
            cluster_keywords[cluster_id] = keywords
        
        return cluster_keywords

