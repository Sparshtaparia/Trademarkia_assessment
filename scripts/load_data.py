"""
Script to load 20 Newsgroups and index in Pinecone
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import DataPipeline
from src.embeddings import EmbeddingManager
from src.clustering import GMMClusterer
from src.pinecone_client import PineconeVectorDB
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def main():
    print("=" * 60)
    print("Loading 20 Newsgroups Data Pipeline")
    print("=" * 60)
    
    # Initialize components
    print("\n[1] Initializing components...")
    pipeline = DataPipeline(cache_dir="./data")
    embedding_manager = EmbeddingManager()
    gmm_clusterer = GMMClusterer(n_components=20)
    pinecone_db = PineconeVectorDB()
    
    # Load data
    print("\n[2] Loading 20 Newsgroups dataset...")
    texts, categories = pipeline.load_20newsgroups(subset="train", remove_headers=True)
    
    # Sample for faster processing (optional - remove for full dataset)
    print("\n[3] Sampling data (5000 documents for demo)...")
    texts = pipeline.sample_texts(texts, n_samples=5000)
    
    # Preprocess
    print("\n[4] Preprocessing texts...")
    texts, valid_indices = pipeline.preprocess_texts(texts)
    
    # Generate embeddings
    print("\n[5] Generating embeddings...")
    embeddings = embedding_manager.encode(texts)
    
    # Fit clustering model
    print("\n[6] Fitting GMM clustering...")
    cluster_metrics = gmm_clusterer.fit(embeddings)
    
    # Create Pinecone index
    print("\n[7] Creating Pinecone index...")
    pinecone_db.create_index(dimension=embeddings.shape[1])
    
    # Index vectors in Pinecone
    print("\n[8] Indexing vectors in Pinecone...")
    vectors = list(zip(range(len(embeddings)), embeddings))
    pinecone_db.upsert_vectors(vectors)
    
    # Get clustering insights
    print("\n[9] Getting clustering insights...")
    cluster_keywords = gmm_clusterer.get_cluster_keywords(embeddings, texts)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total documents loaded: {len(texts)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of clusters: {gmm_clusterer.n_components}")
    print(f"GMM BIC score: {cluster_metrics['bic']:.2f}")
    print(f"GMM AIC score: {cluster_metrics['aic']:.2f}")
    
    # Print Pinecone stats
    try:
        index_stats = pinecone_db.get_index_stats()
        print(f"\nPinecone index stats:")
        print(f"  Total vectors: {index_stats['total_vectors']}")
        print(f"  Index fullness: {index_stats['index_fullness']:.2%}")
    except Exception as e:
        print(f"  Error getting index stats: {e}")
    
    # Print sample cluster keywords
    print(f"\nSample cluster keywords:")
    for cluster_id in range(min(5, gmm_clusterer.n_components)):
        keywords = cluster_keywords.get(cluster_id, [])
        print(f"  Cluster {cluster_id}: {', '.join(keywords[:5])}")
    
    print("\n" + "=" * 60)
    print("Data loading complete! Ready to serve queries.")
    print("=" * 60)


if __name__ == "__main__":
    main()
