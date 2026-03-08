"""Test startup sequence"""
import sys
import os
sys.path.insert(0, 'c:/Users/spars/Desktop/New folder')
os.chdir('c:/Users/spars/Desktop/New folder')

# Set environment
from dotenv import load_dotenv
load_dotenv()

print("1. Loading modules...")
from src.embeddings import EmbeddingManager
from src.clustering import GMMClusterer
from src.semantic_cache import SemanticCache

print("2. Initializing EmbeddingManager...")
embedding_manager = EmbeddingManager()

print("3. Initializing GMMClusterer...")
gmm_clusterer = GMMClusterer(n_components=20)

print("4. Initializing SemanticCache...")
semantic_cache = SemanticCache(similarity_threshold=0.85)

print("5. Checking Pinecone...")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if pinecone_api_key:
    from src.pinecone_client import PineconeVectorDB
    pinecone_db = PineconeVectorDB()
    pinecone_db.create_index(dimension=embedding_manager.embedding_dim)
    print("Pinecone initialized!")
else:
    print("WARNING: PINECONE_API_KEY not set")

print("\nAll components initialized successfully!")
print(f"Embedding dimension: {embedding_manager.embedding_dim}")

