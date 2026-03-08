"""Test script to verify imports work"""
import sys
sys.path.insert(0, 'c:/Users/spars/Desktop/New folder')

from src.embeddings import EmbeddingManager
from src.clustering import GMMClusterer
from src.semantic_cache import SemanticCache
from src.pinecone_client import PineconeVectorDB
from src.data_pipeline import DataPipeline

print('All imports OK')

# Try to initialize components
try:
    print('Testing EmbeddingManager...')
    embedding_manager = EmbeddingManager()
    print('EmbeddingManager OK')
except Exception as e:
    print(f'EmbeddingManager error: {e}')

try:
    print('Testing GMMClusterer...')
    gmm_clusterer = GMMClusterer(n_components=20)
    print('GMMClusterer OK')
except Exception as e:
    print(f'GMMClusterer error: {e}')

try:
    print('Testing SemanticCache...')
    semantic_cache = SemanticCache(similarity_threshold=0.85)
    print('SemanticCache OK')
except Exception as e:
    print(f'SemanticCache error: {e}')

print('Done!')

