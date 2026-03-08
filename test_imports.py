#!/usr/bin/env python
"""Test script to verify imports work correctly"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing imports...")
    
    # Test backend services
    from backend.services.embeddings import EmbeddingManager
    print("✓ backend.services.embeddings")
    
    from backend.services.clustering import GMMClusterer
    print("✓ backend.services.clustering")
    
    from backend.services.semantic_cache import SemanticCache
    print("✓ backend.services.semantic_cache")
    
    from backend.services.vector_db import PineconeVectorDB
    print("✓ backend.services.vector_db")
    
    from backend.pipeline.data_pipeline import DataPipeline
    print("✓ backend.pipeline.data_pipeline")
    
    from backend.api.schemas import QueryRequest, QueryResponse
    print("✓ backend.api.schemas")
    
    from backend.api.routes import router
    print("✓ backend.api.routes")
    
    from backend.main import app
    print("✓ backend.main")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"\n❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

