"""
FastAPI application for semantic search with caching
Entry point for the backend API
"""
from fastapi import FastAPI
import os
from dotenv import load_dotenv

from backend.api.routes import router, set_components
from backend.ml.embeddings import EmbeddingManager
from backend.ml.clustering import GMMClusterer
from backend.ml.semantic_cache import SemanticCache
from backend.ml.vector_db import PineconeVectorDB

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Semantic Search System",
    description="Search with Pinecone, GMM clustering, and semantic caching",
    version="0.1.0"
)

# Global components
embedding_manager = None
gmm_clusterer = None
semantic_cache = None
pinecone_db = None
is_initialized = False


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
        
        # Try to load saved model if exists
        model_path = "gmm_model.pkl"
        if os.path.exists(model_path):
            try:
                gmm_clusterer.load(model_path)
                print("[App] Loaded saved clustering model")
            except Exception as e:
                print(f"[App] Could not load saved model: {e}")
        
        # Only initialize Pinecone if API key is available
        if pinecone_api_key:
            pinecone_db = PineconeVectorDB(index_name="semantic-search")
            pinecone_db.create_index(dimension=embedding_manager.embedding_dim)
            print("[App] Pinecone initialized")
        else:
            pinecone_db = None
            print("[App] WARNING: PINECONE_API_KEY not set, Pinecone disabled")
        
        is_initialized = True
        
        # Set components for routes
        set_components(
            embedding_manager,
            gmm_clusterer,
            semantic_cache,
            pinecone_db,
            is_initialized
        )
        
        print("[App] System initialized successfully")
        
    except Exception as e:
        print(f"[App] Initialization error: {e}")
        import traceback
        traceback.print_exc()
        is_initialized = False


# Include router with /api prefix
app.include_router(router, prefix="/api")


# Root endpoint - basic info
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Semantic Search API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/health"
    }


# Health check at root level (for convenience)
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

