"""
Check Pinecone index status
"""
from pinecone import Pinecone

API_KEY = "pcsk_5Xavyn_BwE2HEcGHhvytr7drAYKLof9woLZjX25z8LFPPgn9X6p3EFNubfNyZELMeFo3Qn"

pc = Pinecone(api_key=API_KEY)

# Check semantic-search (384 dimension)
print("=" * 50)
print("Checking 'semantic-search' index (384-dim):")
print("=" * 50)
index = pc.Index("semantic-search")
stats = index.describe_index_stats()
print(f"Total vectors: {stats.get('total_vector_count', 0)}")
print(f"Dimension: {stats.get('dimension', 0)}")
print()

# Check semanticsearch (1024 dimension)  
print("=" * 50)
print("Checking 'semanticsearch' index (1024-dim):")
print("=" * 50)
index2 = pc.Index("semanticsearch")
stats2 = index2.describe_index_stats()
print(f"Total vectors: {stats2.get('total_vector_count', 0)}")
print(f"Dimension: {stats2.get('dimension', 0)}")

