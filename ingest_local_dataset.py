"""
Script to ingest the 20 Newsgroups dataset into Pinecone vector database.
This script scans the dataset directory, generates embeddings, applies GMM clustering, 
and uploads them to Pinecone.
"""
import os
import numpy as np
from src.embeddings import EmbeddingManager
from src.pinecone_client import PineconeVectorDB
from src.clustering import GMMClusterer

DATASET_PATH = r"C:\Users\spars\Downloads\twenty+newsgroups\20_newsgroups\20_newsgroups"

print("Scanning dataset...")

texts = []
ids = []

doc_id = 0

for category in os.listdir(DATASET_PATH):

    category_path = os.path.join(DATASET_PATH, category)

    if not os.path.isdir(category_path):
        continue

    print("Reading category:", category)

    for filename in os.listdir(category_path):

        file_path = os.path.join(category_path, filename)

        try:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()

            texts.append(text)
            ids.append(f"doc_{doc_id}")
            doc_id += 1

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            pass

print("Total documents loaded:", len(texts))

# Generate embeddings
embedder = EmbeddingManager()
embeddings = embedder.encode(texts)

print("Embeddings shape:", embeddings.shape)

# Train GMM clustering on embeddings
print("\nTraining GMM clustering model...")
n_clusters = 20  # Number of clusters (can be adjusted)
clusterer = GMMClusterer(n_components=n_clusters, random_state=42)
metrics = clusterer.fit(embeddings)

# Get cluster assignments for all documents
cluster_labels = clusterer.predict(embeddings)
cluster_probs = clusterer.predict_proba(embeddings)

print("GMM Clustering complete - BIC: {:.2f}, Converged: {}".format(metrics['bic'], metrics['converged']))

# Save the trained GMM model for later use
model_path = clusterer.save("gmm_model.pkl")
print("GMM model saved to {}".format(model_path))

# Print summary
print("\n" + "="*60)
print("INGESTION SUMMARY")
print("="*60)
print(f"Total documents processed: {len(texts)}")
print(f"Embedding dimension: {embedder.embedding_dim}")
print(f"Number of clusters: {n_clusters}")
print(f"GMM model saved to: gmm_model.pkl")
print(f"Pinecone index: semantic-search")
print("="*60)

# Connect to Pinecone
db = PineconeVectorDB(index_name="semantic-search")
db.create_index(dimension=embedder.embedding_dim)

# Prepare vectors with cluster metadata
records = []
for i, vec in enumerate(embeddings):
    # Include cluster_id and cluster_probs in metadata
    # Truncate text for metadata (Pinecone has size limits)
    text_sample = texts[i][:1000] if len(texts[i]) > 1000 else texts[i]
    metadata = {
        "cluster_id": int(cluster_labels[i]),
        "cluster_prob": float(np.max(cluster_probs[i])),
        "category": ids[i].split("_")[0] if "_" in ids[i] else "unknown",
        "text": text_sample
    }
    records.append((ids[i], vec, metadata))

# Upload
db.upsert_vectors(records)

print("Dataset uploaded successfully")

