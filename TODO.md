# TODO: Fix GMMClusterer Training Issue

## Task
Fix the GMMClusterer so it can be trained and doesn't crash when used before training.

## Steps
- [x] 1. Read and understand the current code (src/clustering.py, src/app.py, src/pinecone_client.py)
- [x] 2. Modify src/clustering.py - Add safe fallbacks when GMM is not fitted
- [x] 3. Modify src/pinecone_client.py - Add fetch_vectors() method
- [x] 4. Modify src/app.py - Add /clustering/train endpoint
- [ ] 5. Test the changes

