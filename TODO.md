# TODO: Fix GMMClusterer Training Issue

## Task
Fix the GMMClusterer so it can be trained and doesn't crash when used before training.

## Steps
- [x] 1. Read and understand the current code (src/clustering.py, src/app.py, src/pinecone_client.py)
- [x] 2. Modify src/clustering.py - Add safe fallbacks when GMM is not fitted
- [x] 3. Modify src/pinecone_client.py - Add fetch_vectors() method
- [x] 4. Modify src/app.py - Add /clustering/train endpoint
- [x] 5. Add model persistence (save/load methods) - src/clustering.py
- [x] 6. Load saved model on startup - src/app.py
- [x] 7. Save model after training - src/app.py
- [ ] 8. Test the changes

