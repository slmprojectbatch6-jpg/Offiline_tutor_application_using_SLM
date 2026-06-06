import sys
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

os.environ["HF_HOME"] = "models"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="models/sentence-transformers", local_files_only=True)
index = faiss.read_index("vector_db/faiss_index")
with open("vector_db/metadata.pkl", "rb") as f:
    chunks = pickle.load(f)

for q in ["what is the Indian constitution", "latest iPhone specifications", "explain quantum computing", "how did the mughal empire decline", "who won the cricket world cup 2024"]:
    vec = model.encode([q], normalize_embeddings=True).astype('float32')
    scores, idxs = index.search(vec, 1)
    print(f"Q: '{q}' -> Best Score: {scores[0][0]*100:.2f}%")
