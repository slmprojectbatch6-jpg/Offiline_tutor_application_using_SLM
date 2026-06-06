import sys, os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HOME"] = "models"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.path.insert(0, ".")

from sentence_transformers import SentenceTransformer
import faiss, pickle, numpy as np

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="models/sentence-transformers",
    local_files_only=True
)
index = faiss.read_index("vector_db/faiss_index")
with open("vector_db/metadata.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Total chunks in DB: {len(chunks)}")
print("=" * 60)

ncert_questions = [
    "world war 2",
    "world war",
    "mughal empire",
    "fundamental rights india",
    "indian constitution",
    "water cycle geography",
    "GDP economics",
    "french revolution",
    "british colonialism india",
    "rivers in india",
]

bad_questions = [
    "latest iPhone specifications",
    "explain quantum computing",
    "who won cricket world cup 2024",
    "what is ChatGPT",
]

print("\n--- NCERT QUESTIONS (should score HIGH) ---")
for q in ncert_questions:
    vec = model.encode([q], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(vec, 1)
    best = scores[0][0] * 100
    preview = chunks[idxs[0][0]].get("text", "")[:60] if idxs[0][0] >= 0 else "N/A"
    print(f"  [{best:5.1f}%] '{q}' -> {preview}...")

print("\n--- BAD QUESTIONS (should score LOW) ---")
for q in bad_questions:
    vec = model.encode([q], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(vec, 1)
    best = scores[0][0] * 100
    print(f"  [{best:5.1f}%] '{q}'")

print("\n" + "=" * 60)
print("RECOMMENDATION: Set HARD_THRESHOLD just above highest bad score")
