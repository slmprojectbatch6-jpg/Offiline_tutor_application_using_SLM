import sys
import os

# Add the project directory to sys.path so we can import backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag.pipeline import get_embedding_model, get_faiss_index
from backend.rag.agent_rag_fixes import calibrate_thresholds

embed_model = get_embedding_model()
faiss_index, metadata = get_faiss_index()

calibrate_thresholds(faiss_index, embed_model, metadata)
