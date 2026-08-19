import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Generates embeddings from chunk representations and populates FAISS."""
        self.chunks = chunks
        texts = [c["text_representation"] for c in chunks]

        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # L2-normalize for Cosine Similarity matching via Inner Product index
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        print(f"[VectorStore] Indexed {len(chunks)} chunks with dimension {dimension}.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Returns the top_k relevant chunks and similarity scores for a natural language query."""
        if self.index is None or not self.chunks:
            raise ValueError("Index is not initialized. Run build_index() first.")

        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec.astype(np.float32), top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results