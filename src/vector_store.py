import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def _get_text(self, chunk: Dict[str, Any]) -> str:
        if "text_representation" in chunk and chunk["text_representation"]:
            return chunk["text_representation"]
        if "code_content" in chunk and chunk["code_content"]:
            return chunk["code_content"]
        return chunk.get("source_code", str(chunk))

    def build_index(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        if not chunks:
            return

        texts = [self._get_text(c) for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Normalize vectors for cosine similarity (IndexFlatIP)
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.array(embeddings).astype("float32"))
        print(f"[VectorStore] Indexed {len(chunks)} chunks with dimension {dimension}.")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        if not self.index or len(self.chunks) == 0:
            return []

        query_vector = self.model.encode([query])
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(np.array(query_vector).astype("float32"), top_k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results