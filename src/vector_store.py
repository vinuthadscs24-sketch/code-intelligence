import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

    def build_index(self, chunks, batch_size=128):
        if not chunks:
            print("[VectorStore] Warning: No chunks to index.")
            return

        self.chunks = chunks
        texts = []
        for c in chunks:
            class_name = c.get("class_name") or ""
            method_name = c.get("method_name") or ""
            annotations = " ".join(c.get("annotations", []))
            calls = " ".join(c.get("calls", []))
            code = c.get("code_content") or ""
            
            text_repr = f"Class: {class_name} Method: {method_name} Annotations: {annotations} Calls: {calls}\n{code}".strip()
            texts.append(text_repr)

        print(f"[VectorStore] Generating embeddings for {len(texts)} chunks (batch size {batch_size})...")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        print(f"[VectorStore] Successfully indexed {len(chunks)} chunks into FAISS IndexFlatIP.")

    def search(self, query, top_k=3):
        if self.index.ntotal == 0:
            return []

        query_vector = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                # Returns tuple: (chunk_dict, float_score)
                results.append((self.chunks[idx], float(score)))

        return results