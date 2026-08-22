import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

    def build_index(self, chunks):
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
            
            # Rich semantic string combining code and metadata
            text_repr = f"Class: {class_name} Method: {method_name} Annotations: {annotations} Calls: {calls}\n{code}".strip()
            texts.append(text_repr)

        # Generate embeddings and normalize for cosine similarity / IndexFlatIP
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        print(f"[VectorStore] Indexed {len(chunks)} chunks with dimension {self.dimension}.")

    def search(self, query, top_k=3):
        if self.index.ntotal == 0:
            return []

        query_vector = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((float(score), self.chunks[idx]))

        return results