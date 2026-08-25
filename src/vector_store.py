import os
import pickle
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
            annotations_list = c.get("annotations", [])
            annotations = " ".join(annotations_list)
            calls = " ".join(c.get("calls", []))
            code = c.get("code_content") or ""
            chunk_type = c.get("chunk_type", "METHOD")
            
            # Boost annotation prominence in embedding space by tripling their appearance
            annotation_boost = f"ANNOTATIONS: {annotations} {annotations} {annotations}" if annotations else ""
            
            text_repr = (
                f"Type: {chunk_type} | Class: {class_name} | Method: {method_name}\n"
                f"{annotation_boost}\n"
                f"Calls: {calls}\n"
                f"Code:\n{code}"
            ).strip()
            
            texts.append(text_repr)

        print(f"[VectorStore] Generating embeddings for {len(texts)} chunks (batch size {batch_size})...")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        # Reset index before adding new embeddings
        self.index = faiss.IndexFlatIP(self.dimension)
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
                results.append((self.chunks[idx], float(score)))

        return results

    def save_index(self, index_path="cache/faiss_index.bin", metadata_path="cache/chunks_meta.pkl"):
        """Saves the FAISS index and chunk metadata to disk."""
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[VectorStore] Cached FAISS index ({self.index.ntotal} vectors) and metadata to disk.")

    def load_index(self, index_path="cache/faiss_index.bin", metadata_path="cache/chunks_meta.pkl") -> bool:
        """Loads FAISS index and metadata from disk if both files exist."""
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(metadata_path, "rb") as f:
                    self.chunks = pickle.load(f)
                print(f"[VectorStore] Successfully loaded cached FAISS index ({self.index.ntotal} items).")
                return True
            except Exception as e:
                print(f"[VectorStore] Failed to load cached index: {e}. Rebuilding index...")
                return False
        return False