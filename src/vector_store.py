import os
import pickle
import requests
import faiss
import numpy as np


class VectorStore:
    """
    FAISS vector store using Ollama's nomic-embed-text embeddings.

    Ollama:
        http://localhost:11434
        Model: nomic-embed-text
        Dimension: 768
    """

    def __init__(
        self,
        model_name="nomic-embed-text",
        ollama_url="http://localhost:11434",
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")

        self.embedding_url = f"{self.ollama_url}/api/embeddings"

        # nomic-embed-text produces 768-dimensional embeddings
        self.dimension = 768

        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

        self._check_ollama()

    def _check_ollama(self):
        """Check that Ollama is running and the embedding model is available."""
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=5,
            )
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            if not any(
                name == self.model_name
                or name.startswith(f"{self.model_name}:")
                for name in model_names
            ):
                raise RuntimeError(
                    f"Ollama model '{self.model_name}' is not installed. "
                    f"Run: ollama pull {self.model_name}"
                )

            print(
                f"[VectorStore] Ollama connected. "
                f"Embedding model: {self.model_name}"
            )

        except requests.RequestException as e:
            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self.ollama_url}. Make sure Ollama is running."
            ) from e

    def _embed(self, texts, batch_size=32):
        """
        Generate embeddings using Ollama.

        Ollama's /api/embeddings endpoint accepts one text at a time,
        so requests are sent individually.
        """

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        total = len(texts)

        for i, text in enumerate(texts, start=1):
            try:
                response = requests.post(
                    self.embedding_url,
                    json={
                        "model": self.model_name,
                        "prompt": text,
                    },
                    timeout=120,
                )

                response.raise_for_status()

                data = response.json()

                if "embedding" not in data:
                    raise RuntimeError(
                        f"Ollama response does not contain an embedding: {data}"
                    )

                embedding = data["embedding"]

                if len(embedding) != self.dimension:
                    raise RuntimeError(
                        f"Unexpected embedding dimension: {len(embedding)}. "
                        f"Expected {self.dimension}."
                    )

                embeddings.append(embedding)

                if i % batch_size == 0 or i == total:
                    print(
                        f"[VectorStore] Generated embeddings: "
                        f"{i}/{total}"
                    )

            except requests.RequestException as e:
                raise RuntimeError(
                    f"Failed to generate embedding {i}/{total}: {e}"
                ) from e

        embeddings = np.array(
            embeddings,
            dtype="float32",
        )

        # Normalize vectors so inner product becomes cosine similarity
        faiss.normalize_L2(embeddings)

        return embeddings

    def _build_text_representation(self, chunk):
        """Convert a code chunk into text suitable for embedding."""

        class_name = chunk.get("class_name") or ""
        method_name = chunk.get("method_name") or ""

        annotations_list = chunk.get("annotations", [])
        annotations = " ".join(annotations_list)

        calls = " ".join(chunk.get("calls", []))

        code = chunk.get("code_content") or ""

        chunk_type = chunk.get(
            "chunk_type",
            "METHOD",
        )

        # Give annotations additional semantic importance.
        annotation_boost = ""

        if annotations:
            annotation_boost = (
                f"ANNOTATIONS: "
                f"{annotations} "
                f"{annotations} "
                f"{annotations}"
            )

        text_repr = (
            f"Type: {chunk_type} | "
            f"Class: {class_name} | "
            f"Method: {method_name}\n"
            f"{annotation_boost}\n"
            f"Calls: {calls}\n"
            f"Code:\n{code}"
        ).strip()

        return text_repr

    def build_index(self, chunks, batch_size=32):
        """Generate embeddings for chunks and build the FAISS index."""

        if not chunks:
            print("[VectorStore] Warning: No chunks to index.")
            return

        self.chunks = chunks

        texts = [
            self._build_text_representation(chunk)
            for chunk in chunks
        ]

        print(
            f"[VectorStore] Generating Ollama embeddings "
            f"for {len(texts)} chunks..."
        )

        embeddings = self._embed(
            texts,
            batch_size=batch_size,
        )

        # Reset index before adding new embeddings
        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.index.add(embeddings)

        print(
            f"[VectorStore] Successfully indexed "
            f"{len(chunks)} chunks into "
            f"FAISS IndexFlatIP."
        )

        print(
            f"[VectorStore] Embedding dimension: "
            f"{self.dimension}"
        )

    def search(self, query, top_k=3):
        """Search the FAISS index using an Ollama query embedding."""

        if self.index.ntotal == 0:
            return []

        query_embedding = self._embed(
            [query]
        )

        # Never request more results than vectors available
        k = min(
            top_k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):
            if (
                idx != -1
                and idx < len(self.chunks)
            ):
                results.append(
                    (
                        self.chunks[idx],
                        float(score),
                    )
                )

        return results

    def save_index(
        self,
        index_path="cache/faiss_index.bin",
        metadata_path="cache/chunks_meta.pkl",
    ):
        """Save FAISS index and chunk metadata to disk."""

        os.makedirs(
            os.path.dirname(index_path) or ".",
            exist_ok=True,
        )

        os.makedirs(
            os.path.dirname(metadata_path) or ".",
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            index_path,
        )

        with open(
            metadata_path,
            "wb",
        ) as f:
            pickle.dump(
                self.chunks,
                f,
            )

        print(
            f"[VectorStore] Cached FAISS index "
            f"({self.index.ntotal} vectors) "
            f"and metadata to disk."
        )

    def load_index(
        self,
        index_path="cache/faiss_index.bin",
        metadata_path="cache/chunks_meta.pkl",
    ):
        """Load FAISS index and metadata from disk."""

        if not (
            os.path.exists(index_path)
            and os.path.exists(metadata_path)
        ):
            return False

        try:
            loaded_index = faiss.read_index(
                index_path
            )

            # Make sure cached index matches
            # the current embedding model.
            if loaded_index.d != self.dimension:
                print(
                    "[VectorStore] Cached FAISS index "
                    f"dimension is {loaded_index.d}, "
                    f"but current embedding dimension "
                    f"is {self.dimension}."
                )

                print(
                    "[VectorStore] Ignoring incompatible "
                    "cached index."
                )

                return False

            self.index = loaded_index

            with open(
                metadata_path,
                "rb",
            ) as f:
                self.chunks = pickle.load(f)

            print(
                f"[VectorStore] Successfully loaded "
                f"cached FAISS index "
                f"({self.index.ntotal} items)."
            )

            return True

        except Exception as e:
            print(
                f"[VectorStore] Failed to load cached "
                f"index: {e}. Rebuilding index..."
            )

            return False