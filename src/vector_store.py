import os
import pickle
import requests
import faiss
import numpy as np
from typing import List, Dict, Any, Union


class VectorStore:
    """
    FAISS vector store using Ollama's nomic-embed-text embeddings.

    Ollama:
        http://localhost:11434

    Model:
        nomic-embed-text

    Dimension:
        768
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")

        self.embedding_url = (
            f"{self.ollama_url}/api/embeddings"
        )

        self.dimension = 768

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.chunks: List[Dict[str, Any]] = []

        self._check_ollama()

    # ============================================================
    # OLLAMA
    # ============================================================

    def _check_ollama(self):
        """Verify Ollama is running and the embedding model exists."""

        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=5,
            )

            response.raise_for_status()

            models = response.json().get(
                "models",
                [],
            )

            model_names = [
                model.get("name", "")
                for model in models
            ]

            model_available = any(
                name == self.model_name
                or name.startswith(
                    f"{self.model_name}:"
                )
                for name in model_names
            )

            if not model_available:
                raise RuntimeError(
                    f"Ollama model '{self.model_name}' "
                    "is not installed. "
                    f"Run: ollama pull {self.model_name}"
                )

            print(
                "[VectorStore] Ollama connected. "
                f"Embedding model: {self.model_name}"
            )

        except requests.RequestException as e:

            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self.ollama_url}. "
                "Make sure Ollama is running."
            ) from e

    # ============================================================
    # EMBEDDINGS
    # ============================================================

    def _embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate embeddings using Ollama.

        Ollama's embeddings endpoint accepts one text
        at a time, so requests are performed sequentially.
        """

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype="float32",
            )

        embeddings = []

        total = len(texts)

        for index, text in enumerate(
            texts,
            start=1,
        ):

            if text is None:
                text = ""

            text = str(text)

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
                        "Ollama response does not contain "
                        f"an embedding: {data}"
                    )

                embedding = data["embedding"]

                if len(embedding) != self.dimension:

                    raise RuntimeError(
                        "Unexpected embedding dimension: "
                        f"{len(embedding)}. "
                        f"Expected {self.dimension}."
                    )

                embeddings.append(
                    embedding
                )

                if (
                    index % batch_size == 0
                    or index == total
                ):

                    print(
                        "[VectorStore] Generated embeddings: "
                        f"{index}/{total}"
                    )

            except requests.RequestException as e:

                raise RuntimeError(
                    f"Failed to generate embedding "
                    f"{index}/{total}: {e}"
                ) from e

        embeddings_array = np.asarray(
            embeddings,
            dtype="float32",
        )

        if embeddings_array.ndim != 2:

            raise RuntimeError(
                "Invalid embedding matrix shape: "
                f"{embeddings_array.shape}"
            )

        if (
            embeddings_array.shape[1]
            != self.dimension
        ):

            raise RuntimeError(
                "Embedding matrix dimension mismatch. "
                f"Got {embeddings_array.shape[1]}, "
                f"expected {self.dimension}."
            )

        # Normalize vectors so inner product behaves
        # like cosine similarity.
        faiss.normalize_L2(
            embeddings_array
        )

        return embeddings_array

    # ============================================================
    # TEXT REPRESENTATION
    # ============================================================

    def _build_text_representation(
        self,
        chunk: Dict[str, Any],
    ) -> str:
        """
        Convert a code chunk into semantic text
        used for embedding generation.
        """

        class_name = str(
            chunk.get("class_name")
            or ""
        )

        method_name = str(
            chunk.get("method_name")
            or ""
        )

        chunk_type = str(
            chunk.get(
                "chunk_type",
                "METHOD",
            )
        )

        annotations = chunk.get(
            "annotations",
            [],
        )

        if isinstance(
            annotations,
            str,
        ):

            annotations_text = annotations

        else:

            annotations_text = " ".join(
                str(annotation)
                for annotation in annotations
            )

        calls = chunk.get(
            "calls",
            [],
        )

        if isinstance(
            calls,
            str,
        ):

            calls_text = calls

        else:

            calls_text = " ".join(
                str(call)
                for call in calls
            )

        code = str(
            chunk.get(
                "code_content",
                ""
            )
            or chunk.get(
                "source_code",
                ""
            )
            or ""
        )

        annotation_boost = ""

        if annotations_text:

            annotation_boost = (
                f"ANNOTATIONS: "
                f"{annotations_text} "
                f"{annotations_text} "
                f"{annotations_text}"
            )

        return (
            f"Type: {chunk_type}\n"
            f"Class: {class_name}\n"
            f"Method: {method_name}\n"
            f"{annotation_boost}\n"
            f"Calls: {calls_text}\n"
            f"Code:\n{code}"
        ).strip()

    # ============================================================
    # BUILD INDEX
    # ============================================================

    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32,
    ):
        """
        Build a fresh FAISS index from code chunks.

        IMPORTANT:
        The metadata list and FAISS vector count are always
        rebuilt together to prevent mismatches.
        """

        if not chunks:

            print(
                "[VectorStore] Warning: "
                "No chunks to index."
            )

            self.chunks = []

            self.index = faiss.IndexFlatIP(
                self.dimension
            )

            return

        # Always replace metadata before building.
        self.chunks = list(chunks)

        texts = [
            self._build_text_representation(
                chunk
            )
            for chunk in self.chunks
        ]

        print(
            "[VectorStore] Generating Ollama "
            f"embeddings for {len(texts)} chunks..."
        )

        embeddings = self._embed(
            texts,
            batch_size=batch_size,
        )

        if len(embeddings) != len(
            self.chunks
        ):

            raise RuntimeError(
                "Embedding count does not match "
                "chunk count. "
                f"Embeddings={len(embeddings)}, "
                f"Chunks={len(self.chunks)}."
            )

        # Always create a completely fresh index.
        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.index.add(
            embeddings
        )

        if (
            self.index.ntotal
            != len(self.chunks)
        ):

            raise RuntimeError(
                "FAISS index count does not match "
                "metadata count after indexing. "
                f"Vectors={self.index.ntotal}, "
                f"Metadata={len(self.chunks)}."
            )

        print(
            "[VectorStore] Successfully indexed "
            f"{len(self.chunks)} chunks into "
            "FAISS IndexFlatIP."
        )

        print(
            "[VectorStore] Embedding dimension: "
            f"{self.dimension}"
        )

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        top_k: int = 3,
    ):
        """
        Search the FAISS index using an Ollama
        query embedding.
        """

        if not query or not str(
            query
        ).strip():

            return []

        if top_k <= 0:
            return []

        if self.index.ntotal == 0:
            return []

        # Critical consistency check.
        if self.index.ntotal != len(
            self.chunks
        ):

            raise RuntimeError(
                "FAISS index and metadata are out of sync. "
                f"Vectors={self.index.ntotal}, "
                f"Metadata={len(self.chunks)}. "
                "Rebuild the index."
            )

        query_embedding = self._embed(
            [str(query)]
        )

        k = min(
            int(top_k),
            self.index.ntotal,
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                k,
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index == -1:
                continue

            if index >= len(
                self.chunks
            ):
                continue

            chunk = self.chunks[
                int(index)
            ]

            results.append(
                (
                    chunk,
                    float(score),
                )
            )

        return results

    # ============================================================
    # SAVE
    # ============================================================

    def save_index(
        self,
        index_path: str = (
            "cache/faiss_index.bin"
        ),
        metadata_path: str = (
            "cache/chunks_meta.pkl"
        ),
    ):
        """
        Save FAISS index and metadata.

        Refuses to save inconsistent state.
        """

        if self.index.ntotal != len(
            self.chunks
        ):

            raise RuntimeError(
                "Cannot save inconsistent vector store. "
                f"Vectors={self.index.ntotal}, "
                f"Metadata={len(self.chunks)}. "
                "Rebuild the index first."
            )

        index_parent = os.path.dirname(
            index_path
        )

        metadata_parent = os.path.dirname(
            metadata_path
        )

        if index_parent:
            os.makedirs(
                index_parent,
                exist_ok=True,
            )

        if metadata_parent:
            os.makedirs(
                metadata_parent,
                exist_ok=True,
            )

        faiss.write_index(
            self.index,
            index_path,
        )

        with open(
            metadata_path,
            "wb",
        ) as file:

            pickle.dump(
                self.chunks,
                file,
            )

        print(
            "[VectorStore] Cached FAISS index "
            f"({self.index.ntotal} vectors) "
            "and metadata to disk."
        )

    # ============================================================
    # LOAD
    # ============================================================

    def load_index(
        self,
        index_path: str = (
            "cache/faiss_index.bin"
        ),
        metadata_path: str = (
            "cache/chunks_meta.pkl"
        ),
    ) -> bool:
        """
        Load FAISS index and metadata.

        IMPORTANT:
        Cached vectors are accepted only when the number
        of FAISS vectors exactly matches the metadata count.
        """

        if not (
            os.path.exists(index_path)
            and os.path.exists(metadata_path)
        ):

            return False

        try:

            loaded_index = faiss.read_index(
                index_path
            )

            # ----------------------------------------------------
            # Dimension validation
            # ----------------------------------------------------

            if (
                loaded_index.d
                != self.dimension
            ):

                print(
                    "[VectorStore] Cached FAISS index "
                    f"dimension is {loaded_index.d}, "
                    "but current embedding dimension "
                    f"is {self.dimension}."
                )

                print(
                    "[VectorStore] Ignoring "
                    "incompatible cached index."
                )

                return False

            with open(
                metadata_path,
                "rb",
            ) as file:

                loaded_chunks = pickle.load(
                    file
                )

            if not isinstance(
                loaded_chunks,
                list,
            ):

                print(
                    "[VectorStore] Cached metadata "
                    "is not a list."
                )

                return False

            # ----------------------------------------------------
            # CRITICAL FIX:
            # FAISS count must equal metadata count.
            # ----------------------------------------------------

            vector_count = (
                loaded_index.ntotal
            )

            metadata_count = len(
                loaded_chunks
            )

            if (
                vector_count
                != metadata_count
            ):

                print(
                    "[VectorStore] Cached index "
                    "is inconsistent."
                )

                print(
                    "[VectorStore] "
                    f"Vectors={vector_count}, "
                    f"Metadata={metadata_count}."
                )

                print(
                    "[VectorStore] Ignoring cached "
                    "index. Rebuilding..."
                )

                return False

            # ----------------------------------------------------
            # Accept cache only after all checks pass.
            # ----------------------------------------------------

            self.index = loaded_index
            self.chunks = loaded_chunks

            print(
                "[VectorStore] Successfully loaded "
                "cached FAISS index "
                f"({self.index.ntotal} vectors)."
            )

            print(
                "[VectorStore] Metadata count: "
                f"{len(self.chunks)}"
            )

            return True

        except Exception as e:

            print(
                "[VectorStore] Failed to load "
                f"cached index: {e}"
            )

            print(
                "[VectorStore] Rebuilding index..."
            )

            # Reset to a clean state.
            self.index = faiss.IndexFlatIP(
                self.dimension
            )

            self.chunks = []

            return False