import os
import pickle
import time
import logging
from typing import List, Dict, Any, Union

import requests
import faiss
import numpy as np


logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS vector store using Ollama's nomic-embed-text embeddings.

    Ollama:
        http://localhost:11434

    Embedding model:
        nomic-embed-text

    Expected embedding dimension:
        768

    Important:
        nomic-embed-text is running with a limited context window,
        so code sent for embedding is intentionally kept compact.
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

        # nomic-embed-text produces 768-dimensional vectors
        self.dimension = 768

        # Keep embedding input safely below Ollama's
        # 2048-token context window.
        #
        # This is CHARACTER based, not token based.
        self.max_embedding_chars = 2000

        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.5

        # FAISS inner-product index.
        # Vectors are L2-normalized before insertion,
        # therefore inner product behaves as cosine similarity.
        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        # Metadata corresponding to FAISS vectors
        self.chunks: List[Dict[str, Any]] = []

        self._check_ollama()

    # ============================================================
    # OLLAMA
    # ============================================================

    def _check_ollama(self):
        """
        Check whether Ollama is running and the embedding
        model is installed.
        """

        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=10,
            )

            response.raise_for_status()

            data = response.json()

            models = data.get(
                "models",
                []
            )

            model_names = [
                str(model.get("name", ""))
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
                    f"is not installed.\n\n"
                    f"Run:\n"
                    f"ollama pull {self.model_name}"
                )

            print(
                "[VectorStore] Ollama connected. "
                f"Embedding model: {self.model_name}"
            )

        except requests.RequestException as e:

            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self.ollama_url}.\n"
                "Make sure Ollama is running."
            ) from e

    # ============================================================
    # TEXT PREPARATION
    # ============================================================

    def _safe_string(
        self,
        value: Any,
    ) -> str:
        """
        Convert arbitrary values to strings safely.
        """

        if value is None:
            return ""

        if isinstance(value, list):

            return " ".join(
                str(item)
                for item in value
            )

        return str(value)

    def _build_text_representation(
        self,
        chunk: Dict[str, Any],
    ) -> str:
        """
        Build a compact semantic representation of a code chunk.

        IMPORTANT:

        We preserve metadata such as:

            class
            method
            signature
            annotations
            calls

        and truncate ONLY the code.

        This is better than blindly truncating the complete
        representation because metadata is highly valuable
        for semantic retrieval.
        """

        # --------------------------------------------------------
        # Basic metadata
        # --------------------------------------------------------

        chunk_type = self._safe_string(
            chunk.get(
                "chunk_type",
                "METHOD"
            )
        )

        class_name = self._safe_string(
            chunk.get(
                "class_name",
                ""
            )
        )

        method_name = self._safe_string(
            chunk.get(
                "method_name",
                ""
            )
        )

        signature = self._safe_string(
            chunk.get(
                "signature",
                ""
            )
        )

        # --------------------------------------------------------
        # Annotations
        # --------------------------------------------------------

        annotations_value = chunk.get(
            "annotations",
            []
        )

        annotations = self._safe_string(
            annotations_value
        )

        # --------------------------------------------------------
        # Calls
        # --------------------------------------------------------

        calls_value = chunk.get(
            "calls",
            []
        )

        calls = self._safe_string(
            calls_value
        )

        # --------------------------------------------------------
        # Source/code
        # --------------------------------------------------------

        code = self._safe_string(
            chunk.get(
                "code_content",
                ""
            )
        )

        if not code:

            code = self._safe_string(
                chunk.get(
                    "source_code",
                    ""
                )
            )

        # --------------------------------------------------------
        # Metadata comes first.
        #
        # This ensures important semantic information remains
        # even when code needs to be shortened.
        # --------------------------------------------------------

        metadata = (
            f"Type: {chunk_type}\n"
            f"Class: {class_name}\n"
            f"Method: {method_name}\n"
            f"Signature: {signature}\n"
            f"Annotations: {annotations}\n"
            f"Calls: {calls}\n"
            f"Code:\n"
        )

        # --------------------------------------------------------
        # Calculate how much code we can keep.
        # --------------------------------------------------------

        available_code_chars = (
            self.max_embedding_chars
            - len(metadata)
        )

        # Always leave at least a small amount for code.
        if available_code_chars < 200:

            available_code_chars = 200

        # --------------------------------------------------------
        # Truncate ONLY code.
        # --------------------------------------------------------

        if len(code) > available_code_chars:

            print(
                "[VectorStore] Truncating code for "
                f"{class_name}.{method_name}: "
                f"{len(code)} -> "
                f"{available_code_chars} chars"
            )

            code = code[
                :available_code_chars
            ]

            code += "\n// [code truncated for embedding]"

        # --------------------------------------------------------
        # Final representation
        # --------------------------------------------------------

        text = (
            metadata
            + code
        )

        return text.strip()

    # ============================================================
    # EMBEDDING
    # ============================================================

    def _generate_single_embedding(
        self,
        text: str,
    ) -> List[float]:
        """
        Generate one embedding using Ollama.
        """

        response = requests.post(
            self.embedding_url,
            json={
                "model": self.model_name,
                "prompt": text,
            },
            timeout=180,
        )

        # Handle HTTP errors ourselves so the message is clearer.
        if response.status_code != 200:

            try:
                error_data = response.json()

                error_message = (
                    error_data.get(
                        "error",
                        response.text
                    )
                )

            except Exception:

                error_message = response.text

            raise RuntimeError(
                f"Ollama HTTP {response.status_code}: "
                f"{error_message}"
            )

        try:

            data = response.json()

        except Exception as e:

            raise RuntimeError(
                "Ollama returned an invalid JSON response."
            ) from e

        if "embedding" not in data:

            raise RuntimeError(
                "Ollama response does not contain "
                f"'embedding': {data}"
            )

        embedding = data["embedding"]

        if not isinstance(
            embedding,
            list
        ):

            raise RuntimeError(
                "Ollama embedding is not a list."
            )

        if len(embedding) != self.dimension:

            raise RuntimeError(
                "Unexpected embedding dimension: "
                f"{len(embedding)}. "
                f"Expected {self.dimension}."
            )

        return embedding

    def _embed(
        self,
        texts: Union[
            str,
            List[str]
        ],
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Ollama's /api/embeddings endpoint is called once
        per text.

        batch_size controls progress logging only.
        """

        if isinstance(
            texts,
            str
        ):

            texts = [texts]

        if not texts:

            return np.empty(
                (0, self.dimension),
                dtype="float32"
            )

        embeddings = []

        total = len(texts)

        print(
            "[VectorStore] Generating Ollama "
            f"embeddings for {total} chunks..."
        )

        for i, text in enumerate(
            texts,
            start=1
        ):

            last_error = None

            for attempt in range(
                1,
                self.max_retries + 1
            ):

                try:

                    embedding = (
                        self._generate_single_embedding(
                            text
                        )
                    )

                    embeddings.append(
                        embedding
                    )

                    break

                except Exception as e:

                    last_error = e

                    print(
                        "[VectorStore] Embedding "
                        f"{i}/{total} failed "
                        f"(attempt "
                        f"{attempt}/"
                        f"{self.max_retries}): "
                        f"{e}"
                    )

                    if attempt < self.max_retries:

                        time.sleep(
                            self.retry_delay
                        )

            else:

                raise RuntimeError(
                    f"Failed to generate embedding "
                    f"{i}/{total} after "
                    f"{self.max_retries} attempts: "
                    f"{last_error}"
                )

            # ----------------------------------------------------
            # Progress
            # ----------------------------------------------------

            if (
                i % batch_size == 0
                or i == total
            ):

                print(
                    "[VectorStore] Generated "
                    f"embeddings: {i}/{total}"
                )

        # --------------------------------------------------------
        # Convert to NumPy
        # --------------------------------------------------------

        matrix = np.asarray(
            embeddings,
            dtype="float32"
        )

        # --------------------------------------------------------
        # Safety check
        # --------------------------------------------------------

        if matrix.ndim != 2:

            raise RuntimeError(
                "Embedding matrix has invalid shape: "
                f"{matrix.shape}"
            )

        if matrix.shape[1] != self.dimension:

            raise RuntimeError(
                "Embedding matrix dimension "
                f"{matrix.shape[1]} does not match "
                f"expected dimension "
                f"{self.dimension}."
            )

        # --------------------------------------------------------
        # Normalize for cosine similarity
        # --------------------------------------------------------

        faiss.normalize_L2(
            matrix
        )

        return matrix

    # ============================================================
    # BUILD INDEX
    # ============================================================

    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 8,
    ):
        """
        Generate embeddings for code chunks
        and build the FAISS index.
        """

        if not chunks:

            print(
                "[VectorStore] Warning: "
                "No chunks to index."
            )

            return

        print(
            "[VectorStore] Preparing "
            f"{len(chunks)} chunks..."
        )

        # --------------------------------------------------------
        # Keep metadata
        # --------------------------------------------------------

        self.chunks = list(
            chunks
        )

        # --------------------------------------------------------
        # Build compact representations
        # --------------------------------------------------------

        texts = []

        for chunk in self.chunks:

            text = (
                self._build_text_representation(
                    chunk
                )
            )

            if not text.strip():

                continue

            texts.append(
                text
            )

        if not texts:

            raise RuntimeError(
                "No valid text was generated "
                "for embedding."
            )

        print(
            "[VectorStore] Valid chunks: "
            f"{len(texts)}/{len(chunks)}"
        )

        # --------------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------------

        embeddings = self._embed(
            texts,
            batch_size=batch_size,
        )

        # --------------------------------------------------------
        # Reset FAISS index
        # --------------------------------------------------------

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        # --------------------------------------------------------
        # Add vectors
        # --------------------------------------------------------

        self.index.add(
            embeddings
        )

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # Metadata count must match FAISS vector count.
        #
        # If all chunks were valid this is identical to chunks.
        # --------------------------------------------------------

        if len(texts) != len(self.chunks):

            print(
                "[VectorStore] Warning: "
                f"{len(texts)} texts generated "
                f"from {len(self.chunks)} chunks."
            )

        print(
            "[VectorStore] Successfully indexed "
            f"{self.index.ntotal} chunks into "
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
        Search FAISS using an Ollama query embedding.
        """

        if not query:

            return []

        if self.index.ntotal == 0:

            return []

        # --------------------------------------------------------
        # Embed query
        # --------------------------------------------------------

        query_embedding = self._embed(
            [query]
        )

        # --------------------------------------------------------
        # Never request more vectors than exist
        # --------------------------------------------------------

        k = min(
            max(1, top_k),
            self.index.ntotal,
        )

        # --------------------------------------------------------
        # FAISS search
        # --------------------------------------------------------

        scores, indices = (
            self.index.search(
                query_embedding,
                k
            )
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):

            if idx == -1:

                continue

            if idx >= len(
                self.chunks
            ):

                continue

            results.append(
                (
                    self.chunks[idx],
                    float(score),
                )
            )

        return results

    # ============================================================
    # SAVE INDEX
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
        Save FAISS index and chunk metadata.
        """

        # --------------------------------------------------------
        # Create directories
        # --------------------------------------------------------

        index_directory = (
            os.path.dirname(
                index_path
            )
        )

        metadata_directory = (
            os.path.dirname(
                metadata_path
            )
        )

        if index_directory:

            os.makedirs(
                index_directory,
                exist_ok=True
            )

        if metadata_directory:

            os.makedirs(
                metadata_directory,
                exist_ok=True
            )

        # --------------------------------------------------------
        # Save FAISS
        # --------------------------------------------------------

        faiss.write_index(
            self.index,
            index_path
        )

        # --------------------------------------------------------
        # Save metadata
        # --------------------------------------------------------

        with open(
            metadata_path,
            "wb"
        ) as f:

            pickle.dump(
                self.chunks,
                f
            )

        print(
            "[VectorStore] Cached FAISS index "
            f"({self.index.ntotal} vectors) "
            "and metadata to disk."
        )

    # ============================================================
    # LOAD INDEX
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
        Load FAISS index and metadata from disk.
        """

        if not (
            os.path.exists(index_path)
            and os.path.exists(metadata_path)
        ):

            print(
                "[VectorStore] No cached index found."
            )

            return False

        try:

            # ----------------------------------------------------
            # Load FAISS
            # ----------------------------------------------------

            loaded_index = (
                faiss.read_index(
                    index_path
                )
            )

            # ----------------------------------------------------
            # Dimension check
            # ----------------------------------------------------

            if (
                loaded_index.d
                != self.dimension
            ):

                print(
                    "[VectorStore] Cached FAISS "
                    "index dimension is "
                    f"{loaded_index.d}, "
                    "but current embedding "
                    "dimension is "
                    f"{self.dimension}."
                )

                print(
                    "[VectorStore] Ignoring "
                    "incompatible cached index."
                )

                return False

            # ----------------------------------------------------
            # Load metadata
            # ----------------------------------------------------

            with open(
                metadata_path,
                "rb"
            ) as f:

                loaded_chunks = (
                    pickle.load(f)
                )

            # ----------------------------------------------------
            # Validate metadata
            # ----------------------------------------------------

            if not isinstance(
                loaded_chunks,
                list
            ):

                print(
                    "[VectorStore] Invalid "
                    "metadata format."
                )

                return False

            # ----------------------------------------------------
            # IMPORTANT:
            #
            # FAISS vector count must match
            # metadata count.
            # ----------------------------------------------------

            if (
                loaded_index.ntotal
                != len(loaded_chunks)
            ):

                print(
                    "[VectorStore] Cached index "
                    "metadata mismatch:"
                )

                print(
                    f"  FAISS vectors: "
                    f"{loaded_index.ntotal}"
                )

                print(
                    f"  Metadata chunks: "
                    f"{len(loaded_chunks)}"
                )

                print(
                    "[VectorStore] Ignoring "
                    "inconsistent cache."
                )

                return False

            # ----------------------------------------------------
            # Assign loaded state
            # ----------------------------------------------------

            self.index = loaded_index

            self.chunks = (
                loaded_chunks
            )

            print(
                "[VectorStore] Successfully "
                "loaded cached FAISS index "
                f"({self.index.ntotal} items)."
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

            return False

    # ============================================================
    # UTILITY
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Return useful vector store statistics.
        """

        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "vectors": int(
                self.index.ntotal
            ),
            "metadata_chunks": len(
                self.chunks
            ),
            "max_embedding_chars": (
                self.max_embedding_chars
            ),
            "ollama_url": self.ollama_url,
        }