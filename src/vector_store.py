import os
import pickle
import time
import logging
from typing import Any, Dict, List, Optional

import requests
import faiss
import numpy as np


logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store using Ollama embeddings.

    Default embedding model:
        nomic-embed-text

    Default embedding dimension:
        768
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
        dimension: int = 768,
        index_dir: str = "data/indexes",
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.dimension = dimension
        self.index_dir = index_dir

        # IMPORTANT:
        # Keep the final embedding text safely below the
        # Ollama model context length.
        #
        # This is a CHARACTER limit, not a token limit.
        # 1200 chars is intentionally conservative.
        self.max_embedding_chars = 1200

        self.embedding_url = (
            f"{self.ollama_url}/api/embeddings"
        )

        self.max_retries = 3
        self.retry_delay = 1.5

        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []

        os.makedirs(
            self.index_dir,
            exist_ok=True,
        )

    # ============================================================
    # OLLAMA
    # ============================================================

    def _check_ollama(self) -> bool:
        """
        Check whether Ollama is running and reachable.
        """

        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=10,
            )

            response.raise_for_status()

            return True

        except requests.RequestException as exc:

            logger.error(
                "Ollama is not reachable: %s",
                exc,
            )

            return False

    # ============================================================
    # SAFE STRING
    # ============================================================

    def _safe_string(
        self,
        value: Any,
    ) -> str:
        """
        Convert arbitrary values into safe strings.
        """

        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, list):

            parts = []

            for item in value:

                if isinstance(item, dict):

                    parts.append(
                        str(item)
                    )

                else:

                    parts.append(
                        str(item)
                    )

            return ", ".join(parts)

        if isinstance(value, dict):
            return str(value)

        return str(value)

    # ============================================================
    # BUILD EMBEDDING TEXT
    # ============================================================

    def _build_text_representation(
        self,
        chunk: Dict[str, Any],
    ) -> str:
        """
        Build a compact semantic representation of a code chunk.

        The returned text is ALWAYS capped at
        self.max_embedding_chars.

        Important design decisions:

        1. Structural metadata is preserved.
        2. Calls are represented compactly.
        3. Complete call dictionaries are NOT dumped into the
           embedding text.
        4. Code is truncated first.
        5. Metadata is progressively shortened if necessary.
        6. A final hard safety limit is always applied.
        """

        # --------------------------------------------------------
        # BASIC METADATA
        # --------------------------------------------------------

        chunk_type = self._safe_string(
            chunk.get(
                "chunk_type",
                "METHOD",
            )
        )

        class_name = self._safe_string(
            chunk.get(
                "class_name",
                "",
            )
        )

        method_name = self._safe_string(
            chunk.get(
                "method_name",
                "",
            )
        )

        signature = self._safe_string(
            chunk.get(
                "signature",
                "",
            )
        )

        annotations_value = chunk.get(
            "annotations",
            [],
        )

        annotations = self._safe_string(
            annotations_value
        )

        # --------------------------------------------------------
        # COMPACT CALL INFORMATION
        # --------------------------------------------------------

        calls_value = chunk.get(
            "calls",
            [],
        )

        call_names: List[str] = []

        if isinstance(
            calls_value,
            list,
        ):

            for call in calls_value:

                if isinstance(
                    call,
                    dict,
                ):

                    method_called = str(
                        call.get(
                            "method_called",
                            "",
                        )
                        or ""
                    ).strip()

                    object_expression = str(
                        call.get(
                            "object_expression",
                            "",
                        )
                        or ""
                    ).strip()

                    if (
                        object_expression
                        and method_called
                    ):

                        call_names.append(
                            f"{object_expression}.{method_called}"
                        )

                    elif method_called:

                        call_names.append(
                            method_called
                        )

                    else:

                        target_method = str(
                            call.get(
                                "target_method",
                                "",
                            )
                            or ""
                        ).strip()

                        if target_method:
                            call_names.append(
                                target_method
                            )

                else:

                    value = str(
                        call
                    ).strip()

                    if value:
                        call_names.append(
                            value
                        )

        else:

            calls_string = self._safe_string(
                calls_value
            )

            if calls_string:
                call_names.append(
                    calls_string
                )

        # Remove duplicates while preserving order.
        call_names = list(
            dict.fromkeys(
                call_names
            )
        )

        calls = ", ".join(
            call_names
        )

        # --------------------------------------------------------
        # CODE
        # --------------------------------------------------------

        code = self._safe_string(
            chunk.get(
                "code_content",
                "",
            )
        )

        if not code:

            code = self._safe_string(
                chunk.get(
                    "source_code",
                    "",
                )
            )

        # --------------------------------------------------------
        # HARD LIMIT
        # --------------------------------------------------------

        limit = int(
            self.max_embedding_chars
        )

        if limit <= 0:

            limit = 1200

        # --------------------------------------------------------
        # FIRST METADATA VERSION
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
        # IF METADATA IS TOO LARGE
        # --------------------------------------------------------

        if len(metadata) >= limit:

            compact_calls = calls

            if len(
                compact_calls
            ) > 300:

                compact_calls = (
                    compact_calls[:300]
                    + "..."
                )

            compact_annotations = annotations

            if len(
                compact_annotations
            ) > 150:

                compact_annotations = (
                    compact_annotations[:150]
                    + "..."
                )

            compact_signature = signature

            if len(
                compact_signature
            ) > 250:

                compact_signature = (
                    compact_signature[:250]
                    + "..."
                )

            metadata = (
                f"Type: {chunk_type}\n"
                f"Class: {class_name}\n"
                f"Method: {method_name}\n"
                f"Signature: {compact_signature}\n"
                f"Annotations: {compact_annotations}\n"
                f"Calls: {compact_calls}\n"
                f"Code:\n"
            )

        # --------------------------------------------------------
        # SECONDARY METADATA FALLBACK
        # --------------------------------------------------------

        if len(metadata) >= limit:

            metadata = (
                f"Type: {chunk_type}\n"
                f"Class: {class_name}\n"
                f"Method: {method_name}\n"
                f"Signature: {signature[:200]}\n"
                f"Calls: {calls[:250]}\n"
                f"Code:\n"
            )

        # --------------------------------------------------------
        # FINAL METADATA SAFETY
        # --------------------------------------------------------

        if len(metadata) >= limit:

            metadata = (
                f"Type: {chunk_type}\n"
                f"Class: {class_name}\n"
                f"Method: {method_name}\n"
                f"Code:\n"
            )

        # --------------------------------------------------------
        # CALCULATE SPACE AVAILABLE FOR CODE
        # --------------------------------------------------------

        available_code_chars = (
            limit - len(metadata)
        )

        if available_code_chars < 0:

            available_code_chars = 0

        # --------------------------------------------------------
        # TRUNCATE CODE
        # --------------------------------------------------------

        if len(code) > available_code_chars:

            if available_code_chars > 40:

                code = (
                    code[
                        : available_code_chars - 40
                    ]
                    + "\n// [code truncated]"
                )

            else:

                code = code[
                    :available_code_chars
                ]

        # --------------------------------------------------------
        # FINAL TEXT
        # --------------------------------------------------------

        text = (
            metadata
            + code
        ).strip()

        # --------------------------------------------------------
        # ABSOLUTE FINAL SAFETY GUARD
        # --------------------------------------------------------

        if len(text) > limit:

            text = text[:limit]

        return text

    # ============================================================
    # SINGLE EMBEDDING
    # ============================================================

    def _generate_single_embedding(
        self,
        text: str,
    ) -> List[float]:
        """
        Generate one embedding using Ollama.

        The text is hard-capped before being sent to Ollama.
        """

        if not isinstance(
            text,
            str,
        ):

            text = str(text)

        # Absolute safety guard.
        text = text[
            : self.max_embedding_chars
        ]

        if not text.strip():

            raise ValueError(
                "Cannot generate embedding for empty text."
            )

        payload = {
            "model": self.model_name,
            "prompt": text,
        }

        last_error = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):

            try:

                response = requests.post(
                    self.embedding_url,
                    json=payload,
                    timeout=120,
                )

                if not response.ok:

                    try:
                        error_data = (
                            response.json()
                        )

                        error_message = (
                            error_data.get(
                                "error",
                                response.text,
                            )
                        )

                    except Exception:

                        error_message = (
                            response.text
                        )

                    raise RuntimeError(
                        "Ollama HTTP "
                        f"{response.status_code}: "
                        f"{error_message}"
                    )

                data = response.json()

                embedding = data.get(
                    "embedding"
                )

                if not embedding:

                    raise RuntimeError(
                        "Ollama response did not "
                        "contain an embedding."
                    )

                embedding = [
                    float(value)
                    for value in embedding
                ]

                if len(embedding) != self.dimension:

                    raise RuntimeError(
                        "Embedding dimension mismatch. "
                        f"Expected {self.dimension}, "
                        f"got {len(embedding)}."
                    )

                return embedding

            except Exception as exc:

                last_error = exc

                logger.warning(
                    "Embedding attempt %s/%s failed: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

                if attempt < self.max_retries:

                    time.sleep(
                        self.retry_delay
                    )

        raise RuntimeError(
            "Failed to generate embedding "
            f"after {self.max_retries} attempts: "
            f"{last_error}"
        )

    # ============================================================
    # EMBED MULTIPLE TEXTS
    # ============================================================

    def _embed(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        """

        if not texts:

            return np.empty(
                (
                    0,
                    self.dimension,
                ),
                dtype=np.float32,
            )

        embeddings = []

        total = len(texts)

        for index, text in enumerate(
            texts,
            start=1,
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

                if (
                    index == 1
                    or index % 10 == 0
                    or index == total
                ):

                    logger.info(
                        "Generated embedding "
                        "%s/%s",
                        index,
                        total,
                    )

            except Exception as exc:

                logger.error(
                    "Failed to generate embedding "
                    "%s/%s after %s attempts: %s",
                    index,
                    total,
                    self.max_retries,
                    exc,
                )

                raise

        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        # Safety check.
        if matrix.ndim != 2:

            raise RuntimeError(
                "Embedding matrix must be 2-dimensional."
            )

        if matrix.shape[1] != self.dimension:

            raise RuntimeError(
                "Embedding matrix dimension mismatch. "
                f"Expected {self.dimension}, "
                f"got {matrix.shape[1]}."
            )

        return matrix

    # ============================================================
    # BUILD INDEX
    # ============================================================

    def build_index(
        self,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """
        Build a FAISS index from code chunks.
        """

        if not chunks:

            raise ValueError(
                "Cannot build vector index from zero chunks."
            )

        logger.info(
            "Building vector index for %s chunks...",
            len(chunks),
        )

        # --------------------------------------------------------
        # Build embedding texts
        # --------------------------------------------------------

        texts = []

        valid_chunks = []

        for chunk in chunks:

            text = (
                self._build_text_representation(
                    chunk
                )
            )

            if not text.strip():

                logger.warning(
                    "Skipping empty chunk: %s",
                    chunk.get(
                        "chunk_id",
                        "unknown",
                    ),
                )

                continue

            # Absolute final safety check.
            text = text[
                : self.max_embedding_chars
            ]

            texts.append(
                text
            )

            valid_chunks.append(
                chunk
            )

        if not texts:

            raise ValueError(
                "No valid text available for embedding."
            )

        # --------------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------------

        embeddings = self._embed(
            texts
        )

        # --------------------------------------------------------
        # Verify metadata/vector alignment
        # --------------------------------------------------------

        if len(embeddings) != len(
            valid_chunks
        ):

            raise RuntimeError(
                "Vector count does not match "
                "metadata count. "
                f"vectors={len(embeddings)}, "
                f"metadata={len(valid_chunks)}"
            )

        # --------------------------------------------------------
        # Normalize embeddings
        #
        # IndexFlatIP + normalized vectors gives cosine similarity.
        # --------------------------------------------------------

        faiss.normalize_L2(
            embeddings
        )

        # --------------------------------------------------------
        # Create FAISS index
        # --------------------------------------------------------

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.index.add(
            embeddings
        )

        # IMPORTANT:
        # Store exactly the chunks corresponding to vectors.
        self.chunks = valid_chunks

        logger.info(
            "FAISS index built successfully. "
            "Vectors: %s, Dimensions: %s",
            self.index.ntotal,
            self.dimension,
        )

        if (
            self.index.ntotal
            != len(self.chunks)
        ):

            raise RuntimeError(
                "FAISS vector count does not match "
                "chunk metadata count after index build. "
                f"vectors={self.index.ntotal}, "
                f"chunks={len(self.chunks)}"
            )

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search the vector index.
        """

        if self.index is None:

            raise RuntimeError(
                "Vector index has not been built or loaded."
            )

        if not self.chunks:

            return []

        query = self._safe_string(
            query
        ).strip()

        if not query:

            return []

        query_embedding = (
            self._generate_single_embedding(
                query
            )
        )

        query_vector = np.asarray(
            [query_embedding],
            dtype=np.float32,
        )

        faiss.normalize_L2(
            query_vector
        )

        actual_k = min(
            max(
                int(top_k),
                1,
            ),
            self.index.ntotal,
        )

        scores, indices = (
            self.index.search(
                query_vector,
                actual_k,
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index < 0:
                continue

            if index >= len(
                self.chunks
            ):
                continue

            chunk = dict(
                self.chunks[index]
            )

            chunk["score"] = float(
                score
            )

            results.append(
                chunk
            )

        return results

    # ============================================================
    # SAVE INDEX
    # ============================================================

    def save_index(
        self,
        repo_name: str,
    ) -> Dict[str, str]:
        """
        Save FAISS index and chunk metadata.
        """

        if self.index is None:

            raise RuntimeError(
                "Cannot save an empty vector index."
            )

        repo_dir = os.path.join(
            self.index_dir,
            repo_name,
        )

        os.makedirs(
            repo_dir,
            exist_ok=True,
        )

        vector_index_path = os.path.join(
            repo_dir,
            "vector_index.faiss",
        )

        metadata_path = os.path.join(
            repo_dir,
            "metadata.pkl",
        )

        faiss.write_index(
            self.index,
            vector_index_path,
        )

        with open(
            metadata_path,
            "wb",
        ) as file:

            pickle.dump(
                self.chunks,
                file,
            )

        logger.info(
            "Vector index saved: %s",
            vector_index_path,
        )

        logger.info(
            "Metadata saved: %s",
            metadata_path,
        )

        return {
            "vector_index": vector_index_path,
            "metadata": metadata_path,
        }

    # ============================================================
    # LOAD INDEX
    # ============================================================

    def load_index(
        self,
        repo_name: str,
    ) -> bool:
        """
        Load FAISS index and metadata.
        """

        repo_dir = os.path.join(
            self.index_dir,
            repo_name,
        )

        vector_index_path = os.path.join(
            repo_dir,
            "vector_index.faiss",
        )

        metadata_path = os.path.join(
            repo_dir,
            "metadata.pkl",
        )

        if not os.path.exists(
            vector_index_path
        ):

            logger.warning(
                "FAISS index not found: %s",
                vector_index_path,
            )

            return False

        if not os.path.exists(
            metadata_path
        ):

            logger.warning(
                "Metadata not found: %s",
                metadata_path,
            )

            return False

        try:

            loaded_index = faiss.read_index(
                vector_index_path
            )

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

                raise RuntimeError(
                    "Loaded metadata is not a list."
                )

            if (
                loaded_index.ntotal
                != len(loaded_chunks)
            ):

                raise RuntimeError(
                    "Loaded FAISS index and metadata "
                    "are out of sync. "
                    f"vectors={loaded_index.ntotal}, "
                    f"metadata={len(loaded_chunks)}"
                )

            if (
                loaded_index.d
                != self.dimension
            ):

                raise RuntimeError(
                    "Loaded FAISS dimension mismatch. "
                    f"Expected {self.dimension}, "
                    f"got {loaded_index.d}"
                )

            self.index = loaded_index
            self.chunks = loaded_chunks

            logger.info(
                "FAISS index loaded successfully. "
                "Vectors: %s, Dimensions: %s",
                self.index.ntotal,
                self.index.d,
            )

            logger.info(
                "Metadata loaded: %s chunks",
                len(self.chunks),
            )

            return True

        except Exception as exc:

            logger.error(
                "Failed to load vector index: %s",
                exc,
            )

            self.index = None
            self.chunks = []

            return False

    # ============================================================
    # STATS
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Return vector store statistics.
        """

        return {
            "index_loaded": self.index is not None,
            "vector_count": (
                self.index.ntotal
                if self.index is not None
                else 0
            ),
            "dimension": (
                self.index.d
                if self.index is not None
                else self.dimension
            ),
            "metadata_count": len(
                self.chunks
            ),
            "model_name": self.model_name,
            "max_embedding_chars": (
                self.max_embedding_chars
            ),
        }

    # ============================================================
    # CLEAR
    # ============================================================

    def clear(self) -> None:
        """
        Clear the current in-memory index.
        """

        self.index = None
        self.chunks = []

        logger.info(
            "Vector store cleared."
        )