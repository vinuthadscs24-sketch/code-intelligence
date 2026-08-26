import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import faiss
import numpy as np
import requests
from loguru import logger

from src import config
from src.data_processing.chunkers import DocumentChunk


# ============================================================
# OLLAMA EMBEDDINGS
# ============================================================

def _process_batch(
    batch_info: Tuple[int, List[str]],
    model_name: str,
    dimensions: int,
    apikey: Optional[str] = None
) -> Tuple[int, Optional[List[List[float]]]]:
    """
    Generate embeddings for one batch using Ollama.
    """

    batch_idx, batch_texts = batch_info

    try:
        url = f"{config.OLLAMA_BASE_URL}/api/embed"

        response = requests.post(
            url,
            json={
                "model": model_name,
                "input": batch_texts
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        embeddings = data.get("embeddings")

        if not embeddings:
            raise ValueError(
                "Ollama returned no embeddings."
            )

        # Check embedding dimension
        actual_dimension = len(embeddings[0])

        if actual_dimension != dimensions:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {dimensions}, "
                f"got {actual_dimension}."
            )

        logger.debug(
            f"Generated Ollama embeddings for batch "
            f"{batch_idx + 1}"
        )

        return batch_idx, embeddings

    except Exception as e:

        logger.error(
            f"Error generating Ollama embeddings "
            f"for batch {batch_idx + 1}: {e}"
        )

        return batch_idx, None


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(
    texts: List[str],
    model_name: str = config.EMBEDDING_MODEL_NAME,
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
    dimensions: int = config.EMBEDDING_DIMENSIONS,
    apikey: Optional[str] = None,
    max_workers: int = 4
) -> Optional[np.ndarray]:
    """
    Generate embeddings using Ollama.

    Returns:
        NumPy float32 array with shape:
        (number_of_texts, embedding_dimension)
    """

    if not texts:
        return np.array([], dtype=np.float32)

    logger.info(
        f"Generating embeddings for {len(texts)} texts "
        f"using Ollama model '{model_name}' "
        f"(batch size: {batch_size})"
    )

    # --------------------------------------------------------
    # Create batches
    # --------------------------------------------------------

    batches = []

    for i in range(0, len(texts), batch_size):

        batch_texts = texts[i:i + batch_size]

        batches.append(
            (
                i // batch_size,
                batch_texts
            )
        )

    logger.info(
        f"Created {len(batches)} embedding batches."
    )

    # --------------------------------------------------------
    # Process batches in parallel
    # --------------------------------------------------------

    all_embeddings = [None] * len(batches)

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        future_to_batch = {
            executor.submit(
                _process_batch,
                batch_info,
                model_name,
                dimensions,
                apikey
            ): batch_info[0]
            for batch_info in batches
        }

        for future in as_completed(
            future_to_batch
        ):

            batch_idx = future_to_batch[future]

            try:

                result_batch_idx, batch_embeddings = (
                    future.result()
                )

                if batch_embeddings is None:

                    logger.error(
                        f"Failed to generate embeddings "
                        f"for batch {batch_idx + 1}"
                    )

                    return None

                all_embeddings[
                    result_batch_idx
                ] = batch_embeddings

            except Exception as e:

                logger.error(
                    f"Exception in batch processing: {e}"
                )

                return None

    # --------------------------------------------------------
    # Flatten embeddings while preserving order
    # --------------------------------------------------------

    embeddings_list = []

    for batch_embeddings in all_embeddings:

        if batch_embeddings is None:

            logger.error(
                "One or more embedding batches failed."
            )

            return None

        embeddings_list.extend(
            batch_embeddings
        )

    if not embeddings_list:

        logger.warning(
            "No embeddings were generated."
        )

        return np.array([], dtype=np.float32)

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    try:

        embeddings_array = np.array(
            embeddings_list,
            dtype=np.float32
        )

        logger.info(
            f"Embedding matrix shape: "
            f"{embeddings_array.shape}"
        )

        return embeddings_array

    except ValueError as e:

        logger.error(
            f"Could not convert embeddings "
            f"to NumPy array: {e}"
        )

        return None


# ============================================================
# FAISS VECTOR INDEX
# ============================================================

class FaissVectorIndex:
    """
    Manages the FAISS vector index and associated metadata.
    """

    def __init__(
        self,
        index_dir: Path,
        embedding_dim: int = config.EMBEDDING_DIMENSIONS,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        batch_size: int = config.EMBEDDING_BATCH_SIZE
    ):

        self.index_dir = index_dir

        self.embedding_dim = embedding_dim

        self.index_file_path = (
            self.index_dir
            / config.FAISS_INDEX_FILENAME
        )

        self.metadata_file_path = (
            self.index_dir
            / config.METADATA_FILENAME
        )

        self.model = model_name

        self.batch_size = batch_size

        self.index: Optional[faiss.Index] = None

        self.chunk_metadata: List[
            Dict[str, Any]
        ] = []

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # BUILD INDEX
    # ========================================================

    def build_index(
        self,
        chunks: List[DocumentChunk],
        force_rebuild: bool = False,
        apikey: Optional[str] = None
    ) -> bool:

        if (
            not force_rebuild
            and self.index_file_path.exists()
            and self.metadata_file_path.exists()
        ):

            logger.info(
                f"Index already exists at "
                f"{self.index_dir}."
            )

            return True

        if not chunks:

            logger.warning(
                "No chunks provided to build "
                "the index."
            )

            return False

        logger.info(
            f"Building FAISS index with "
            f"{len(chunks)} chunks..."
        )

        # ----------------------------------------------------
        # Extract chunk text
        # ----------------------------------------------------

        chunk_texts = [
            chunk.content
            for chunk in chunks
        ]

        # ----------------------------------------------------
        # Generate Ollama embeddings
        # ----------------------------------------------------

        embeddings = generate_embeddings(
            texts=chunk_texts,
            model_name=self.model,
            batch_size=self.batch_size,
            dimensions=self.embedding_dim,
            apikey=apikey,
            max_workers=4
        )

        if (
            embeddings is None
            or embeddings.shape[0] == 0
        ):

            logger.error(
                "Failed to generate embeddings. "
                "Index not built."
            )

            return False

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if embeddings.shape[1] != self.embedding_dim:

            logger.error(
                f"Generated embedding dimension "
                f"({embeddings.shape[1]}) does not match "
                f"configured dimension "
                f"({self.embedding_dim})."
            )

            return False

        # ----------------------------------------------------
        # Create FAISS index
        # ----------------------------------------------------

        self.index = faiss.IndexFlatL2(
            self.embedding_dim
        )

        self.index.add(embeddings)

        logger.info(
            f"FAISS index built successfully. "
            f"Total vectors: {self.index.ntotal}"
        )

        # ----------------------------------------------------
        # Create metadata
        # ----------------------------------------------------

        self.chunk_metadata = []

        for i, chunk in enumerate(chunks):

            meta_item = chunk.model_dump(
                exclude={"absolute_path"}
            )

            meta_item[
                "original_file_path"
            ] = str(
                chunk.original_file_path
            )

            meta_item[
                "file_path"
            ] = str(
                chunk.file_path
            )

            meta_item[
                "vector_id"
            ] = i

            self.chunk_metadata.append(
                meta_item
            )

        # ----------------------------------------------------
        # Save index
        # ----------------------------------------------------

        self.save_index()

        return True

    # ========================================================
    # SAVE INDEX
    # ========================================================

    def save_index(self):

        if self.index is not None:

            logger.info(
                f"Saving FAISS index to: "
                f"{self.index_file_path}"
            )

            faiss.write_index(
                self.index,
                str(self.index_file_path)
            )

        else:

            logger.warning(
                "No FAISS index to save."
            )

        if self.chunk_metadata:

            logger.info(
                f"Saving metadata to: "
                f"{self.metadata_file_path}"
            )

            with open(
                self.metadata_file_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    self.chunk_metadata,
                    f,
                    indent=2
                )

        else:

            logger.warning(
                "No metadata to save."
            )

    # ========================================================
    # LOAD INDEX
    # ========================================================

    def load_index(self) -> bool:

        if (
            not self.index_file_path.exists()
            or not self.metadata_file_path.exists()
        ):

            logger.warning(
                f"Index files not found in "
                f"{self.index_dir}."
            )

            return False

        try:

            logger.info(
                f"Loading FAISS index from: "
                f"{self.index_file_path}"
            )

            self.index = faiss.read_index(
                str(self.index_file_path)
            )

            logger.info(
                f"FAISS index loaded. "
                f"Vectors: {self.index.ntotal}, "
                f"Dimensions: {self.index.d}"
            )

            if self.index.d != self.embedding_dim:

                logger.warning(
                    f"Loaded index dimension "
                    f"({self.index.d}) differs from "
                    f"configured dimension "
                    f"({self.embedding_dim})."
                )

            logger.info(
                f"Loading metadata from: "
                f"{self.metadata_file_path}"
            )

            with open(
                self.metadata_file_path,
                "r",
                encoding="utf-8"
            ) as f:

                self.chunk_metadata = json.load(f)

            logger.info(
                f"Metadata loaded for "
                f"{len(self.chunk_metadata)} chunks."
            )

            # Validate vector/metadata count

            if (
                self.index.ntotal
                != len(self.chunk_metadata)
            ):

                logger.warning(
                    f"Vector count "
                    f"({self.index.ntotal}) does not "
                    f"match metadata count "
                    f"({len(self.chunk_metadata)})."
                )

            return True

        except Exception as e:

            logger.error(
                f"Error loading FAISS index "
                f"or metadata: {e}"
            )

            self.index = None

            self.chunk_metadata = []

            return False

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query_text: str,
        top_k: int = config.RETRIEVAL_VECTOR_TOP_K,
        model: str = config.EMBEDDING_MODEL_NAME,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        dimensions: int = config.EMBEDDING_DIMENSIONS,
        apikey: Optional[str] = None
    ) -> List[
        Tuple[float, Dict[str, Any]]
    ]:

        # ----------------------------------------------------
        # Make sure index is loaded
        # ----------------------------------------------------

        if self.index is None:

            logger.info(
                "FAISS index not loaded. "
                "Attempting to load..."
            )

            if not self.load_index():

                logger.warning(
                    "Could not load FAISS index."
                )

                return []

        # ----------------------------------------------------
        # Generate query embedding
        # ----------------------------------------------------

        logger.debug(
            f"Searching for query: "
            f"'{query_text[:80]}...'"
        )

        query_embedding = generate_embeddings(
            texts=[query_text],
            model_name=model,
            batch_size=batch_size,
            dimensions=dimensions,
            apikey=apikey,
            max_workers=1
        )

        if (
            query_embedding is None
            or query_embedding.shape[0] == 0
        ):

            logger.error(
                "Failed to generate query embedding."
            )

            return []

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if query_embedding.shape[1] != self.index.d:

            logger.error(
                f"Query embedding dimension "
                f"({query_embedding.shape[1]}) does not "
                f"match FAISS index dimension "
                f"({self.index.d})."
            )

            return []

        # ----------------------------------------------------
        # Limit top_k
        # ----------------------------------------------------

        actual_top_k = min(
            top_k,
            self.index.ntotal
        )

        if actual_top_k <= 0:

            return []

        # ----------------------------------------------------
        # Search FAISS
        # ----------------------------------------------------

        distances, indices = self.index.search(
            query_embedding,
            actual_top_k
        )

        results = []

        for i in range(
            indices.shape[1]
        ):

            vector_id = int(
                indices[0, i]
            )

            distance = float(
                distances[0, i]
            )

            if (
                vector_id < 0
                or vector_id
                >= len(self.chunk_metadata)
            ):

                logger.warning(
                    f"Invalid vector ID "
                    f"{vector_id}. Skipping."
                )

                continue

            metadata = (
                self.chunk_metadata[
                    vector_id
                ]
            )

            results.append(
                (
                    distance,
                    metadata
                )
            )

        logger.debug(
            f"Search returned "
            f"{len(results)} results."
        )

        return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    logger.remove()

    logger.add(
        lambda msg: print(
            msg,
            end=""
        ),
        level="INFO"
    )

    logger.info(
        "========================================"
    )

    logger.info(
        "Testing Ollama + FAISS Vector Index"
    )

    logger.info(
        "========================================"
    )

    # --------------------------------------------------------
    # Test Ollama connection
    # --------------------------------------------------------

    try:

        response = requests.get(
            f"{config.OLLAMA_BASE_URL}/api/tags",
            timeout=10
        )

        response.raise_for_status()

        models = response.json().get(
            "models",
            []
        )

        model_names = [
            model.get("name")
            for model in models
        ]

        logger.info(
            f"Ollama models available: "
            f"{model_names}"
        )

    except Exception as e:

        logger.error(
            f"Could not connect to Ollama: {e}"
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Create dummy chunks
    # --------------------------------------------------------

    dummy_chunks_data = [

        {
            "file_path": Path(
                "test.py_chunk_1"
            ),

            "content":
                "def hello(): "
                "return 'world'",

            "language":
                "python",

            "original_file_path":
                Path("test.py"),

            "chunk_id":
                1,

            "size_bytes":
                30,

            "absolute_path":
                Path("/abs/test.py")
        },

        {
            "file_path": Path(
                "test.py_chunk_2"
            ),

            "content":
                "class Greeter: "
                "def greet(self): "
                "print('Hello')",

            "language":
                "python",

            "original_file_path":
                Path("test.py"),

            "chunk_id":
                2,

            "size_bytes":
                50,

            "absolute_path":
                Path("/abs/test.py")
        },

        {
            "file_path": Path(
                "other.js_chunk_1"
            ),

            "content":
                "function test() "
                "{ return 1+1; }",

            "language":
                "javascript",

            "original_file_path":
                Path("other.js"),

            "chunk_id":
                1,

            "size_bytes":
                40,

            "absolute_path":
                Path("/abs/other.js")
        }
    ]

    test_chunks = [
        DocumentChunk(**data)
        for data in dummy_chunks_data
    ]

    # --------------------------------------------------------
    # Initialize index
    # --------------------------------------------------------

    test_index_dir = (
        config.INDEX_DIR
        / "test_vector_index"
    )

    vector_indexer = FaissVectorIndex(
        index_dir=test_index_dir,
        embedding_dim=
            config.EMBEDDING_DIMENSIONS
    )

    # --------------------------------------------------------
    # Build index
    # --------------------------------------------------------

    logger.info(
        "\n--- Building Vector Index ---"
    )

    build_success = (
        vector_indexer.build_index(
            test_chunks,
            force_rebuild=True
        )
    )

    if not build_success:

        logger.error(
            "Failed to build vector index."
        )

        raise SystemExit(1)

    logger.info(
        "Index built successfully."
    )

    # --------------------------------------------------------
    # Load index
    # --------------------------------------------------------

    logger.info(
        "\n--- Testing Index Loading ---"
    )

    loaded_vector_indexer = (
        FaissVectorIndex(
            index_dir=test_index_dir,
            embedding_dim=
                config.EMBEDDING_DIMENSIONS
        )
    )

    load_success = (
        loaded_vector_indexer.load_index()
    )

    if not load_success:

        logger.error(
            "Failed to load vector index."
        )

        raise SystemExit(1)

    logger.info(
        "Index loaded successfully."
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    logger.info(
        "\n--- Testing Search ---"
    )

    query = (
        "python hello world function"
    )

    search_results = (
        loaded_vector_indexer.search(
            query,
            top_k=2
        )
    )

    logger.info(
        f"Search results for: "
        f"'{query}'"
    )

    for score, meta in search_results:

        logger.info(
            f"  Distance: {score:.4f}"
        )

        logger.info(
            f"  File: "
            f"{meta['original_file_path']}"
        )

        logger.info(
            f"  Chunk ID: "
            f"{meta['chunk_id']}"
        )

    if search_results:

        logger.info(
            "Basic vector search "
            "assertion passed."
        )

    else:

        logger.error(
            "No search results returned."
        )

        raise SystemExit(1)

    logger.info(
        "========================================"
    )

    logger.info(
        "OLLAMA + FAISS TEST PASSED"
    )

    logger.info(
        "========================================"
    )