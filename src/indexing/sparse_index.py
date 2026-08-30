import pickle
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from src import config
from src.data_processing.chunkers import DocumentChunk
from src.utils.nlp_utils import NLPUtils as nlp


# ============================================================
# BM25 PREPROCESSING
# ============================================================

def preprocess_text_for_bm25(text: str) -> List[str]:
    """
    Preprocess text for BM25 retrieval.

    Steps:
    - Remove extra spaces
    - Stem text
    - Tokenize
    - Keep word tokens
    - Remove unwanted words
    - Remove duplicate words
    """

    if not text:
        return []

    # Remove extra spaces
    text = nlp.removeExtraSpaces(text)

    # Stem text
    text = nlp.stem(text)

    # Tokenize
    tokens = nlp.tokenize(text, True)

    # Keep only word tokens
    word_tokens = [
        token
        for token in tokens
        if token["type"] == "word"
    ]

    # Remove stopwords/unwanted words
    cleaned_tokens = nlp.removeWords(word_tokens)

    # Remove duplicate words
    unique_tokens = nlp.setOfWords(cleaned_tokens)

    # Extract token values
    return [
        str(token["value"])
        for token in unique_tokens
    ]


# ============================================================
# BM25 INDEX
# ============================================================

class BM25Index:
    """
    Manages the BM25 sparse retrieval index.
    """

    def __init__(self, index_dir: Path):

        self.index_dir = index_dir

        self.bm25_model_file_path = (
            self.index_dir /
            config.BM25_INDEX_FILENAME
        )

        self.metadata_file_path = (
            self.index_dir /
            config.BM25_METADATA_FILENAME
        )

        self.bm25: Optional[BM25Okapi] = None

        self.chunk_corpus_tokenized: List[List[str]] = []

        self.chunk_metadata: List[Dict[str, Any]] = []

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
        force_rebuild: bool = False
    ) -> bool:

        if (
            not force_rebuild
            and self.bm25_model_file_path.exists()
            and self.metadata_file_path.exists()
        ):
            logger.info(
                f"BM25 index already exists at "
                f"{self.index_dir}. "
                f"Skipping build."
            )
            return True

        if not chunks:
            logger.warning(
                "No chunks provided to build BM25 index."
            )
            return False

        logger.info(
            f"Building BM25 index with "
            f"{len(chunks)} chunks..."
        )

        # ----------------------------------------------------
        # Tokenize chunks
        # ----------------------------------------------------

        all_tokenized_chunks = [
            preprocess_text_for_bm25(
                chunk.content
            )
            for chunk in chunks
        ]

        # ----------------------------------------------------
        # Remove empty documents
        # ----------------------------------------------------

        valid_indices = [
            i
            for i, tokens in enumerate(
                all_tokenized_chunks
            )
            if tokens
        ]

        if not valid_indices:
            logger.warning(
                "All chunks became empty after preprocessing."
            )
            return False

        self.chunk_corpus_tokenized = [
            all_tokenized_chunks[i]
            for i in valid_indices
        ]

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        self.chunk_metadata = []

        for original_index in valid_indices:

            chunk = chunks[original_index]

            meta_item = chunk.model_dump(
                exclude={"absolute_path"}
            )

            meta_item["original_file_path"] = str(
                chunk.original_file_path
            )

            meta_item["file_path"] = str(
                chunk.file_path
            )

            meta_item["bm25_doc_id"] = len(
                self.chunk_metadata
            )

            self.chunk_metadata.append(
                meta_item
            )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if len(self.chunk_corpus_tokenized) != len(
            self.chunk_metadata
        ):
            logger.error(
                f"BM25 corpus count "
                f"({len(self.chunk_corpus_tokenized)}) "
                f"does not match metadata count "
                f"({len(self.chunk_metadata)})."
            )
            return False

        # ----------------------------------------------------
        # Build BM25
        # ----------------------------------------------------

        try:

            self.bm25 = BM25Okapi(
                self.chunk_corpus_tokenized
            )

            logger.info(
                f"BM25 index built successfully. "
                f"Indexed "
                f"{len(self.chunk_corpus_tokenized)} "
                f"documents."
            )

        except Exception as e:

            logger.error(
                f"Failed to initialize BM25Okapi: {e}"
            )

            self.bm25 = None
            return False

        self.save_index()

        return True

    # ========================================================
    # SAVE INDEX
    # ========================================================

    def save_index(self):

        if self.bm25 is not None:

            logger.info(
                f"Saving BM25 model to: "
                f"{self.bm25_model_file_path}"
            )

            with open(
                self.bm25_model_file_path,
                "wb"
            ) as f:

                pickle.dump(
                    {
                        "bm25_model": self.bm25,
                        "corpus_tokenized_checksum":
                            len(
                                self.chunk_corpus_tokenized
                            )
                    },
                    f
                )

        if self.chunk_metadata:

            logger.info(
                f"Saving BM25 metadata to: "
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

    # ========================================================
    # LOAD INDEX
    # ========================================================

    def load_index(self) -> bool:

        if (
            not self.bm25_model_file_path.exists()
            or not self.metadata_file_path.exists()
        ):
            logger.warning(
                f"BM25 index or metadata not found "
                f"in {self.index_dir}."
            )
            return False

        try:

            logger.info(
                f"Loading BM25 model from: "
                f"{self.bm25_model_file_path}"
            )

            with open(
                self.bm25_model_file_path,
                "rb"
            ) as f:

                saved_data = pickle.load(f)

            self.bm25 = saved_data.get(
                "bm25_model"
            )

            if self.bm25 is None:
                logger.error(
                    "Saved BM25 file does not contain "
                    "a valid BM25 model."
                )
                return False

            logger.info(
                f"Loading BM25 metadata from: "
                f"{self.metadata_file_path}"
            )

            with open(
                self.metadata_file_path,
                "r",
                encoding="utf-8"
            ) as f:

                self.chunk_metadata = json.load(f)

            if not isinstance(
                self.chunk_metadata,
                list
            ):
                logger.error(
                    "BM25 metadata is not a list."
                )

                self.chunk_metadata = []
                return False

            expected_count = saved_data.get(
                "corpus_tokenized_checksum"
            )

            actual_count = len(
                self.chunk_metadata
            )

            if (
                expected_count is not None
                and expected_count != actual_count
            ):
                logger.error(
                    f"BM25 model document count "
                    f"({expected_count}) does not "
                    f"match metadata count "
                    f"({actual_count})."
                )

                self.bm25 = None
                self.chunk_metadata = []

                return False

            logger.info(
                f"BM25 index and metadata loaded. "
                f"Model ready for "
                f"{len(self.chunk_metadata)} documents."
            )

            return True

        except Exception as e:

            logger.error(
                f"Error loading BM25 index: {e}"
            )

            self.bm25 = None
            self.chunk_metadata = []

            return False

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query_text: str,
        top_k: int = config.RETRIEVAL_BM25_TOP_K
    ) -> List[
        Tuple[
            float,
            Dict[str, Any]
        ]
    ]:

        if not query_text or not query_text.strip():

            logger.warning(
                "Empty BM25 search query."
            )

            return []

        # Load index if necessary
        if self.bm25 is None:

            if not self.load_index():

                logger.error(
                    "Could not load BM25 index."
                )

                return []

        if self.bm25 is None:
            return []

        if not self.chunk_metadata:
            logger.error(
                "BM25 metadata is empty."
            )
            return []

        if top_k <= 0:
            return []

        # ----------------------------------------------------
        # Preprocess query
        # ----------------------------------------------------

        tokenized_query = (
            preprocess_text_for_bm25(
                query_text
            )
        )

        if not tokenized_query:
            return []

        # ----------------------------------------------------
        # Calculate scores
        # ----------------------------------------------------

        try:

            doc_scores = self.bm25.get_scores(
                tokenized_query
            )

        except Exception as e:

            logger.error(
                f"Error getting BM25 scores: {e}"
            )

            return []

        # ----------------------------------------------------
        # Top K
        # ----------------------------------------------------

        actual_top_k = min(
            top_k,
            len(doc_scores),
            len(self.chunk_metadata)
        )

        if actual_top_k <= 0:
            return []

        top_n_indices = (
            np.argsort(doc_scores)[::-1]
            [:actual_top_k]
        )

        # ----------------------------------------------------
        # Build results
        # ----------------------------------------------------

        results = []

        for index in top_n_indices:

            index = int(index)

            if (
                index < 0
                or index >= len(self.chunk_metadata)
            ):
                continue

            score = float(
                doc_scores[index]
            )

            metadata = (
                self.chunk_metadata[index]
            )

            results.append(
                (
                    score,
                    metadata
                )
            )

        return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    logger.remove()

    logger.add(
        lambda message: print(
            message,
            end=""
        ),
        level="INFO"
    )

    # --------------------------------------------------------
    # Dummy chunks
    # --------------------------------------------------------

    dummy_chunks_data = [

        {
            "file_path": Path(
                "test.py_chunk_1"
            ),

            "content":
                "def hello(): "
                "return 'world'",

            "language": "python",

            "original_file_path":
                Path("test.py"),

            "chunk_id": 1,

            "size_bytes": 30,

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
                "print('Python Greeter "
                "class hello')",

            "language": "python",

            "original_file_path":
                Path("test.py"),

            "chunk_id": 2,

            "size_bytes": 50,

            "absolute_path":
                Path("/abs/test.py")
        },

        {
            "file_path": Path(
                "other.js_chunk_1"
            ),

            "content":
                "function test() "
                "{ return 1+1; } "
                "// javascript test "
                "function",

            "language": "javascript",

            "original_file_path":
                Path("other.js"),

            "chunk_id": 1,

            "size_bytes": 40,

            "absolute_path":
                Path("/abs/other.js")
        },

        {
            "file_path": Path(
                "empty_content.txt_chunk_1"
            ),

            "content": "",

            "language": "text",

            "original_file_path":
                Path("empty_content.txt"),

            "chunk_id": 1,

            "size_bytes": 0,

            "absolute_path":
                Path("/abs/empty_content.txt")
        }
    ]

    # --------------------------------------------------------
    # Create DocumentChunk objects
    # --------------------------------------------------------

    test_chunks = [
        DocumentChunk(**data)
        for data in dummy_chunks_data
    ]

    # --------------------------------------------------------
    # Test directory
    # --------------------------------------------------------

    test_bm25_index_dir = (
        config.INDEX_DIR /
        "test_bm25_index"
    )

    bm25_indexer = BM25Index(
        index_dir=test_bm25_index_dir
    )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    logger.info(
        "\n--- Building BM25 Index ---"
    )

    build_success = (
        bm25_indexer.build_index(
            test_chunks,
            force_rebuild=True
        )
    )

    if not build_success:

        logger.error(
            "Failed to build BM25 index."
        )

        raise SystemExit(1)

    logger.info(
        "BM25 Index built successfully."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    logger.info(
        "\n--- Testing BM25 Index Loading ---"
    )

    loaded_bm25_indexer = BM25Index(
        index_dir=test_bm25_index_dir
    )

    load_success = (
        loaded_bm25_indexer.load_index()
    )

    if not load_success:

        logger.error(
            "Failed to load BM25 index."
        )

        raise SystemExit(1)

    logger.info(
        "BM25 Index loaded successfully."
    )

    # --------------------------------------------------------
    # Search tests
    # --------------------------------------------------------

    logger.info(
        "\n--- Testing BM25 Search ---"
    )

    queries = [
        "python hello function",
        "javascript test",
        "empty",
        "Greeter class"
    ]

    for query in queries:

        search_results = (
            loaded_bm25_indexer.search(
                query,
                top_k=2
            )
        )

        logger.info(
            f"Search results for '{query}':"
        )

        if not search_results:

            logger.info(
                "  No results found."
            )

        for score, metadata in search_results:

            logger.info(
                f"  Score (BM25): "
                f"{score:.4f}, "
                f"Chunk Original Path: "
                f"{metadata['original_file_path']}, "
                f"Chunk ID: "
                f"{metadata['chunk_id']}"
            )

        logger.info("---")

    # --------------------------------------------------------
    # Greeter assertion
    # --------------------------------------------------------

    results_for_greeter = (
        loaded_bm25_indexer.search(
            "Greeter class",
            top_k=1
        )
    )

    if results_for_greeter:

        top_result = results_for_greeter[0][1]

        expected_path = "test.py_chunk_2"

        actual_path = str(
            top_result["file_path"]
        )

        if expected_path in actual_path:

            logger.info(
                "Basic BM25 search assertion passed."
            )

        else:

            logger.warning(
                "BM25 search returned a different "
                "top result for 'Greeter class'."
            )

            logger.warning(
                f"Expected: {expected_path}"
            )

            logger.warning(
                f"Got: {actual_path}"
            )

    else:

        logger.error(
            "BM25 search for "
            "'Greeter class' returned no results."
        )