
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from loguru import logger

from src import config
from src.indexing.vector_index import FaissVectorIndex
from src.indexing.sparse_index import BM25Index


SearchResultItem = Tuple[float, Dict[str, Any]]


class HybridRetriever:
    """
    Hybrid retrieval using:
    1. FAISS dense retrieval
    2. BM25 sparse retrieval
    3. Reciprocal Rank Fusion (RRF)

    FAISS and BM25 maintain separate metadata files.
    Their metadata counts are allowed to differ because BM25
    can remove chunks that become empty after preprocessing.
    """

    def __init__(
        self,
        index_dir: Path = config.INDEX_DIR,
        indexes: Optional[List[str]] = None,
        vector_top_k: int = config.RETRIEVAL_VECTOR_TOP_K,
        bm25_top_k: int = config.RETRIEVAL_BM25_TOP_K,
        rrf_k_constant: int = config.RRF_CONSTANT_K,
    ):
        self.index_dir = index_dir

        self.indexes = (
            indexes
            if indexes is not None
            else config.RETRIEVAL_INDEXES
        )

        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k_constant = rrf_k_constant

        self.vector_index: Optional[FaissVectorIndex] = None
        self.bm25_index: Optional[BM25Index] = None

        self._load_indexes()

    # ========================================================
    # LOAD INDEXES
    # ========================================================

    def _load_indexes(self) -> bool:
        """Load FAISS and BM25 indexes independently."""

        logger.info(
            f"Initializing HybridRetriever from: {self.index_dir}"
        )

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        if "vector" in self.indexes:

            try:
                self.vector_index = FaissVectorIndex(
                    index_dir=self.index_dir,
                    embedding_dim=config.EMBEDDING_DIMENSIONS,
                )

                if self.vector_index.load_index():

                    logger.info(
                        f"FAISS loaded successfully. "
                        f"Vectors: {self.vector_index.index.ntotal}"
                    )

                else:

                    logger.error(
                        "Failed to load FAISS index."
                    )

                    self.vector_index = None

            except Exception as e:

                logger.error(
                    f"Exception loading FAISS index: {e}"
                )

                self.vector_index = None

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        if "bm25" in self.indexes:

            try:
                self.bm25_index = BM25Index(
                    index_dir=self.index_dir
                )

                if self.bm25_index.load_index():

                    logger.info(
                        f"BM25 loaded successfully. "
                        f"Documents: "
                        f"{len(self.bm25_index.chunk_metadata)}"
                    )

                else:

                    logger.error(
                        "Failed to load BM25 index."
                    )

                    self.bm25_index = None

            except Exception as e:

                logger.error(
                    f"Exception loading BM25 index: {e}"
                )

                self.bm25_index = None

        # ----------------------------------------------------
        # Check availability
        # ----------------------------------------------------

        if (
            self.vector_index is None
            and self.bm25_index is None
        ):

            logger.error(
                "Neither FAISS nor BM25 index is available."
            )

            return False

        if self.vector_index is not None:

            logger.info(
                f"FAISS metadata chunks: "
                f"{len(self.vector_index.chunk_metadata)}"
            )

        if self.bm25_index is not None:

            logger.info(
                f"BM25 metadata chunks: "
                f"{len(self.bm25_index.chunk_metadata)}"
            )

        # Do NOT compare FAISS and BM25 metadata counts.
        #
        # FAISS keeps all chunks.
        # BM25 removes chunks that become empty after preprocessing.

        return True

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query_text: str,
        apikey: Optional[str] = None,
        top_n_final: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval.

        Query
          |
          +----> FAISS
          |
          +----> BM25
          |
          v
         RRF
          |
          v
      Final ranked chunks
        """

        if not query_text or not query_text.strip():

            logger.warning(
                "Empty retrieval query."
            )

            return []

        if (
            self.vector_index is None
            and self.bm25_index is None
        ):

            logger.error(
                "No retrieval indexes are available."
            )

            return []

        # ----------------------------------------------------
        # Determine K values
        # ----------------------------------------------------

        vector_k = (
            vector_top_k
            if vector_top_k is not None
            else self.vector_top_k
        )

        bm25_k = (
            bm25_top_k
            if bm25_top_k is not None
            else self.bm25_top_k
        )

        if top_n_final is None:

            top_n_final = max(
                vector_k,
                bm25_k,
                5
            )

        logger.info(
            f"Hybrid retrieval for query: "
            f"'{query_text[:100]}'"
        )

        # ====================================================
        # FAISS SEARCH
        # ====================================================

        vector_results: List[SearchResultItem] = []

        if self.vector_index is not None:

            try:

                vector_results = self.vector_index.search(
                    query_text=query_text,
                    top_k=vector_k,
                    apikey=apikey,
                )

                logger.debug(
                    f"FAISS returned "
                    f"{len(vector_results)} results."
                )

            except Exception as e:

                logger.error(
                    f"FAISS retrieval failed: {e}"
                )

        # ====================================================
        # BM25 SEARCH
        # ====================================================

        bm25_results: List[SearchResultItem] = []

        if self.bm25_index is not None:

            try:

                bm25_results = self.bm25_index.search(
                    query_text=query_text,
                    top_k=bm25_k,
                )

                logger.debug(
                    f"BM25 returned "
                    f"{len(bm25_results)} results."
                )

            except Exception as e:

                logger.error(
                    f"BM25 retrieval failed: {e}"
                )

        # ====================================================
        # NO RESULTS
        # ====================================================

        if not vector_results and not bm25_results:

            logger.info(
                "Neither FAISS nor BM25 returned results."
            )

            return []

        # ====================================================
        # PREPARE RRF INPUT
        # ====================================================

        result_lists = []

        if vector_results:

            result_lists.append(
                ("vector", vector_results)
            )

        if bm25_results:

            result_lists.append(
                ("bm25", bm25_results)
            )

        # ====================================================
        # RRF
        # ====================================================

        fused_results = self._reciprocal_rank_fusion(
            result_lists
        )

        # ====================================================
        # FINAL RESULTS
        # ====================================================

        fused_results.sort(
            key=lambda item: item[0],
            reverse=True
        )

        final_results = [
            metadata
            for _, metadata in fused_results[:top_n_final]
        ]

        logger.info(
            f"Hybrid retrieval returned "
            f"{len(final_results)} results."
        )

        return final_results

    # ========================================================
    # RECIPROCAL RANK FUSION
    # ========================================================

    def _reciprocal_rank_fusion(
        self,
        search_results_lists: List[
            Tuple[str, List[SearchResultItem]]
        ],
    ) -> List[SearchResultItem]:
        """
        Combine FAISS and BM25 rankings using RRF.

        RRF formula:

            RRF(d) = sum(1 / (k + rank))

        Raw FAISS distances and BM25 scores are NOT combined
        directly. Only their rankings are used.
        """

        if not search_results_lists:

            return []

        fused_scores: Dict[
            Tuple[str, int],
            float
        ] = {}

        metadata_cache: Dict[
            Tuple[str, int],
            Dict[str, Any]
        ] = {}

        # ====================================================
        # PROCESS EACH RETRIEVER
        # ====================================================

        for retriever_name, results in search_results_lists:

            if not results:
                continue

            # ------------------------------------------------
            # FAISS
            # ------------------------------------------------
            #
            # IndexFlatL2:
            # lower distance = better

            if retriever_name == "vector":

                ranked_results = sorted(
                    results,
                    key=lambda item: item[0]
                )

            # ------------------------------------------------
            # BM25
            # ------------------------------------------------
            #
            # higher score = better

            else:

                ranked_results = sorted(
                    results,
                    key=lambda item: item[0],
                    reverse=True
                )

            # =================================================
            # CALCULATE RRF
            # =================================================

            for rank, (_, metadata) in enumerate(
                ranked_results,
                start=1
            ):

                doc_id = self._get_document_id(
                    metadata
                )

                rrf_score = (
                    1.0
                    /
                    (
                        self.rrf_k_constant
                        + rank
                    )
                )

                fused_scores[doc_id] = (
                    fused_scores.get(
                        doc_id,
                        0.0
                    )
                    + rrf_score
                )

                if doc_id not in metadata_cache:

                    metadata_cache[
                        doc_id
                    ] = metadata

        # ====================================================
        # BUILD FUSED RESULT LIST
        # ====================================================

        fused_results = [

            (
                score,
                metadata_cache[doc_id]
            )

            for doc_id, score
            in fused_scores.items()

            if doc_id in metadata_cache
        ]

        fused_results.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return fused_results

    # ========================================================
    # DOCUMENT ID
    # ========================================================

    @staticmethod
    def _get_document_id(
        metadata: Dict[str, Any]
    ) -> Tuple[str, int]:
        """
        Generate a stable identifier for a code chunk.

        original_file_path + chunk_id
        """

        file_path = str(
            metadata.get(
                "original_file_path",
                ""
            )
        )

        chunk_id = metadata.get(
            "chunk_id",
            -1
        )

        try:

            chunk_id = int(chunk_id)

        except (TypeError, ValueError):

            chunk_id = -1

        return (
            file_path,
            chunk_id
        )


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

    logger.info(
        "========================================"
    )

    logger.info(
        "       TESTING HYBRID RETRIEVER"
    )

    logger.info(
        "========================================"
    )

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    retriever = HybridRetriever(
        index_dir=config.INDEX_DIR
    )

    # --------------------------------------------------------
    # Check indexes
    # --------------------------------------------------------

    if (
        retriever.vector_index is None
        and retriever.bm25_index is None
    ):

        logger.error(
            "No indexes available."
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Test queries
    # --------------------------------------------------------

    queries = [
        "python hello function",
        "javascript test function",
        "Greeter class",
    ]

    for query in queries:

        logger.info(
            f"\n--- Query: {query} ---"
        )

        results = retriever.retrieve(
            query_text=query,
            top_n_final=5
        )

        if not results:

            logger.info(
                "No results found."
            )

            continue

        for rank, metadata in enumerate(
            results,
            start=1
        ):

            logger.info(
                f"Rank {rank}: "
                f"file={metadata.get('original_file_path')}, "
                f"chunk_id={metadata.get('chunk_id')}"
            )

