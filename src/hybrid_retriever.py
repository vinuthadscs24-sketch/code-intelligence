from typing import List, Dict, Any

from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph


WEB_ANNOTATIONS = {
    "RestController",
    "Controller",
    "RequestMapping",
    "GetMapping",
    "PostMapping",
    "PutMapping",
    "DeleteMapping",
    "PatchMapping",
    "WebServlet",
}


class HybridRetriever:
    """
    Combines semantic vector retrieval and structural graph retrieval.

    Pipeline:

        Query
          |
          +------------------+
          |                  |
          v                  v
      Vector Search      Graph Search
          |                  |
          +--------+---------+
                   |
             Weighted RRF
                   |
            Endpoint Boost
                   |
            File Diversity
                   |
                Top-K
    """

    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: CodeKnowledgeGraph,
        rrf_k: int = 60,
        max_per_file: int = 2,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
    ):
        self.vector_store = vector_store
        self.kg = knowledge_graph

        self.rrf_k = rrf_k
        self.max_per_file = max_per_file

        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

        # Prevent accidental invalid configurations.
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0.")

        if self.max_per_file <= 0:
            raise ValueError("max_per_file must be greater than 0.")

        if self.vector_weight < 0 or self.graph_weight < 0:
            raise ValueError("Retrieval weights cannot be negative.")

        if self.vector_weight == 0 and self.graph_weight == 0:
            raise ValueError(
                "At least one of vector_weight or graph_weight must be greater than 0."
            )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _get_chunk_id(self, chunk: Dict[str, Any]) -> str:
        """
        Returns a stable identifier for a retrieved chunk.

        Priority:
            1. chunk_id
            2. id
            3. derived identifier
        """

        if chunk.get("chunk_id"):
            return str(chunk["chunk_id"])

        if chunk.get("id"):
            return str(chunk["id"])

        file_name = chunk.get(
            "file_name",
            chunk.get("file", chunk.get("file_path", "unknown")),
        )

        class_name = chunk.get(
            "class_name",
            chunk.get("class", ""),
        )

        method_name = chunk.get(
            "method_name",
            chunk.get("method", ""),
        )

        start_line = chunk.get("start_line", 0)

        return f"{file_name}::{class_name}::{method_name}:{start_line}"

    def _get_file_key(self, chunk: Dict[str, Any]) -> str:
        """
        Returns the file identifier used for diversity control.
        """

        return str(
            chunk.get("file_name")
            or chunk.get("file_path")
            or chunk.get("file")
            or chunk.get("source")
            or "unknown"
        )

    def _clean_annotations(self, annotations: Any) -> set:
        """
        Normalizes annotations.

        Examples:
            @RestController -> RestController
            RestController  -> RestController
        """

        if not annotations:
            return set()

        if isinstance(annotations, str):
            annotations = [annotations]

        return {
            str(annotation)
            .replace("@", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
            for annotation in annotations
        }

    # ------------------------------------------------------------------
    # Graph Retrieval
    # ------------------------------------------------------------------

    def _get_graph_tokens(self, query: str) -> List[str]:
        """
        Extracts simple searchable tokens from the query.

        This is intentionally lightweight for V1.
        A future version can replace this with entity extraction.
        """

        normalized_query = (
            query.replace(".", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace(",", " ")
            .replace(":", " ")
            .replace("/", " ")
            .replace("_", " ")
            .replace("-", " ")
        )

        tokens = [
            token.lower()
            for token in normalized_query.split()
            if len(token) > 2
        ]

        # Remove common natural-language words that don't help
        # identify code entities.
        stop_words = {
            "the",
            "and",
            "how",
            "what",
            "where",
            "which",
            "who",
            "does",
            "with",
            "from",
            "this",
            "that",
            "are",
            "was",
            "were",
            "for",
            "into",
            "about",
            "show",
            "find",
            "tell",
            "give",
            "explain",
            "work",
            "works",
        }

        return [
            token
            for token in tokens
            if token not in stop_words
        ]

    def _independent_graph_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Performs graph retrieval independently of vector retrieval.

        Current V1 strategy:
            Query tokens
                ↓
            Match graph node names
                ↓
            Retrieve graph relationships

        This intentionally does NOT depend on FAISS results.
        """

        if not hasattr(self.kg, "graph") or self.kg.graph is None:
            return []

        tokens = self._get_graph_tokens(query)

        if not tokens:
            return []

        candidates = []

        for node, data in self.kg.graph.nodes(data=True):
            node_string = str(node)
            node_lower = node_string.lower()

            # Calculate a simple lexical match score.
            matched_tokens = [
                token
                for token in tokens
                if token in node_lower
            ]

            if not matched_tokens:
                continue

            # More matched query tokens = stronger graph candidate.
            match_score = len(matched_tokens) / len(tokens)

            callers = []
            callees = []

            if hasattr(self.kg, "get_callers_of"):
                try:
                    callers = self.kg.get_callers_of(node) or []
                except Exception:
                    callers = []

            if hasattr(self.kg, "get_calls_from"):
                try:
                    callees = self.kg.get_calls_from(node) or []
                except Exception:
                    callees = []

            chunk_id = (
                data.get("chunk_id")
                or data.get("id")
                or str(node)
            )

            file_name = (
                data.get("file_name")
                or data.get("file")
                or data.get("file_path")
            )

            if not file_name:
                chunk_id_string = str(chunk_id)

                if "::" in chunk_id_string:
                    file_name = chunk_id_string.split("::")[0]
                else:
                    file_name = "unknown"

            annotations = data.get("annotations", [])

            code_content = (
                data.get("code_content")
                or data.get("code")
                or ""
            )

            candidates.append(
                {
                    "chunk_id": str(chunk_id),
                    "file_name": str(file_name),
                    "method_name": str(
                        data.get("method_name")
                        or node
                    ),
                    "class_name": data.get("class_name", ""),
                    "annotations": annotations,
                    "graph_callers": callers,
                    "graph_callees": callees,
                    "code_content": code_content,

                    # Useful for debugging/evaluation.
                    "graph_match_score": round(match_score, 6),
                    "graph_matched_tokens": matched_tokens,
                }
            )

        # Stronger lexical graph matches first.
        candidates.sort(
            key=lambda item: (
                item.get("graph_match_score", 0),
                len(item.get("graph_callers", []))
                + len(item.get("graph_callees", [])),
            ),
            reverse=True,
        )

        return candidates[:top_k]

    # ------------------------------------------------------------------
    # RRF
    # ------------------------------------------------------------------

    def _rrf_score(
        self,
        rank: int,
        weight: float,
    ) -> float:
        """
        Weighted Reciprocal Rank Fusion contribution.

        Formula:

            weight / (rrf_k + rank)
        """

        return weight / (self.rrf_k + rank)

    # ------------------------------------------------------------------
    # Main Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval.

        Returns ranked code chunks containing:

            combined_score
            sources
            vector_rank
            graph_rank
            graph_callers
            graph_callees
            code metadata
        """

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        # Retrieve extra candidates because:
        #
        #   1. Some results may be duplicates.
        #   2. File diversity may remove results.
        #   3. Graph and vector retrieval may overlap.
        #
        fetch_limit = max(top_k * 4, 10)

        # ==============================================================
        # 1. VECTOR RETRIEVAL
        # ==============================================================

        raw_vector_results = self.vector_store.search(
            query,
            top_k=fetch_limit,
        )

        vector_chunks: List[Dict[str, Any]] = []

        for result in raw_vector_results:
            if isinstance(result, tuple):
                if len(result) > 0 and isinstance(result[0], dict):
                    vector_chunks.append(result[0])

            elif isinstance(result, dict):
                vector_chunks.append(result)

        # ==============================================================
        # 2. INDEPENDENT GRAPH RETRIEVAL
        # ==============================================================

        graph_chunks = self._independent_graph_search(
            query,
            top_k=fetch_limit,
        )

        # ==============================================================
        # 3. QUERY INTENT
        # ==============================================================

        query_lower = query.lower()

        endpoint_keywords = {
            "controller",
            "endpoint",
            "api",
            "route",
            "http",
            "rest",
            "web",
            "mapping",
        }

        is_endpoint_query = any(
            keyword in query_lower
            for keyword in endpoint_keywords
        )

        # ==============================================================
        # 4. WEIGHTED RRF
        # ==============================================================

        rrf_scores: Dict[str, float] = {}

        doc_map: Dict[str, Dict[str, Any]] = {}

        def merge_result(
            chunk: Dict[str, Any],
            rank: int,
            weight: float,
            source_type: str,
        ) -> None:

            chunk_id = self._get_chunk_id(chunk)

            if chunk_id not in doc_map:

                # Copy so we never mutate the original result.
                doc_map[chunk_id] = dict(chunk)

                doc_map[chunk_id]["sources"] = []

                doc_map[chunk_id]["vector_rank"] = None
                doc_map[chunk_id]["graph_rank"] = None

            document = doc_map[chunk_id]

            # ----------------------------------------------------------
            # Source tracking
            # ----------------------------------------------------------

            if source_type not in document["sources"]:
                document["sources"].append(source_type)

            # ----------------------------------------------------------
            # Rank tracking
            # ----------------------------------------------------------

            if source_type == "vector":
                document["vector_rank"] = rank

            elif source_type == "graph":
                document["graph_rank"] = rank

            # ----------------------------------------------------------
            # Merge graph information
            # ----------------------------------------------------------

            for graph_key in (
                "graph_callers",
                "graph_callees",
                "graph_match_score",
                "graph_matched_tokens",
            ):

                if graph_key in chunk and chunk[graph_key]:

                    document[graph_key] = chunk[graph_key]

            # ----------------------------------------------------------
            # Merge useful metadata
            # ----------------------------------------------------------

            for key in (
                "file_name",
                "file_path",
                "class_name",
                "method_name",
                "annotations",
                "code_content",
                "start_line",
                "end_line",
            ):

                if not document.get(key) and chunk.get(key):
                    document[key] = chunk[key]

            # ----------------------------------------------------------
            # RRF contribution
            # ----------------------------------------------------------

            contribution = self._rrf_score(
                rank=rank,
                weight=weight,
            )

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + contribution
            )

        # Vector ranking
        for rank, chunk in enumerate(
            vector_chunks,
            start=1,
        ):
            merge_result(
                chunk=chunk,
                rank=rank,
                weight=self.vector_weight,
                source_type="vector",
            )

        # Graph ranking
        for rank, chunk in enumerate(
            graph_chunks,
            start=1,
        ):
            merge_result(
                chunk=chunk,
                rank=rank,
                weight=self.graph_weight,
                source_type="graph",
            )

        # ==============================================================
        # 5. ENDPOINT BOOST
        # ==============================================================

        if is_endpoint_query:

            for chunk_id in rrf_scores:

                annotations = self._clean_annotations(
                    doc_map[chunk_id].get(
                        "annotations",
                        [],
                    )
                )

                if annotations.intersection(
                    WEB_ANNOTATIONS
                ):
                    rrf_scores[chunk_id] *= 1.5

        # ==============================================================
        # 6. SORT BY FINAL SCORE
        # ==============================================================

        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        # ==============================================================
        # 7. FILE DIVERSITY
        # ==============================================================

        final_results: List[Dict[str, Any]] = []

        file_counts: Dict[str, int] = {}

        deferred_results = []

        for chunk_id, score in sorted_chunks:

            chunk = doc_map.get(chunk_id)

            if not chunk:
                continue

            file_key = self._get_file_key(chunk)

            current_count = file_counts.get(
                file_key,
                0,
            )

            enriched_chunk = dict(chunk)

            enriched_chunk["combined_score"] = round(
                score,
                6,
            )

            enriched_chunk.setdefault(
                "graph_callers",
                [],
            )

            enriched_chunk.setdefault(
                "graph_callees",
                [],
            )

            enriched_chunk.setdefault(
                "sources",
                [],
            )

            # ----------------------------------------------------------
            # Prefer diverse files
            # ----------------------------------------------------------

            if current_count < self.max_per_file:

                file_counts[file_key] = (
                    current_count + 1
                )

                final_results.append(
                    enriched_chunk
                )

            else:

                deferred_results.append(
                    (
                        enriched_chunk,
                        score,
                    )
                )

            if len(final_results) >= top_k:
                break

        # ==============================================================
        # 8. FALLBACK
        # ==============================================================

        if len(final_results) < top_k:

            for chunk, _score in deferred_results:

                if len(final_results) >= top_k:
                    break

                final_results.append(chunk)

        # ==============================================================
        # 9. FINAL RANK METADATA
        # ==============================================================

        for index, result in enumerate(
            final_results,
            start=1,
        ):
            result["final_rank"] = index

        return final_results