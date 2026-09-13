
from typing import List, Dict, Any
import re

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
    Hybrid code retrieval using:

        Vector Search
             +
        Graph Search
             +
        Semantic Matching
             +
        Weighted RRF
             +
        Code Intent Matching
             +
        Endpoint Boost
             +
        File Diversity
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

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0.")

        if self.max_per_file <= 0:
            raise ValueError(
                "max_per_file must be greater than 0."
            )

        if self.vector_weight < 0 or self.graph_weight < 0:
            raise ValueError(
                "Retrieval weights cannot be negative."
            )

        if self.vector_weight == 0 and self.graph_weight == 0:
            raise ValueError(
                "At least one of vector_weight or graph_weight "
                "must be greater than 0."
            )

    # ==============================================================
    # UTILITY
    # ==============================================================

    def _get_chunk_id(
        self,
        chunk: Dict[str, Any],
    ) -> str:

        if chunk.get("chunk_id"):
            return str(chunk["chunk_id"])

        if chunk.get("id"):
            return str(chunk["id"])

        file_name = chunk.get(
            "file_name",
            chunk.get(
                "file",
                chunk.get(
                    "file_path",
                    "unknown",
                ),
            ),
        )

        class_name = chunk.get(
            "class_name",
            chunk.get("class", ""),
        )

        method_name = chunk.get(
            "method_name",
            chunk.get("method", ""),
        )

        start_line = chunk.get(
            "start_line",
            0,
        )

        return (
            f"{file_name}::"
            f"{class_name}::"
            f"{method_name}:"
            f"{start_line}"
        )

    def _get_file_key(
        self,
        chunk: Dict[str, Any],
    ) -> str:

        return str(
            chunk.get("file_name")
            or chunk.get("file_path")
            or chunk.get("file")
            or chunk.get("source")
            or "unknown"
        )

    def _clean_annotations(
        self,
        annotations: Any,
    ) -> set:

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

    # ==============================================================
    # QUERY TOKENIZATION
    # ==============================================================

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:

        text = re.sub(
            r"[^a-zA-Z0-9_]",
            " ",
            text,
        )

        return [
            token.lower()
            for token in text.split()
            if len(token) > 2
        ]

    def _get_graph_tokens(
        self,
        query: str,
    ) -> List[str]:

        tokens = self._tokenize(query)

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
            "can",
            "all",
            "records",
            "data",
            "code",
            "using",
            "use",
        }

        return [
            token
            for token in tokens
            if token not in stop_words
        ]

    # ==============================================================
    # QUERY INTENT
    # ==============================================================

    def _query_intent_tokens(
        self,
        query: str,
    ) -> List[str]:

        tokens = self._tokenize(query)

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
            "can",
            "all",
            "is",
            "to",
            "of",
            "a",
            "an",
        }

        return [
            token
            for token in tokens
            if token not in stop_words
        ]

    # ==============================================================
    # CODE INTENT
    # ==============================================================

    def _detect_code_intent(
        self,
        query: str,
    ) -> str | None:

        query_lower = query.lower()

        intent_aliases = {
            "search": [
                "find",
                "search",
                "retrieve",
                "matching",
                "satisfy a condition",
                "find records",
                "find documents",
                "search records",
                "search documents",
            ],
            "insert": [
                "insert",
                "add",
                "store",
                "create record",
                "add record",
            ],
            "remove": [
                "delete",
                "remove",
                "erase",
            ],
            "update": [
                "update",
                "modify",
                "change",
            ],
            "count": [
                "count",
                "how many",
                "number of",
            ],
            "contains": [
                "contains",
                "exists",
                "check whether",
            ],
        }

        for intent, aliases in intent_aliases.items():
            if any(
                alias in query_lower
                for alias in aliases
            ):
                return intent

        return None

    # ==============================================================
    # SEMANTIC / TEXT MATCHING
    # ==============================================================

    def _semantic_score(
        self,
        query: str,
        chunk: Dict[str, Any],
    ) -> float:

        query_tokens = set(
            self._query_intent_tokens(query)
        )

        if not query_tokens:
            return 0.0

        method_name = str(
            chunk.get("method_name")
            or ""
        ).lower()

        class_name = str(
            chunk.get("class_name")
            or ""
        ).lower()

        code = str(
            chunk.get("code_content")
            or chunk.get("text_representation")
            or ""
        ).lower()

        file_name = str(
            chunk.get("file_name")
            or ""
        ).lower()

        method_tokens = set(
            self._tokenize(method_name)
        )

        class_tokens = set(
            self._tokenize(class_name)
        )

        code_tokens = set(
            self._tokenize(code)
        )

        file_tokens = set(
            self._tokenize(file_name)
        )

        score = 0.0

        # ----------------------------------------------------------
        # Method name
        # ----------------------------------------------------------

        method_matches = query_tokens.intersection(
            method_tokens
        )

        score += len(method_matches) * 0.30

        # ----------------------------------------------------------
        # Class name
        # ----------------------------------------------------------

        class_matches = query_tokens.intersection(
            class_tokens
        )

        score += len(class_matches) * 0.15

        # ----------------------------------------------------------
        # Code content
        # ----------------------------------------------------------

        code_matches = query_tokens.intersection(
            code_tokens
        )

        score += len(code_matches) * 0.05

        # ----------------------------------------------------------
        # File name
        # ----------------------------------------------------------

        file_matches = query_tokens.intersection(
            file_tokens
        )

        score += len(file_matches) * 0.05

        # ----------------------------------------------------------
        # CODE INTENT BOOST
        # ----------------------------------------------------------

        intent = self._detect_code_intent(query)

        if intent == method_name:
            score += 0.50

        return min(score, 1.0)

    # ==============================================================
    # GRAPH RETRIEVAL
    # ==============================================================

    def _independent_graph_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:

        if (
            not hasattr(self.kg, "graph")
            or self.kg.graph is None
        ):
            return []

        tokens = self._get_graph_tokens(query)

        if not tokens:
            return []

        candidates = []

        for node, data in self.kg.graph.nodes(
            data=True
        ):

            node_string = str(node)
            node_lower = node_string.lower()

            matched_tokens = [
                token
                for token in tokens
                if token in node_lower
            ]

            if not matched_tokens:
                continue

            match_score = (
                len(matched_tokens)
                / len(tokens)
            )

            callers = []
            callees = []

            if hasattr(
                self.kg,
                "get_callers_of",
            ):
                try:
                    callers = (
                        self.kg.get_callers_of(node)
                        or []
                    )
                except Exception:
                    callers = []

            if hasattr(
                self.kg,
                "get_calls_from",
            ):
                try:
                    callees = (
                        self.kg.get_calls_from(node)
                        or []
                    )
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
                    file_name = (
                        chunk_id_string.split(
                            "::"
                        )[0]
                    )
                else:
                    file_name = "unknown"

            annotations = data.get(
                "annotations",
                [],
            )

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
                    "class_name": data.get(
                        "class_name",
                        "",
                    ),
                    "annotations": annotations,
                    "graph_callers": callers,
                    "graph_callees": callees,
                    "code_content": code_content,
                    "graph_match_score": round(
                        match_score,
                        6,
                    ),
                    "graph_matched_tokens": (
                        matched_tokens
                    ),
                }
            )

        candidates.sort(
            key=lambda item: (
                item.get(
                    "graph_match_score",
                    0,
                ),
                len(
                    item.get(
                        "graph_callers",
                        [],
                    )
                )
                + len(
                    item.get(
                        "graph_callees",
                        [],
                    )
                ),
            ),
            reverse=True,
        )

        return candidates[:top_k]

    # ==============================================================
    # RRF
    # ==============================================================

    def _rrf_score(
        self,
        rank: int,
        weight: float,
    ) -> float:

        return weight / (
            self.rrf_k + rank
        )

    # ==============================================================
    # MAIN SEARCH
    # ==============================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        # IMPORTANT:
        # Fetch deeper candidates so that a relevant method
        # such as Table.search is not lost before reranking.
        fetch_limit = max(
            top_k * 10,
            50,
        )

        # ==========================================================
        # 1. VECTOR RETRIEVAL
        # ==========================================================

        raw_vector_results = (
            self.vector_store.search(
                query,
                top_k=fetch_limit,
            )
        )

        vector_chunks: List[
            Dict[str, Any]
        ] = []

        for result in raw_vector_results:

            if isinstance(
                result,
                tuple,
            ):
                if (
                    len(result) > 0
                    and isinstance(
                        result[0],
                        dict,
                    )
                ):
                    vector_chunks.append(
                        result[0]
                    )

            elif isinstance(
                result,
                dict,
            ):
                vector_chunks.append(
                    result
                )

        # ==========================================================
        # 2. GRAPH RETRIEVAL
        # ==========================================================

        graph_chunks = (
            self._independent_graph_search(
                query,
                top_k=fetch_limit,
            )
        )

        # ==========================================================
        # 3. QUERY INTENT
        # ==========================================================

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

        # ==========================================================
        # 4. CODE-SYMBOL QUERY
        # ==========================================================

        code_symbol_query = bool(
            re.search(
                r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
                query,
            )
        )

        # ==========================================================
        # 5. MERGE RESULTS
        # ==========================================================

        rrf_scores: Dict[
            str,
            float,
        ] = {}

        doc_map: Dict[
            str,
            Dict[str, Any],
        ] = {}

        def merge_result(
            chunk: Dict[str, Any],
            rank: int,
            weight: float,
            source_type: str,
        ) -> None:

            chunk_id = self._get_chunk_id(
                chunk
            )

            if chunk_id not in doc_map:

                doc_map[chunk_id] = dict(
                    chunk
                )

                doc_map[chunk_id][
                    "sources"
                ] = []

                doc_map[chunk_id][
                    "vector_rank"
                ] = None

                doc_map[chunk_id][
                    "graph_rank"
                ] = None

            document = doc_map[
                chunk_id
            ]

            # ------------------------------------------------------
            # Source
            # ------------------------------------------------------

            is_new_source = (
                source_type
                not in document["sources"]
            )

            if is_new_source:
                document[
                    "sources"
                ].append(
                    source_type
                )

            # ------------------------------------------------------
            # Rank
            # ------------------------------------------------------

            if source_type == "vector":

                # Keep the BEST vector rank.
                existing_rank = document.get(
                    "vector_rank"
                )

                if (
                    existing_rank is None
                    or rank < existing_rank
                ):
                    document[
                        "vector_rank"
                    ] = rank

                # Preserve actual vector similarity.
                if (
                    chunk.get("score")
                    is not None
                ):
                    try:

                        current_score = float(
                            chunk["score"]
                        )

                        previous_score = (
                            document.get(
                                "vector_score"
                            )
                        )

                        if (
                            previous_score is None
                            or current_score
                            > float(previous_score)
                        ):
                            document[
                                "vector_score"
                            ] = current_score

                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

            elif source_type == "graph":

                existing_rank = document.get(
                    "graph_rank"
                )

                if (
                    existing_rank is None
                    or rank < existing_rank
                ):
                    document[
                        "graph_rank"
                    ] = rank

            # ------------------------------------------------------
            # Graph metadata
            # ------------------------------------------------------

            for graph_key in (
                "graph_callers",
                "graph_callees",
                "graph_match_score",
                "graph_matched_tokens",
            ):

                if (
                    graph_key in chunk
                    and chunk[graph_key]
                ):
                    document[
                        graph_key
                    ] = chunk[
                        graph_key
                    ]

            # ------------------------------------------------------
            # Metadata
            # ------------------------------------------------------

            for key in (
                "file_name",
                "file_path",
                "class_name",
                "method_name",
                "annotations",
                "code_content",
                "text_representation",
                "start_line",
                "end_line",
            ):

                if (
                    not document.get(key)
                    and chunk.get(key)
                ):
                    document[key] = chunk[key]

            # ------------------------------------------------------
            # RRF
            #
            # IMPORTANT:
            # Only count each source once per chunk.
            # This prevents duplicate vector entries from
            # artificially inflating the score.
            # ------------------------------------------------------

            if is_new_source:

                contribution = (
                    self._rrf_score(
                        rank=rank,
                        weight=weight,
                    )
                )

                rrf_scores[
                    chunk_id
                ] = (
                    rrf_scores.get(
                        chunk_id,
                        0.0,
                    )
                    + contribution
                )

        # ----------------------------------------------------------
        # Vector results
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Graph results
        # ----------------------------------------------------------

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

        # ==========================================================
        # 6. BUILD FINAL SCORE
        # ==========================================================

        final_scores: Dict[
            str,
            float,
        ] = {}

        for chunk_id, rrf_score in (
            rrf_scores.items()
        ):

            chunk = doc_map[
                chunk_id
            ]

            # ------------------------------------------------------
            # Vector score
            # ------------------------------------------------------

            vector_score = 0.0

            if chunk.get(
                "vector_score"
            ) is not None:

                try:
                    vector_score = float(
                        chunk[
                            "vector_score"
                        ]
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    vector_score = 0.0

            vector_score = max(
                0.0,
                min(
                    1.0,
                    vector_score,
                ),
            )

            # ------------------------------------------------------
            # Semantic score
            # ------------------------------------------------------

            semantic_score = (
                self._semantic_score(
                    query,
                    chunk,
                )
            )

            # ------------------------------------------------------
            # Graph score
            # ------------------------------------------------------

            graph_score = float(
                chunk.get(
                    "graph_match_score",
                    0.0,
                )
                or 0.0
            )

            # ------------------------------------------------------
            # Base score
            # ------------------------------------------------------

            score = rrf_score

            # Vector similarity.
            score += (
                vector_score * 0.20
            )

            # Textual relevance + code intent.
            score += (
                semantic_score * 0.20
            )

            # Graph relevance.
            if graph_score > 0:

                graph_multiplier = 0.10

                if code_symbol_query:
                    graph_multiplier = 0.20

                score += (
                    graph_score
                    * graph_multiplier
                )

            final_scores[
                chunk_id
            ] = score

            chunk[
                "semantic_score"
            ] = round(
                semantic_score,
                6,
            )

            chunk[
                "vector_score"
            ] = round(
                vector_score,
                6,
            )

        # ==========================================================
        # 7. ENDPOINT BOOST
        # ==========================================================

        if is_endpoint_query:

            for chunk_id in final_scores:

                annotations = (
                    self._clean_annotations(
                        doc_map[
                            chunk_id
                        ].get(
                            "annotations",
                            [],
                        )
                    )
                )

                if annotations.intersection(
                    WEB_ANNOTATIONS
                ):

                    final_scores[
                        chunk_id
                    ] *= 1.5

        # ==========================================================
        # 8. SORT
        # ==========================================================

        sorted_chunks = sorted(
            final_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        # ==========================================================
        # 9. FILE DIVERSITY
        # ==========================================================

        final_results: List[
            Dict[str, Any]
        ] = []

        file_counts: Dict[
            str,
            int,
        ] = {}

        deferred_results = []

        for chunk_id, score in (
            sorted_chunks
        ):

            chunk = doc_map.get(
                chunk_id
            )

            if not chunk:
                continue

            file_key = (
                self._get_file_key(
                    chunk
                )
            )

            current_count = (
                file_counts.get(
                    file_key,
                    0,
                )
            )

            enriched_chunk = dict(
                chunk
            )

            enriched_chunk[
                "combined_score"
            ] = round(
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

            if (
                current_count
                < self.max_per_file
            ):

                file_counts[
                    file_key
                ] = (
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

            if (
                len(final_results)
                >= top_k
            ):
                break

        # ==========================================================
        # 10. FALLBACK
        # ==========================================================

        if (
            len(final_results)
            < top_k
        ):

            for chunk, _score in (
                deferred_results
            ):

                if (
                    len(final_results)
                    >= top_k
                ):
                    break

                final_results.append(
                    chunk
                )

        # ==========================================================
        # 11. FINAL RANK
        # ==========================================================

        for index, result in enumerate(
            final_results,
            start=1,
        ):

            result[
                "final_rank"
            ] = index

        # ==========================================================
        # 12. DEBUG
        # ==========================================================

        print(
            "\n========== RETRIEVAL DEBUG =========="
        )

        print(
            "QUERY:",
            query,
        )

        print(
            "DETECTED INTENT:",
            self._detect_code_intent(query),
        )

        print(
            "\n--- QUERY TOKENS ---"
        )

        print(
            self._get_graph_tokens(
                query
            )
        )

        print(
            "\n--- VECTOR RESULTS ---"
        )

        for i, chunk in enumerate(
            vector_chunks[:30],
            start=1,
        ):

            print(
                i,
                "|",
                chunk.get(
                    "chunk_id"
                ),
                "| class=",
                chunk.get(
                    "class_name"
                ),
                "| method=",
                chunk.get(
                    "method_name"
                ),
                "| score=",
                chunk.get(
                    "score"
                ),
            )

        print(
            "\n--- GRAPH RESULTS ---"
        )

        for i, chunk in enumerate(
            graph_chunks[:15],
            start=1,
        ):

            print(
                i,
                "|",
                chunk.get(
                    "chunk_id"
                ),
                "| class=",
                chunk.get(
                    "class_name"
                ),
                "| method=",
                chunk.get(
                    "method_name"
                ),
                "| match_score=",
                chunk.get(
                    "graph_match_score"
                ),
                "| matched_tokens=",
                chunk.get(
                    "graph_matched_tokens"
                ),
            )

        print(
            "\n--- FINAL RESULTS ---"
        )

        for i, chunk in enumerate(
            final_results,
            start=1,
        ):

            print(
                i,
                "|",
                chunk.get(
                    "chunk_id"
                ),
                "| class=",
                chunk.get(
                    "class_name"
                ),
                "| method=",
                chunk.get(
                    "method_name"
                ),
                "| combined_score=",
                chunk.get(
                    "combined_score"
                ),
                "| vector_score=",
                chunk.get(
                    "vector_score"
                ),
                "| semantic_score=",
                chunk.get(
                    "semantic_score"
                ),
                "| sources=",
                chunk.get(
                    "sources"
                ),
                "| vector_rank=",
                chunk.get(
                    "vector_rank"
                ),
                "| graph_rank=",
                chunk.get(
                    "graph_rank"
                ),
            )

        print(
            "====================================\n"
        )

        return final_results