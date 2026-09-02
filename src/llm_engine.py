from typing import Any, Dict, List, Optional, Tuple
import re

from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph
from src.hybrid_retriever import HybridRetriever
from src.context_builder import CodeIntelligenceContextBuilder


class CodeIntelligenceEngine:
    """
    Main intelligence layer for the AI Codebase Intelligence system.

    Responsibilities:
        1. Hybrid code retrieval
        2. Graph-aware context enrichment
        3. Git/context integration
        4. Deterministic contextual answers
        5. Caller/callee queries
        6. Impact-aware queries
    """

    def __init__(
        self,
        repo_path: str,
        vector_store: VectorStore,
        graph_db: CodeKnowledgeGraph,
        context_builder: Optional[CodeIntelligenceContextBuilder] = None,
    ):
        self.repo_path = str(repo_path)
        self.vector_store = vector_store
        self.graph_db = graph_db
        self.context_builder = context_builder

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            knowledge_graph=self.graph_db,
        )

    # ===============================================================
    # MAIN QUERY
    # ===============================================================

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:

        if not query or not query.strip():
            return {
                "query": query,
                "retrieved_chunks": [],
                "answer": "Please provide a valid codebase query.",
            }

        query = query.strip()

        # -----------------------------------------------------------
        # 1. Caller query
        # -----------------------------------------------------------

        caller_target = self._extract_caller_target(query)

        if caller_target:
            return self._answer_caller_query(
                query=query,
                method_name=caller_target,
                top_k=top_k,
            )

        # -----------------------------------------------------------
        # 2. Callee query
        # -----------------------------------------------------------

        callee_target = self._extract_callee_target(query)

        if callee_target:
            return self._answer_callee_query(
                query=query,
                method_name=callee_target,
                top_k=top_k,
            )

        # -----------------------------------------------------------
        # 3. Normal hybrid retrieval
        # -----------------------------------------------------------

        retrieved_chunks = self.retriever.search(
            query=query,
            top_k=top_k,
        )

        # -----------------------------------------------------------
        # 4. Contextual answer
        # -----------------------------------------------------------

        answer = self._build_contextual_answer(
            query=query,
            chunks=retrieved_chunks,
        )

        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
        }

    # ===============================================================
    # CALLER QUERY
    # ===============================================================

    def _answer_caller_query(
        self,
        query: str,
        method_name: str,
        top_k: int,
    ) -> Dict[str, Any]:

        callers = self._resolve_method_callers(method_name)

        retrieved_chunks: List[Dict[str, Any]] = []

        for caller in callers[:top_k]:

            class_name, caller_method = self._split_method_id(caller)

            retrieved_chunks.append(
                {
                    "chunk_id": caller,
                    "chunk_type": "METHOD",
                    "class_name": class_name,
                    "method_name": caller_method,
                    "file_name": self._get_node_file(caller),
                    "graph_callers": self.graph_db.get_callers_of(caller),
                    "graph_callees": self.graph_db.get_calls_from(caller),
                    "sources": ["graph"],
                    "graph_rank": len(retrieved_chunks) + 1,
                    "final_rank": len(retrieved_chunks) + 1,
                }
            )

        if not callers:
            answer = (
                f"No methods calling '{method_name}()' "
                f"were found in the code graph."
            )
        else:
            lines = [
                f"Methods that call '{method_name}()':"
            ]

            for caller in callers[:top_k]:
                lines.append(f"- {caller}")

            answer = "\n".join(lines)

        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
        }

    # ===============================================================
    # CALLEE QUERY
    # ===============================================================

    def _answer_callee_query(
        self,
        query: str,
        method_name: str,
        top_k: int,
    ) -> Dict[str, Any]:

        callees = self._resolve_method_callees(method_name)

        retrieved_chunks: List[Dict[str, Any]] = []

        for callee in callees[:top_k]:

            class_name, callee_method = self._split_method_id(callee)

            retrieved_chunks.append(
                {
                    "chunk_id": callee,
                    "chunk_type": "METHOD",
                    "class_name": class_name,
                    "method_name": callee_method,
                    "file_name": self._get_node_file(callee),
                    "graph_callers": self.graph_db.get_callers_of(callee),
                    "graph_callees": self.graph_db.get_calls_from(callee),
                    "sources": ["graph"],
                    "graph_rank": len(retrieved_chunks) + 1,
                    "final_rank": len(retrieved_chunks) + 1,
                }
            )

        if not callees:
            answer = (
                f"No methods called by '{method_name}()' "
                f"were found in the code graph."
            )
        else:
            lines = [
                f"Methods called by '{method_name}()':"
            ]

            for callee in callees[:top_k]:
                lines.append(f"- {callee}")

            answer = "\n".join(lines)

        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
        }

    # ===============================================================
    # QUERY DETECTION
    # ===============================================================

    @classmethod
    def _extract_caller_target(
        cls,
        query: str,
    ) -> Optional[str]:

        if not query:
            return None

        patterns = [
            # Who calls bookEquipment?
            r"\bwho\s+calls\s+([A-Za-z_$][\w$]*(?:\(\))?)\b",

            # What calls bookEquipment?
            r"\bwhat\s+calls\s+([A-Za-z_$][\w$]*(?:\(\))?)\b",

            # Callers of bookEquipment
            r"\bcallers?\s+of\s+([A-Za-z_$][\w$]*(?:\(\))?)\b",

            # Which methods call bookEquipment?
            r"\bwhich\s+(?:methods?|functions?)\s+call\s+([A-Za-z_$][\w$]*(?:\(\))?)\b",

            # Which classes call bookEquipment?
            r"\bwhich\s+classes?\s+call\s+([A-Za-z_$][\w$]*(?:\(\))?)\b",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            candidate = match.group(1)

            candidate = cls._clean_query_method_name(
                candidate
            )

            if not cls._is_valid_method_candidate(candidate):
                continue

            if cls._is_stopword_candidate(candidate):
                continue

            # -------------------------------------------------------
            # CRITICAL VALIDATION
            #
            # Do not route a natural-language word to the graph.
            # Example:
            #
            # "Which methods call other methods related to booking?"
            #
            # "other" must NOT become a method target.
            # -------------------------------------------------------

            return candidate

        return None

    @classmethod
    def _extract_callee_target(
        cls,
        query: str,
    ) -> Optional[str]:

        if not query:
            return None

        patterns = [
            # What does bookEquipment call?
            r"\bwhat\s+does\s+([A-Za-z_$][\w$]*(?:\(\))?)\s+call\b",

            # Which methods does bookEquipment call?
            r"\bwhich\s+(?:methods?|functions?)\s+does\s+([A-Za-z_$][\w$]*(?:\(\))?)\s+call\b",

            # Callees of bookEquipment
            r"\bcallees?\s+of\s+([A-Za-z_$][\w$]*(?:\(\))?)\b",

            # What methods are called by bookEquipment?
            r"\bwhat\s+(?:methods?|functions?)\s+(?:are|does)\s+called\s+by\s+([A-Za-z_$][\w$]*(?:\(\))?)\b",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            candidate = match.group(1)

            candidate = cls._clean_query_method_name(
                candidate
            )

            if not cls._is_valid_method_candidate(candidate):
                continue

            if cls._is_stopword_candidate(candidate):
                continue

            return candidate

        return None

    # ===============================================================
    # QUERY METHOD NORMALIZATION
    # ===============================================================

    @staticmethod
    def _clean_query_method_name(
        method_name: str,
    ) -> str:

        if not method_name:
            return ""

        value = str(method_name).strip()

        # Remove surrounding punctuation.
        value = value.strip(
            " \t\r\n.,;:!?\"'`()[]{}"
        )

        # Remove trailing parentheses.
        if value.endswith("()"):
            value = value[:-2]

        return value.strip()

    @staticmethod
    def _is_valid_method_candidate(
        method_name: str,
    ) -> bool:

        if not method_name:
            return False

        # Supports:
        #
        # booking
        # bookEquipment
        # _bookEquipment
        # $helper
        #
        # Also supports class-qualified names:
        #
        # BookingService.bookEquipment

        return bool(
            re.fullmatch(
                r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*",
                method_name,
            )
        )

    @staticmethod
    def _is_stopword_candidate(
        method_name: str,
    ) -> bool:

        stopwords = {
            "a",
            "an",
            "the",
            "other",
            "another",
            "methods",
            "method",
            "function",
            "functions",
            "class",
            "classes",
            "code",
            "codes",
            "anything",
            "something",
            "it",
            "this",
            "that",
            "these",
            "those",
            "related",
            "used",
            "use",
            "work",
            "working",
            "functionality",
            "some",
            "any",
            "all",
            "things",
        }

        return method_name.lower() in stopwords

    # ===============================================================
    # GRAPH RESOLUTION
    # ===============================================================

    def _resolve_method_callers(
        self,
        method_name: str,
    ) -> List[str]:

        method_name = self._clean_query_method_name(
            method_name
        )

        if not method_name:
            return []

        results: List[str] = []

        # -----------------------------------------------------------
        # 1. Exact graph lookup
        # -----------------------------------------------------------

        try:
            exact = self.graph_db.get_callers_of(
                method_name
            )

            if exact:
                results.extend(exact)

        except Exception:
            pass

        # -----------------------------------------------------------
        # 2. Fuzzy method-node lookup
        # -----------------------------------------------------------

        if not results:
            results.extend(
                self._find_callers_fuzzy(
                    method_name
                )
            )

        return list(dict.fromkeys(results))

    def _resolve_method_callees(
        self,
        method_name: str,
    ) -> List[str]:

        method_name = self._clean_query_method_name(
            method_name
        )

        if not method_name:
            return []

        results: List[str] = []

        try:
            exact = self.graph_db.get_calls_from(
                method_name
            )

            if exact:
                results.extend(exact)

        except Exception:
            pass

        if not results:
            results.extend(
                self._find_callees_fuzzy(
                    method_name
                )
            )

        return list(dict.fromkeys(results))

    # ===============================================================
    # FUZZY CALLER SEARCH
    # ===============================================================

    def _find_callers_fuzzy(
        self,
        method_name: str,
    ) -> List[str]:

        results: List[str] = []

        target = self._clean_query_method_name(
            method_name
        ).lower()

        if not target:
            return results

        for node in self.graph_db.graph.nodes():

            node_data = self.graph_db.graph.nodes[node]

            # Only real method nodes.
            if node_data.get("type") != "METHOD":
                continue

            node_str = str(node)

            clean_node = node_str

            if clean_node.endswith("()"):
                clean_node = clean_node[:-2]

            node_name = (
                clean_node.rsplit(".", 1)[-1]
                .strip()
                .lower()
            )

            if node_name != target:
                continue

            try:
                callers = self.graph_db.get_callers_of(
                    node_str
                )

                results.extend(callers)

            except Exception:
                continue

        return list(dict.fromkeys(results))

    # ===============================================================
    # FUZZY CALLEE SEARCH
    # ===============================================================

    def _find_callees_fuzzy(
        self,
        method_name: str,
    ) -> List[str]:

        results: List[str] = []

        target = self._clean_query_method_name(
            method_name
        ).lower()

        if not target:
            return results

        for node in self.graph_db.graph.nodes():

            node_data = self.graph_db.graph.nodes[node]

            # Only real method nodes.
            if node_data.get("type") != "METHOD":
                continue

            node_str = str(node)

            clean_node = node_str

            if clean_node.endswith("()"):
                clean_node = clean_node[:-2]

            node_name = (
                clean_node.rsplit(".", 1)[-1]
                .strip()
                .lower()
            )

            if node_name != target:
                continue

            try:
                callees = self.graph_db.get_calls_from(
                    node_str
                )

                results.extend(callees)

            except Exception:
                continue

        return list(dict.fromkeys(results))

    # ===============================================================
    # GRAPH HELPERS
    # ===============================================================

    def _get_node_file(
        self,
        node_id: str,
    ) -> str:

        if self.graph_db.graph.has_node(node_id):

            node_data = self.graph_db.graph.nodes[node_id]

            return (
                node_data.get("file", "")
                or node_data.get("file_name", "")
                or node_data.get("file_path", "")
            )

        return ""

    @staticmethod
    def _split_method_id(
        method_id: str,
    ) -> Tuple[str, str]:

        clean = str(method_id).strip()

        if clean.endswith("()"):
            clean = clean[:-2]

        if "." in clean:
            class_name, method_name = clean.rsplit(
                ".",
                1,
            )

            return class_name, method_name

        return "", clean

    # ===============================================================
    # NORMAL CONTEXTUAL ANSWER
    # ===============================================================

    def _build_contextual_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> str:

        if not chunks:
            return (
                "No relevant code was found in the indexed repository."
            )

        lines: List[str] = []

        lines.append(
            f'Based on the indexed codebase, the most relevant code '
            f'for the query "{query}" is:'
        )

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            file_name = (
                chunk.get("file_name")
                or chunk.get("file_path")
                or "Unknown file"
            )

            class_name = chunk.get(
                "class_name",
                "",
            )

            method_name = chunk.get(
                "method_name",
                "",
            )

            chunk_type = chunk.get(
                "chunk_type",
                "METHOD",
            )

            callers = chunk.get(
                "graph_callers",
                [],
            )

            callees = chunk.get(
                "graph_callees",
                [],
            )

            lines.append("")

            lines.append(
                f"{index}. {file_name}"
            )

            if class_name:
                lines.append(
                    f"   Class: {class_name}"
                )

            if method_name:
                lines.append(
                    f"   Method: {method_name}"
                )

            lines.append(
                f"   Type: {chunk_type}"
            )

            if callers:
                lines.append(
                    "   Callers: "
                    + ", ".join(
                        map(str, callers)
                    )
                )

            if callees:
                lines.append(
                    "   Callees: "
                    + ", ".join(
                        map(str, callees)
                    )
                )

            code = (
                chunk.get("code_content")
                or chunk.get("text_representation")
                or ""
            )

            if code:

                code_preview = str(
                    code
                ).strip()

                if len(code_preview) > 500:
                    code_preview = (
                        code_preview[:500]
                        + "..."
                    )

                lines.append(
                    "   Code:\n"
                    + self._indent_code(
                        code_preview
                    )
                )

        return "\n".join(lines)

    # ===============================================================
    # UTILITY
    # ===============================================================

    @staticmethod
    def _indent_code(
        code: str,
    ) -> str:

        return "\n".join(
            "      " + line
            for line in code.splitlines()
        )

    # ===============================================================
    # IMPACT ANALYSIS
    # ===============================================================

    def get_impact(
        self,
        method_name: str,
    ) -> Dict[str, Any]:

        if not method_name:
            return {
                "error": "Method name cannot be empty."
            }

        from src.impact_analysis import ImpactAnalyzer

        analyzer = ImpactAnalyzer(
            self.graph_db
        )

        return analyzer.analyze_blast_radius(
            method_name
        )

    # ===============================================================
    # GIT HISTORY
    # ===============================================================

    def get_file_history(
        self,
        file_path: str,
    ) -> Any:

        if self.context_builder is None:
            return {
                "error": (
                    "Git context builder is not initialized."
                )
            }

        git_intel = getattr(
            self.context_builder,
            "git_intel",
            None,
        )

        if git_intel is None:
            return {
                "error": (
                    "Git intelligence is not available."
                )
            }

        return git_intel.get_file_history(
            file_path
        )

    # ===============================================================
    # WHY CHANGED / PROVENANCE
    # ===============================================================

    def explain_why_changed(
        self,
        file_path: str,
        method_name: str,
        start_line: int,
        end_line: int,
    ) -> Dict[str, Any]:

        if self.context_builder is None:
            return {
                "answer": (
                    "Git context builder is not initialized."
                )
            }

        if hasattr(
            self.context_builder,
            "explain_why_changed",
        ):

            try:

                return (
                    self.context_builder
                    .explain_why_changed(
                        file_path=file_path,
                        method_name=method_name,
                        start_line=start_line,
                        end_line=end_line,
                    )
                )

            except Exception as e:

                return {
                    "answer": (
                        f"Unable to perform provenance "
                        f"analysis: {e}"
                    )
                }

        return {
            "answer": (
                f"Provenance analysis requested for "
                f"{method_name} in {file_path}, "
                f"lines {start_line}-{end_line}, "
                f"but the current context builder does not "
                f"implement explain_why_changed()."
            )
        }