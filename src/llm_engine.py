from typing import Any, Dict, List, Optional
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
        5. Caller/callee and impact-aware queries
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

    # ---------------------------------------------------------------
    # Query
    # ---------------------------------------------------------------

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
        # 1. Detect graph-specific caller query
        # -----------------------------------------------------------

        caller_target = self._extract_caller_target(query)

        if caller_target:
            return self._answer_caller_query(
                query=query,
                method_name=caller_target,
                top_k=top_k,
            )

        # -----------------------------------------------------------
        # 2. Detect graph-specific callee query
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
        # 4. Build contextual answer
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

    # ---------------------------------------------------------------
    # Caller Query
    # ---------------------------------------------------------------

    def _answer_caller_query(
        self,
        query: str,
        method_name: str,
        top_k: int,
    ) -> Dict[str, Any]:

        callers = self.graph_db.get_callers_of(
            method_name
        )

        # Try alternate method forms if exact lookup fails
        if not callers:
            callers = self._find_callers_fuzzy(
                method_name
            )

        retrieved_chunks = []

        # -----------------------------------------------------------
        # Build useful graph results
        # -----------------------------------------------------------

        for caller in callers[:top_k]:

            class_name, caller_method = (
                self._split_method_id(caller)
            )

            retrieved_chunks.append({
                "chunk_id": caller,
                "chunk_type": "METHOD",
                "class_name": class_name,
                "method_name": caller_method,
                "file_name": self._get_node_file(caller),
                "graph_callers": [],
                "graph_callees": self.graph_db.get_calls_from(
                    caller
                ),
                "sources": ["graph"],
                "graph_rank": len(retrieved_chunks) + 1,
                "final_rank": len(retrieved_chunks) + 1,
            })

        # -----------------------------------------------------------
        # Construct answer
        # -----------------------------------------------------------

        if not callers:
            answer = (
                f"No methods calling '{method_name}' "
                f"were found in the code graph."
            )

        else:
            lines = [
                f"Methods that call '{method_name}()':"
            ]

            for caller in callers[:top_k]:
                lines.append(
                    f"- {caller}"
                )

            answer = "\n".join(lines)

        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
        }

    # ---------------------------------------------------------------
    # Callee Query
    # ---------------------------------------------------------------

    def _answer_callee_query(
        self,
        query: str,
        method_name: str,
        top_k: int,
    ) -> Dict[str, Any]:

        callees = self.graph_db.get_calls_from(
            method_name
        )

        if not callees:
            callees = self._find_callees_fuzzy(
                method_name
            )

        retrieved_chunks = []

        for callee in callees[:top_k]:

            class_name, callee_method = (
                self._split_method_id(callee)
            )

            retrieved_chunks.append({
                "chunk_id": callee,
                "chunk_type": "METHOD",
                "class_name": class_name,
                "method_name": callee_method,
                "file_name": self._get_node_file(callee),
                "graph_callers": self.graph_db.get_callers_of(
                    callee
                ),
                "graph_callees": [],
                "sources": ["graph"],
                "graph_rank": len(retrieved_chunks) + 1,
                "final_rank": len(retrieved_chunks) + 1,
            })

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
                lines.append(
                    f"- {callee}"
                )

            answer = "\n".join(lines)

        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
        }

    # ---------------------------------------------------------------
    # Query Detection
    # ---------------------------------------------------------------

    @staticmethod
    def _extract_caller_target(
        query: str,
    ) -> Optional[str]:

        patterns = [
            r"which\s+(?:methods?|functions?|classes?)\s+call\s+([A-Za-z_$][\w$]*)",
            r"who\s+calls\s+([A-Za-z_$][\w$]*)",
            r"what\s+calls\s+([A-Za-z_$][\w$]*)",
            r"callers?\s+of\s+([A-Za-z_$][\w$]*)",
        ]

        query_lower = query.lower()

        for pattern in patterns:

            match = re.search(
                pattern,
                query_lower,
            )

            if match:
                return match.group(1)

        return None

    @staticmethod
    def _extract_callee_target(
        query: str,
    ) -> Optional[str]:

        patterns = [
            r"which\s+(?:methods?|functions?)\s+does\s+([A-Za-z_$][\w$]*)\s+call",
            r"what\s+does\s+([A-Za-z_$][\w$]*)\s+call",
            r"callees?\s+of\s+([A-Za-z_$][\w$]*)",
        ]

        query_lower = query.lower()

        for pattern in patterns:

            match = re.search(
                pattern,
                query_lower,
            )

            if match:
                return match.group(1)

        return None

    # ---------------------------------------------------------------
    # Fuzzy Caller Search
    # ---------------------------------------------------------------

    def _find_callers_fuzzy(
        self,
        method_name: str,
    ) -> List[str]:

        results = []

        target = method_name.lower()

        for node in self.graph_db.graph.nodes():

            node_str = str(node)

            if (
                node_str.lower() == target
                or node_str.lower().endswith(
                    f".{target}()"
                )
            ):

                callers = self.graph_db.get_callers_of(
                    node_str
                )

                results.extend(callers)

        return list(dict.fromkeys(results))

    # ---------------------------------------------------------------
    # Fuzzy Callee Search
    # ---------------------------------------------------------------

    def _find_callees_fuzzy(
        self,
        method_name: str,
    ) -> List[str]:

        results = []

        target = method_name.lower()

        for node in self.graph_db.graph.nodes():

            node_str = str(node)

            if (
                node_str.lower() == target
                or node_str.lower().endswith(
                    f".{target}()"
                )
            ):

                results.extend(
                    self.graph_db.get_calls_from(
                        node_str
                    )
                )

        return list(dict.fromkeys(results))

    # ---------------------------------------------------------------
    # Graph Helpers
    # ---------------------------------------------------------------

    def _get_node_file(
        self,
        node_id: str,
    ) -> str:

        if self.graph_db.graph.has_node(node_id):

            return self.graph_db.graph.nodes[
                node_id
            ].get("file", "")

        return ""

    @staticmethod
    def _split_method_id(
        method_id: str,
    ) -> tuple:

        clean = method_id.replace(
            "()",
            "",
        )

        if "." in clean:

            class_name, method_name = (
                clean.rsplit(".", 1)
            )

            return class_name, method_name

        return "", clean

    # ---------------------------------------------------------------
    # Normal Contextual Answer
    # ---------------------------------------------------------------

    def _build_contextual_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> str:

        if not chunks:
            return (
                "No relevant code was found in the indexed repository."
            )

        lines = []

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

                code_preview = code.strip()

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

    # ---------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------

    @staticmethod
    def _indent_code(
        code: str,
    ) -> str:

        return "\n".join(
            "      " + line
            for line in code.splitlines()
        )

    # ---------------------------------------------------------------
    # Impact Analysis
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Git History
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Why Changed / Provenance
    # ---------------------------------------------------------------

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