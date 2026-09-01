from typing import Any, Dict, List, Optional
import re

from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph
from src.hybrid_retriever import HybridRetriever
from src.context_builder import CodeIntelligenceContextBuilder


class CodeIntelligenceEngine:
    """
    Main intelligence layer for the AI Codebase Intelligence system.

    Supported query types:
        - Normal code questions
        - Caller queries
        - Callee queries
        - Impact analysis
        - Git history
        - Change provenance

    Every response contains a response_type field so the frontend
    can render the appropriate visual panel.
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

    # ============================================================
    # MAIN QUERY ROUTER
    # ============================================================

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:

        if not query or not query.strip():
            return {
                "query": query,
                "response_type": "text",
                "retrieved_chunks": [],
                "answer": "Please provide a valid codebase query.",
            }

        query = query.strip()

        # --------------------------------------------------------
        # 1. CALLER QUERY
        # --------------------------------------------------------

        caller_target = self._extract_caller_target(query)

        if caller_target:
            return self._answer_caller_query(
                query=query,
                method_name=caller_target,
                top_k=top_k,
            )

        # --------------------------------------------------------
        # 2. CALLEE QUERY
        # --------------------------------------------------------

        callee_target = self._extract_callee_target(query)

        if callee_target:
            return self._answer_callee_query(
                query=query,
                method_name=callee_target,
                top_k=top_k,
            )

        # --------------------------------------------------------
        # 3. IMPACT QUERY
        # --------------------------------------------------------

        impact_target = self._extract_impact_target(query)

        if impact_target:
            return self._answer_impact_query(
                query=query,
                method_name=impact_target,
                top_k=top_k,
            )

        # --------------------------------------------------------
        # 4. PROVENANCE QUERY
        # --------------------------------------------------------

        provenance_target = self._extract_provenance_target(query)

        if provenance_target:
            return self._answer_provenance_query(
                query=query,
                target=provenance_target,
            )

        # --------------------------------------------------------
        # 5. GIT HISTORY QUERY
        # --------------------------------------------------------

        history_target = self._extract_history_target(query)

        if history_target:
            return self._answer_history_query(
                query=query,
                file_path=history_target,
            )

        # --------------------------------------------------------
        # 6. NORMAL HYBRID RETRIEVAL
        # --------------------------------------------------------

        retrieved_chunks = self.retriever.search(
            query=query,
            top_k=top_k,
        )

        # --------------------------------------------------------
        # 7. NORMAL ANSWER
        # --------------------------------------------------------

        answer = self._build_contextual_answer(
            query=query,
            chunks=retrieved_chunks,
        )

        return {
            "query": query,
            "response_type": "text",
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
        }

    # ============================================================
    # CALLER QUERY
    # ============================================================

    def _answer_caller_query(
        self,
        query: str,
        method_name: str,
        top_k: int,
    ) -> Dict[str, Any]:

        callers = self.graph_db.get_callers_of(
            method_name
        )

        if not callers:
            callers = self._find_callers_fuzzy(
                method_name
            )

        callers = list(dict.fromkeys(callers))

        retrieved_chunks = []

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
                "graph_callers": self.graph_db.get_callers_of(
                    caller
                ),
                "graph_callees": self.graph_db.get_calls_from(
                    caller
                ),
                "sources": ["graph"],
                "graph_rank": len(retrieved_chunks) + 1,
                "final_rank": len(retrieved_chunks) + 1,
            })

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

        # Include target node so frontend can draw the graph
        nodes = [
            {
                "id": method_name,
                "type": "METHOD",
                "role": "target",
            }
        ]

        nodes.extend([
            {
                "id": caller,
                "type": "METHOD",
                "role": "caller",
            }
            for caller in callers[:top_k]
        ])

        edges = [
            {
                "source": caller,
                "target": method_name,
                "relation": "CALLS",
            }
            for caller in callers[:top_k]
        ]

        return {
            "query": query,
            "response_type": "call_graph",
            "target": method_name,
            "direction": "callers",
            "retrieved_chunks": retrieved_chunks,
            "graph": {
                "target": method_name,
                "direction": "callers",
                "nodes": nodes,
                "edges": edges,
            },
            "answer": answer,
        }

    # ============================================================
    # CALLEE QUERY
    # ============================================================

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

        callees = list(dict.fromkeys(callees))

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
                "graph_callees": self.graph_db.get_calls_from(
                    callee
                ),
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

        nodes = [
            {
                "id": method_name,
                "type": "METHOD",
                "role": "target",
            }
        ]

        nodes.extend([
            {
                "id": callee,
                "type": "METHOD",
                "role": "callee",
            }
            for callee in callees[:top_k]
        ])

        edges = [
            {
                "source": method_name,
                "target": callee,
                "relation": "CALLS",
            }
            for callee in callees[:top_k]
        ]

        return {
            "query": query,
            "response_type": "call_graph",
            "target": method_name,
            "direction": "callees",
            "retrieved_chunks": retrieved_chunks,
            "graph": {
                "target": method_name,
                "direction": "callees",
                "nodes": nodes,
                "edges": edges,
            },
            "answer": answer,
        }

    # ============================================================
    # IMPACT QUERY
    # ============================================================

    def _answer_impact_query(
        self,
        query: str,
        method_name: str,
        top_k: int,
    ) -> Dict[str, Any]:

        try:

            impact = self.get_impact(
                method_name
            )

            if not isinstance(impact, dict):
                impact = {
                    "result": impact
                }

            answer = self._format_impact_answer(
                method_name,
                impact,
            )

            return {
                "query": query,
                "response_type": "impact_summary",
                "target": method_name,
                "impact": impact,
                "retrieved_chunks": [],
                "answer": answer,
            }

        except Exception as e:

            return {
                "query": query,
                "response_type": "impact_summary",
                "target": method_name,
                "impact": {},
                "retrieved_chunks": [],
                "answer": (
                    f"Unable to perform impact analysis "
                    f"for '{method_name}': {e}"
                ),
            }

    # ============================================================
    # GIT HISTORY QUERY
    # ============================================================

    def _answer_history_query(
        self,
        query: str,
        file_path: str,
    ) -> Dict[str, Any]:

        try:

            history = self.get_file_history(
                file_path
            )

            if isinstance(history, dict):

                answer = self._format_history_answer(
                    file_path,
                    history,
                )

            elif isinstance(history, list):

                answer = (
                    f"Git history for {file_path}:\n"
                    + "\n".join(
                        f"- {item}"
                        for item in history
                    )
                )

            else:

                answer = (
                    f"Git history for {file_path}:\n"
                    f"{history}"
                )

            return {
                "query": query,
                "response_type": "git_timeline",
                "target": file_path,
                "history": history,
                "retrieved_chunks": [],
                "answer": answer,
            }

        except Exception as e:

            return {
                "query": query,
                "response_type": "git_timeline",
                "target": file_path,
                "history": [],
                "retrieved_chunks": [],
                "answer": (
                    f"Unable to retrieve Git history "
                    f"for '{file_path}': {e}"
                ),
            }

    # ============================================================
    # PROVENANCE QUERY
    # ============================================================

    def _answer_provenance_query(
        self,
        query: str,
        target: Dict[str, Any],
    ) -> Dict[str, Any]:

        file_path = target.get(
            "file_path",
            "",
        )

        method_name = target.get(
            "method_name",
            "",
        )

        start_line = target.get(
            "start_line",
            1,
        )

        end_line = target.get(
            "end_line",
            start_line,
        )

        try:

            result = self.explain_why_changed(
                file_path=file_path,
                method_name=method_name,
                start_line=start_line,
                end_line=end_line,
            )

            answer = (
                result.get(
                    "answer",
                    str(result),
                )
                if isinstance(result, dict)
                else str(result)
            )

            return {
                "query": query,
                "response_type": "provenance",
                "target": target,
                "provenance": result,
                "retrieved_chunks": [],
                "answer": answer,
            }

        except Exception as e:

            return {
                "query": query,
                "response_type": "provenance",
                "target": target,
                "provenance": {},
                "retrieved_chunks": [],
                "answer": (
                    f"Unable to perform provenance "
                    f"analysis: {e}"
                ),
            }

    # ============================================================
    # QUERY DETECTION
    # ============================================================

    @staticmethod
    def _extract_caller_target(
        query: str,
    ) -> Optional[str]:

        patterns = [

            r"which\s+(?:methods?|functions?|classes?)\s+call\s+([A-Za-z_$][\w$]*)",

            r"who\s+calls\s+([A-Za-z_$][\w$]*)",

            r"what\s+calls\s+([A-Za-z_$][\w$]*)",

            r"callers?\s+of\s+([A-Za-z_$][\w$]*)",

            r"who\s+invokes\s+([A-Za-z_$][\w$]*)",

            r"what\s+invokes\s+([A-Za-z_$][\w$]*)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                query,
                re.IGNORECASE,
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

            r"what\s+does\s+([A-Za-z_$][\w$]*)\s+invoke",

            r"what\s+methods?\s+are\s+called\s+by\s+([A-Za-z_$][\w$]*)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                query,
                re.IGNORECASE,
            )

            if match:
                return match.group(1)

        return None

    @staticmethod
    def _extract_impact_target(
        query: str,
    ) -> Optional[str]:

        patterns = [

            r"impact\s+(?:of|from|if)\s+(?:changing|modifying)?\s*([A-Za-z_$][\w$]*)",

            r"impact\s+(?:analysis|of)\s+([A-Za-z_$][\w$]*)",

            r"what\s+(?:would\s+be\s+)?affected\s+(?:if\s+we\s+change|by\s+changing)\s+([A-Za-z_$][\w$]*)",

            r"what\s+depends\s+on\s+([A-Za-z_$][\w$]*)",

            r"blast\s+radius\s+(?:of|for)\s+([A-Za-z_$][\w$]*)",

            r"what\s+happens\s+if\s+([A-Za-z_$][\w$]*)\s+is\s+changed",

            r"dependencies\s+affected\s+by\s+([A-Za-z_$][\w$]*)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                query,
                re.IGNORECASE,
            )

            if match:
                return match.group(1)

        return None

    @staticmethod
    def _extract_history_target(
        query: str,
    ) -> Optional[str]:

        file_pattern = (
            r"([A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+)"
        )

        patterns = [

            rf"(?:git\s+)?history\s+(?:of|for)\s+{file_pattern}",

            rf"commits?\s+(?:for|of)\s+{file_pattern}",

            rf"git\s+log\s+(?:for|of)\s+{file_pattern}",

            rf"changes?\s+(?:to|in)\s+{file_pattern}",

            rf"show\s+(?:git\s+)?history\s+(?:of|for)\s+{file_pattern}",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                query,
                re.IGNORECASE,
            )

            if match:
                return match.group(1)

        return None

    @staticmethod
    def _extract_provenance_target(
        query: str,
    ) -> Optional[Dict[str, Any]]:

        # Detect questions such as:
        #
        # Why was UserService.java changed?
        # Why was calculateTotal modified?
        # What is the reason this method changed?
        # What caused lines 20-40 in UserService.java to change?

        provenance_pattern = (
            r"(why|reason|provenance|what\s+caused|"
            r"why\s+was).*(changed|change|modified|modify|updated|update)"
        )

        if not re.search(
            provenance_pattern,
            query,
            re.IGNORECASE,
        ):
            return None

        # File
        file_match = re.search(
            r"([A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+)",
            query,
        )

        # Method/function
        method_match = re.search(
            r"(?:method|function)\s+([A-Za-z_$][\w$]*)",
            query,
            re.IGNORECASE,
        )

        if not method_match:

            method_match = re.search(
                r"(?:for|of|in)\s+([A-Za-z_$][\w$]*)",
                query,
                re.IGNORECASE,
            )

        # Lines
        line_match = re.search(
            r"lines?\s+(\d+)(?:\s*[-–]\s*(\d+))?",
            query,
            re.IGNORECASE,
        )

        start_line = (
            int(line_match.group(1))
            if line_match
            else 1
        )

        end_line = (
            int(line_match.group(2))
            if line_match and line_match.group(2)
            else start_line
        )

        return {
            "file_path": (
                file_match.group(1)
                if file_match
                else ""
            ),

            "method_name": (
                method_match.group(1)
                if method_match
                else ""
            ),

            "start_line": start_line,

            "end_line": end_line,
        }

    # ============================================================
    # FUZZY CALLER SEARCH
    # ============================================================

    def _find_callers_fuzzy(
        self,
        method_name: str,
    ) -> List[str]:

        results = []

        target = method_name.lower()

        for node in self.graph_db.graph.nodes():

            node_str = str(node)

            clean_node = node_str.replace(
                "()",
                "",
            )

            node_method = (
                clean_node.rsplit(".", 1)[-1]
            )

            if node_method.lower() == target:

                results.extend(
                    self.graph_db.get_callers_of(
                        node_str
                    )
                )

        return list(
            dict.fromkeys(results)
        )

    # ============================================================
    # FUZZY CALLEE SEARCH
    # ============================================================

    def _find_callees_fuzzy(
        self,
        method_name: str,
    ) -> List[str]:

        results = []

        target = method_name.lower()

        for node in self.graph_db.graph.nodes():

            node_str = str(node)

            clean_node = node_str.replace(
                "()",
                "",
            )

            node_method = (
                clean_node.rsplit(".", 1)[-1]
            )

            if node_method.lower() == target:

                results.extend(
                    self.graph_db.get_calls_from(
                        node_str
                    )
                )

        return list(
            dict.fromkeys(results)
        )

    # ============================================================
    # GRAPH HELPERS
    # ============================================================

    def _get_node_file(
        self,
        node_id: str,
    ) -> str:

        if self.graph_db.graph.has_node(
            node_id
        ):

            return self.graph_db.graph.nodes[
                node_id
            ].get(
                "file",
                "",
            )

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

    # ============================================================
    # NORMAL CONTEXTUAL ANSWER
    # ============================================================

    def _build_contextual_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> str:

        if not chunks:

            return (
                "No relevant code was found "
                "in the indexed repository."
            )

        lines = []

        lines.append(
            f'Based on the indexed codebase, the most relevant '
            f'code for the query "{query}" is:'
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

    # ============================================================
    # IMPACT ANALYSIS
    # ============================================================

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

    # ============================================================
    # GIT HISTORY
    # ============================================================

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

    # ============================================================
    # WHY CHANGED / PROVENANCE
    # ============================================================

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

    # ============================================================
    # FORMATTING HELPERS
    # ============================================================

    @staticmethod
    def _format_impact_answer(
        method_name: str,
        impact: Dict[str, Any],
    ) -> str:

        lines = [
            f"Impact analysis for '{method_name}()':"
        ]

        if not impact:

            lines.append(
                "No impact information was returned."
            )

            return "\n".join(lines)

        for key, value in impact.items():

            if isinstance(value, list):

                lines.append(
                    f"- {key}:"
                )

                for item in value:

                    lines.append(
                        f"  - {item}"
                    )

            else:

                lines.append(
                    f"- {key}: {value}"
                )

        return "\n".join(lines)

    @staticmethod
    def _format_history_answer(
        file_path: str,
        history: Dict[str, Any],
    ) -> str:

        lines = [
            f"Git history for '{file_path}':"
        ]

        if not history:

            lines.append(
                "No Git history information was returned."
            )

            return "\n".join(lines)

        for key, value in history.items():

            if isinstance(value, list):

                lines.append(
                    f"- {key}:"
                )

                for item in value[:10]:

                    lines.append(
                        f"  - {item}"
                    )

            else:

                lines.append(
                    f"- {key}: {value}"
                )

        return "\n".join(lines)

    # ============================================================
    # CODE INDENTATION
    # ============================================================

    @staticmethod
    def _indent_code(
        code: str,
    ) -> str:

        return "\n".join(
            "      " + line
            for line in code.splitlines()
        )