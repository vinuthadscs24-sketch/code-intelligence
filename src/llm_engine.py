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
            r"\bwho\s+calls\s+([A-Za-z_$][\w$]*(?:\(\))?)\b",
            r"\bwhat\s+calls\s+([A-Za-z_$][\w$]*(?:\(\))?)\b",
            r"\bcallers?\s+of\s+([A-Za-z_$][\w$]*(?:\(\))?)\b",
            r"\bwhich\s+(?:methods?|functions?)\s+call\s+([A-Za-z_$][\w$]*(?:\(\))?)\b",
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
            r"\bwhat\s+does\s+([A-Za-z_$][\w$]*(?:\(\))?)\s+call\b",
            r"\bwhich\s+(?:methods?|functions?)\s+does\s+([A-Za-z_$][\w$]*(?:\(\))?)\s+call\b",
            r"\bcallees?\s+of\s+([A-Za-z_$][\w$]*(?:\(\))?)\b",
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

        value = value.strip(
            " \t\r\n.,;:!?\"'`()[]{}"
        )

        if value.endswith("()"):
            value = value[:-2]

        return value.strip()

    @staticmethod
    def _is_valid_method_candidate(
        method_name: str,
    ) -> bool:

        if not method_name:
            return False

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

        # -----------------------------------------------------------
        # Use the highest-ranked retrieved chunk as the main answer.
        # The frontend already displays all retrieved chunks below
        # the answer, so we should NOT duplicate them here.
        # -----------------------------------------------------------

        top_chunk = chunks[0]

        class_name = (
            top_chunk.get("class_name")
            or ""
        )

        method_name = (
            top_chunk.get("method_name")
            or ""
        )

        code = (
            top_chunk.get("code_content")
            or top_chunk.get("text_representation")
            or ""
        )

        # -----------------------------------------------------------
        # Identify the main symbol
        # -----------------------------------------------------------

        if class_name and method_name:
            symbol = f"{class_name}.{method_name}"
        elif method_name:
            symbol = method_name
        elif class_name:
            symbol = class_name
        else:
            symbol = "The retrieved code"

        # -----------------------------------------------------------
        # Special case: count queries
        # -----------------------------------------------------------

        count_words = (
            "count",
            "number",
            "how many",
            "matching documents",
            "matching records",
        )

        query_lower = query.lower()

        if (
            any(word in query_lower for word in count_words)
            and method_name.lower() == "count"
        ):
            if "search(" in code:
                return (
                    f"`{symbol}()` counts matching documents "
                    f"by searching for documents that satisfy "
                    f"the given condition and returning the "
                    f"number of results."
                )

            return (
                f"`{symbol}()` is responsible for counting "
                f"documents that match the given condition."
            )

        # -----------------------------------------------------------
        # Insert / add / create queries
        # -----------------------------------------------------------

        insert_words = (
            "insert",
            "add",
            "create a document",
            "add a document",
            "new record",
            "new document",
        )

        if (
            any(word in query_lower for word in insert_words)
            and method_name.lower() in {
                "insert",
                "insert_multiple",
            }
        ):
            return (
                f"`{symbol}()` handles adding documents "
                f"to the database."
            )

        # -----------------------------------------------------------
        # Search / find queries
        # -----------------------------------------------------------

        search_words = (
            "search",
            "find",
            "matching",
            "satisfy a condition",
            "satisfy the condition",
        )

        if (
            any(word in query_lower for word in search_words)
            and method_name.lower() == "search"
        ):
            return (
                f"`{symbol}()` finds documents that match "
                f"the supplied query or condition."
            )

        # -----------------------------------------------------------
        # Delete queries
        # -----------------------------------------------------------

        delete_words = (
            "delete",
            "remove",
            "removing",
        )

        if (
            any(word in query_lower for word in delete_words)
            and method_name.lower() in {
                "remove",
                "remove_multiple",
            }
        ):
            return (
                f"`{symbol}()` handles removing documents "
                f"from the database."
            )

        # -----------------------------------------------------------
        # Update queries
        # -----------------------------------------------------------

        update_words = (
            "update",
            "modify",
            "change",
        )

        if (
            any(word in query_lower for word in update_words)
            and method_name.lower() in {
                "update",
                "update_multiple",
                "upsert",
            }
        ):
            return (
                f"`{symbol}()` handles updating documents "
                f"that match the supplied condition."
            )

        # -----------------------------------------------------------
        # Generic method explanation
        # -----------------------------------------------------------

        if method_name:

            if code:

                first_line = ""

                for line in str(code).splitlines():
                    cleaned = line.strip()

                    if not cleaned:
                        continue

                    if cleaned.startswith("def "):
                        continue

                    if cleaned.startswith("class "):
                        continue

                    if cleaned.startswith("#"):
                        continue

                    first_line = cleaned
                    break

                if first_line:
                    return (
                        f"The most relevant implementation is "
                        f"`{symbol}()`. "
                        f"It contains the logic related to your query. "
                        f"The retrieved code shows: "
                        f"`{first_line}`"
                    )

            return (
                f"The most relevant implementation for your "
                f"query is `{symbol}()`."
            )

        # -----------------------------------------------------------
        # Final fallback
        # -----------------------------------------------------------

        return (
            "The indexed codebase contains relevant code for "
            f"your query. The most relevant result is `{symbol}`."
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