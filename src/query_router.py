
from dataclasses import dataclass
from enum import Enum
import re
from typing import Optional


class QueryIntent(str, Enum):
    CALL_GRAPH = "call_graph"
    IMPACT = "impact"
    GIT_HISTORY = "git_history"
    FLOW = "flow"
    RETRIEVAL_TRACE = "retrieval_trace"
    ANSWER = "answer"


@dataclass
class QueryRoute:
    intent: QueryIntent
    target: Optional[str] = None


class QueryRouter:
    """
    Determines which intelligence mode should answer a query.

    The router only classifies the user's intent and extracts
    an optional method/entity target. It does not execute
    graph, Git, retrieval, or LLM logic.
    """

    # =========================================================
    # CALLER PATTERNS
    # =========================================================

    CALLER_PATTERNS = [
        # Who calls Table.insert()?
        r"\bwho\s+calls\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)",

        # What calls Table.insert()?
        r"\bwhat\s+calls\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)",

        # Callers of Table.insert()
        r"\bcallers?\s+of\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)",

        # Which methods call Table.insert()?
        r"\bwhich\s+(?:methods?|functions?|classes?)\s+call\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)",
    ]

    # =========================================================
    # CALLEE PATTERNS
    # =========================================================

    CALLEE_PATTERNS = [
        # What does Table.insert() call?
        r"\bwhat\s+does\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\s+call\b",

        # What methods does Table.insert() call?
        r"\bwhat\s+(?:methods?|functions?)\s+does\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\s+call\b",

        # Which methods does Table.insert() call?
        r"\bwhich\s+(?:methods?|functions?)\s+does\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\s+call\b",

        # Callees of Table.insert()
        r"\bcallees?\s+of\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)",

        # What methods are called by Table.insert()?
        r"\bwhat\s+(?:methods?|functions?)\s+(?:are|is)\s+called\s+by\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)",

        # Which methods are called by Table.insert()?
        r"\bwhich\s+(?:methods?|functions?)\s+(?:are|is)\s+called\s+by\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)",
    ]

    # =========================================================
    # IMPACT PATTERNS
    # =========================================================

    IMPACT_PATTERNS = [
        r"\bwhat\s+(?:will\s+be\s+)?affected\b",
        r"\bwhat\s+(?:could\s+)?break\b",
        r"\bwhat\s+breaks\b",
        r"\bimpact\b",
        r"\bblast\s+radius\b",
        r"\bdependents?\b",
        r"\bdependencies\b",
        r"\baffected\s+(?:methods?|functions?|classes?|code)\b",
    ]

    # =========================================================
    # GIT HISTORY PATTERNS
    # =========================================================

    GIT_PATTERNS = [
        r"\bwhy\s+was\b",
        r"\bwhy\s+did\b",
        r"\bwhy\s+has\b",
        r"\bwhy\s+was\s+.+\s+changed\b",
        r"\bwhy\s+did\s+.+\s+change\b",
        r"\bwhat\s+changed\b",
        r"\bchange\s+history\b",
        r"\bgit\s+history\b",
        r"\bcommit\s+history\b",
        r"\bprovenance\b",
        r"\bwhen\s+was\s+.+\s+changed\b",
    ]

    # =========================================================
    # FLOW PATTERNS
    # =========================================================

    FLOW_PATTERNS = [
        r"\bhow\s+does\s+.+\s+work\b",
        r"\bhow\s+does\s+.+\s+flow\b",
        r"\bhow\s+.+\s+flow\b",
        r"\bexplain\s+the\s+flow\b",
        r"\bexplain\s+how\s+.+\s+works\b",
        r"\bwalk\s+me\s+through\b",
        r"\bend[- ]to[- ]end\b",
        r"\bexecution\s+flow\b",
        r"\brequest\s+flow\b",
        r"\bdata\s+flow\b",
    ]

    # =========================================================
    # RETRIEVAL TRACE PATTERNS
    # =========================================================

    RETRIEVAL_PATTERNS = [
        r"\bhow\s+did\s+you\s+find\b",
        r"\bhow\s+did\s+you\s+arrive\b",
        r"\bwhat\s+sources\b",
        r"\bshow\s+sources\b",
        r"\bshow\s+evidence\b",
        r"\bretrieval\b",
        r"\bretrieved\b",
        r"\bwhy\s+is\s+this\s+relevant\b",
    ]

    # =========================================================
    # STOPWORDS
    # =========================================================

    STOPWORDS = {
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

    # =========================================================
    # MAIN ROUTER
    # =========================================================

    @classmethod
    def route(cls, query: str) -> QueryRoute:

        query = (query or "").strip()

        if not query:
            return QueryRoute(QueryIntent.ANSWER)

        # =====================================================
        # CALLER QUERIES
        # =====================================================

        target = cls._extract_target(
            query,
            cls.CALLER_PATTERNS,
        )

        if target:
            return QueryRoute(
                QueryIntent.CALL_GRAPH,
                target,
            )

        # =====================================================
        # CALLEE QUERIES
        # =====================================================

        target = cls._extract_target(
            query,
            cls.CALLEE_PATTERNS,
        )

        if target:
            return QueryRoute(
                QueryIntent.CALL_GRAPH,
                target,
            )

        # =====================================================
        # IMPACT QUERIES
        # =====================================================

        if cls._matches_any(
            query,
            cls.IMPACT_PATTERNS,
        ):
            return QueryRoute(
                QueryIntent.IMPACT,
                cls._extract_entity_target(query),
            )

        # =====================================================
        # GIT HISTORY QUERIES
        # =====================================================

        if cls._matches_any(
            query,
            cls.GIT_PATTERNS,
        ):
            return QueryRoute(
                QueryIntent.GIT_HISTORY,
                cls._extract_entity_target(query),
            )

        # =====================================================
        # RETRIEVAL TRACE QUERIES
        # =====================================================

        if cls._matches_any(
            query,
            cls.RETRIEVAL_PATTERNS,
        ):
            return QueryRoute(
                QueryIntent.RETRIEVAL_TRACE,
            )

        # =====================================================
        # FLOW QUERIES
        # =====================================================

        if cls._matches_any(
            query,
            cls.FLOW_PATTERNS,
        ):
            return QueryRoute(
                QueryIntent.FLOW,
                cls._extract_flow_target(query),
            )

        # =====================================================
        # NORMAL RAG ANSWER
        # =====================================================

        return QueryRoute(QueryIntent.ANSWER)

    # =========================================================
    # EXTRACT TARGET
    # =========================================================

    @classmethod
    def _extract_target(
        cls,
        query: str,
        patterns: list[str],
    ) -> Optional[str]:

        for pattern in patterns:

            match = re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            candidate = cls._clean_target(
                match.group(1)
            )

            if cls._valid_target(candidate):
                return candidate

        return None

    # =========================================================
    # MATCH ANY
    # =========================================================

    @staticmethod
    def _matches_any(
        query: str,
        patterns: list[str],
    ) -> bool:

        return any(
            re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        )

    # =========================================================
    # EXTRACT FLOW TARGET
    # =========================================================

    @classmethod
    def _extract_flow_target(
        cls,
        query: str,
    ) -> Optional[str]:

        patterns = [
            r"\bflow\s+(?:of|for)\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)",

            r"\b(?:how|explain)\s+does\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\s+flow\b",

            r"\b(?:how|explain)\s+does\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\s+work\b",

            r"\bflow\s+(?:through|from)\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            candidate = cls._clean_target(
                match.group(1)
            )

            if cls._valid_target(candidate):
                return candidate

        return None

    # =========================================================
    # EXTRACT ENTITY TARGET
    # =========================================================

    @classmethod
    def _extract_entity_target(
        cls,
        query: str,
    ) -> Optional[str]:

        contextual_patterns = [
            r"\b(?:change|changing|modify|modifying|edit|editing)\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)",

            r"\b(?:impact|affect|affects|affected|breaks|break)\s+(?:of\s+)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)",

            r"\b(?:about|of|for|on)\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)",

            r"\b(?:flow|workflow)\s+(?:of|for)\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)",
        ]

        for pattern in contextual_patterns:

            match = re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            candidate = cls._clean_target(
                match.group(1)
            )

            if cls._valid_target(candidate):
                return candidate

        # Git questions such as:
        # "Why was createBooking changed?"

        git_patterns = [
            r"\b(?:was|were)\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\s+changed\b",

            r"\b(?:did)\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\s+change\b",

            r"\b(?:has|have)\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\s+changed\b",
        ]

        for pattern in git_patterns:

            match = re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            candidate = cls._clean_target(
                match.group(1)
            )

            if cls._valid_target(candidate):
                return candidate

        # Last-resort scan for an explicit code-style identifier.

        candidates = re.findall(
            r"\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?\b",
            query,
        )

        for candidate in candidates:

            if cls._valid_target(candidate):
                return candidate

        return None

    # =========================================================
    # CLEAN TARGET
    # =========================================================

    @classmethod
    def _clean_target(
        cls,
        value: str,
    ) -> str:

        value = str(value).strip()

        # Table.insert() -> Table.insert
        if value.endswith("()"):
            value = value[:-2]

        return value.strip(
            " \t\r\n.,;:!?\"'`()[]{}"
        )

    # =========================================================
    # VALID TARGET
    # =========================================================

    @classmethod
    def _valid_target(
        cls,
        value: Optional[str],
    ) -> bool:

        if not value:
            return False

        value = value.strip()

        if not value:
            return False

        if value.lower() in cls.STOPWORDS:
            return False

        return bool(
            re.fullmatch(
                r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?",
                value,
            )
        )

