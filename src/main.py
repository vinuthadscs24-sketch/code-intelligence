import shutil
from pathlib import Path
from typing import Any, Optional
from collections import deque

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.repo_utils import clone_repo_if_url
from src.multi_ast_parser import CodeParserFactory
from src.chunker import CodeChunker
from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph
from src.git_intelligence import GitIntelligence
from src.context_builder import CodeIntelligenceContextBuilder
from src.llm_engine import CodeIntelligenceEngine
from src.query_router import QueryRouter, QueryIntent
from src.structured_response import StructuredResponse


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="AI Codebase Intelligence Engine",
    description=(
        "REST API for Hybrid RRF Code Retrieval, "
        "Knowledge Graph Queries, Impact Analysis, "
        "Git Provenance and LLM Reasoning."
    ),
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# GLOBAL ENGINE STATE
# =========================================================

class EngineState:
    def __init__(self):
        self.repo_path: Optional[Path] = None
        self.is_temp: bool = False
        self.parser: Optional[Any] = None
        self.chunker: Optional[CodeChunker] = None
        self.graph_db: Optional[CodeKnowledgeGraph] = None
        self.vector_store: Optional[VectorStore] = None
        self.git_intel: Optional[GitIntelligence] = None
        self.context_builder: Optional[
            CodeIntelligenceContextBuilder
        ] = None
        self.engine: Optional[CodeIntelligenceEngine] = None

        # Adaptive response routing
        self.query_router = QueryRouter()

        self.is_indexed: bool = False


state = EngineState()


# =========================================================
# REQUEST MODELS
# =========================================================

class IndexRequest(BaseModel):
    repo_path_or_url: Optional[str] = Field(
        default=None,
        example="https://github.com/user/repository.git",
    )

    repo_url: Optional[str] = None

    rebuild_index: bool = Field(
        default=False,
        description="Force rebuilding the FAISS index.",
    )

    @property
    def url(self) -> str:
        target = self.repo_path_or_url or self.repo_url

        if not target:
            raise ValueError(
                "Must provide either repo_path_or_url or repo_url."
            )

        return target


class QueryRequest(BaseModel):
    question: Optional[str] = Field(
        default=None,
        example="How does authentication work?",
    )

    query: Optional[str] = None

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    repo_id: Optional[str] = "default"

    @property
    def text(self) -> str:
        q = self.question or self.query

        if not q:
            raise ValueError(
                "Must provide either question or query field."
            )

        return q


class ProvenanceRequest(BaseModel):
    file: str
    method: str
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _normalise_graph_result(value):
    """
    Convert graph results into JSON-safe lists.

    Handles lists, tuples, sets, dictionaries and None.
    """

    if value is None:
        return []

    if isinstance(value, dict):
        return value

    if isinstance(value, (list, tuple, set)):
        return list(value)

    return [value]


def _call_graph_method(
    graph,
    method_name: str,
    entity_name: str,
):
    """
    Safely call a graph method.

    Some versions of graph_builder.py may return different
    structures, so this wrapper keeps the API stable.
    """

    method = getattr(
        graph,
        method_name,
        None,
    )

    if method is None:
        return []

    try:

        result = method(
            entity_name
        )

        return _normalise_graph_result(
            result
        )

    except TypeError:

        return []

    except Exception as exc:

        print(
            f"[Graph] {method_name} failed for "
            f"{entity_name}: {exc}"
        )

        return []


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "online",
        "indexed": state.is_indexed,
        "repo_path": (
            str(state.repo_path)
            if state.repo_path
            else None
        ),
    }


# =========================================================
# INDEX REPOSITORY
# =========================================================

@app.post("/api/repository/index")
@app.post("/index")
def index_repository(
    payload: IndexRequest,
):

    try:

        target_url = payload.url

        print(
            f"[Index] Starting repository indexing: "
            f"{target_url}"
        )

        # -------------------------------------------------
        # Cleanup previous temporary repository
        # -------------------------------------------------

        if (
            state.is_temp
            and state.repo_path
            and state.repo_path.exists()
        ):

            print(
                "[Index] Removing previous temporary repository..."
            )

            shutil.rmtree(
                state.repo_path,
                ignore_errors=True,
            )

        # -------------------------------------------------
        # Reset state
        # -------------------------------------------------

        state.is_indexed = False
        state.repo_path = None
        state.is_temp = False
        state.parser = None
        state.graph_db = None
        state.vector_store = None
        state.git_intel = None
        state.context_builder = None
        state.engine = None

        # -------------------------------------------------
        # Clone / load repository
        # -------------------------------------------------

        repo_path, is_temp = clone_repo_if_url(
            target_url
        )

        repo_path = Path(
            repo_path
        )

        if not repo_path.exists():

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Repository path does not exist: "
                    f"{repo_path}"
                ),
            )

        if not repo_path.is_dir():

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Repository path is not a directory: "
                    f"{repo_path}"
                ),
            )

        print(
            f"[Index] Repository path: {repo_path}"
        )

        # -------------------------------------------------
        # Repository name
        #
        # This is used for the FAISS index directory.
        # -------------------------------------------------

        repo_name = repo_path.name

        print(
            f"[Index] Repository name: {repo_name}"
        )

        # -------------------------------------------------
        # Supported source extensions
        # -------------------------------------------------

        supported_extensions = {
            ".java",
            ".dart",
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".kt",
            ".kts",
        }

        ignored_dirs = {
            ".git",
            ".idea",
            ".vscode",
            "node_modules",
            "build",
            ".dart_tool",
            "__pycache__",
            "venv",
            ".venv",
            "dist",
            "target",
            ".gradle",
        }

        # -------------------------------------------------
        # Parse source files
        # -------------------------------------------------

        extracted_data = []

        skipped_files = 0
        parsed_files = 0

        for source_file in repo_path.rglob("*"):

            if not source_file.is_file():
                continue

            parts_lower = {
                part.lower()
                for part in source_file.parts
            }

            if parts_lower.intersection(
                ignored_dirs
            ):
                continue

            extension = (
                source_file.suffix.lower()
            )

            if extension not in supported_extensions:
                continue

            if source_file.name in {
                "module-info.java",
                "package-info.java",
            }:
                continue

            # -------------------------------------------------
            # Select parser
            # -------------------------------------------------

            parser = CodeParserFactory.get_parser(
                str(source_file)
            )

            if parser is None:

                skipped_files += 1
                continue

            try:

                result = parser.parse_file(
                    str(source_file)
                )

                if not result:

                    skipped_files += 1
                    continue

                tree = result[0]

                source_code = (
                    result[1]
                    if len(result) > 1
                    else ""
                )

                if not source_code.strip():

                    skipped_files += 1
                    continue

                symbols = (
                    parser.extract_symbols_and_relations(
                        tree,
                        source_code,
                    )
                )

                extracted_data.append(
                    (
                        Path(source_file),
                        {
                            "tree": tree,
                            "source_code": source_code,
                            "symbols": symbols,
                        },
                    )
                )

                parsed_files += 1

            except Exception as exc:

                print(
                    f"[Parser] Skipping "
                    f"{source_file}: {exc}"
                )

                skipped_files += 1

        print(
            f"[Index] Parsed files: "
            f"{parsed_files}"
        )

        print(
            f"[Index] Skipped files: "
            f"{skipped_files}"
        )

        if not extracted_data:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No supported source files could be "
                    "parsed from the repository."
                ),
            )

        # -------------------------------------------------
        # Chunking
        # -------------------------------------------------

        print(
            "[Index] Creating code chunks..."
        )

        chunker = CodeChunker()

        chunks = chunker.create_chunks(
            extracted_data
        )

        print(
            f"[Index] Created {len(chunks)} chunks."
        )

        if not chunks:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No code chunks were generated."
                ),
            )

        # -------------------------------------------------
        # Knowledge Graph
        # -------------------------------------------------

        print(
            "[Index] Building knowledge graph..."
        )

        kg = CodeKnowledgeGraph()

        kg.build_graph_from_chunks(
            chunks
        )

        # -------------------------------------------------
        # Vector Store
        # -------------------------------------------------

        print(
            "[Index] Initializing vector store..."
        )

        store = VectorStore()

        index_loaded = False

        # -------------------------------------------------
        # Load existing index
        #
        # IMPORTANT:
        # VectorStore.load_index() requires repo_name.
        # -------------------------------------------------

        if not payload.rebuild_index:

            try:

                index_loaded = store.load_index(
                    repo_name
                )

            except Exception as exc:

                print(
                    "[Index] Existing index could not "
                    f"be loaded: {exc}"
                )

                index_loaded = False

        # -------------------------------------------------
        # Build new FAISS index
        # -------------------------------------------------

        if (
            payload.rebuild_index
            or not index_loaded
        ):

            print(
                "[Index] Building FAISS vector index..."
            )

            store.build_index(
                chunks
            )

            # IMPORTANT:
            # save_index() requires repo_name.
            store.save_index(
                repo_name
            )

            print(
                "[Index] FAISS vector index saved."
            )

        else:

            print(
                "[Index] Existing FAISS index loaded."
            )

        # -------------------------------------------------
        # Verify vector store
        # -------------------------------------------------

        vector_stats = store.get_stats()

        print(
            "[Index] Vector store stats: "
            f"{vector_stats}"
        )

        # -------------------------------------------------
        # Git intelligence
        # -------------------------------------------------

        print(
            "[Index] Initializing Git intelligence..."
        )

        git_intel = GitIntelligence(
            repo_path=str(
                repo_path
            )
        )

        context_builder = (
            CodeIntelligenceContextBuilder(
                git_intel=git_intel
            )
        )

        # -------------------------------------------------
        # LLM engine
        # -------------------------------------------------

        print(
            "[Index] Initializing LLM engine..."
        )

        engine = CodeIntelligenceEngine(
            repo_path=str(
                repo_path
            ),
            vector_store=store,
            graph_db=kg,
            context_builder=context_builder,
        )

        # -------------------------------------------------
        # Update global state
        # -------------------------------------------------

        state.repo_path = repo_path
        state.is_temp = is_temp
        state.parser = None
        state.chunker = chunker
        state.graph_db = kg
        state.vector_store = store
        state.git_intel = git_intel
        state.context_builder = context_builder
        state.engine = engine
        state.is_indexed = True

        # -------------------------------------------------
        # Graph summary
        # -------------------------------------------------

        try:

            summary = kg.get_summary()

        except Exception as exc:

            print(
                f"[Index] Graph summary failed: "
                f"{exc}"
            )

            summary = {
                "total_nodes": 0,
                "total_edges": 0,
            }

        # -------------------------------------------------
        # Success response
        # -------------------------------------------------

        return {
            "status": "success",
            "message": (
                "Repository indexed successfully."
            ),
            "repoName": repo_name,
            "total_files": len(
                extracted_data
            ),
            "total_chunks": len(
                chunks
            ),
            "vector_stats": vector_stats,
            "graph_summary": summary,
        }

    except HTTPException:

        raise

    except ValueError as exc:

        state.is_indexed = False

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        state.is_indexed = False

        print(
            f"[Index ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Indexing failed: {str(exc)}"
            ),
        )


# =========================================================
# ADAPTIVE QUERY RESPONSE
# =========================================================


def _safe_name(value: Any) -> str:
    """
    Convert graph values into a stable display name.
    """
    if isinstance(value, dict):
        return (
            value.get("name")
            or value.get("entity")
            or value.get("id")
            or value.get("method")
            or str(value)
        )

    return str(value)


def _build_call_graph_response(
    query: str,
    target: str,
) -> StructuredResponse:
    """Build a structured call-graph response using the knowledge graph."""

    graph = state.graph_db

    if not graph:
        return StructuredResponse(
            query=query,
            response_type=QueryIntent.CALL_GRAPH.value,
            answer="Knowledge graph is not available.",
            data={},
            evidence=[],
        )

    callers = _call_graph_method(
        graph,
        "get_callers_of",
        target,
    )

    callees = _call_graph_method(
        graph,
        "get_calls_from",
        target,
    )

    nodes = [
        {
            "id": target,
            "label": target,
            "type": "method",
        }
    ]

    edges = []

    for caller in callers:
        caller_name = _safe_name(caller)
        if not caller_name:
            continue

        nodes.append(
            {
                "id": caller_name,
                "label": caller_name,
                "type": "method",
            }
        )

        edges.append(
            {
                "source": caller_name,
                "target": target,
                "type": "CALLS",
            }
        )

    for callee in callees:
        callee_name = _safe_name(callee)
        if not callee_name:
            continue

        nodes.append(
            {
                "id": callee_name,
                "label": callee_name,
                "type": "method",
            }
        )

        edges.append(
            {
                "source": target,
                "target": callee_name,
                "type": "CALLS",
            }
        )

    unique_nodes = {}
    for node in nodes:
        unique_nodes[node["id"]] = node

    unique_edges = []
    seen_edges = set()

    for edge in edges:
        edge_key = (
            edge["source"],
            edge["target"],
            edge["type"],
        )

        if edge_key in seen_edges:
            continue

        seen_edges.add(edge_key)
        unique_edges.append(edge)

    if callers and callees:
        answer = (
            f"{target} is called by {len(callers)} method(s) "
            f"and calls {len(callees)} method(s)."
        )
    elif callers:
        answer = f"{target} is called by {len(callers)} method(s)."
    elif callees:
        answer = f"{target} calls {len(callees)} method(s)."
    else:
        answer = (
            f"No direct caller or callee relationships were found "
            f"for {target}."
        )

    return StructuredResponse(
        query=query,
        response_type=QueryIntent.CALL_GRAPH.value,
        answer=answer,
        data={
            "target": target,
            "nodes": list(unique_nodes.values()),
            "edges": unique_edges,
            "callers": [_safe_name(caller) for caller in callers],
            "callees": [_safe_name(callee) for callee in callees],
        },
        evidence=[],
    )


def _build_impact_response(
    query: str,
    target: str,
) -> StructuredResponse:
    """Build a structured impact-analysis response."""

    graph = state.graph_db

    if not graph:
        return StructuredResponse(
            query=query,
            response_type=QueryIntent.IMPACT.value,
            answer="Knowledge graph is not available.",
            data={},
            evidence=[],
        )

    max_depth = 3
    visited = {target}
    queue = deque([(target, 0)])
    affected_entities = []

    while queue:
        current_entity, distance = queue.popleft()

        if distance >= max_depth:
            continue

        callers = _call_graph_method(
            graph,
            "get_callers_of",
            current_entity,
        )

        for caller in callers:
            caller_name = _safe_name(caller)

            if not caller_name or caller_name in visited:
                continue

            visited.add(caller_name)
            next_distance = distance + 1

            affected_entities.append(
                {
                    "entity": caller_name,
                    "distance": next_distance,
                }
            )

            queue.append((caller_name, next_distance))

    return StructuredResponse(
        query=query,
        response_type=QueryIntent.IMPACT.value,
        answer=(
            f"Changing {target} could directly or indirectly affect "
            f"{len(affected_entities)} method(s) within "
            f"{max_depth} call level(s)."
        ),
        data={
            "target": target,
            "max_depth": max_depth,
            "affected": affected_entities,
        },
        evidence=[],
    )


def _run_rag_response(
    query: str,
    top_k: int,
):
    """Run the existing RAG engine without changing its behavior."""

    response = state.engine.answer_query(
        query,
        top_k=top_k,
    )

    if isinstance(response, dict):
        answer = response.get(
            "answer",
            "No response generated.",
        )
        retrieved_chunks = response.get(
            "retrieved_chunks",
            [],
        )
        context_used = response.get(
            "context",
            {},
        )
    else:
        answer = str(response)
        retrieved_chunks = []
        context_used = {}

    return answer, retrieved_chunks, context_used


def _build_retrieval_trace_response(
    query: str,
    top_k: int,
) -> StructuredResponse:
    """Return the normal RAG answer plus retrieval evidence."""

    answer, retrieved_chunks, context_used = _run_rag_response(
        query,
        top_k,
    )

    evidence = (
        retrieved_chunks
        if isinstance(retrieved_chunks, list)
        else []
    )

    return StructuredResponse(
        query=query,
        response_type=QueryIntent.RETRIEVAL_TRACE.value,
        answer=answer,
        data={
            "retrieved_chunks": retrieved_chunks,
            "context_used": context_used,
        },
        evidence=evidence,
    )


def _build_standard_answer_response(
    query: str,
    top_k: int,
) -> StructuredResponse:
    """Preserve the existing RAG answer behavior."""

    answer, retrieved_chunks, context_used = _run_rag_response(
        query,
        top_k,
    )

    evidence = (
        retrieved_chunks
        if isinstance(retrieved_chunks, list)
        else []
    )

    return StructuredResponse(
        query=query,
        response_type=QueryIntent.ANSWER.value,
        answer=answer,
        data={
            "retrieved_chunks": retrieved_chunks,
            "context_used": context_used,
        },
        evidence=evidence,
    )


def _build_git_history_response(
    query: str,
    target: Optional[str],
    top_k: int,
) -> StructuredResponse:
    """
    Preserve the existing Git/RAG reasoning behavior while exposing
    the response through the structured response contract.
    """

    answer, retrieved_chunks, context_used = _run_rag_response(
        query,
        top_k,
    )

    evidence = (
        retrieved_chunks
        if isinstance(retrieved_chunks, list)
        else []
    )

    return StructuredResponse(
        query=query,
        response_type=QueryIntent.GIT_HISTORY.value,
        answer=answer,
        data={
            "target": target,
            "retrieved_chunks": retrieved_chunks,
            "context_used": context_used,
        },
        evidence=evidence,
    )


def _build_flow_response(
    query: str,
    target: Optional[str],
    top_k: int,
) -> StructuredResponse:
    """
    Build a flow response from actual graph relationships.

    For a generic flow question with no explicit target, the existing
    RAG engine is used to preserve semantic/business-flow reasoning.
    For a targeted flow question, direct graph caller/callee edges are
    exposed as structured steps and edges.
    """

    graph = state.graph_db

    if not graph:
        return StructuredResponse(
            query=query,
            response_type=QueryIntent.FLOW.value,
            answer="Knowledge graph is not available.",
            data={},
            evidence=[],
        )

    # ---------------------------------------------------------
    # Generic flow question
    # ---------------------------------------------------------

    if not target:
        answer, retrieved_chunks, context_used = _run_rag_response(
            query,
            top_k,
        )

        evidence = (
            retrieved_chunks
            if isinstance(retrieved_chunks, list)
            else []
        )

        return StructuredResponse(
            query=query,
            response_type=QueryIntent.FLOW.value,
            answer=answer,
            data={
                "steps": [],
                "edges": [],
                "mode": "semantic",
                "context_used": context_used,
            },
            evidence=evidence,
        )

    # ---------------------------------------------------------
    # Targeted flow question
    # ---------------------------------------------------------

    steps = [
        {
            "id": target,
            "label": target,
            "type": "method",
        }
    ]

    edges = []

    callers = _call_graph_method(
        graph,
        "get_callers_of",
        target,
    )

    callees = _call_graph_method(
        graph,
        "get_calls_from",
        target,
    )

    for caller in callers:
        caller_name = _safe_name(caller)
        if not caller_name:
            continue

        steps.append(
            {
                "id": caller_name,
                "label": caller_name,
                "type": "method",
            }
        )

        edges.append(
            {
                "source": caller_name,
                "target": target,
                "type": "CALLS",
            }
        )

    for callee in callees:
        callee_name = _safe_name(callee)
        if not callee_name:
            continue

        steps.append(
            {
                "id": callee_name,
                "label": callee_name,
                "type": "method",
            }
        )

        edges.append(
            {
                "source": target,
                "target": callee_name,
                "type": "CALLS",
            }
        )

    unique_steps = {}
    for step in steps:
        unique_steps[step["id"]] = step

    unique_edges = []
    seen_edges = set()

    for edge in edges:
        edge_key = (
            edge["source"],
            edge["target"],
            edge["type"],
        )

        if edge_key in seen_edges:
            continue

        seen_edges.add(edge_key)
        unique_edges.append(edge)

    answer = (
        f"The flow around {target} contains "
        f"{len(unique_steps)} connected method(s) "
        f"through direct caller/callee relationships."
    )

    return StructuredResponse(
        query=query,
        response_type=QueryIntent.FLOW.value,
        answer=answer,
        data={
            "target": target,
            "steps": list(unique_steps.values()),
            "edges": unique_edges,
            "mode": "graph",
        },
        evidence=[],
    )


@app.post("/api/query", response_model=StructuredResponse)
@app.post("/ask", response_model=StructuredResponse)
def query_codebase(
    payload: QueryRequest,
):

    if (
        not state.is_indexed
        or not state.engine
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "No repository indexed. "
                "Call POST /index first."
            ),
        )

    try:

        q_text = payload.text

        # -----------------------------------------------------
        # 1. Route the query by intent
        # -----------------------------------------------------

        route = state.query_router.route(q_text)

        print(
            f"[Query Router] intent={route.intent.value} "
            f"target={route.target}"
        )

        # -----------------------------------------------------
        # 2. Build the appropriate structured response
        # -----------------------------------------------------

        if route.intent == QueryIntent.CALL_GRAPH:

            if route.target:
                result = _build_call_graph_response(
                    q_text,
                    route.target,
                )
            else:
                result = _build_standard_answer_response(
                    q_text,
                    payload.top_k,
                )

        elif route.intent == QueryIntent.IMPACT:

            if route.target:
                result = _build_impact_response(
                    q_text,
                    route.target,
                )
            else:
                result = _build_standard_answer_response(
                    q_text,
                    payload.top_k,
                )

        elif route.intent == QueryIntent.GIT_HISTORY:

            result = _build_git_history_response(
                q_text,
                route.target,
                payload.top_k,
            )

        elif route.intent == QueryIntent.FLOW:

            result = _build_flow_response(
                q_text,
                route.target,
                payload.top_k,
            )

        elif route.intent == QueryIntent.RETRIEVAL_TRACE:

            result = _build_retrieval_trace_response(
                q_text,
                payload.top_k,
            )

        else:

            result = _build_standard_answer_response(
                q_text,
                payload.top_k,
            )

        # -----------------------------------------------------
        # 3. Return the Pydantic structured response
        # -----------------------------------------------------

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        print(
            f"[Query ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Query failed: {str(exc)}"
            ),
        )


# =========================================================
# DEPENDENCIES
# =========================================================

@app.get(
    "/dependencies/{entity_name}"
)
def get_entity_dependencies(
    entity_name: str,
):

    if (
        not state.is_indexed
        or not state.graph_db
    ):

        raise HTTPException(
            status_code=400,
            detail="No repository indexed.",
        )

    try:

        graph = state.graph_db

        # -------------------------------------------------
        # Callers
        # -------------------------------------------------

        callers = _call_graph_method(
            graph,
            "get_callers_of",
            entity_name,
        )

        # -------------------------------------------------
        # Callees
        # -------------------------------------------------

        callees = _call_graph_method(
            graph,
            "get_calls_from",
            entity_name,
        )

        return {
            "entity": entity_name,
            "callers": callers,
            "callees": callees,
        }

    except Exception as exc:

        print(
            f"[Dependencies ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Dependency lookup failed: "
                f"{str(exc)}"
            ),
        )


# =========================================================
# IMPACT ANALYSIS
# =========================================================

@app.get(
    "/impact/{entity_name}"
)
def get_impact_analysis(
    entity_name: str,
    max_depth: int = Query(
        default=3,
        ge=1,
        le=5,
    ),
):

    if (
        not state.is_indexed
        or not state.graph_db
    ):

        raise HTTPException(
            status_code=400,
            detail="No repository indexed.",
        )

    try:

        graph = state.graph_db

        visited = {
            entity_name
        }

        queue = deque(
            [
                (
                    entity_name,
                    0,
                )
            ]
        )

        affected_entities = []

        while queue:

            current_entity, distance = (
                queue.popleft()
            )

            if distance >= max_depth:
                continue

            callers = _call_graph_method(
                graph,
                "get_callers_of",
                current_entity,
            )

            for caller in callers:

                if isinstance(
                    caller,
                    dict,
                ):

                    caller_name = (
                        caller.get("name")
                        or caller.get("entity")
                        or caller.get("id")
                    )

                else:

                    caller_name = str(
                        caller
                    )

                if not caller_name:
                    continue

                if caller_name in visited:
                    continue

                visited.add(
                    caller_name
                )

                affected_entities.append(
                    {
                        "entity": caller_name,
                        "distance": (
                            distance + 1
                        ),
                    }
                )

                queue.append(
                    (
                        caller_name,
                        distance + 1,
                    )
                )

        return {
            "target_entity": entity_name,
            "max_depth": max_depth,
            "total_affected": len(
                affected_entities
            ),
            "affected": affected_entities,
        }

    except Exception as exc:

        print(
            f"[Impact ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Impact analysis failed: "
                f"{str(exc)}"
            ),
        )


# =========================================================
# GIT PROVENANCE
# =========================================================

@app.post(
    "/history/why-changed"
)
def explain_method_provenance(
    payload: ProvenanceRequest,
):

    if (
        not state.is_indexed
        or not state.engine
    ):

        raise HTTPException(
            status_code=400,
            detail="No repository indexed.",
        )

    try:

        response = (
            state.engine.explain_why_changed(
                file_path=payload.file,
                method_name=payload.method,
                start_line=payload.start_line,
                end_line=payload.end_line,
            )
        )

        return response

    except Exception as exc:

        print(
            f"[Provenance ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Provenance lookup failed: "
                f"{str(exc)}"
            ),
        )


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )