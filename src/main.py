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


def _call_graph_method(graph, method_name: str, entity_name: str):
    """
    Safely call a graph method.

    Some versions of graph_builder.py may return different
    structures, so this wrapper keeps the API stable.
    """
    method = getattr(graph, method_name, None)

    if method is None:
        return []

    try:
        result = method(entity_name)
        return _normalise_graph_result(result)
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
def index_repository(payload: IndexRequest):

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

        repo_path = Path(repo_path)

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

            if parts_lower.intersection(ignored_dirs):
                continue

            extension = source_file.suffix.lower()

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
            f"[Index] Parsed files: {parsed_files}"
        )

        print(
            f"[Index] Skipped files: {skipped_files}"
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
                detail="No code chunks were generated.",
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

        if not payload.rebuild_index:
            try:
                index_loaded = store.load_index()
            except Exception as exc:
                print(
                    f"[Index] Existing index could not "
                    f"be loaded: {exc}"
                )
                index_loaded = False

        if payload.rebuild_index or not index_loaded:

            print(
                "[Index] Building FAISS vector index..."
            )

            store.build_index(
                chunks
            )

            store.save_index()

        else:

            print(
                "[Index] Existing FAISS index loaded."
            )

        # -------------------------------------------------
        # Git intelligence
        # -------------------------------------------------

        print(
            "[Index] Initializing Git intelligence..."
        )

        git_intel = GitIntelligence(
            repo_path=str(repo_path)
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
            repo_path=str(repo_path),
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
                f"[Index] Graph summary failed: {exc}"
            )

            summary = {
                "total_nodes": 0,
                "total_edges": 0,
            }

        return {
            "status": "success",
            "message": "Repository indexed successfully.",
            "repoName": repo_path.name,
            "total_files": len(extracted_data),
            "total_chunks": len(chunks),
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
            detail=f"Indexing failed: {str(exc)}",
        )


# =========================================================
# QUERY
# =========================================================

@app.post("/api/query")
@app.post("/ask")
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

        response = state.engine.answer_query(
            q_text,
            top_k=payload.top_k,
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

        return {
            "query": q_text,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "context_used": context_used,
        }

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
            detail=f"Query failed: {str(exc)}",
        )


# =========================================================
# DEPENDENCIES
# =========================================================

@app.get("/dependencies/{entity_name}")
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

@app.get("/impact/{entity_name}")
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

                # Graph results may sometimes be dictionaries.
                if isinstance(caller, dict):

                    caller_name = (
                        caller.get("name")
                        or caller.get("entity")
                        or caller.get("id")
                    )

                else:

                    caller_name = str(caller)

                if not caller_name:
                    continue

                if caller_name in visited:
                    continue

                visited.add(caller_name)

                affected_entities.append(
                    {
                        "entity": caller_name,
                        "distance": distance + 1,
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

@app.post("/history/why-changed")
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