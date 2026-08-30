import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware

# Import core engine modules
from src.repo_utils import clone_repo_if_url
from src.parser import JavaASTParser
from src.chunker import CodeChunker
from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph
from src.git_intelligence import GitIntelligence
from src.context_builder import CodeIntelligenceContextBuilder
from src.llm_engine import CodeIntelligenceEngine


app = FastAPI(
    title="AI Codebase Intelligence Engine",
    description="REST API for Hybrid RRF Code Retrieval, Knowledge Graph Queries, Transitive Impact Analysis, and LLM Reasoning.",
    version="1.0.0"
)

# Enable CORS for React/Tailwind frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container
class EngineState:
    repo_path: Optional[Path] = None
    is_temp: bool = False
    parser: Optional[JavaASTParser] = None
    chunker: Optional[CodeChunker] = None
    graph_db: Optional[CodeKnowledgeGraph] = None
    vector_store: Optional[VectorStore] = None
    git_intel: Optional[GitIntelligence] = None
    context_builder: Optional[CodeIntelligenceContextBuilder] = None
    engine: Optional[CodeIntelligenceEngine] = None
    is_indexed: bool = False


state = EngineState()


# --- Request/Response Models ---

class IndexRequest(BaseModel):
    repo_path_or_url: Optional[str] = Field(None, example="https://github.com/spring-projects/spring-petclinic.git")
    repo_url: Optional[str] = None
    rebuild_index: bool = Field(False, description="Force rebuilding FAISS vector index")

    @property
    def url(self) -> str:
        target = self.repo_path_or_url or self.repo_url
        if not target:
            raise ValueError("Must provide either repo_path_or_url or repo_url.")
        return target


class QueryRequest(BaseModel):
    question: Optional[str] = Field(None, example="How does user authentication work in this codebase?")
    query: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)
    repo_id: Optional[str] = "default"

    @property
    def text(self) -> str:
        q = self.question or self.query
        if not q:
            raise ValueError("Must provide either question or query field.")
        return q


class ProvenanceRequest(BaseModel):
    file: str = Field(..., example="src/main/java/com/example/UserService.java")
    method: str = Field(..., example="createUser")
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)


# --- API Endpoints ---

@app.get("/health")
def health_check():
    """Returns engine indexing and repository readiness status."""
    return {
        "status": "online",
        "indexed": state.is_indexed,
        "repo_path": str(state.repo_path) if state.repo_path else None
    }


@app.post("/api/repository/index")
@app.post("/index")
def index_repository(payload: IndexRequest):
    """
    Clones (if URL) or loads a local Java repository, parses ASTs, builds
    the Knowledge Graph, and populates FAISS vector embeddings.
    """
    try:
        target_url = payload.url
        
        # Clean up existing temporary repository if active
        if state.is_temp and state.repo_path and state.repo_path.exists():
            shutil.rmtree(state.repo_path, ignore_errors=True)

        repo_path, is_temp = clone_repo_if_url(target_url)
        if not repo_path.exists() or not repo_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Directory '{repo_path}' does not exist.")

        # 1. Parse AST
        parser = JavaASTParser()
        extracted_data = []
        for java_file in repo_path.rglob("*.java"):
            if java_file.name in {"module-info.java", "package-info.java"}:
                continue
            try:
                result = parser.parse_file(str(java_file))
                if result:
                    if isinstance(result, dict):
                        extracted_data.append((Path(java_file), result))
                    elif isinstance(result, (tuple, list)):
                        tree = result[0]
                        source_code = result[1] if len(result) > 1 else ""
                        symbols = parser.extract_symbols_and_relations(tree, source_code)
                        extracted_data.append((Path(java_file), {
                            "tree": tree,
                            "source_code": source_code,
                            "symbols": symbols
                        }))
            except Exception:
                pass

        if not extracted_data:
            raise HTTPException(status_code=400, detail="No valid Java files found or parsed.")

        # 2. Chunking
        chunker = CodeChunker(parser=parser)
        chunks = chunker.create_chunks(extracted_data)

        # 3. Knowledge Graph
        kg = CodeKnowledgeGraph()
        kg.build_graph_from_chunks(chunks)

        # 4. Vector Store
        store = VectorStore()
        if payload.rebuild_index or not store.load_index():
            store.build_index(chunks)
            store.save_index()

        # 5. Git & Context Engine
        git_intel = GitIntelligence(repo_path=str(repo_path))
        context_builder = CodeIntelligenceContextBuilder(git_intel=git_intel)
        engine = CodeIntelligenceEngine(
            repo_path=str(repo_path),
            vector_store=store,
            graph_db=kg,
            context_builder=context_builder
        )

        # Update State
        state.repo_path = repo_path
        state.is_temp = is_temp
        state.parser = parser
        state.chunker = chunker
        state.graph_db = kg
        state.vector_store = store
        state.git_intel = git_intel
        state.context_builder = context_builder
        state.engine = engine
        state.is_indexed = True

        summary = kg.get_summary()
        return {
            "status": "success",
            "message": "Repository indexed successfully.",
            "repoName": repo_path.name,
            "total_files": len(extracted_data),
            "total_chunks": len(chunks),
            "graph_summary": summary
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/api/query")
@app.post("/ask")
def query_codebase(payload: QueryRequest):
    """
    Unified end-to-end Q&A endpoint.
    Performs Hybrid RRF search, builds contextual prompts, and queries LLM.
    """
    if not state.is_indexed or not state.engine:
        raise HTTPException(status_code=400, detail="No repository indexed. Call POST /index first.")

    try:
        q_text = payload.text
        response = state.engine.answer_query(q_text, top_k=payload.top_k)
        return {
            "query": q_text,
            "answer": response.get("answer", "No response generated."),
            "retrieved_chunks": response.get("retrieved_chunks", []),
            "context_used": response.get("context", {})
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/dependencies/{entity_name}")
def get_entity_dependencies(entity_name: str):
    """Retrieves callers, callees, and imports for a class or method entity."""
    if not state.is_indexed or not state.graph_db:
        raise HTTPException(status_code=400, detail="No repository indexed.")

    callers = state.graph_db.get_callers(entity_name)
    callees = state.graph_db.get_callees(entity_name)

    return {
        "entity": entity_name,
        "callers": callers,
        "callees": callees
    }


@app.get("/impact/{entity_name}")
def get_impact_analysis(
    entity_name: str, 
    max_depth: int = Query(3, ge=1, le=5)
):
    """Calculates multi-level transitive impact analysis using BFS across the Knowledge Graph."""
    if not state.is_indexed or not state.graph_db:
        raise HTTPException(status_code=400, detail="No repository indexed.")

    from collections import deque
    visited = {entity_name}
    queue = deque([(entity_name, 0)])
    affected_entities = []

    while queue:
        current_entity, distance = queue.popleft()
        if distance >= max_depth:
            continue

        callers = state.graph_db.get_callers(current_entity)
        for caller in callers:
            if caller not in visited:
                visited.add(caller)
                affected_entities.append({
                    "entity": caller,
                    "distance": distance + 1
                })
                queue.append((caller, distance + 1))

    return {
        "target_entity": entity_name,
        "max_depth": max_depth,
        "total_affected": len(affected_entities),
        "affected": affected_entities
    }


@app.post("/history/why-changed")
def explain_method_provenance(payload: ProvenanceRequest):
    """Explains why a specific method/line range changed using Git blame, show, and diff context."""
    if not state.is_indexed or not state.engine:
        raise HTTPException(status_code=400, detail="No repository indexed.")

    try:
        response = state.engine.explain_why_changed(
            file_path=payload.file,
            method_name=payload.method,
            start_line=payload.start_line,
            end_line=payload.end_line
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Provenance lookup failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)