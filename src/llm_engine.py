from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


from src.parser import JavaASTParser
from src.chunker import CodeChunker
from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph
from src.git_intelligence import GitIntelligence
from src.context_builder import CodeIntelligenceContextBuilder

from src.impact_analysis import ImpactAnalyzer


app = FastAPI(
    title="AI Codebase Intelligence",
    version="1.0"
)

vector_store = VectorStore()
kg = CodeKnowledgeGraph()
repo_dir = None
engine = None


class IndexRequest(BaseModel):
    repo_url: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/index")
def index_repo(req: IndexRequest):
    global repo_dir, engine, kg, vector_store

    # Resolve repository
    if req.repo_url in [".", "./"]:
        repo_path = Path(".").resolve()
    else:
        repo_path, _ = clone_repo_if_url(req.repo_url)

    repo_dir = repo_path

    print(f"[Index] Repository: {repo_path}")

    # Reset graph for a fresh repository
    kg = CodeKnowledgeGraph()

    parser = JavaASTParser()
    extracted = []

    EXCLUDE_DIRS = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "target",
        "build",
        ".idea",
    }

    # Parse Java files
    for java_file in repo_path.rglob("*.java"):

        if any(
            part in EXCLUDE_DIRS
            for part in java_file.parts
        ):
            continue

        if java_file.name in {
            "module-info.java",
            "package-info.java",
        }:
            continue

        try:
            result = parser.parse_file(str(java_file))

            if not result:
                continue

            if isinstance(result, dict):
                extracted.append(
                    (Path(java_file), result)
                )

            elif isinstance(result, (tuple, list)):
                tree = result[0]
                source_code = (
                    result[1]
                    if len(result) > 1
                    else ""
                )

                symbols = parser.extract_symbols_and_relations(
                    tree,
                    source_code
                )

                extracted.append(
                    (
                        Path(java_file),
                        {
                            "tree": tree,
                            "source_code": source_code,
                            "symbols": symbols,
                        },
                    )
                )

        except Exception as e:
            print(
                f"[Warning] Failed to parse {java_file}: {e}"
            )

    if not extracted:
        raise HTTPException(
            status_code=400,
            detail="No Java files could be parsed."
        )

    print(
        f"[Index] Parsed {len(extracted)} Java files."
    )

    # Create chunks
    chunker = CodeChunker(parser=parser)

    chunks = chunker.create_chunks(
        extracted
    )

    print(
        f"[Index] Created {len(chunks)} chunks."
    )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No code chunks were created."
        )

    # Build Knowledge Graph
    kg.build_graph_from_chunks(chunks)

    summary = kg.get_summary()

    print(
        f"[Index] Graph: "
        f"{summary['total_nodes']} nodes, "
        f"{summary['total_edges']} edges."
    )

    # Build FAISS index
    vector_store = VectorStore()
    vector_store.build_index(chunks)

    print("[Index] FAISS index ready.")

    # Git Intelligence
    git_intel = GitIntelligence(
        repo_path=str(repo_path)
    )

    # Context Builder
    context_builder = CodeIntelligenceContextBuilder(
        git_intel=git_intel
    )

    # Full Intelligence Engine
    engine = CodeIntelligenceEngine(
        repo_path=str(repo_path),
        vector_store=vector_store,
        graph_db=kg,
        context_builder=context_builder,
    )

    app.state.git_intel = git_intel
    app.state.engine = engine

    return {
        "status": "ready",
        "chunks_indexed": len(chunks),
        "graph_nodes": summary["total_nodes"],
        "graph_edges": summary["total_edges"],
    }


@app.post("/query")
def query_engine(req: QueryRequest):

    if engine is None:
        raise HTTPException(
            status_code=400,
            detail="Please index a repository first using /index."
        )

    if not req.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    try:
        result = engine.answer_query(
            query=req.query,
            top_k=req.top_k,
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@app.get("/impact/{method_name:path}")
def get_impact(method_name: str):

    if kg is None:
        raise HTTPException(
            status_code=400,
            detail="Please index a repository first."
        )

    analyzer = ImpactAnalyzer(kg)

    result = analyzer.analyze_blast_radius(
        method_name
    )

    if "error" in result:
        raise HTTPException(
            status_code=404,
            detail=result["error"]
        )

    return result


@app.get("/history/{file_path:path}")
def get_file_history(file_path: str):

    if not hasattr(
        app.state,
        "git_intel"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please index a repository first."
        )

    try:
        return app.state.git_intel.get_file_history(
            file_path
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Git history error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )