from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.main import clone_repo_if_url
from src.parser import JavaASTParser
from src.chunker import CodeChunker
from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph
from src.hybrid_retriever import HybridRetriever
from src.impact_analysis import ImpactAnalyzer
from src.git_intelligence import GitIntelligence

app = FastAPI(title="AI Codebase Intelligence", version="1.0")

# Global State
vector_store = VectorStore()
kg = CodeKnowledgeGraph()
repo_dir = None

class IndexRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/index")
def index_repo(req: IndexRequest):
    global repo_dir
    
    # Resolve local path or remote URL
    if req.repo_url in [".", "./"] or req.repo_url.startswith("./"):
        repo_path = Path(".").resolve()
    else:
        repo_path, _ = clone_repo_if_url(req.repo_url)
        
    repo_dir = repo_path
    
    # Store persistent git intelligence in app state
    app.state.git_intel = GitIntelligence(repo_path)
    
    parser = JavaASTParser()
    extracted = []
    
    EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "target", "build", ".idea"}
    
    for jf in repo_path.rglob("*.java"):
        if any(part in EXCLUDE_DIRS for part in jf.parts):
            continue
        if jf.name in ["module-info.java", "package-info.java"]:
            continue
            
        try:
            res = parser.parse_file(str(jf))
            if isinstance(res, dict):
                extracted.append((Path(jf), res))
            elif isinstance(res, (tuple, list)): 
                extracted.append((Path(jf), {"tree": res[0], "source_code": res[1] if len(res) > 1 else ""}))
        except Exception:
            pass
            
    chunker = CodeChunker(parser=parser)
    chunks = chunker.create_chunks(extracted)
    
    kg.build_graph_from_chunks(chunks)
    vector_store.build_index(chunks)
    
    app.state.kg = kg
    
    summary = kg.get_summary()
    return {
        "status": "ready",
        "chunks_indexed": len(chunks),
        "graph_nodes": summary["total_nodes"],
        "graph_edges": summary["total_edges"]
    }

@app.post("/query")
def query_engine(req: QueryRequest):
    retriever = HybridRetriever(vector_store, kg)
    results = retriever.search(req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}

@app.get("/impact/{method_name:path}")
def get_impact(method_name: str):
    analyzer = ImpactAnalyzer(kg)
    result = analyzer.analyze_blast_radius(method_name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/history/{file_path:path}")
def get_file_history(file_path: str):
    if not hasattr(app.state, "git_intel") or app.state.git_intel is None:
        app.state.git_intel = GitIntelligence(Path(".").resolve())
        
    try:
        result = app.state.git_intel.get_file_history(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Git history error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)