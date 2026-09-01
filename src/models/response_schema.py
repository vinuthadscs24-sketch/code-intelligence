from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class EvidenceItem(BaseModel):
    file_path: str
    symbol_name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    code_snippet: str

class RetrievalTrace(BaseModel):
    query_raw: str
    rewritten_queries: List[str] = Field(default_factory=list)
    faiss_count: int = 0
    bm25_count: int = 0
    graph_expanded_count: int = 0
    top_rrf_matches: List[str] = Field(default_factory=list)

class UnifiedAdaptiveResponse(BaseModel):
    answer_text: str
    query_type: str        # "dependency" | "execution" | "impact" | "git" | "explanation"
    panel_type: str        # "dependency_graph" | "execution_graph" | "impact_summary" | "git_timeline" | "text"
    panel_data: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    retrieval_trace: Optional[RetrievalTrace] = None