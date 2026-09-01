import re
from typing import Dict, Any
from src.models.response_schema import UnifiedAdaptiveResponse

class AdaptiveResponseOrchestrator:
    def __init__(self, llm_engine, impact_analyzer, git_intelligence):
        # Leverage existing, fully functional modules
        self.llm_engine = llm_engine
        self.impact_analyzer = impact_analyzer
        self.git_intelligence = git_intelligence

    def process_query(self, query: str, repo_id: str, debug: bool = False) -> UnifiedAdaptiveResponse:
        q_lower = query.lower()
        target_symbol = self._extract_symbol(query)

        # 1. DEPENDENCY / CALL GRAPH INTENT
        if any(k in q_lower for k in ["who calls", "called by", "dependents of", "calls from"]):
            raw_callers = self.llm_engine.knowledge_graph.get_callers_of(target_symbol)
            
            nodes = [{"id": target_symbol, "label": target_symbol, "type": "target"}]
            edges = []
            for caller in raw_callers:
                nodes.append({"id": caller, "label": caller, "type": "caller"})
                edges.append({"source": caller, "target": target_symbol, "label": "CALLS"})

            panel_data = {"nodes": nodes, "edges": edges, "target": target_symbol}
            answer_text = f"Found {len(raw_callers)} methods calling `{target_symbol}` in `{repo_id}`."
            
            return UnifiedAdaptiveResponse(
                answer_text=answer_text,
                query_type="dependency",
                panel_type="dependency_graph",
                panel_data=panel_data,
                evidence=[]
            )

        # 2. IMPACT ANALYSIS INTENT
        elif any(k in q_lower for k in ["what breaks", "impact of", "affect", "blast radius"]):
            # Harness existing impact_analysis.py!
            impact = self.impact_analyzer.analyze_blast_radius(target_symbol, max_depth=3)
            
            panel_data = {
                "target_symbol": target_symbol,
                "direct_dependents": len(impact.get("direct_callers", [])),
                "affected_classes": len(impact.get("affected_classes", [])),
                "affected_files": len(impact.get("affected_files", [])),
                "risk_level": "HIGH" if len(impact.get("affected_files", [])) > 4 else "MEDIUM",
                "nodes": impact.get("nodes", []),
                "edges": impact.get("edges", [])
            }
            answer_text = f"Modifying `{target_symbol}` affects up to {panel_data['affected_files']} files with a {panel_data['risk_level']} risk level."

            return UnifiedAdaptiveResponse(
                answer_text=answer_text,
                query_type="impact",
                panel_type="impact_summary",
                panel_data=panel_data
            )

        # 3. GIT TIMELINE INTENT
        elif any(k in q_lower for k in ["why changed", "git history", "who modified", "commits"]):
            # Harness existing git_intelligence.py!
            commits = self.git_intelligence.get_file_history(target_symbol)
            panel_data = {"symbol_or_file": target_symbol, "commits": commits}
            answer_text = f"Retrieved commit provenance timeline for `{target_symbol}`."

            return UnifiedAdaptiveResponse(
                answer_text=answer_text,
                query_type="git",
                panel_type="git_timeline",
                panel_data=panel_data
            )

        # 4. DEFAULT (HYBRID RAG + LLM SYNTHESIS)
        else:
            rag_result = self.llm_engine.query(query, repo_id=repo_id)
            
            evidence = [
                {
                    "file_path": c.get("file_path", "unknown"),
                    "symbol_name": c.get("symbol_name"),
                    "start_line": c.get("start_line", 0),
                    "end_line": c.get("end_line", 0),
                    "code_snippet": c.get("text", "")[:250]
                }
                for c in rag_result.get("context_chunks", [])
            ]

            trace = None
            if debug:
                trace = {
                    "query_raw": query,
                    "rewritten_queries": rag_result.get("rewritten_queries", []),
                    "faiss_count": rag_result.get("faiss_count", 0),
                    "bm25_count": rag_result.get("bm25_count", 0),
                    "graph_expanded_count": rag_result.get("graph_count", 0),
                    "top_rrf_matches": [e["file_path"] for e in evidence[:3]]
                }

            return UnifiedAdaptiveResponse(
                answer_text=rag_result.get("answer", "No synthesis returned."),
                query_type="explanation",
                panel_type="text",
                panel_data={},
                evidence=evidence,
                retrieval_trace=trace
            )

    def _extract_symbol(self, query: str) -> str:
        words = re.findall(r'[A-Z][a-zA-Z0-9_]+', query)
        return words[0] if words else "BookingService"