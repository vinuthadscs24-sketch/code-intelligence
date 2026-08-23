from typing import List, Dict, Any
from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph

class HybridRetriever:
    def __init__(self, vector_store: VectorStore, knowledge_graph: CodeKnowledgeGraph, rrf_k: int = 60):
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 1. Semantic Vector Search
        raw_vector_results = self.vector_store.search(query, top_k=top_k * 2)
        
        # Standardize vector results (handles tuple vs dictionary returns)
        vector_chunks = []
        for res in raw_vector_results:
            if isinstance(res, tuple):
                chunk = res[0]
            elif isinstance(res, dict):
                chunk = res
            else:
                continue
            vector_chunks.append(chunk)

        if not vector_chunks:
            return []

        # 2. Graph Context Expansion
        graph_results = []
        for chunk in vector_chunks:
            method_name = chunk.get("method_name") or chunk.get("name") or chunk.get("id")
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or str(chunk)
            
            if method_name and hasattr(self.kg, "get_calls_to"):
                try:
                    callers = self.kg.get_calls_to(method_name)
                    callees = self.kg.get_calls_from(method_name)
                    if callers or callees:
                        graph_results.append({
                            "chunk_id": chunk_id,
                            "method_name": method_name,
                            "callers": callers,
                            "callees": callees
                        })
                except Exception:
                    pass

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        # Score Vector Results
        for rank, chunk in enumerate(vector_chunks, start=1):
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or str(rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            
        # Score Graph Connectivity
        for rank, g_res in enumerate(graph_results, start=1):
            chunk_id = g_res["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            
        # Sort chunks by fused RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Reconstruct final enriched payload
        final_results = []
        chunk_map = {
            (c.get("chunk_id") or c.get("id") or str(idx + 1)): c 
            for idx, c in enumerate(vector_chunks)
        }
        
        for chunk_id, score in sorted_chunks[:top_k]:
            if chunk_id in chunk_map:
                enriched_chunk = chunk_map[chunk_id].copy()
                enriched_chunk["rrf_score"] = score
                
                # Attach graph context if present
                graph_node = next((g for g in graph_results if g["chunk_id"] == chunk_id), None)
                if graph_node:
                    enriched_chunk["graph_callers"] = graph_node["callers"]
                    enriched_chunk["graph_callees"] = graph_node["callees"]
                    
                final_results.append(enriched_chunk)
                
        return final_results