from typing import List, Dict, Any
from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph

class HybridRetriever:
    def __init__(self, vector_store: VectorStore, knowledge_graph: CodeKnowledgeGraph, rrf_k: int = 60, max_per_file: int = 1):
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.rrf_k = rrf_k
        self.max_per_file = max_per_file

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 1. Semantic Vector Search (fetch extra candidates for deduplication headroom)
        fetch_limit = top_k * 4
        raw_vector_results = self.vector_store.search(query, top_k=fetch_limit)
        
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
        
        # 4. Reconstruct & Apply File-Level Deduplication
        chunk_map = {
            (c.get("chunk_id") or c.get("id") or str(idx + 1)): c 
            for idx, c in enumerate(vector_chunks)
        }
        
        final_results = []
        file_counts = {}

        for chunk_id, score in sorted_chunks:
            if len(final_results) >= top_k:
                break

            if chunk_id in chunk_map:
                chunk = chunk_map[chunk_id]
                
                # Extract file path/name key
                file_key = chunk.get("file_name") or chunk.get("file_path") or chunk.get("source") or "unknown"
                
                # Skip if file instance limit reached
                if file_counts.get(file_key, 0) >= self.max_per_file:
                    continue

                file_counts[file_key] = file_counts.get(file_key, 0) + 1

                enriched_chunk = chunk.copy()
                enriched_chunk["rrf_score"] = score
                
                # Attach graph context if present
                graph_node = next((g for g in graph_results if g["chunk_id"] == chunk_id), None)
                if graph_node:
                    enriched_chunk["graph_callers"] = graph_node["callers"]
                    enriched_chunk["graph_callees"] = graph_node["callees"]
                    
                final_results.append(enriched_chunk)
                
        return final_results