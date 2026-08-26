from typing import List, Dict, Any
from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph

WEB_ANNOTATIONS = {
    "RestController", "Controller", "RequestMapping", 
    "GetMapping", "PostMapping", "PutMapping", 
    "DeleteMapping", "PatchMapping", "WebServlet"
}

class HybridRetriever:
    def __init__(self, vector_store: VectorStore, knowledge_graph: CodeKnowledgeGraph, rrf_k: int = 60, max_per_file: int = 1):
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.rrf_k = rrf_k
        self.max_per_file = max_per_file

    def _independent_graph_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Searches graph nodes independently based on token matches in node keys."""
        if not hasattr(self.kg, "graph"):
            return []

        tokens = [t.lower() for t in query.replace(".", " ").split() if len(t) > 2]
        if not tokens:
            return []

        matched_chunks = []
        for node, data in self.kg.graph.nodes(data=True):
            node_str = str(node).lower()
            if any(token in node_str for token in tokens):
                callers = self.kg.get_callers_of(node) if hasattr(self.kg, "get_callers_of") else []
                callees = self.kg.get_calls_from(node) if hasattr(self.kg, "get_calls_from") else []
                
                # Use canonical ID matching vector store format (falling back to node string)
                chunk_id = data.get("chunk_id") or str(node)
                
                matched_chunks.append({
                    "chunk_id": chunk_id,
                    "file_name": data.get("file", "unknown"),
                    "method_name": str(node),
                    "annotations": data.get("annotations", []),
                    "graph_callers": callers,
                    "graph_callees": callees,
                    "code_content": f"// Method: {node}\n// Callers: {callers}\n// Callees: {callees}"
                })
                if len(matched_chunks) >= top_k:
                    break

        return matched_chunks

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        fetch_limit = top_k * 4
        
        # 1. Semantic Vector Search
        raw_vector_results = self.vector_store.search(query, top_k=fetch_limit)
        vector_chunks = [res[0] if isinstance(res, tuple) else res for res in raw_vector_results if isinstance(res, (tuple, dict))]

        # 2. Independent Graph Search
        independent_graph_chunks = self._independent_graph_search(query, top_k=fetch_limit)

        # 3. Detect intent for API endpoints
        query_lower = query.lower()
        is_endpoint_query = any(k in query_lower for k in ["controller", "endpoint", "api", "route", "http", "rest", "web"])

        # 4. Score Fusion via RRF
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        def apply_rrf(chunk: Dict[str, Any], rank: int):
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or str(chunk.get("method_name"))
            doc_map[chunk_id] = chunk
            
            base_score = 1.0 / (self.rrf_k + rank)
            
            # Apply boost for web controllers if query matches intent
            if is_endpoint_query:
                raw_annotations = chunk.get("annotations", [])
                cleaned_annotations = {a.replace("@", "").strip() for a in raw_annotations}
                if cleaned_annotations.intersection(WEB_ANNOTATIONS):
                    base_score *= 1.5  # Moderated multiplier

            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + base_score

        for rank, chunk in enumerate(vector_chunks, start=1):
            apply_rrf(chunk, rank)

        for rank, chunk in enumerate(independent_graph_chunks, start=1):
            apply_rrf(chunk, rank)

        # 5. Result Selection & File Deduplication
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_results = []
        file_counts: Dict[str, int] = {}

        for chunk_id, score in sorted_chunks:
            if len(final_results) >= top_k:
                break

            chunk = doc_map.get(chunk_id)
            if not chunk:
                continue

            file_key = chunk.get("file_name") or chunk.get("file_path") or chunk.get("source") or "unknown"
            if file_counts.get(file_key, 0) >= self.max_per_file:
                continue

            file_counts[file_key] = file_counts.get(file_key, 0) + 1
            
            enriched_chunk = chunk.copy()
            enriched_chunk["rrf_score"] = score
            final_results.append(enriched_chunk)

        return final_results