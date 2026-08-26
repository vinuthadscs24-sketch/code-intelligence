from typing import List, Dict, Any, Optional
from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph

WEB_ANNOTATIONS = {
    "RestController", "Controller", "RequestMapping", 
    "GetMapping", "PostMapping", "PutMapping", 
    "DeleteMapping", "PatchMapping", "WebServlet"
}

class HybridRetriever:
    def __init__(
        self, 
        vector_store: VectorStore, 
        knowledge_graph: CodeKnowledgeGraph, 
        rrf_k: int = 60, 
        max_per_file: int = 1,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4
    ):
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.rrf_k = rrf_k
        self.max_per_file = max_per_file
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

    def _get_chunk_id(self, chunk: Dict[str, Any]) -> str:
        """Derives a consistent unique identifier for deduplication across stores."""
        if chunk.get("chunk_id"):
            return str(chunk["chunk_id"])
        if chunk.get("id"):
            return str(chunk["id"])
            
        file_name = chunk.get("file_name", chunk.get("file", "unknown"))
        class_name = chunk.get("class_name", "")
        method_name = chunk.get("method_name", "")
        start_line = chunk.get("start_line", 0)
        
        return f"{file_name}::{class_name}::{method_name}:{start_line}"

    def _independent_graph_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Searches graph nodes independently based on token matches in node keys."""
        if not hasattr(self.kg, "graph") or self.kg.graph is None:
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
                
                chunk_id = data.get("chunk_id") or str(node)
                file_name = data.get("file", chunk_id.split("::")[0] if "::" in chunk_id else "unknown")
                
                matched_chunks.append({
                    "chunk_id": chunk_id,
                    "file_name": file_name,
                    "method_name": str(node),
                    "annotations": data.get("annotations", []),
                    "graph_callers": callers,
                    "graph_callees": callees,
                    "code_content": data.get("code_content", f"// Method: {node}\n// Callers: {callers}\n// Callees: {callees}")
                })
                if len(matched_chunks) >= top_k:
                    break

        return matched_chunks

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        fetch_limit = top_k * 4
        
        # 1. Semantic Vector Search
        raw_vector_results = self.vector_store.search(query, top_k=fetch_limit)
        vector_chunks = []
        for res in raw_vector_results:
            if isinstance(res, tuple):
                vector_chunks.append(res[0])
            elif isinstance(res, dict):
                vector_chunks.append(res)

        # 2. Independent Graph Search
        independent_graph_chunks = self._independent_graph_search(query, top_k=fetch_limit)

        # 3. Detect intent for API endpoints
        query_lower = query.lower()
        is_endpoint_query = any(k in query_lower for k in ["controller", "endpoint", "api", "route", "http", "rest", "web"])

        # 4. Score Fusion via Weighted RRF
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        def merge_and_apply_rrf(chunk: Dict[str, Any], rank: int, source_weight: float, source_type: str):
            chunk_id = self._get_chunk_id(chunk)
            
            # Merge dictionary representation safely
            if chunk_id not in doc_map:
                doc_map[chunk_id] = dict(chunk)
                doc_map[chunk_id]["sources"] = [source_type]
            else:
                doc_map[chunk_id]["sources"].append(source_type)
                # Preserve graph relational context if vector result updates existing chunk
                for graph_key in ["graph_callers", "graph_callees"]:
                    if graph_key in chunk and chunk[graph_key]:
                        doc_map[chunk_id][graph_key] = chunk[graph_key]

            base_score = source_weight * (1.0 / (self.rrf_k + rank))
            
            # Apply boost for web controllers if query matches endpoint intent
            if is_endpoint_query:
                raw_annotations = doc_map[chunk_id].get("annotations", [])
                cleaned_annotations = {str(a).replace("@", "").strip() for a in raw_annotations}
                if cleaned_annotations.intersection(WEB_ANNOTATIONS):
                    base_score *= 1.5

            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + base_score

        for rank, chunk in enumerate(vector_chunks, start=1):
            merge_and_apply_rrf(chunk, rank, self.vector_weight, "vector")

        for rank, chunk in enumerate(independent_graph_chunks, start=1):
            merge_and_apply_rrf(chunk, rank, self.graph_weight, "graph")

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
            
            enriched_chunk = dict(chunk)
            enriched_chunk["combined_score"] = round(score, 6)
            enriched_chunk.setdefault("graph_callers", [])
            enriched_chunk.setdefault("graph_callees", [])
            final_results.append(enriched_chunk)

        return final_results