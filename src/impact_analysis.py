from typing import List, Dict, Any
from src.graph_builder import CodeKnowledgeGraph

class ImpactAnalyzer:
    def __init__(self, kg: CodeKnowledgeGraph):
        self.kg = kg

    def analyze_blast_radius(self, target_method: str, depth: int = 2) -> Dict[str, Any]:
        """
        Traverses upstream graph edges to identify which methods break
        if the target_method is modified.
        """
        # Find matching node(s) flexible to chunk_id or bare method name
        matched_nodes = [
            node for node in self.kg.graph.nodes
            if node == target_method or node.endswith(f"::{target_method}") or target_method in node
        ]

        if not matched_nodes:
            available_nodes = list(self.kg.graph.nodes)[:10]
            return {
                "error": f"Method '{target_method}' not found in Knowledge Graph.",
                "available_sample_nodes": available_nodes
            }

        affected_nodes = set()
        current_level = set(matched_nodes)
        
        for level in range(1, depth + 1):
            next_level = set()
            for node in current_level:
                # Predecessors are components/methods that call this node
                predecessors = list(self.kg.graph.predecessors(node))
                next_level.update(predecessors)
                affected_nodes.update(predecessors)
            current_level = next_level
            if not current_level:
                break

        return {
            "target": target_method,
            "matched_nodes": matched_nodes,
            "depth_analyzed": depth,
            "total_affected": len(affected_nodes),
            "affected_components": list(affected_nodes)
        }