import networkx as nx
from typing import List, Dict, Any

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, chunks: List[Dict[str, Any]]):
        self.graph.clear()
        
        # Phase 1: Create all known CLASS and METHOD nodes
        for chunk in chunks:
            class_name = chunk.get("class_name")
            method_name = chunk.get("method_name")
            chunk_id = chunk.get("chunk_id")
            implements_list = chunk.get("implements", [])

            if not class_name or not method_name or not chunk_id:
                continue

            if not self.graph.has_node(class_name):
                self.graph.add_node(class_name, node_type="CLASS")

            for impl in implements_list:
                if not self.graph.has_node(impl):
                    self.graph.add_node(impl, node_type="INTERFACE")
                self.graph.add_edge(class_name, impl, relationship="IMPLEMENTS")

            self.graph.add_node(chunk_id, node_type="METHOD", method_name=method_name, class_name=class_name)
            self.graph.add_edge(class_name, chunk_id, relationship="HAS_METHOD")

        # Phase 2: Add CALLS and INSTANTIATES edges
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id or not self.graph.has_node(chunk_id):
                continue

            calls = chunk.get("relationships", {}).get("CALLS", [])
            instantiates = chunk.get("relationships", {}).get("INSTANTIATES", [])

            for target_call in calls:
                # Store target method node or placeholder
                if not self.graph.has_node(target_call):
                    self.graph.add_node(target_call, node_type="METHOD_CALL", method_name=target_call)
                self.graph.add_edge(chunk_id, target_call, relationship="CALLS")

            for target_inst in instantiates:
                if not self.graph.has_node(target_inst):
                    self.graph.add_node(target_inst, node_type="CLASS_INSTANTIATION", class_name=target_inst)
                self.graph.add_edge(chunk_id, target_inst, relationship="INSTANTIATES")

    def get_class_methods(self, class_name: str) -> List[str]:
        if not self.graph.has_node(class_name):
            return []
        return [
            self.graph.nodes[target].get("method_name", target)
            for _, target, data in self.graph.out_edges(class_name, data=True)
            if data.get("relationship") == "HAS_METHOD"
        ]

    def get_calls_from(self, method_name: str) -> List[str]:
        calls = set()
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "METHOD" and data.get("method_name") == method_name:
                for _, target, edge_data in self.graph.out_edges(node, data=True):
                    if edge_data.get("relationship") == "CALLS":
                        target_label = self.graph.nodes[target].get("method_name", target)
                        calls.add(target_label)
        return sorted(list(calls))

    def get_callers_of(self, target_method: str) -> List[str]:
        callers = set()
        for node, data in self.graph.nodes(data=True):
            # Strict node matching for target method
            if data.get("method_name") == target_method or node == target_method:
                for source, _, edge_data in self.graph.in_edges(node, data=True):
                    if edge_data.get("relationship") == "CALLS":
                        src_name = self.graph.nodes[source].get("method_name", source)
                        src_cls = self.graph.nodes[source].get("class_name", "")
                        callers.add(f"{src_cls}::{src_name}" if src_cls else src_name)
        return sorted(list(callers))

    def get_class_containing(self, method_name: str) -> List[str]:
        classes = {
            data.get("class_name")
            for _, data in self.graph.nodes(data=True)
            if data.get("method_name") == method_name and data.get("class_name")
        }
        return sorted(list(classes))