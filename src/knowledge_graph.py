import networkx as nx
from typing import List, Dict, Any

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, chunks: List[Dict[str, Any]]):
        self.graph.clear()
        for chunk in chunks:
            class_name = chunk.get("class_name")
            method_name = chunk.get("method_name")
            chunk_id = chunk.get("chunk_id")
            calls = chunk.get("relationships", {}).get("CALLS", [])
            instantiates = chunk.get("relationships", {}).get("INSTANTIATES", [])
            implements_list = chunk.get("implements", [])

            if not class_name or not method_name:
                continue

            if not self.graph.has_node(class_name):
                self.graph.add_node(class_name, node_type="CLASS")

            for impl in implements_list:
                if not self.graph.has_node(impl):
                    self.graph.add_node(impl, node_type="INTERFACE")
                self.graph.add_edge(class_name, impl, relationship="IMPLEMENTS")

            self.graph.add_node(chunk_id, node_type="METHOD", method_name=method_name, class_name=class_name)
            self.graph.add_edge(class_name, chunk_id, relationship="HAS_METHOD")

            for target_call in calls:
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
        calls = []
        for node, data in self.graph.nodes(data=True):
            if data.get("method_name") == method_name or node.endswith(f"::{method_name}") or f"::{method_name}(" in node:
                for _, target, edge_data in self.graph.out_edges(node, data=True):
                    if edge_data.get("relationship") == "CALLS":
                        calls.append(target)
        return sorted(list(set(calls)))

    def get_callers_of(self, target_method: str) -> List[str]:
        callers = []
        for node in self.graph.nodes():
            if target_method in node or self.graph.nodes[node].get("method_name") == target_method:
                for source, _, edge_data in self.graph.in_edges(node, data=True):
                    if edge_data.get("relationship") == "CALLS":
                        caller_name = self.graph.nodes[source].get("method_name", source)
                        caller_cls = self.graph.nodes[source].get("class_name", "")
                        callers.append(f"{caller_cls}::{caller_name}" if caller_cls else caller_name)
        return sorted(list(set(callers)))

    def get_class_containing(self, method_name: str) -> List[str]:
        classes = set()
        for node, data in self.graph.nodes(data=True):
            if data.get("method_name") == method_name and data.get("class_name"):
                classes.add(data.get("class_name"))
        return sorted(list(classes))
