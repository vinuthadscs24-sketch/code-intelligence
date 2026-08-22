import json
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

class CodeKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, repo_name: str, extracted_data: List[Tuple[Union[Path, str], dict]]):
        """Constructs a directed graph with Spring DI, inheritance, and resolved call chains."""
        self.graph.clear()
        self.graph.add_node(repo_name, type="REPOSITORY")

        class_type_map = {}   # class_name -> package/details
        field_type_map = {}   # (class_name, field_name) -> field_type

        # Phase 1: Register Classes, Interfaces, Inheritance, and Fields
        for file_path, symbols in extracted_data:
            package = symbols.get("package", "default")
            file_str = file_path.name if isinstance(file_path, Path) else str(file_path)

            for cls in symbols.get("classes", []):
                cls_name = cls["name"]
                class_type_map[cls_name] = package

                self.graph.add_node(
                    cls_name,
                    type="CLASS",
                    package=package,
                    annotations=json.dumps(cls.get("annotations", [])),
                    file=file_str
                )
                self.graph.add_edge(repo_name, cls_name, relation="CONTAINS")

                if cls.get("extends"):
                    parent = cls["extends"]
                    self.graph.add_node(parent, type="CLASS")
                    self.graph.add_edge(cls_name, parent, relation="EXTENDS")

                for iface in cls.get("implements", []):
                    self.graph.add_node(iface, type="INTERFACE")
                    self.graph.add_edge(cls_name, iface, relation="IMPLEMENTS")

            for iface in symbols.get("interfaces", []):
                self.graph.add_node(iface, type="INTERFACE", package=package, file=file_str)
                self.graph.add_edge(repo_name, iface, relation="CONTAINS")

            for f in symbols.get("fields", []):
                enclosing = f["enclosing_class"]
                field_name = f["name"]
                field_type = f["type"]
                annotations = f.get("annotations", [])

                field_type_map[(enclosing, field_name)] = field_type

                is_injected = any(
                    a for a in annotations if any(kw in a for kw in ["@Autowired", "@Inject", "@Resource"])
                )

                if is_injected or field_type in class_type_map:
                    self.graph.add_node(field_type, type="CLASS")
                    self.graph.add_edge(
                        enclosing, 
                        field_type, 
                        relation="INJECTS", 
                        field_name=field_name,
                        annotations=json.dumps(annotations)
                    )

        # Phase 2: Add Methods & Resolve Method Calls
        for file_path, symbols in extracted_data:
            for m in symbols.get("methods", []):
                m_name = m["name"]
                enclosing = m.get("enclosing_class")

                if enclosing:
                    method_node_id = f"{enclosing}.{m_name}()"
                    self.graph.add_node(
                        method_node_id,
                        type="METHOD",
                        name=m_name,
                        annotations=json.dumps(m.get("annotations", []))
                    )
                    self.graph.add_edge(enclosing, method_node_id, relation="HAS_METHOD")

            for call in symbols.get("method_calls", []):
                caller_class = call.get("caller_class")
                obj_expr = call.get("object_expression")
                method_called = call.get("method_called")

                if not caller_class:
                    continue

                target_class = None
                if obj_expr and (caller_class, obj_expr) in field_type_map:
                    target_class = field_type_map[(caller_class, obj_expr)]
                elif obj_expr and obj_expr in class_type_map:
                    target_class = obj_expr

                if target_class:
                    target_method_id = f"{target_class}.{method_called}()"
                    self.graph.add_node(target_method_id, type="METHOD")
                    self.graph.add_edge(caller_class, target_method_id, relation="CALLS")
                else:
                    unresolved_id = f"{obj_expr}.{method_called}()" if obj_expr else method_called
                    self.graph.add_node(unresolved_id, type="METHOD_CALL")
                    self.graph.add_edge(caller_class, unresolved_id, relation="CALLS")

    def build_graph_from_chunks(self, chunks: List[Dict[str, Any]]):
        """Adapter allowing method chunks created by CodeChunker to populate the graph directly."""
        self.graph.clear()
        repo_node = "CodeRepository"
        self.graph.add_node(repo_node, type="REPOSITORY")

        for chunk in chunks:
            file_name = chunk.get("file_name", chunk.get("file", "unknown"))
            class_name = chunk.get("class_name", "Global")
            method_name = chunk.get("method_name", "unknown")

            self.graph.add_node(class_name, type="CLASS", file=file_name)
            self.graph.add_edge(repo_node, class_name, relation="CONTAINS")

            method_id = f"{class_name}.{method_name}()"
            self.graph.add_node(
                method_id, 
                type="METHOD", 
                file=file_name, 
                line_start=chunk.get("start_line", 1), 
                line_end=chunk.get("end_line", 1)
            )
            self.graph.add_edge(class_name, method_id, relation="HAS_METHOD")

            for call in chunk.get("calls", []):
                call_id = f"{call}()" if not call.endswith("()") else call
                self.graph.add_node(call_id, type="METHOD")
                self.graph.add_edge(method_id, call_id, relation="CALLS")

    def get_calls_from(self, method_name: str) -> List[str]:
        """Returns outgoing method calls from a specified method node."""
        target = method_name if method_name.endswith("()") else f"{method_name}()"
        if not self.graph.has_node(target):
            target = method_name
        if not self.graph.has_node(target):
            return []
        return [
            dst for _, dst, data in self.graph.out_edges(target, data=True) 
            if data.get("relation") == "CALLS"
        ]

    def get_callers_of(self, method_name: str) -> List[str]:
        """Returns incoming callers for a specified method node."""
        target = method_name if method_name.endswith("()") else f"{method_name}()"
        if not self.graph.has_node(target):
            target = method_name
        if not self.graph.has_node(target):
            return []
        return [
            src for src, _, data in self.graph.in_edges(target, data=True) 
            if data.get("relation") == "CALLS"
        ]

    def get_summary(self) -> dict:
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "relationship_types": self._count_edge_relations()
        }

    def _count_node_types(self) -> dict:
        counts = {}
        for _, attrs in self.graph.nodes(data=True):
            ntype = attrs.get("type", "UNKNOWN")
            counts[ntype] = counts.get(ntype, 0) + 1
        return counts

    def _count_edge_relations(self) -> dict:
        counts = {}
        for _, _, attrs in self.graph.edges(data=True):
            rel = attrs.get("relation", "UNKNOWN")
            counts[rel] = counts.get(rel, 0) + 1
        return counts

    def export_graphml(self, output_path: str = "workspace/code_graph.graphml"):
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.graph, out_file)
        print(f"[Graph] Exported GraphML to: {out_file.resolve()}")
    def get_class_methods(self, class_name: str) -> List[str]:
        """Returns all method names belonging to a specified class."""
        if not self.graph.has_node(class_name):
            return []
        return [
            dst for _, dst, data in self.graph.out_edges(class_name, data=True)
            if data.get("relation") == "HAS_METHOD"
        ]

    def get_class_containing(self, method_name: str) -> List[str]:
        """Returns class names that contain the specified method."""
        results = []
        for src, dst, data in self.graph.edges(data=True):
            if data.get("relation") == "HAS_METHOD":
                if method_name in dst:
                    results.append(src)
        return list(set(results))
        

    def inspect_class_dependencies(self, class_name: str) -> dict:
        if class_name not in self.graph:
            return {"error": f"Class '{class_name}' not in graph."}

        relations = {"HAS_METHOD": [], "INJECTS": [], "CALLS": [], "EXTENDS": [], "IMPLEMENTS": []}
        for successor in self.graph.successors(class_name):
            edge_data = self.graph.get_edge_data(class_name, successor)
            rel = edge_data.get("relation") if edge_data else None
            if rel in relations:
                relations[rel].append(successor)

        return {"class": class_name, "relationships": relations}