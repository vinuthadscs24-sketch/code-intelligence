import json
import networkx as nx
from pathlib import Path

class CodeKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, repo_name: str, extracted_data: list[tuple[Path, dict]]):
        """Constructs a directed graph with Spring DI, inheritance, and resolved call chains."""
        self.graph.clear()
        self.graph.add_node(repo_name, type="REPOSITORY")

        class_type_map = {}   # class_name -> package/details
        field_type_map = {}   # (class_name, field_name) -> field_type

        # --- Phase 1: Register Classes, Interfaces, Inheritance, and Fields ---
        for file_path, symbols in extracted_data:
            package = symbols.get("package", "default")

            for cls in symbols["classes"]:
                cls_name = cls["name"]
                class_type_map[cls_name] = package

                self.graph.add_node(
                    cls_name,
                    type="CLASS",
                    package=package,
                    annotations=json.dumps(cls["annotations"]),
                    file=file_path.name
                )
                self.graph.add_edge(repo_name, cls_name, relation="CONTAINS")

                # Inheritance: EXTENDS
                if cls.get("extends"):
                    parent = cls["extends"]
                    self.graph.add_node(parent, type="CLASS")
                    self.graph.add_edge(cls_name, parent, relation="EXTENDS")

                # Inheritance: IMPLEMENTS
                for iface in cls.get("implements", []):
                    self.graph.add_node(iface, type="INTERFACE")
                    self.graph.add_edge(cls_name, iface, relation="IMPLEMENTS")

            for iface in symbols["interfaces"]:
                self.graph.add_node(iface, type="INTERFACE", package=package, file=file_path.name)
                self.graph.add_edge(repo_name, iface, relation="CONTAINS")

            # Store Fields & Build INJECTS Relationships
            for f in symbols["fields"]:
                enclosing = f["enclosing_class"]
                field_name = f["name"]
                field_type = f["type"]
                annotations = f["annotations"]

                field_type_map[(enclosing, field_name)] = field_type

                # Check if @Autowired, @Inject, or @Resource is present
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

        # --- Phase 2: Add Methods & Resolve Method Calls ---
        for file_path, symbols in extracted_data:
            for m in symbols["methods"]:
                m_name = m["name"]
                enclosing = m["enclosing_class"]

                if enclosing:
                    method_node_id = f"{enclosing}.{m_name}()"
                    self.graph.add_node(
                        method_node_id,
                        type="METHOD",
                        name=m_name,
                        annotations=json.dumps(m["annotations"])
                    )
                    self.graph.add_edge(enclosing, method_node_id, relation="HAS_METHOD")

            for call in symbols["method_calls"]:
                caller_class = call["caller_class"]
                obj_expr = call["object_expression"]
                method_called = call["method_called"]

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