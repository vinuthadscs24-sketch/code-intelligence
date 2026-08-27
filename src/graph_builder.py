import json
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union


class CodeKnowledgeGraph:

    def __init__(self):
        self.graph = nx.DiGraph()

    # ============================================================
    # BUILD GRAPH FROM PARSED DATA
    # ============================================================

    def build_graph(
        self,
        repo_name: str,
        extracted_data: List[
            Tuple[Union[Path, str], dict]
        ]
    ):

        self.graph.clear()

        self.graph.add_node(
            repo_name,
            type="REPOSITORY"
        )

        class_type_map = {}
        field_type_map = {}

        # --------------------------------------------------------
        # Phase 1: Classes
        # --------------------------------------------------------

        for file_path, symbols in extracted_data:

            package = symbols.get(
                "package",
                "default"
            )

            file_str = (
                file_path.name
                if isinstance(file_path, Path)
                else str(file_path)
            )

            for cls in symbols.get(
                "classes",
                []
            ):

                if isinstance(cls, str):
                    cls_name = cls
                    annotations = []
                    extends = None
                    implements = []

                else:
                    cls_name = cls.get(
                        "name",
                        "UnknownClass"
                    )

                    annotations = cls.get(
                        "annotations",
                        []
                    )

                    extends = cls.get(
                        "extends"
                    )

                    implements = cls.get(
                        "implements",
                        []
                    )

                class_type_map[
                    cls_name
                ] = package

                self.graph.add_node(
                    cls_name,
                    type="CLASS",
                    package=package,
                    annotations=json.dumps(
                        annotations
                    ),
                    file=file_str
                )

                self.graph.add_edge(
                    repo_name,
                    cls_name,
                    relation="CONTAINS"
                )

                if extends:

                    self.graph.add_node(
                        extends,
                        type="CLASS"
                    )

                    self.graph.add_edge(
                        cls_name,
                        extends,
                        relation="EXTENDS"
                    )

                for iface in implements:

                    self.graph.add_node(
                        iface,
                        type="INTERFACE"
                    )

                    self.graph.add_edge(
                        cls_name,
                        iface,
                        relation="IMPLEMENTS"
                    )

            # ----------------------------------------------------
            # Fields / dependencies
            # ----------------------------------------------------

            for field in symbols.get(
                "fields",
                []
            ):

                enclosing = field.get(
                    "enclosing_class"
                )

                field_name = field.get(
                    "name"
                )

                field_type = field.get(
                    "type"
                )

                if not enclosing or not field_name:
                    continue

                field_type_map[
                    (enclosing, field_name)
                ] = field_type

                annotations = field.get(
                    "annotations",
                    []
                )

                is_injected = any(
                    any(
                        keyword in str(annotation)
                        for keyword in [
                            "@Autowired",
                            "@Inject",
                            "@Resource"
                        ]
                    )
                    for annotation in annotations
                )

                if is_injected:

                    self.graph.add_node(
                        field_type,
                        type="CLASS"
                    )

                    self.graph.add_edge(
                        enclosing,
                        field_type,
                        relation="INJECTS",
                        field_name=field_name,
                        annotations=json.dumps(
                            annotations
                        )
                    )

        # --------------------------------------------------------
        # Phase 2: Methods
        # --------------------------------------------------------

        for file_path, symbols in extracted_data:

            file_str = (
                file_path.name
                if isinstance(file_path, Path)
                else str(file_path)
            )

            for method in symbols.get(
                "methods",
                []
            ):

                method_name = method.get(
                    "name",
                    "unknown"
                )

                enclosing = method.get(
                    "enclosing_class",
                    "Global"
                )

                method_id = (
                    f"{enclosing}.{method_name}()"
                )

                self.graph.add_node(
                    method_id,
                    type="METHOD",
                    name=method_name,
                    class_name=enclosing,
                    file=file_str,
                    annotations=json.dumps(
                        method.get(
                            "annotations",
                            []
                        )
                    ),
                    code_content=method.get(
                        "source_code",
                        ""
                    ),
                    start_line=method.get(
                        "start_line",
                        1
                    ),
                    end_line=method.get(
                        "end_line",
                        1
                    )
                )

                self.graph.add_edge(
                    enclosing,
                    method_id,
                    relation="HAS_METHOD"
                )

        # --------------------------------------------------------
        # Phase 3: Resolve method calls
        # --------------------------------------------------------

        for file_path, symbols in extracted_data:

            for method in symbols.get(
                "methods",
                []
            ):

                caller_class = method.get(
                    "enclosing_class",
                    "Global"
                )

                caller_method = method.get(
                    "name",
                    "unknown"
                )

                caller_id = (
                    f"{caller_class}.{caller_method}()"
                )

                calls = method.get(
                    "calls",
                    []
                )

                if not calls:
                    continue

                for called_method in calls:

                    called_method = str(
                        called_method
                    ).strip()

                    if not called_method:
                        continue

                    # ------------------------------------------------
                    # Find every method named like the called method
                    # ------------------------------------------------

                    targets = []

                    for node, attrs in self.graph.nodes(
                        data=True
                    ):

                        if attrs.get(
                            "type"
                        ) != "METHOD":
                            continue

                        node_method_name = attrs.get(
                            "name"
                        )

                        if (
                            node_method_name
                            and
                            node_method_name.lower()
                            == called_method.lower()
                        ):

                            targets.append(node)

                    # ------------------------------------------------
                    # Connect to matching real methods
                    # ------------------------------------------------

                    if targets:

                        for target in targets:

                            self.graph.add_edge(
                                caller_id,
                                target,
                                relation="CALLS"
                            )

                    else:

                        # ------------------------------------------------
                        # Keep unresolved call in graph
                        # ------------------------------------------------

                        unresolved_id = (
                            f"{called_method}()"
                        )

                        self.graph.add_node(
                            unresolved_id,
                            type="METHOD_CALL",
                            name=called_method
                        )

                        self.graph.add_edge(
                            caller_id,
                            unresolved_id,
                            relation="CALLS"
                        )

    # ============================================================
    # BUILD GRAPH FROM CHUNKS
    # ============================================================

    def build_graph_from_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ):

        self.graph.clear()

        repo_node = "CodeRepository"

        self.graph.add_node(
            repo_node,
            type="REPOSITORY"
        )

        # --------------------------------------------------------
        # Phase 1: Create classes and methods
        # --------------------------------------------------------

        for chunk in chunks:

            file_name = chunk.get(
                "file_name",
                chunk.get(
                    "file",
                    "unknown"
                )
            )

            class_name = chunk.get(
                "class_name",
                "Global"
            )

            method_name = chunk.get(
                "method_name",
                "unknown"
            )

            self.graph.add_node(
                class_name,
                type="CLASS",
                file=file_name
            )

            self.graph.add_edge(
                repo_node,
                class_name,
                relation="CONTAINS"
            )

            method_id = (
                f"{class_name}.{method_name}()"
            )

            self.graph.add_node(
                method_id,
                type="METHOD",
                name=method_name,
                class_name=class_name,
                file=file_name,
                code_content=chunk.get(
                    "code_content",
                    ""
                ),
                start_line=chunk.get(
                    "start_line",
                    1
                ),
                end_line=chunk.get(
                    "end_line",
                    1
                ),
                annotations=chunk.get(
                    "annotations",
                    []
                )
            )

            self.graph.add_edge(
                class_name,
                method_id,
                relation="HAS_METHOD"
            )

        # --------------------------------------------------------
        # Phase 2: Create CALLS edges
        # --------------------------------------------------------

        for chunk in chunks:

            class_name = chunk.get(
                "class_name",
                "Global"
            )

            method_name = chunk.get(
                "method_name",
                "unknown"
            )

            caller_id = (
                f"{class_name}.{method_name}()"
            )

            calls = chunk.get(
                "calls",
                []
            )

            # Compatibility with old chunks
            if not calls:

                calls = chunk.get(
                    "relationships",
                    []
                )

            if not calls:
                continue

            for called_method in calls:

                called_method = str(
                    called_method
                ).strip()

                if not called_method:
                    continue

                # Remove ()
                clean_name = (
                    called_method
                    .replace("()", "")
                    .strip()
                )

                # ------------------------------------------------
                # Find actual method nodes
                # ------------------------------------------------

                targets = []

                for node, attrs in self.graph.nodes(
                    data=True
                ):

                    if attrs.get(
                        "type"
                    ) != "METHOD":
                        continue

                    node_method_name = attrs.get(
                        "name"
                    )

                    if (
                        node_method_name
                        and
                        node_method_name.lower()
                        == clean_name.lower()
                    ):

                        targets.append(node)

                # ------------------------------------------------
                # Create CALLS relationship
                # ------------------------------------------------

                if targets:

                    for target in targets:

                        if target != caller_id:

                            self.graph.add_edge(
                                caller_id,
                                target,
                                relation="CALLS"
                            )

                else:

                    # ------------------------------------------------
                    # Unresolved method call
                    # ------------------------------------------------

                    unresolved_id = (
                        f"{clean_name}()"
                    )

                    self.graph.add_node(
                        unresolved_id,
                        type="METHOD_CALL",
                        name=clean_name
                    )

                    self.graph.add_edge(
                        caller_id,
                        unresolved_id,
                        relation="CALLS"
                    )

    # ============================================================
    # CALLERS
    # ============================================================

    def get_callers_of(
        self,
        method_name: str
    ) -> List[str]:

        target = (
            method_name
            if method_name.endswith("()")
            else f"{method_name}()"
        )

        # Exact match
        if self.graph.has_node(target):

            return [
                src
                for src, _, data
                in self.graph.in_edges(
                    target,
                    data=True
                )
                if data.get(
                    "relation"
                ) == "CALLS"
            ]

        # --------------------------------------------------------
        # Fuzzy match by method name
        # --------------------------------------------------------

        target_name = (
            method_name
            .replace("()", "")
            .lower()
        )

        results = []

        for node, attrs in self.graph.nodes(
            data=True
        ):

            if attrs.get(
                "type"
            ) != "METHOD":
                continue

            node_name = attrs.get(
                "name",
                ""
            )

            if (
                node_name.lower()
                == target_name
            ):

                for src, _, data in self.graph.in_edges(
                    node,
                    data=True
                ):

                    if data.get(
                        "relation"
                    ) == "CALLS":

                        results.append(src)

        return list(
            dict.fromkeys(results)
        )

    # ============================================================
    # CALLEES
    # ============================================================

    def get_calls_from(
        self,
        method_name: str
    ) -> List[str]:

        target = (
            method_name
            if method_name.endswith("()")
            else f"{method_name}()"
        )

        if self.graph.has_node(target):

            return [
                dst
                for _, dst, data
                in self.graph.out_edges(
                    target,
                    data=True
                )
                if data.get(
                    "relation"
                ) == "CALLS"
            ]

        return []

    # ============================================================
    # SUMMARY
    # ============================================================

    def get_summary(self) -> dict:

        return {
            "total_nodes":
                self.graph.number_of_nodes(),

            "total_edges":
                self.graph.number_of_edges(),

            "node_types":
                self._count_node_types(),

            "relationship_types":
                self._count_edge_relations()
        }

    # ============================================================
    # NODE COUNTS
    # ============================================================

    def _count_node_types(
        self
    ) -> dict:

        counts = {}

        for _, attrs in self.graph.nodes(
            data=True
        ):

            node_type = attrs.get(
                "type",
                "UNKNOWN"
            )

            counts[node_type] = (
                counts.get(
                    node_type,
                    0
                ) + 1
            )

        return counts

    # ============================================================
    # EDGE COUNTS
    # ============================================================

    def _count_edge_relations(
        self
    ) -> dict:

        counts = {}

        for _, _, attrs in self.graph.edges(
            data=True
        ):

            relation = attrs.get(
                "relation",
                "UNKNOWN"
            )

            counts[relation] = (
                counts.get(
                    relation,
                    0
                ) + 1
            )

        return counts

    # ============================================================
    # CLASS METHODS
    # ============================================================

    def get_class_methods(
        self,
        class_name: str
    ) -> List[str]:

        if not self.graph.has_node(
            class_name
        ):
            return []

        return [
            dst
            for _, dst, data
            in self.graph.out_edges(
                class_name,
                data=True
            )
            if data.get(
                "relation"
            ) == "HAS_METHOD"
        ]

    # ============================================================
    # CLASS CONTAINING METHOD
    # ============================================================

    def get_class_containing(
        self,
        method_name: str
    ) -> List[str]:

        results = []

        for src, dst, data in self.graph.edges(
            data=True
        ):

            if data.get(
                "relation"
            ) == "HAS_METHOD":

                if method_name in dst:

                    results.append(src)

        return list(
            set(results)
        )

    # ============================================================
    # CLASS DEPENDENCIES
    # ============================================================

    def inspect_class_dependencies(
        self,
        class_name: str
    ) -> dict:

        if class_name not in self.graph:

            return {
                "error":
                    f"Class '{class_name}' not in graph."
            }

        relations = {
            "HAS_METHOD": [],
            "INJECTS": [],
            "CALLS": [],
            "EXTENDS": [],
            "IMPLEMENTS": []
        }

        for successor in self.graph.successors(
            class_name
        ):

            edge_data = self.graph.get_edge_data(
                class_name,
                successor
            )

            relation = (
                edge_data.get("relation")
                if edge_data
                else None
            )

            if relation in relations:

                relations[
                    relation
                ].append(
                    successor
                )

        return {
            "class": class_name,
            "relationships": relations
        }

    # ============================================================
    # EXPORT GRAPH
    # ============================================================

    def export_graphml(
        self,
        output_path: str =
            "workspace/code_graph.graphml"
    ):

        out_file = Path(
            output_path
        )

        out_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        nx.write_graphml(
            self.graph,
            out_file
        )

        print(
            f"[Graph] Exported GraphML to: "
            f"{out_file.resolve()}"
        )