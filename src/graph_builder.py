import json
import re
import networkx as nx

from pathlib import Path
from typing import List, Dict, Any, Tuple, Union


class CodeKnowledgeGraph:

    def __init__(self):
        self.graph = nx.DiGraph()

    # ============================================================
    # AST-BASED GRAPH
    # ============================================================

    def build_graph(
        self,
        repo_name: str,
        extracted_data: List[
            Tuple[Union[Path, str], dict]
        ],
    ):
        """
        Construct a directed code knowledge graph.

        Supports:
        - Classes
        - Interfaces
        - Inheritance
        - Implementations
        - Dependency injection
        - Methods
        - Method calls
        """

        self.graph.clear()

        self.graph.add_node(
            repo_name,
            type="REPOSITORY"
        )

        class_type_map = {}
        field_type_map = {}

        # ========================================================
        # PHASE 1
        # Register classes, interfaces, inheritance and fields
        # ========================================================

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

            # ----------------------------------------------------
            # Classes
            # ----------------------------------------------------

            for cls in symbols.get(
                "classes",
                []
            ):

                cls_name = cls["name"]

                class_type_map[
                    cls_name
                ] = package

                self.graph.add_node(
                    cls_name,
                    type="CLASS",
                    package=package,
                    annotations=json.dumps(
                        cls.get(
                            "annotations",
                            []
                        )
                    ),
                    file=file_str,
                )

                self.graph.add_edge(
                    repo_name,
                    cls_name,
                    relation="CONTAINS"
                )

                # Inheritance
                if cls.get("extends"):

                    parent = cls["extends"]

                    self.graph.add_node(
                        parent,
                        type="CLASS"
                    )

                    self.graph.add_edge(
                        cls_name,
                        parent,
                        relation="EXTENDS"
                    )

                # Interfaces
                for iface in cls.get(
                    "implements",
                    []
                ):

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
            # Interfaces
            # ----------------------------------------------------

            for iface in symbols.get(
                "interfaces",
                []
            ):

                self.graph.add_node(
                    iface,
                    type="INTERFACE",
                    package=package,
                    file=file_str,
                )

                self.graph.add_edge(
                    repo_name,
                    iface,
                    relation="CONTAINS"
                )

            # ----------------------------------------------------
            # Fields / Dependencies
            # ----------------------------------------------------

            for field in symbols.get(
                "fields",
                []
            ):

                enclosing = field[
                    "enclosing_class"
                ]

                field_name = field[
                    "name"
                ]

                field_type = field[
                    "type"
                ]

                annotations = field.get(
                    "annotations",
                    []
                )

                field_type_map[
                    (enclosing, field_name)
                ] = field_type

                is_injected = any(
                    any(
                        keyword in annotation
                        for keyword in [
                            "@Autowired",
                            "@Inject",
                            "@Resource",
                        ]
                    )
                    for annotation in annotations
                )

                if (
                    is_injected
                    or field_type in class_type_map
                ):

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
                        ),
                    )

        # ========================================================
        # PHASE 2
        # Methods and method calls
        # ========================================================

        method_lookup = {}

        # --------------------------------------------------------
        # Register all methods FIRST
        # --------------------------------------------------------

        for file_path, symbols in extracted_data:

            for method in symbols.get(
                "methods",
                []
            ):

                method_name = method[
                    "name"
                ]

                enclosing = method.get(
                    "enclosing_class"
                )

                if not enclosing:
                    continue

                method_node_id = (
                    f"{enclosing}.{method_name}()"
                )

                self.graph.add_node(
                    method_node_id,
                    type="METHOD",
                    name=method_name,
                    class_name=enclosing,
                    annotations=json.dumps(
                        method.get(
                            "annotations",
                            []
                        )
                    ),
                    file=str(file_path),
                    line_start=method.get(
                        "start_line",
                        1
                    ),
                    line_end=method.get(
                        "end_line",
                        1
                    ),
                    code_content=method.get(
                        "source_code",
                        ""
                    ),
                )

                self.graph.add_edge(
                    enclosing,
                    method_node_id,
                    relation="HAS_METHOD"
                )

                method_lookup.setdefault(
                    method_name,
                    []
                ).append(
                    method_node_id
                )

        # --------------------------------------------------------
        # Resolve method calls
        # --------------------------------------------------------

        for file_path, symbols in extracted_data:

            for call in symbols.get(
                "method_calls",
                []
            ):

                caller_class = call.get(
                    "caller_class"
                )

                caller_method = call.get(
                    "caller_method"
                )

                obj_expr = call.get(
                    "object_expression"
                )

                method_called = call.get(
                    "method_called"
                )

                if not caller_class or not method_called:
                    continue

                # Try to identify the actual caller method
                caller_id = None

                if caller_method:

                    possible_caller = (
                        f"{caller_class}.{caller_method}()"
                    )

                    if self.graph.has_node(
                        possible_caller
                    ):
                        caller_id = possible_caller

                # If caller method is unavailable,
                # fall back to class-level CALLS edge.
                if not caller_id:
                    caller_id = caller_class

                target_class = None

                # ------------------------------------------------
                # Resolve object expression through fields
                # ------------------------------------------------

                if (
                    obj_expr
                    and
                    (
                        caller_class,
                        obj_expr
                    ) in field_type_map
                ):

                    target_class = field_type_map[
                        (
                            caller_class,
                            obj_expr
                        )
                    ]

                # ------------------------------------------------
                # Direct class reference
                # ------------------------------------------------

                elif (
                    obj_expr
                    and
                    obj_expr in class_type_map
                ):

                    target_class = obj_expr

                # ------------------------------------------------
                # Resolve target
                # ------------------------------------------------

                if target_class:

                    target_method_id = (
                        f"{target_class}.{method_called}()"
                    )

                    if not self.graph.has_node(
                        target_method_id
                    ):

                        self.graph.add_node(
                            target_method_id,
                            type="METHOD",
                            name=method_called,
                            class_name=target_class,
                        )

                    self.graph.add_edge(
                        caller_id,
                        target_method_id,
                        relation="CALLS"
                    )

                else:

                    # Try method-name lookup
                    candidates = method_lookup.get(
                        method_called,
                        []
                    )

                    if len(candidates) == 1:

                        self.graph.add_edge(
                            caller_id,
                            candidates[0],
                            relation="CALLS"
                        )

                    else:

                        unresolved_id = (
                            f"{method_called}()"
                        )

                        self.graph.add_node(
                            unresolved_id,
                            type="METHOD_CALL",
                            name=method_called,
                        )

                        self.graph.add_edge(
                            caller_id,
                            unresolved_id,
                            relation="CALLS"
                        )

    # ============================================================
    # CHUNK-BASED GRAPH
    # ============================================================

    def build_graph_from_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ):
        """
        Build graph from CodeChunker output.

        This is the method currently used by api.py.
        """

        self.graph.clear()

        repo_node = "CodeRepository"

        self.graph.add_node(
            repo_node,
            type="REPOSITORY"
        )

        # ========================================================
        # PHASE 1
        # Register ALL classes and methods
        # ========================================================

        method_lookup = {}

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

            # ----------------------------------------------------
            # Class node
            # ----------------------------------------------------

            self.graph.add_node(
                class_name,
                type="CLASS",
                file=file_name,
            )

            self.graph.add_edge(
                repo_node,
                class_name,
                relation="CONTAINS"
            )

            # ----------------------------------------------------
            # Method node
            # ----------------------------------------------------

            method_id = (
                f"{class_name}.{method_name}()"
            )

            self.graph.add_node(
                method_id,
                type="METHOD",
                name=method_name,
                class_name=class_name,
                file=file_name,
                line_start=chunk.get(
                    "start_line",
                    1
                ),
                line_end=chunk.get(
                    "end_line",
                    1
                ),
                code_content=chunk.get(
                    "code_content",
                    ""
                ),
                annotations=chunk.get(
                    "annotations",
                    "[]"
                ),
            )

            self.graph.add_edge(
                class_name,
                method_id,
                relation="HAS_METHOD"
            )

            # ----------------------------------------------------
            # Method lookup
            # ----------------------------------------------------

            method_lookup.setdefault(
                method_name,
                []
            ).append(
                method_id
            )

        # ========================================================
        # PHASE 2
        # Resolve CALLS relationships
        # ========================================================

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

            if not self.graph.has_node(
                caller_id
            ):
                continue

            # CodeChunker currently stores calls under
            # "relationships", but support both.
            calls = (
                chunk.get("calls")
                or chunk.get("relationships")
                or []
            )

            for call in calls:

                if not call:
                    continue

                # Handle dictionaries as well as strings
                if isinstance(call, dict):

                    called_method = (
                        call.get("method_called")
                        or call.get("name")
                        or call.get("method")
                    )

                    object_expression = (
                        call.get(
                            "object_expression"
                        )
                    )

                else:

                    called_method = str(
                        call
                    )

                    object_expression = None

                if not called_method:
                    continue

                called_method = (
                    str(called_method)
                    .strip()
                )

                # Remove ()
                if called_method.endswith(
                    "()"
                ):
                    called_method = (
                        called_method[:-2]
                    )

                # ------------------------------------------------
                # Extract final method name
                #
                # Examples:
                # save
                # helper.save
                # this.save
                # OuterRepository.save
                # ------------------------------------------------

                method_only = (
                    called_method
                    .split(".")[-1]
                )

                # ------------------------------------------------
                # Try exact method lookup
                # ------------------------------------------------

                candidates = method_lookup.get(
                    method_only,
                    []
                )

                # ------------------------------------------------
                # Exact object/class resolution
                # ------------------------------------------------

                target = None

                if object_expression:

                    object_name = str(
                        object_expression
                    ).strip()

                    # Search for Class.method()
                    possible_target = (
                        f"{object_name}.{method_only}()"
                    )

                    if self.graph.has_node(
                        possible_target
                    ):
                        target = possible_target

                # ------------------------------------------------
                # Single method candidate
                # ------------------------------------------------

                if not target and len(
                    candidates
                ) == 1:

                    target = candidates[0]

                # ------------------------------------------------
                # Multiple candidates
                # ------------------------------------------------

                if not target and len(
                    candidates
                ) > 1:

                    # Prefer same class
                    same_class_target = (
                        f"{class_name}.{method_only}()"
                    )

                    if self.graph.has_node(
                        same_class_target
                    ):

                        target = (
                            same_class_target
                        )

                # ------------------------------------------------
                # Add resolved CALLS edge
                # ------------------------------------------------

                if target:

                    self.graph.add_edge(
                        caller_id,
                        target,
                        relation="CALLS"
                    )

                # ------------------------------------------------
                # Otherwise unresolved call
                # ------------------------------------------------

                else:

                    unresolved_id = (
                        f"{method_only}()"
                    )

                    self.graph.add_node(
                        unresolved_id,
                        type="METHOD_CALL",
                        name=method_only,
                    )

                    self.graph.add_edge(
                        caller_id,
                        unresolved_id,
                        relation="CALLS"
                    )

    # ============================================================
    # METHOD RESOLUTION
    # ============================================================

    def _resolve_method_nodes(
        self,
        method_name: str
    ) -> List[str]:
        """
        Resolve a user-provided method name to actual
        METHOD nodes.

        Examples:

            save
            save()
            OuterRepository.save
            OuterRepository.save()
        """

        if not method_name:
            return []

        name = str(
            method_name
        ).strip()

        if name.endswith(
            "()"
        ):
            name = name[:-2]

        results = []

        # --------------------------------------------------------
        # Exact node
        # --------------------------------------------------------

        if self.graph.has_node(
            f"{name}()"
        ):

            node = f"{name}()"

            if self.graph.nodes[node].get(
                "type"
            ) == "METHOD":

                results.append(
                    node
                )

        # --------------------------------------------------------
        # Search actual METHOD nodes
        # --------------------------------------------------------

        for node, data in self.graph.nodes(
            data=True
        ):

            if data.get(
                "type"
            ) != "METHOD":
                continue

            node_str = str(
                node
            )

            node_method_name = data.get(
                "name"
            )

            # Exact method name
            if node_method_name == name:

                results.append(
                    node_str
                )
                continue

            # Class.method
            if node_str == name:

                results.append(
                    node_str
                )
                continue

            # Class.method()
            if node_str == f"{name}()":

                results.append(
                    node_str
                )

        return list(
            dict.fromkeys(
                results
            )
        )

    # ============================================================
    # CALLS
    # ============================================================

    def get_calls_from(
        self,
        method_name: str
    ) -> List[str]:

        targets = self._resolve_method_nodes(
            method_name
        )

        if not targets:
            return []

        results = []

        for target in targets:

            for _, dst, data in self.graph.out_edges(
                target,
                data=True
            ):

                if data.get(
                    "relation"
                ) == "CALLS":

                    results.append(
                        dst
                    )

        return list(
            dict.fromkeys(
                results
            )
        )

    # ============================================================
    # CALLERS
    # ============================================================

    def get_callers_of(
        self,
        method_name: str
    ) -> List[str]:

        targets = self._resolve_method_nodes(
            method_name
        )

        if not targets:
            return []

        results = []

        for target in targets:

            for src, _, data in self.graph.in_edges(
                target,
                data=True
            ):

                if data.get(
                    "relation"
                ) == "CALLS":

                    results.append(
                        src
                    )

        return list(
            dict.fromkeys(
                results
            )
        )

    # ============================================================
    # GRAPH SEARCH
    # ============================================================

    def search_graph(
        self,
        query: str,
        top_k: int = 5,
        vector_store=None
    ) -> List[Dict[str, Any]]:

        raw_words = set(
            re.findall(
                r"[A-Za-z0-9]+",
                query
            )
        )

        query_tokens = {
            word.lower()
            for word in raw_words
            if len(word) > 2
        }

        seed_nodes = set()

        # --------------------------------------------------------
        # Symbol matching
        # --------------------------------------------------------

        for node in self.graph.nodes():

            node_str = str(
                node
            )

            split_words = set(
                re.findall(
                    r"[A-Z]?[a-z]+|"
                    r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|"
                    r"[0-9]+",
                    node_str
                )
            )

            node_tokens = {
                word.lower()
                for word in split_words
                if len(word) > 1
            }

            if (
                query_tokens.intersection(
                    node_tokens
                )
                or any(
                    token in node_str.lower()
                    for token in query_tokens
                )
            ):

                seed_nodes.add(
                    node
                )

        # --------------------------------------------------------
        # Vector fallback
        # --------------------------------------------------------

        if (
            len(seed_nodes) < 3
            and vector_store
            and hasattr(
                vector_store,
                "search"
            )
        ):

            raw_vec = vector_store.search(
                query,
                top_k=5
            )

            for item in raw_vec:

                chunk = (
                    item[0]
                    if isinstance(
                        item,
                        tuple
                    )
                    else item
                )

                cls = chunk.get(
                    "class_name"
                )

                mth = chunk.get(
                    "method_name"
                )

                if cls and self.graph.has_node(
                    cls
                ):

                    seed_nodes.add(
                        cls
                    )

                if cls and mth:

                    method_id = (
                        f"{cls}.{mth}()"
                    )

                    if self.graph.has_node(
                        method_id
                    ):

                        seed_nodes.add(
                            method_id
                        )

        # --------------------------------------------------------
        # Traverse graph
        # --------------------------------------------------------

        visited = set()

        matched_chunks = []

        for seed in seed_nodes:

            if seed in visited:
                continue

            visited.add(
                seed
            )

            node_attrs = (
                self.graph.nodes[
                    seed
                ]
            )

            chunk_dict = {
                "id": seed,
                "class_name": (
                    seed
                    if node_attrs.get(
                        "type"
                    ) == "CLASS"
                    else str(seed).split(
                        "."
                    )[0]
                ),
                "method_name": (
                    node_attrs.get(
                        "name"
                    )
                    or str(seed)
                    .split(".")[-1]
                    .replace(
                        "()",
                        ""
                    )
                ),
                "file_name": node_attrs.get(
                    "file",
                    ""
                ),
                "node_type": node_attrs.get(
                    "type",
                    "UNKNOWN"
                ),
            }

            matched_chunks.append(
                chunk_dict
            )

            for neighbor in self.graph.successors(
                seed
            ):

                if neighbor in visited:
                    continue

                visited.add(
                    neighbor
                )

                attrs = (
                    self.graph.nodes[
                        neighbor
                    ]
                )

                matched_chunks.append({
                    "id": neighbor,
                    "class_name": (
                        neighbor
                        if attrs.get(
                            "type"
                        ) == "CLASS"
                        else str(
                            neighbor
                        ).split(".")[0]
                    ),
                    "method_name": (
                        attrs.get(
                            "name"
                        )
                        or str(
                            neighbor
                        )
                        .split(".")[-1]
                        .replace(
                            "()",
                            ""
                        )
                    ),
                    "file_name": attrs.get(
                        "file",
                        ""
                    ),
                    "node_type": attrs.get(
                        "type",
                        "UNKNOWN"
                    ),
                })

        return matched_chunks[
            :top_k
        ]

    # ============================================================
    # SUMMARY
    # ============================================================

    def get_summary(
        self
    ) -> dict:

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "relationship_types": self._count_edge_relations(),
        }

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
    # GRAPH EXPORT
    # ============================================================

    def export_graphml(
        self,
        output_path: str = (
            "workspace/code_graph.graphml"
        )
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
            "[Graph] Exported GraphML to: "
            f"{out_file.resolve()}"
        )

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
            for _, dst, data in self.graph.out_edges(
                class_name,
                data=True
            )
            if data.get(
                "relation"
            ) == "HAS_METHOD"
        ]

    # ============================================================
    # FIND CLASS CONTAINING METHOD
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

                if method_name in str(
                    dst
                ):

                    results.append(
                        src
                    )

        return list(
            set(
                results
            )
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
                "error": (
                    f"Class '{class_name}' "
                    "not in graph."
                )
            }

        relations = {
            "HAS_METHOD": [],
            "INJECTS": [],
            "CALLS": [],
            "EXTENDS": [],
            "IMPLEMENTS": [],
        }

        for successor in self.graph.successors(
            class_name
        ):

            edge_data = (
                self.graph.get_edge_data(
                    class_name,
                    successor
                )
            )

            relation = (
                edge_data.get(
                    "relation"
                )
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
            "relationships": relations,
        }