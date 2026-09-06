
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import networkx as nx
from loguru import logger

from src.data_processing.chunkers import get_tree_sitter_parser


class CodeKnowledgeGraph:
    """
    Generic AST-based Code Knowledge Graph.

    Supported:
        Python
        Java
        JavaScript
        TypeScript

    Main node types:
        FILE
        CLASS
        INTERFACE
        FUNCTION
        METHOD
        CONSTRUCTOR

    Main relationships:
        CONTAINS
        CALLS
        IMPORTS
        EXTENDS
        IMPLEMENTS

    The graph is repository-independent.
    """

    def __init__(self):
        self.graph = nx.DiGraph()

        # --------------------------------------------------------
        # Symbol indexes
        # --------------------------------------------------------

        self.symbols_by_name: Dict[str, List[str]] = {}
        self.functions_by_name: Dict[str, List[str]] = {}
        self.classes_by_name: Dict[str, List[str]] = {}

        self.methods_by_class: Dict[
            Tuple[str, str],
            List[str]
        ] = {}

        self.symbols_by_file_name: Dict[
            Tuple[str, str],
            List[str]
        ] = {}

        # Used to resolve calls more reliably.
        self.symbols_by_qualified_name: Dict[
            str,
            List[str]
        ] = {}

    # ============================================================
    # PUBLIC API
    # ============================================================

    def build_graph_from_documents(
        self,
        documents
    ) -> nx.DiGraph:
        """
        Build the knowledge graph directly from source documents.

        This is the preferred graph-building path.

        Each document should expose:

            file_path
            content
            language
        """

        self._reset()

        documents = list(documents or [])

        logger.info(
            f"[Graph] Building AST graph from "
            f"{len(documents)} documents"
        )

        parsed_documents = []

        # ========================================================
        # PASS 1
        # Parse files and register ALL definitions.
        # ========================================================

        for document in documents:

            try:
                file_path = getattr(
                    document,
                    "file_path",
                    None
                )

                content = getattr(
                    document,
                    "content",
                    None
                )

                language = getattr(
                    document,
                    "language",
                    None
                )

                if not file_path:
                    logger.debug(
                        "[Graph] Skipping document without file_path"
                    )
                    continue

                if content is None:
                    logger.debug(
                        f"[Graph] Skipping empty document: "
                        f"{file_path}"
                    )
                    continue

                if not language:
                    logger.debug(
                        f"[Graph] Skipping document without language: "
                        f"{file_path}"
                    )
                    continue

                language = str(language).lower().strip()

                parser = get_tree_sitter_parser(
                    language
                )

                if parser is None:
                    logger.debug(
                        f"[Graph] No Tree-sitter parser for "
                        f"{file_path} ({language})"
                    )
                    continue

                source_bytes = content.encode(
                    "utf-8"
                )

                tree = parser.parse(
                    source_bytes
                )

                parsed_documents.append(
                    (
                        document,
                        tree,
                        source_bytes,
                        language
                    )
                )

                # ------------------------------------------------
                # Add FILE node
                # ------------------------------------------------

                self._add_file_node(
                    document
                )

                # ------------------------------------------------
                # Register classes/functions/methods
                # ------------------------------------------------

                self._register_definitions(
                    tree.root_node,
                    document,
                    source_bytes,
                    language
                )

            except Exception as exc:

                logger.exception(
                    f"[Graph] Failed parsing "
                    f"{getattr(document, 'file_path', 'unknown')}: "
                    f"{exc}"
                )

        logger.info(
            f"[Graph] Pass 1 complete: "
            f"{self.graph.number_of_nodes()} nodes"
        )

        # ========================================================
        # PASS 2
        # Extract CALLS / IMPORTS / inheritance.
        # ========================================================

        for (
            document,
            tree,
            source_bytes,
            language
        ) in parsed_documents:

            try:

                self._extract_relationships(
                    tree.root_node,
                    document,
                    source_bytes,
                    language
                )

            except Exception as exc:

                logger.exception(
                    f"[Graph] Relationship extraction failed for "
                    f"{document.file_path}: {exc}"
                )

        logger.info(
            f"[Graph] Graph construction complete: "
            f"{self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

        logger.info(
            f"[Graph] Summary: {self.get_summary()}"
        )

        return self.graph

    # ============================================================
    # RESET
    # ============================================================

    def _reset(self):

        self.graph.clear()

        self.symbols_by_name.clear()
        self.functions_by_name.clear()
        self.classes_by_name.clear()
        self.methods_by_class.clear()
        self.symbols_by_file_name.clear()
        self.symbols_by_qualified_name.clear()

    # ============================================================
    # FILE NODE
    # ============================================================

    def _add_file_node(
        self,
        document
    ):

        file_path = self._normalize_path(
            getattr(
                document,
                "file_path",
                ""
            )
        )

        if not file_path:
            return

        file_id = self._file_id(
            file_path
        )

        content = getattr(
            document,
            "content",
            ""
        )

        self.graph.add_node(
            file_id,
            type="FILE",
            name=Path(file_path).name,
            qualified_name=file_path,
            file=file_path,
            language=getattr(
                document,
                "language",
                None
            ),
            line_start=1,
            line_end=content.count("\n") + 1,
            code_content=content
        )

    # ============================================================
    # DEFINITION REGISTRATION
    # ============================================================

    def _register_definitions(
        self,
        node,
        document,
        source_bytes: bytes,
        language: str,
        current_class: Optional[str] = None,
        current_class_id: Optional[str] = None
    ):
        """
        Recursively discover classes, functions and methods.

        IMPORTANT:
        This pass only registers definitions.

        CALLS are extracted later after every symbol is known.
        """

        node_type = node.type

        file_path = self._normalize_path(
            document.file_path
        )

        # ========================================================
        # CLASS
        # ========================================================

        if node_type in {
            "class_definition",
            "class_declaration"
        }:

            name = self._node_name(
                node
            )

            if name:

                class_id = self._class_id(
                    file_path,
                    name,
                    node.start_point[0] + 1
                )

                self._add_symbol_node(
                    class_id,
                    "CLASS",
                    name,
                    document,
                    node,
                    current_class=None
                )

                parent_id = (
                    current_class_id
                    if current_class_id
                    else self._file_id(file_path)
                )

                self._add_contains(
                    parent_id,
                    class_id
                )

                self._register_class_indexes(
                    name,
                    class_id,
                    file_path
                )

                # Continue inside class.
                for child in node.children:

                    self._register_definitions(
                        child,
                        document,
                        source_bytes,
                        language,
                        current_class=name,
                        current_class_id=class_id
                    )

                return

        # ========================================================
        # INTERFACE
        # ========================================================

        if node_type == "interface_declaration":

            name = self._node_name(
                node
            )

            if name:

                interface_id = self._class_id(
                    file_path,
                    name,
                    node.start_point[0] + 1
                )

                self._add_symbol_node(
                    interface_id,
                    "INTERFACE",
                    name,
                    document,
                    node,
                    current_class=None
                )

                parent_id = (
                    current_class_id
                    if current_class_id
                    else self._file_id(file_path)
                )

                self._add_contains(
                    parent_id,
                    interface_id
                )

                self._register_class_indexes(
                    name,
                    interface_id,
                    file_path
                )

                for child in node.children:

                    self._register_definitions(
                        child,
                        document,
                        source_bytes,
                        language,
                        current_class=name,
                        current_class_id=interface_id
                    )

                return

        # ========================================================
        # PYTHON FUNCTION / METHOD
        # ========================================================

        if (
            language == "python"
            and node_type in {
                "function_definition",
                "async_function_definition"
            }
        ):

            name = self._node_name(
                node
            )

            if name:

                if current_class:

                    symbol_type = "METHOD"

                    symbol_id = self._method_id(
                        file_path,
                        current_class,
                        name,
                        node.start_point[0] + 1
                    )

                    parent_id = current_class_id

                else:

                    symbol_type = "FUNCTION"

                    symbol_id = self._function_id(
                        file_path,
                        name,
                        node.start_point[0] + 1
                    )

                    parent_id = self._file_id(
                        file_path
                    )

                self._add_symbol_node(
                    symbol_id,
                    symbol_type,
                    name,
                    document,
                    node,
                    current_class=current_class
                )

                self._add_contains(
                    parent_id,
                    symbol_id
                )

                self._register_symbol_indexes(
                    name,
                    symbol_id,
                    file_path
                )

                if current_class:

                    self._register_method_index(
                        current_class,
                        name,
                        symbol_id
                    )

                # Do NOT return before scanning nested definitions.
                for child in node.children:

                    self._register_definitions(
                        child,
                        document,
                        source_bytes,
                        language,
                        current_class=current_class,
                        current_class_id=current_class_id
                    )

                return

        # ========================================================
        # JAVA METHOD / CONSTRUCTOR
        # ========================================================

        if node_type in {
            "method_declaration",
            "constructor_declaration"
        }:

            name = self._node_name(
                node
            )

            if not name:
                name = "constructor"

            if current_class:

                symbol_type = (
                    "CONSTRUCTOR"
                    if node_type == "constructor_declaration"
                    else "METHOD"
                )

                symbol_id = self._method_id(
                    file_path,
                    current_class,
                    name,
                    node.start_point[0] + 1
                )

                parent_id = current_class_id

            else:

                symbol_type = "FUNCTION"

                symbol_id = self._function_id(
                    file_path,
                    name,
                    node.start_point[0] + 1
                )

                parent_id = self._file_id(
                    file_path
                )

            self._add_symbol_node(
                symbol_id,
                symbol_type,
                name,
                document,
                node,
                current_class=current_class
            )

            self._add_contains(
                parent_id,
                symbol_id
            )

            self._register_symbol_indexes(
                name,
                symbol_id,
                file_path
            )

            if current_class:

                self._register_method_index(
                    current_class,
                    name,
                    symbol_id
                )

            for child in node.children:

                self._register_definitions(
                    child,
                    document,
                    source_bytes,
                    language,
                    current_class=current_class,
                    current_class_id=current_class_id
                )

            return

        # ========================================================
        # JAVASCRIPT / TYPESCRIPT FUNCTION
        # ========================================================

        if node_type in {
            "function_declaration",
            "generator_function_declaration",
            "function"
        }:

            name = self._node_name(
                node
            )

            if name:

                if current_class:

                    symbol_type = "METHOD"

                    symbol_id = self._method_id(
                        file_path,
                        current_class,
                        name,
                        node.start_point[0] + 1
                    )

                    parent_id = current_class_id

                else:

                    symbol_type = "FUNCTION"

                    symbol_id = self._function_id(
                        file_path,
                        name,
                        node.start_point[0] + 1
                    )

                    parent_id = self._file_id(
                        file_path
                    )

                self._add_symbol_node(
                    symbol_id,
                    symbol_type,
                    name,
                    document,
                    node,
                    current_class=current_class
                )

                self._add_contains(
                    parent_id,
                    symbol_id
                )

                self._register_symbol_indexes(
                    name,
                    symbol_id,
                    file_path
                )

                if current_class:

                    self._register_method_index(
                        current_class,
                        name,
                        symbol_id
                    )

                for child in node.children:

                    self._register_definitions(
                        child,
                        document,
                        source_bytes,
                        language,
                        current_class=current_class,
                        current_class_id=current_class_id
                    )

                return

        # ========================================================
        # JAVASCRIPT / TYPESCRIPT METHOD
        # ========================================================

        if node_type == "method_definition":

            name = self._node_name(
                node
            )

            if name and current_class:

                symbol_id = self._method_id(
                    file_path,
                    current_class,
                    name,
                    node.start_point[0] + 1
                )

                self._add_symbol_node(
                    symbol_id,
                    "METHOD",
                    name,
                    document,
                    node,
                    current_class=current_class
                )

                self._add_contains(
                    current_class_id,
                    symbol_id
                )

                self._register_symbol_indexes(
                    name,
                    symbol_id,
                    file_path
                )

                self._register_method_index(
                    current_class,
                    name,
                    symbol_id
                )

                for child in node.children:

                    self._register_definitions(
                        child,
                        document,
                        source_bytes,
                        language,
                        current_class=current_class,
                        current_class_id=current_class_id
                    )

                return

        # ========================================================
        # GENERIC RECURSION
        # ========================================================

        for child in node.children:

            self._register_definitions(
                child,
                document,
                source_bytes,
                language,
                current_class=current_class,
                current_class_id=current_class_id
            )

    # ============================================================
    # RELATIONSHIP EXTRACTION
    # ============================================================

    def _extract_relationships(
        self,
        node,
        document,
        source_bytes: bytes,
        language: str,
        current_callable: Optional[str] = None,
        current_class: Optional[str] = None
    ):

        node_type = node.type

        callable_id = current_callable
        class_name = current_class

        # ========================================================
        # Detect current class
        # ========================================================

        if node_type in {
            "class_definition",
            "class_declaration",
            "interface_declaration"
        }:

            name = self._node_name(
                node
            )

            if name:
                class_name = name

        # ========================================================
        # Detect current callable
        # ========================================================

        if node_type in {
            "function_definition",
            "async_function_definition",
            "function_declaration",
            "generator_function_declaration",
            "function",
            "method_declaration",
            "method_definition",
            "constructor_declaration"
        }:

            callable_id = self._find_symbol_at_location(
                document.file_path,
                node.start_point[0] + 1,
                {
                    "FUNCTION",
                    "METHOD",
                    "CONSTRUCTOR"
                }
            )

        # ========================================================
        # CALL
        # ========================================================

        if node_type in {
            "call",
            "call_expression",
            "method_invocation"
        }:

            if callable_id:

                target_name, receiver = (
                    self._extract_call_target(
                        node,
                        language,
                        source_bytes
                    )
                )

                if target_name:

                    target_id = (
                        self._resolve_call_target(
                            target_name=target_name,
                            receiver=receiver,
                            source_file=document.file_path,
                            current_class=class_name,
                            language=language
                        )
                    )

                    if (
                        target_id
                        and target_id != callable_id
                    ):

                        self.graph.add_edge(
                            callable_id,
                            target_id,
                            relation="CALLS"
                        )

                        logger.debug(
                            f"[Graph] CALLS: "
                            f"{self.graph.nodes[callable_id].get('qualified_name')} "
                            f"-> "
                            f"{self.graph.nodes[target_id].get('qualified_name')}"
                        )

        # ========================================================
        # IMPORT
        # ========================================================

        if node_type in {
            "import_statement",
            "import_declaration",
            "import_from_statement"
        }:

            self._extract_import(
                node,
                document,
                language,
                source_bytes
            )

        # ========================================================
        # RECURSE
        # ========================================================

        for child in node.children:

            self._extract_relationships(
                child,
                document,
                source_bytes,
                language,
                current_callable=callable_id,
                current_class=class_name
            )

    # ============================================================
    # CALL TARGET EXTRACTION
    # ============================================================

    def _extract_call_target(
        self,
        node,
        language: str,
        source_bytes: bytes
    ) -> Tuple[
        Optional[str],
        Optional[str]
    ]:

        # ========================================================
        # PYTHON
        # ========================================================

        if language == "python":

            function_node = (
                node.child_by_field_name(
                    "function"
                )
            )

            if not function_node:
                return None, None

            text = self._node_text(
                function_node,
                source_bytes
            ).strip()

            if not text:
                return None, None

            # self.foo()
            # obj.foo()
            # module.foo()

            if "." in text:

                parts = text.split(".")

                return (
                    parts[-1],
                    ".".join(parts[:-1])
                )

            return text, None

        # ========================================================
        # JAVASCRIPT / TYPESCRIPT
        # ========================================================

        if language in {
            "javascript",
            "typescript"
        }:

            function_node = (
                node.child_by_field_name(
                    "function"
                )
            )

            if not function_node:
                return None, None

            if function_node.type == "identifier":

                return (
                    self._node_text(
                        function_node,
                        source_bytes
                    ),
                    None
                )

            if function_node.type in {
                "member_expression",
                "optional_member_expression"
            }:

                property_node = (
                    function_node.child_by_field_name(
                        "property"
                    )
                )

                object_node = (
                    function_node.child_by_field_name(
                        "object"
                    )
                )

                if property_node:

                    target_name = (
                        self._node_text(
                            property_node,
                            source_bytes
                        )
                    )

                    receiver = (
                        self._node_text(
                            object_node,
                            source_bytes
                        )
                        if object_node
                        else None
                    )

                    return (
                        target_name,
                        receiver
                    )

        # ========================================================
        # JAVA
        # ========================================================

        if language == "java":

            name_node = (
                node.child_by_field_name(
                    "name"
                )
            )

            if not name_node:
                return None, None

            target_name = self._node_text(
                name_node,
                source_bytes
            )

            object_node = (
                node.child_by_field_name(
                    "object"
                )
            )

            receiver = (
                self._node_text(
                    object_node,
                    source_bytes
                )
                if object_node
                else None
            )

            return (
                target_name,
                receiver
            )

        return None, None

    # ============================================================
    # CALL RESOLUTION
    # ============================================================

    def _resolve_call_target(
        self,
        target_name: str,
        receiver: Optional[str],
        source_file,
        current_class: Optional[str],
        language: str
    ) -> Optional[str]:

        target_name = target_name.strip()

        if not target_name:
            return None

        # ========================================================
        # self.foo() / this.foo()
        # ========================================================

        if receiver in {
            "self",
            "this"
        } and current_class:

            candidates = (
                self.methods_by_class.get(
                    (
                        current_class,
                        target_name
                    ),
                    []
                )
            )

            if candidates:
                return candidates[0]

        # ========================================================
        # ClassName.foo()
        # ========================================================

        if receiver:

            receiver_clean = (
                receiver.split(".")[-1]
            )

            class_candidates = (
                self.classes_by_name.get(
                    receiver_clean,
                    []
                )
            )

            for class_id in class_candidates:

                attrs = self.graph.nodes.get(
                    class_id,
                    {}
                )

                class_name = attrs.get(
                    "name"
                )

                if class_name:

                    candidates = (
                        self.methods_by_class.get(
                            (
                                class_name,
                                target_name
                            ),
                            []
                        )
                    )

                    if candidates:
                        return candidates[0]

        # ========================================================
        # Same class
        # ========================================================

        if current_class:

            candidates = (
                self.methods_by_class.get(
                    (
                        current_class,
                        target_name
                    ),
                    []
                )
            )

            if candidates:
                return candidates[0]

        # ========================================================
        # Same file
        # ========================================================

        file_name = self._normalize_path(
            source_file
        )

        candidates = (
            self.symbols_by_file_name.get(
                (
                    file_name,
                    target_name
                ),
                []
            )
        )

        if len(candidates) == 1:
            return candidates[0]

        # ========================================================
        # Globally unique symbol
        # ========================================================

        candidates = (
            self.symbols_by_name.get(
                target_name,
                []
            )
        )

        if len(candidates) == 1:
            return candidates[0]

        # ========================================================
        # Globally unique function
        # ========================================================

        candidates = (
            self.functions_by_name.get(
                target_name,
                []
            )
        )

        if len(candidates) == 1:
            return candidates[0]

        return None

    # ============================================================
    # IMPORTS
    # ============================================================

    def _extract_import(
        self,
        node,
        document,
        language,
        source_bytes
    ):

        file_path = self._normalize_path(
            document.file_path
        )

        file_id = self._file_id(
            file_path
        )

        if file_id not in self.graph:
            return

        text = self._node_text(
            node,
            source_bytes
        ).strip()

        if not text:
            return

        imports = self.graph.nodes[
            file_id
        ].get(
            "imports",
            []
        )

        if text not in imports:

            imports.append(
                text
            )

        self.graph.nodes[
            file_id
        ]["imports"] = imports

    # ============================================================
    # NODE CREATION
    # ============================================================

    def _add_symbol_node(
        self,
        node_id,
        node_type,
        name,
        document,
        node,
        current_class=None
    ):

        file_path = self._normalize_path(
            document.file_path
        )

        if current_class:

            qualified_name = (
                f"{current_class}.{name}"
            )

        else:

            qualified_name = name

        self.graph.add_node(
            node_id,
            type=node_type,
            name=name,
            qualified_name=qualified_name,
            class_name=current_class,
            file=file_path,
            language=document.language,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            code_content=self._node_text(
                node,
                document.content.encode("utf-8")
            )
        )

    def _add_contains(
        self,
        parent_id,
        child_id
    ):

        if parent_id and child_id:

            self.graph.add_edge(
                parent_id,
                child_id,
                relation="CONTAINS"
            )

    # ============================================================
    # INDEXES
    # ============================================================

    def _register_symbol_indexes(
        self,
        name,
        node_id,
        file_path
    ):

        self.symbols_by_name.setdefault(
            name,
            []
        ).append(
            node_id
        )

        node_type = self.graph.nodes[
            node_id
        ].get(
            "type"
        )

        if node_type == "FUNCTION":

            self.functions_by_name.setdefault(
                name,
                []
            ).append(
                node_id
            )

        normalized_file = (
            self._normalize_path(
                file_path
            )
        )

        self.symbols_by_file_name.setdefault(
            (
                normalized_file,
                name
            ),
            []
        ).append(
            node_id
        )

        qualified_name = (
            self.graph.nodes[node_id].get(
                "qualified_name"
            )
        )

        if qualified_name:

            self.symbols_by_qualified_name.setdefault(
                qualified_name,
                []
            ).append(
                node_id
            )

    def _register_class_indexes(
        self,
        name,
        node_id,
        file_path
    ):

        self.classes_by_name.setdefault(
            name,
            []
        ).append(
            node_id
        )

        self.symbols_by_name.setdefault(
            name,
            []
        ).append(
            node_id
        )

        normalized_file = (
            self._normalize_path(
                file_path
            )
        )

        self.symbols_by_file_name.setdefault(
            (
                normalized_file,
                name
            ),
            []
        ).append(
            node_id
        )

        qualified_name = (
            self.graph.nodes[node_id].get(
                "qualified_name"
            )
        )

        if qualified_name:

            self.symbols_by_qualified_name.setdefault(
                qualified_name,
                []
            ).append(
                node_id
            )

    def _register_method_index(
        self,
        class_name,
        method_name,
        node_id
    ):

        self.methods_by_class.setdefault(
            (
                class_name,
                method_name
            ),
            []
        ).append(
            node_id
        )

    # ============================================================
    # CALLER / CALLEE API
    # ============================================================

    def get_calls_from(
        self,
        method_name: str
    ) -> List[Dict[str, Any]]:

        nodes = self._resolve_method_nodes(
            method_name
        )

        results = []
        seen = set()

        for node_id in nodes:

            if node_id not in self.graph:
                continue

            for _, target_id, data in self.graph.out_edges(
                node_id,
                data=True
            ):

                if data.get(
                    "relation"
                ) != "CALLS":
                    continue

                if target_id in seen:
                    continue

                seen.add(
                    target_id
                )

                results.append(
                    self._node_to_result(
                        target_id
                    )
                )

        return results

    def get_callers_of(
        self,
        method_name: str
    ) -> List[Dict[str, Any]]:

        nodes = self._resolve_method_nodes(
            method_name
        )

        results = []
        seen = set()

        for node_id in nodes:

            if node_id not in self.graph:
                continue

            for source_id, _, data in self.graph.in_edges(
                node_id,
                data=True
            ):

                if data.get(
                    "relation"
                ) != "CALLS":
                    continue

                if source_id in seen:
                    continue

                seen.add(
                    source_id
                )

                results.append(
                    self._node_to_result(
                        source_id
                    )
                )

        return results

    # ============================================================
    # RESOLVE METHOD
    # ============================================================

    def _resolve_method_nodes(
        self,
        method_name: str
    ) -> List[str]:

        method_name = (
            method_name.strip()
        )

        # Exact symbol name.
        if method_name in self.symbols_by_name:

            return self.symbols_by_name[
                method_name
            ]

        # Exact qualified name.
        if method_name in (
            self.symbols_by_qualified_name
        ):

            return self.symbols_by_qualified_name[
                method_name
            ]

        # Class.method
        if "." in method_name:

            class_name, name = (
                method_name.rsplit(
                    ".",
                    1
                )
            )

            candidates = (
                self.methods_by_class.get(
                    (
                        class_name,
                        name
                    ),
                    []
                )
            )

            if candidates:
                return candidates

        # Final scan.
        matches = []

        for node_id, attrs in self.graph.nodes(
            data=True
        ):

            if attrs.get(
                "name"
            ) == method_name:

                if attrs.get(
                    "type"
                ) in {
                    "FUNCTION",
                    "METHOD",
                    "CONSTRUCTOR"
                }:

                    matches.append(
                        node_id
                    )

        return matches

    # ============================================================
    # GRAPH SEARCH
    # ============================================================

    def search_graph(
        self,
        query: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:

        query_lower = (
            query.lower()
        )

        results = []

        for node_id, attrs in self.graph.nodes(
            data=True
        ):

            searchable = " ".join(
                str(
                    attrs.get(
                        key,
                        ""
                    )
                )
                for key in [
                    "name",
                    "qualified_name",
                    "file",
                    "type"
                ]
            ).lower()

            if query_lower in searchable:

                results.append(
                    self._node_to_result(
                        node_id
                    )
                )

                if len(results) >= limit:
                    break

        return results

    # ============================================================
    # CLASS HELPERS
    # ============================================================

    def get_class_methods(
        self,
        class_name: str
    ) -> List[Dict[str, Any]]:

        return [
            self._node_to_result(
                node_id
            )
            for node_id in self.methods_by_class.get(
                class_name,
                []
            )
        ]

    def get_class_containing(
        self,
        method_name: str
    ) -> Optional[Dict[str, Any]]:

        nodes = self._resolve_method_nodes(
            method_name
        )

        for node_id in nodes:

            attrs = self.graph.nodes.get(
                node_id,
                {}
            )

            class_name = attrs.get(
                "class_name"
            )

            if class_name:

                candidates = (
                    self.classes_by_name.get(
                        class_name,
                        []
                    )
                )

                if candidates:

                    return self._node_to_result(
                        candidates[0]
                    )

        return None

    def inspect_class_dependencies(
        self,
        class_name: str
    ) -> Dict[str, Any]:

        classes = self.classes_by_name.get(
            class_name,
            []
        )

        if not classes:

            return {
                "class": class_name,
                "dependencies": []
            }

        class_id = classes[0]

        dependencies = []

        for _, target_id, data in self.graph.out_edges(
            class_id,
            data=True
        ):

            relation = data.get(
                "relation"
            )

            if relation in {
                "CALLS",
                "IMPORTS",
                "EXTENDS",
                "IMPLEMENTS"
            }:

                dependencies.append(
                    self._node_to_result(
                        target_id
                    )
                )

        return {
            "class": class_name,
            "dependencies": dependencies
        }

    # ============================================================
    # SUMMARY
    # ============================================================

    def get_summary(
        self
    ) -> Dict[str, Any]:

        return {
            "total_nodes": (
                self.graph.number_of_nodes()
            ),
            "total_edges": (
                self.graph.number_of_edges()
            ),
            "node_types": (
                self._count_node_types()
            ),
            "relationship_types": (
                self._count_edge_relations()
            )
        }

    def _count_node_types(
        self
    ):

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
    ):

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
    # GRAPHML
    # ============================================================

    def export_graphml(
        self,
        output_path: str
    ):

        graph_copy = self.graph.copy()

        for _, attrs in graph_copy.nodes(
            data=True
        ):

            for key, value in list(
                attrs.items()
            ):

                if isinstance(
                    value,
                    (
                        list,
                        dict,
                        tuple,
                        set
                    )
                ):

                    attrs[key] = str(
                        value
                    )

        for _, _, attrs in graph_copy.edges(
            data=True
        ):

            for key, value in list(
                attrs.items()
            ):

                if isinstance(
                    value,
                    (
                        list,
                        dict,
                        tuple,
                        set
                    )
                ):

                    attrs[key] = str(
                        value
                    )

        nx.write_graphml(
            graph_copy,
            output_path
        )

    # ============================================================
    # LOCATION RESOLUTION
    # ============================================================

    def _find_symbol_at_location(
        self,
        file_path,
        line,
        allowed_types
    ) -> Optional[str]:

        normalized = (
            self._normalize_path(
                file_path
            )
        )

        candidates = []

        for node_id, attrs in self.graph.nodes(
            data=True
        ):

            if attrs.get(
                "file"
            ) != normalized:

                continue

            if attrs.get(
                "type"
            ) not in allowed_types:

                continue

            line_start = attrs.get(
                "line_start",
                0
            )

            line_end = attrs.get(
                "line_end",
                0
            )

            if (
                line_start <= line
                <= line_end
            ):

                size = (
                    line_end
                    - line_start
                )

                candidates.append(
                    (
                        size,
                        node_id
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0]
        )

        return candidates[0][1]

    # ============================================================
    # RESULT
    # ============================================================

    def _node_to_result(
        self,
        node_id: str
    ) -> Dict[str, Any]:

        attrs = dict(
            self.graph.nodes.get(
                node_id,
                {}
            )
        )

        attrs["id"] = node_id

        return attrs

    # ============================================================
    # TREE-SITTER HELPERS
    # ============================================================

    @staticmethod
    def _node_name(
        node
    ) -> Optional[str]:

        try:

            name_node = (
                node.child_by_field_name(
                    "name"
                )
            )

            if name_node:

                text = name_node.text

                if isinstance(
                    text,
                    bytes
                ):

                    return text.decode(
                        "utf-8",
                        errors="replace"
                    )

                return str(text)

        except Exception:
            pass

        # Some grammars use different structures.
        # Search direct identifier children.

        try:

            for child in node.children:

                if child.type in {
                    "identifier",
                    "type_identifier",
                    "property_identifier"
                }:

                    text = child.text

                    if isinstance(
                        text,
                        bytes
                    ):

                        return text.decode(
                            "utf-8",
                            errors="replace"
                        )

                    return str(text)

        except Exception:
            pass

        return None

    @staticmethod
    def _node_text(
        node,
        source_bytes
    ) -> str:

        try:

            return source_bytes[
                node.start_byte:node.end_byte
            ].decode(
                "utf-8",
                errors="replace"
            )

        except Exception:

            return ""

    @staticmethod
    def _normalize_path(
        path
    ) -> str:

        if isinstance(
            path,
            Path
        ):

            path = str(path)

        return str(
            path
        ).replace(
            "\\",
            "/"
        )

    # ============================================================
    # NODE IDS
    # ============================================================

    @staticmethod
    def _file_id(
        file_path
    ) -> str:

        return (
            f"file::{file_path}"
        )

    @staticmethod
    def _class_id(
        file_path,
        name,
        line
    ) -> str:

        return (
            f"class::{file_path}::"
            f"{name}::{line}"
        )

    @staticmethod
    def _function_id(
        file_path,
        name,
        line
    ) -> str:

        return (
            f"function::{file_path}::"
            f"{name}::{line}"
        )

    @staticmethod
    def _method_id(
        file_path,
        class_name,
        name,
        line
    ) -> str:

        return (
            f"method::{file_path}::"
            f"{class_name}::{name}::{line}"
        )

    # ============================================================
    # IDENTIFIER EXTRACTION
    # ============================================================

    @staticmethod
    def _extract_identifiers(
        text: str
    ) -> List[str]:

        return re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            text
        )


# ================================================================
# BACKWARD COMPATIBILITY
# ================================================================

def build_graph(
    repo_name: str,
    extracted_data: Any
) -> CodeKnowledgeGraph:

    """
    Backward-compatible graph builder.

    New code should use:

        CodeKnowledgeGraph().build_graph_from_documents(...)
    """

    graph = CodeKnowledgeGraph()

    logger.warning(
        "[Graph] build_graph(extracted_data) is deprecated. "
        "Use build_graph_from_documents() instead."
    )

    if isinstance(
        extracted_data,
        list
    ):

        graph.build_graph_from_documents(
            extracted_data
        )

    return graph

