from typing import Any, Dict, List, Optional
import tree_sitter

class JavaSymbolExtractor:
    def __init__(self, ast_parser):
        self.parser = ast_parser

    def extract_symbols(self, node: tree_sitter.Node, source_code: str) -> Dict[str, Any]:
        """Extracts packages, imports, classes, interfaces, inheritance, fields, methods, and calls."""
        symbols = {
            "package": None,
            "imports": [],
            "classes": [],
            "interfaces": [],
            "methods": [],
            "fields": [],
            "method_calls": []
        }

        self._walk_tree(node, source_code, symbols, current_class=None)
        return symbols

    def _walk_tree(
        self,
        node: tree_sitter.Node,
        source_code: str,
        symbols: Dict[str, Any],
        current_class: Optional[str] = None
    ):
        node_type = node.type

        # 1. Package Declaration
        if node_type == "package_declaration":
            for child in node.children:
                if child.type == "scoped_identifier" or child.type == "identifier":
                    symbols["package"] = self.parser.get_node_text(child, source_code)

        # 2. Import Declarations
        elif node_type == "import_declaration":
            import_path = self.parser.get_node_text(node, source_code).replace("import ", "").replace(";", "").strip()
            symbols["imports"].append(import_path)

        # 3. Class Declaration
        elif node_type == "class_declaration":
            class_name = None
            annotations = []
            extends_class = None
            implements_interfaces = []

            # Check annotations before class definition
            annotations = self._extract_annotations(node, source_code)

            for child in node.children:
                if child.type == "identifier":
                    class_name = self.parser.get_node_text(child, source_code)
                elif child.type == "superclass":
                    # Superclass identification
                    for sub in child.children:
                        if sub.type in ("type_identifier", "scoped_type_identifier"):
                            extends_class = self.parser.get_node_text(sub, source_code)
                elif child.type == "super_interfaces":
                    # Interface implementations
                    for sub in child.children:
                        if sub.type == "type_list":
                            for type_node in sub.children:
                                if type_node.type in ("type_identifier", "scoped_type_identifier"):
                                    implements_interfaces.append(self.parser.get_node_text(type_node, source_code))

            if class_name:
                symbols["classes"].append({
                    "name": class_name,
                    "annotations": annotations,
                    "extends": extends_class,
                    "implements": implements_interfaces
                })
                current_class = class_name

        # 4. Interface Declaration
        elif node_type == "interface_declaration":
            for child in node.children:
                if child.type == "identifier":
                    symbols["interfaces"].append(self.parser.get_node_text(child, source_code))

        # 5. Field Declaration (Field Injection Detection)
        elif node_type == "field_declaration" and current_class:
            field_annotations = self._extract_annotations(node, source_code)
            field_type = None
            field_name = None

            for child in node.children:
                if child.type in ("type_identifier", "scoped_type_identifier", "generic_type"):
                    field_type = self.parser.get_node_text(child, source_code)
                elif child.type == "variable_declarator":
                    for sub in child.children:
                        if sub.type == "identifier":
                            field_name = self.parser.get_node_text(sub, source_code)

            if field_type and field_name:
                symbols["fields"].append({
                    "name": field_name,
                    "type": field_type,
                    "annotations": field_annotations,
                    "enclosing_class": current_class
                })

        # 6. Method Declaration
        elif node_type == "method_declaration" and current_class:
            method_name = None
            annotations = self._extract_annotations(node, source_code)

            for child in node.children:
                if child.type == "identifier":
                    method_name = self.parser.get_node_text(child, source_code)

            if method_name:
                symbols["methods"].append({
                    "name": method_name,
                    "annotations": annotations,
                    "enclosing_class": current_class
                })

        # 7. Method Invocation (e.g., userService.findUser())
        elif node_type == "method_invocation" and current_class:
            object_expression = None
            method_called = None

            for child in node.children:
                if child.type == "field_access":
                    object_expression = self.parser.get_node_text(child, source_code)
                elif child.type == "identifier":
                    if object_expression is None and child != node.children[-1]:
                        object_expression = self.parser.get_node_text(child, source_code)
                    else:
                        method_called = self.parser.get_node_text(child, source_code)

            if method_called:
                symbols["method_calls"].append({
                    "caller_class": current_class,
                    "object_expression": object_expression,
                    "method_called": method_called
                })

        # Recursively walk children
        for child in node.children:
            self._walk_tree(child, source_code, symbols, current_class)

    def _extract_annotations(self, node: tree_sitter.Node, source_code: str) -> List[str]:
        """Extracts annotation strings attached to a node."""
        annotations = []
        for child in node.children:
            if child.type == "marker_annotation" or child.type == "annotation":
                annotations.append(self.parser.get_node_text(child, source_code))
            elif child.type == "modifiers":
                for sub in child.children:
                    if sub.type in ("marker_annotation", "annotation"):
                        annotations.append(self.parser.get_node_text(sub, source_code))
        return list(set(annotations))