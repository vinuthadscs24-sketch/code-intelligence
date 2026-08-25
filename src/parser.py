import tree_sitter_java as tsjava
from tree_sitter import Language, Parser
from pathlib import Path
from typing import Dict, Any, Tuple, List

JAVA_LANGUAGE = Language(tsjava.language())

class JavaASTParser:
    def __init__(self):
        self.parser = Parser(JAVA_LANGUAGE)

    def parse_file(self, file_path: str) -> Tuple[Any, str]:
        """Parses a Java file using Tree-sitter and returns (tree, source_code)."""
        with open(file_path, "r", encoding="utf-8-sig") as f:
            source_code = f.read()
        
        tree = self.parser.parse(bytes(source_code, "utf-8"))
        return tree, source_code

    def extract_symbols_and_relations(self, tree: Any, source_code: str) -> Dict[str, Any]:
        """Traverses the Tree-sitter AST to extract classes and methods."""
        classes: List[str] = []
        methods: List[Dict[str, Any]] = []

        if tree is None or not tree.root_node:
            return {"classes": classes, "methods": methods}

        source_bytes = bytes(source_code, "utf-8")
        current_class = "Global"

        def traverse(node):
            nonlocal current_class
            
            # Record class declaration names
            if node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    class_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
                    classes.append(class_name)
                    previous_class = current_class
                    current_class = class_name
                    
                    for child in node.children:
                        traverse(child)
                    
                    current_class = previous_class
                    return

            # Record method declarations
            if node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                method_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8") if name_node else "unknown"
                
                # Extract annotations
                annotations = []
                modifiers = node.child_by_field_name("modifiers")
                if modifiers:
                    for child in modifiers.children:
                        if child.type in ["marker_annotation", "annotation"]:
                            ann_name = source_bytes[child.start_byte:child.end_byte].decode("utf-8")
                            annotations.append(ann_name)

                # Extract method calls and instantiations within method body
                calls = []
                instantiations = []
                
                def traverse_body(b_node):
                    if b_node.type == "method_invocation":
                        name_n = b_node.child_by_field_name("name")
                        if name_n:
                            calls.append(source_bytes[name_n.start_byte:name_n.end_byte].decode("utf-8"))
                    elif b_node.type == "object_creation_expression":
                        type_n = b_node.child_by_field_name("type")
                        if type_n:
                            instantiations.append(source_bytes[type_n.start_byte:type_n.end_byte].decode("utf-8"))
                    
                    for c in b_node.children:
                        traverse_body(c)

                body = node.child_by_field_name("body")
                if body:
                    traverse_body(body)

                method_code = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                methods.append({
                    "name": method_name,
                    "enclosing_class": current_class,
                    "annotations": annotations,
                    "calls": calls,
                    "instantiations": instantiations,
                    "source_code": method_code,
                    "start_line": start_line,
                    "end_line": end_line
                })
                return

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)

        return {
            "classes": classes,
            "methods": methods
        }

    @staticmethod
    def get_node_text(node: Any, source_code: str) -> str:
        """Extracts text corresponding to a Tree-sitter AST node."""
        if node is None:
            return ""
        source_bytes = bytes(source_code, "utf-8")
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")