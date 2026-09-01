import os
import logging
from typing import Dict, Any, Tuple, List, Optional
from tree_sitter import Language, Parser

# --- Language Grammar Imports ---
import tree_sitter_java as tsjava
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts

# Optional imports for Dart & Kotlin (with safe fallbacks)
try:
    import tree_sitter_dart as tsdart
    HAS_DART = True
except ImportError:
    HAS_DART = False

try:
    import tree_sitter_kotlin as tskotlin
    HAS_KOTLIN = True
except ImportError:
    HAS_KOTLIN = False

logger = logging.getLogger(__name__)

# --- Initialize Languages ---
JAVA_LANG = Language(tsjava.language())
PY_LANG = Language(tspython.language())
JS_LANG = Language(tsjs.language())
TS_LANG = Language(tsts.language_typescript())
DART_LANG = Language(tsdart.language()) if HAS_DART else None
KOTLIN_LANG = Language(tskotlin.language()) if HAS_KOTLIN else None


class BaseASTParser:
    """Base helper class containing common byte/text extraction tools."""
    
    @staticmethod
    def get_text(node: Any, source_bytes: bytes) -> str:
        if node is None:
            return ""
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    @staticmethod
    def extract_annotations(node: Any, source_bytes: bytes) -> List[str]:
        annotations = []
        modifiers = node.child_by_field_name("modifiers")
        if modifiers:
            for child in modifiers.children:
                if child.type in ("marker_annotation", "annotation", "decorator"):
                    annotations.append(BaseASTParser.get_text(child, source_bytes))
        return annotations


class JavaASTParser(BaseASTParser):
    def __init__(self):
        self.parser = Parser(JAVA_LANG)

    def parse_file(self, file_path: str) -> Tuple[Any, str]:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            source_code = f.read()
        tree = self.parser.parse(bytes(source_code, "utf-8"))
        return tree, source_code

    def extract_symbols_and_relations(self, tree: Any, source_code: str) -> Dict[str, Any]:
        classes: List[Dict[str, Any]] = []
        methods: List[Dict[str, Any]] = []
        method_calls: List[Dict[str, Any]] = []
        fields: List[Dict[str, Any]] = []
        interfaces: List[Dict[str, Any]] = []

        if tree is None or not tree.root_node:
            return {"classes": [], "methods": [], "method_calls": [], "interfaces": [], "fields": []}

        source_bytes = bytes(source_code, "utf-8")
        current_class = "Global"

        def extract_method_calls(body, caller_class, caller_method):
            calls, instantiations = [], []
            def visit(node):
                if node.type == "method_invocation":
                    name_node = node.child_by_field_name("name")
                    object_node = node.child_by_field_name("object")
                    method_called = self.get_text(name_node, source_bytes) if name_node else ""
                    object_expression = self.get_text(object_node, source_bytes) if object_node else None

                    if method_called:
                        calls.append({
                            "method_called": method_called,
                            "object_expression": object_expression,
                            "caller_class": caller_class,
                            "caller_method": caller_method,
                            "line": node.start_point[0] + 1
                        })
                elif node.type == "object_creation_expression":
                    type_node = node.child_by_field_name("type")
                    if type_node:
                        instantiations.append(self.get_text(type_node, source_bytes))

                for child in node.children:
                    visit(child)

            visit(body)
            return calls, instantiations

        def traverse(node):
            nonlocal current_class

            if node.type in ("class_declaration", "interface_declaration"):
                is_interface = node.type == "interface_declaration"
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = self.get_text(name_node, source_bytes)
                    annotations = self.extract_annotations(node, source_bytes)
                    
                    if is_interface:
                        interfaces.append(name)
                        
                    classes.append({
                        "name": name,
                        "type": "INTERFACE" if is_interface else "CLASS",
                        "annotations": annotations
                    })

                    prev_class = current_class
                    current_class = name
                    for child in node.children:
                        traverse(child)
                    current_class = prev_class
                    return

            elif node.type == "field_declaration":
                type_node = node.child_by_field_name("type")
                field_type = self.get_text(type_node, source_bytes) if type_node else ""
                annotations = self.extract_annotations(node, source_bytes)
                
                for child in node.children:
                    if child.type == "variable_declarator":
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            fields.append({
                                "name": self.get_text(name_node, source_bytes),
                                "type": field_type,
                                "enclosing_class": current_class,
                                "annotations": annotations
                            })

            elif node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                method_name = self.get_text(name_node, source_bytes) if name_node else "unknown"
                annotations = self.extract_annotations(node, source_bytes)
                method_code = self.get_text(node, source_bytes)
                body = node.child_by_field_name("body")

                calls, instantiations = [], []
                if body:
                    calls, instantiations = extract_method_calls(body, current_class, method_name)

                methods.append({
                    "name": method_name,
                    "enclosing_class": current_class,
                    "annotations": annotations,
                    "calls": calls,
                    "method_calls": calls,
                    "instantiations": instantiations,
                    "source_code": method_code,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "signature": f"{method_name}()"
                })

                for call in calls:
                    method_calls.append({
                        "caller_class": current_class,
                        "caller_method": method_name,
                        "method_called": call.get("method_called"),
                        "object_expression": call.get("object_expression"),
                        "line": call.get("line")
                    })
                return

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return {
            "classes": classes,
            "methods": methods,
            "method_calls": method_calls,
            "interfaces": interfaces,
            "fields": fields
        }


class GenericASTParser(BaseASTParser):
    """Parser supporting Dart, Kotlin, Python, JS, TS using tree-sitter AST nodes."""

    def __init__(self, language: Language):
        self.parser = Parser(language)

    def parse_file(self, file_path: str) -> Tuple[Any, str]:
        with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            source_code = f.read()
        tree = self.parser.parse(bytes(source_code, "utf-8"))
        return tree, source_code

    def extract_symbols_and_relations(self, tree: Any, source_code: str) -> Dict[str, Any]:
        classes: List[Dict[str, Any]] = []
        methods: List[Dict[str, Any]] = []
        method_calls: List[Dict[str, Any]] = []
        fields: List[Dict[str, Any]] = []
        interfaces: List[Dict[str, Any]] = []

        if tree is None or not tree.root_node:
            return {"classes": [], "methods": [], "method_calls": [], "interfaces": [], "fields": []}

        source_bytes = bytes(source_code, "utf-8")
        current_class = "Global"

        CLASS_TYPES = {"class_declaration", "class_definition", "interface_declaration", "mixin_declaration"}
        FUNC_TYPES = {"method_declaration", "function_declaration", "function_definition", "method_definition"}
        CALL_TYPES = {"call_expression", "method_invocation"}

        def traverse(node):
            nonlocal current_class

            # Classes & Interfaces
            if node.type in CLASS_TYPES:
                name_node = node.child_by_field_name("name")
                if name_node:
                    class_name = self.get_text(name_node, source_bytes)
                    is_interface = "interface" in node.type
                    
                    if is_interface:
                        interfaces.append(class_name)
                        
                    classes.append({
                        "name": class_name,
                        "type": "INTERFACE" if is_interface else "CLASS",
                        "annotations": self.extract_annotations(node, source_bytes)
                    })
                    
                    prev_class = current_class
                    current_class = class_name
                    for child in node.children:
                        traverse(child)
                    current_class = prev_class
                    return

            # Functions / Methods
            elif node.type in FUNC_TYPES:
                name_node = node.child_by_field_name("name")
                func_name = self.get_text(name_node, source_bytes) if name_node else "anonymous"
                func_code = self.get_text(node, source_bytes)
                
                # Extract calls inside function
                calls = []
                def extract_calls(n):
                    if n.type in CALL_TYPES:
                        fn_node = n.child_by_field_name("function") or n.child_by_field_name("name")
                        if fn_node:
                            called_name = self.get_text(fn_node, source_bytes)
                            calls.append({
                                "method_called": called_name,
                                "object_expression": None,
                                "caller_class": current_class,
                                "caller_method": func_name,
                                "line": n.start_point[0] + 1
                            })
                    for c in n.children:
                        extract_calls(c)

                extract_calls(node)

                methods.append({
                    "name": func_name,
                    "enclosing_class": current_class,
                    "annotations": self.extract_annotations(node, source_bytes),
                    "calls": calls,
                    "method_calls": calls,
                    "instantiations": [],
                    "source_code": func_code,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "signature": f"{func_name}()"
                })

                for call in calls:
                    method_calls.append({
                        "caller_class": current_class,
                        "caller_method": func_name,
                        "method_called": call.get("method_called"),
                        "object_expression": None,
                        "line": call.get("line")
                    })
                return

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return {
            "classes": classes,
            "methods": methods,
            "method_calls": method_calls,
            "interfaces": interfaces,
            "fields": fields
        }


class CodeParserFactory:
    """Factory to dispatch the correct AST parser based on file extension."""

    @staticmethod
    def get_parser(file_path: str):
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".java":
            return JavaASTParser()
        elif ext == ".dart" and DART_LANG:
            return GenericASTParser(DART_LANG)
        elif ext in (".kt", ".kts") and KOTLIN_LANG:
            return GenericASTParser(KOTLIN_LANG)
        elif ext == ".py":
            return GenericASTParser(PY_LANG)
        elif ext in (".js", ".jsx"):
            return GenericASTParser(JS_LANG)
        elif ext in (".ts", ".tsx"):
            return GenericASTParser(TS_LANG)
        else:
            return None