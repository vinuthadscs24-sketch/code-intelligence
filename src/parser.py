import tree_sitter_java as tsjava
from tree_sitter import Language, Parser
from typing import Dict, Any, Tuple, List


JAVA_LANGUAGE = Language(tsjava.language())


class JavaASTParser:

    def __init__(self):
        self.parser = Parser(JAVA_LANGUAGE)

    # ============================================================
    # PARSE FILE
    # ============================================================

    def parse_file(
        self,
        file_path: str
    ) -> Tuple[Any, str]:

        with open(
            file_path,
            "r",
            encoding="utf-8-sig"
        ) as f:
            source_code = f.read()

        tree = self.parser.parse(
            bytes(source_code, "utf-8")
        )

        return tree, source_code

    # ============================================================
    # EXTRACT SYMBOLS AND RELATIONS
    # ============================================================

    def extract_symbols_and_relations(
        self,
        tree: Any,
        source_code: str
    ) -> Dict[str, Any]:

        classes: List[Dict[str, Any]] = []
        methods: List[Dict[str, Any]] = []
        method_calls: List[Dict[str, Any]] = []
        fields: List[Dict[str, Any]] = []
        interfaces: List[Dict[str, Any]] = []

        if tree is None or not tree.root_node:
            return {
                "classes": [],
                "methods": [],
                "method_calls": [],
                "interfaces": [],
                "fields": []
            }

        source_bytes = bytes(
            source_code,
            "utf-8"
        )

        current_class = "Global"

        # ========================================================
        # HELPER: GET NODE TEXT
        # ========================================================

        def get_text(node):

            if node is None:
                return ""

            return source_bytes[
                node.start_byte:node.end_byte
            ].decode(
                "utf-8",
                errors="replace"
            )

        # ========================================================
        # HELPER: EXTRACT ANNOTATIONS
        # ========================================================

        def extract_annotations(node):

            annotations = []

            modifiers = node.child_by_field_name(
                "modifiers"
            )

            if modifiers:

                for child in modifiers.children:

                    if child.type in (
                        "marker_annotation",
                        "annotation"
                    ):

                        annotations.append(
                            get_text(child)
                        )

            return annotations

        # ========================================================
        # EXTRACT METHOD CALLS
        # ========================================================

        def extract_method_calls(
            body,
            caller_class,
            caller_method
        ):

            calls = []
            instantiations = []

            def visit(node):

                # ------------------------------------------------
                # METHOD INVOCATION
                # ------------------------------------------------

                if node.type == "method_invocation":

                    name_node = (
                        node.child_by_field_name(
                            "name"
                        )
                    )

                    object_node = (
                        node.child_by_field_name(
                            "object"
                        )
                    )

                    method_called = (
                        get_text(name_node)
                        if name_node
                        else ""
                    )

                    object_expression = (
                        get_text(object_node)
                        if object_node
                        else None
                    )

                    if method_called:

                        calls.append({

                            "method_called":
                                method_called,

                            "object_expression":
                                object_expression,

                            "caller_class":
                                caller_class,

                            "caller_method":
                                caller_method,

                            "line":
                                node.start_point[0] + 1
                        })

                # ------------------------------------------------
                # OBJECT CREATION
                # ------------------------------------------------

                elif node.type == "object_creation_expression":

                    type_node = (
                        node.child_by_field_name(
                            "type"
                        )
                    )

                    if type_node:

                        instantiations.append(
                            get_text(type_node)
                        )

                # ------------------------------------------------
                # CONTINUE AST TRAVERSAL
                # ------------------------------------------------

                for child in node.children:
                    visit(child)

            visit(body)

            return calls, instantiations

        # ========================================================
        # AST TRAVERSAL
        # ========================================================

        def traverse(node):

            nonlocal current_class

            # ====================================================
            # CLASS
            # ====================================================

            if node.type == "class_declaration":

                name_node = (
                    node.child_by_field_name(
                        "name"
                    )
                )

                if name_node:

                    class_name = get_text(
                        name_node
                    )

                    annotations = (
                        extract_annotations(node)
                    )

                    classes.append({

                        "name":
                            class_name,

                        "type":
                            "CLASS",

                        "annotations":
                            annotations
                    })

                    previous_class = current_class

                    current_class = class_name

                    for child in node.children:
                        traverse(child)

                    current_class = previous_class

                    return

            # ====================================================
            # INTERFACE
            # ====================================================

            if node.type == "interface_declaration":

                name_node = (
                    node.child_by_field_name(
                        "name"
                    )
                )

                if name_node:

                    interface_name = get_text(
                        name_node
                    )

                    annotations = (
                        extract_annotations(node)
                    )

                    interfaces.append(
                        interface_name
                    )

                    classes.append({

                        "name":
                            interface_name,

                        "type":
                            "INTERFACE",

                        "annotations":
                            annotations
                    })

                # Continue traversal inside interface
                previous_class = current_class

                if name_node:
                    current_class = get_text(
                        name_node
                    )

                for child in node.children:
                    traverse(child)

                current_class = previous_class

                return

            # ====================================================
            # FIELD DECLARATION
            # ====================================================

            if node.type == "field_declaration":

                type_node = (
                    node.child_by_field_name(
                        "type"
                    )
                )

                field_type = (
                    get_text(type_node)
                    if type_node
                    else ""
                )

                annotations = (
                    extract_annotations(node)
                )

                for child in node.children:

                    if child.type == "variable_declarator":

                        field_name_node = (
                            child.child_by_field_name(
                                "name"
                            )
                        )

                        if field_name_node:

                            field_name = get_text(
                                field_name_node
                            )

                            fields.append({

                                "name":
                                    field_name,

                                "type":
                                    field_type,

                                "enclosing_class":
                                    current_class,

                                "annotations":
                                    annotations
                            })

                # Don't return here because
                # nested structures may exist.

            # ====================================================
            # METHOD
            # ====================================================

            if node.type == "method_declaration":

                name_node = (
                    node.child_by_field_name(
                        "name"
                    )
                )

                method_name = (
                    get_text(name_node)
                    if name_node
                    else "unknown"
                )

                annotations = (
                    extract_annotations(node)
                )

                method_code = get_text(
                    node
                )

                start_line = (
                    node.start_point[0] + 1
                )

                end_line = (
                    node.end_point[0] + 1
                )

                body = (
                    node.child_by_field_name(
                        "body"
                    )
                )

                calls = []
                instantiations = []

                if body:

                    calls, instantiations = (
                        extract_method_calls(
                            body,
                            current_class,
                            method_name
                        )
                    )

                # ------------------------------------------------
                # Store method
                # ------------------------------------------------

                method_data = {

                    "name":
                        method_name,

                    "enclosing_class":
                        current_class,

                    "annotations":
                        annotations,

                    "calls":
                        calls,

                    "method_calls":
                        calls,

                    "instantiations":
                        instantiations,

                    "source_code":
                        method_code,

                    "start_line":
                        start_line,

                    "end_line":
                        end_line,

                    "signature":
                        f"{method_name}()"
                }

                methods.append(
                    method_data
                )

                # ------------------------------------------------
                # Store global method call records
                # ------------------------------------------------

                for call in calls:

                    method_calls.append({

                        "caller_class":
                            current_class,

                        "caller_method":
                            method_name,

                        "method_called":
                            call.get(
                                "method_called"
                            ),

                        "object_expression":
                            call.get(
                                "object_expression"
                            ),

                        "line":
                            call.get(
                                "line"
                            )
                    })

                return

            # ====================================================
            # CONTINUE
            # ====================================================

            for child in node.children:
                traverse(child)

        # ========================================================
        # START TRAVERSAL
        # ========================================================

        traverse(
            tree.root_node
        )

        # ========================================================
        # RETURN
        # ========================================================

        return {

            "classes":
                classes,

            "methods":
                methods,

            "method_calls":
                method_calls,

            "interfaces":
                interfaces,

            "fields":
                fields
        }

    # ============================================================
    # NODE TEXT
    # ============================================================

    @staticmethod
    def get_node_text(
        node: Any,
        source_code: str
    ) -> str:

        if node is None:
            return ""

        source_bytes = bytes(
            source_code,
            "utf-8"
        )

        return source_bytes[
            node.start_byte:node.end_byte
        ].decode(
            "utf-8",
            errors="replace"
        )