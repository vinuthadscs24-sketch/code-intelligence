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
            source_code.encode("utf-8")
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

        classes = []
        methods = []
        method_calls = []
        fields = []
        interfaces = []

        if tree is None or not tree.root_node:
            return {
                "package": "default",
                "classes": [],
                "methods": [],
                "method_calls": [],
                "interfaces": [],
                "fields": []
            }

        source_bytes = source_code.encode("utf-8")

        # ========================================================
        # HELPERS
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

        def find_package(node):
            if node.type == "package_declaration":
                return get_text(node).replace(
                    "package",
                    "",
                    1
                ).replace(
                    ";",
                    ""
                ).strip()

            for child in node.children:
                result = find_package(child)

                if result:
                    return result

            return "default"

        package = find_package(
            tree.root_node
        )

        # ========================================================
        # METHOD CALL EXTRACTION
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
                # TRAVERSE CHILDREN
                # ------------------------------------------------

                for child in node.children:
                    visit(child)

            visit(body)

            return calls, instantiations

        # ========================================================
        # AST TRAVERSAL
        # ========================================================

        def traverse(
            node,
            current_class="Global"
        ):

            # ====================================================
            # CLASS
            # ====================================================

            if node.type == "class_declaration":

                name_node = (
                    node.child_by_field_name(
                        "name"
                    )
                )

                if not name_node:
                    return

                class_name = get_text(
                    name_node
                )

                annotations = (
                    extract_annotations(node)
                )

                # ------------------------------------------------
                # EXTENDS
                # ------------------------------------------------

                extends = None

                superclass_node = (
                    node.child_by_field_name(
                        "superclass"
                    )
                )

                if superclass_node:
                    extends = get_text(
                        superclass_node
                    )

                # ------------------------------------------------
                # IMPLEMENTS
                # ------------------------------------------------

                implements = []

                interfaces_node = (
                    node.child_by_field_name(
                        "interfaces"
                    )
                )

                if interfaces_node:

                    for child in interfaces_node.children:

                        if child.type in (
                            "type_identifier",
                            "generic_type",
                            "scoped_type_identifier"
                        ):
                            implements.append(
                                get_text(child)
                            )

                classes.append({
                    "name": class_name,
                    "type": "CLASS",
                    "package": package,
                    "annotations": annotations,
                    "extends": extends,
                    "implements": implements
                })

                # ------------------------------------------------
                # Traverse class body
                # ------------------------------------------------

                for child in node.children:
                    traverse(
                        child,
                        class_name
                    )

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

                if not name_node:
                    return

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
                    "name": interface_name,
                    "type": "INTERFACE",
                    "package": package,
                    "annotations": annotations,
                    "extends": None,
                    "implements": []
                })

                for child in node.children:
                    traverse(
                        child,
                        interface_name
                    )

                return

            # ====================================================
            # FIELD
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

                    if child.type != "variable_declarator":
                        continue

                    field_name_node = (
                        child.child_by_field_name(
                            "name"
                        )
                    )

                    if not field_name_node:
                        continue

                    field_name = get_text(
                        field_name_node
                    )

                    fields.append({
                        "name": field_name,
                        "type": field_type,
                        "enclosing_class":
                            current_class,
                        "annotations":
                            annotations,
                        "start_line":
                            node.start_point[0] + 1,
                        "end_line":
                            node.end_point[0] + 1
                    })

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
                # PARAMETERS
                # ------------------------------------------------

                parameters = []

                parameters_node = (
                    node.child_by_field_name(
                        "parameters"
                    )
                )

                if parameters_node:

                    for child in parameters_node.children:

                        if child.type in (
                            "formal_parameter",
                            "spread_parameter"
                        ):
                            parameters.append(
                                get_text(child)
                            )

                # ------------------------------------------------
                # RETURN TYPE
                # ------------------------------------------------

                return_type_node = (
                    node.child_by_field_name(
                        "type"
                    )
                )

                return_type = (
                    get_text(return_type_node)
                    if return_type_node
                    else ""
                )

                signature = (
                    f"{return_type} "
                    f"{method_name}"
                    f"({', '.join(parameters)})"
                ).strip()

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

                    "parameters":
                        parameters,

                    "return_type":
                        return_type,

                    "signature":
                        signature
                }

                methods.append(
                    method_data
                )

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

                # Do not recursively traverse method body again.
                return

            # ====================================================
            # CONTINUE TRAVERSAL
            # ====================================================

            for child in node.children:
                traverse(
                    child,
                    current_class
                )

        # ========================================================
        # START
        # ========================================================

        traverse(
            tree.root_node
        )

        # ========================================================
        # RETURN
        # ========================================================

        return {
            "package":
                package,

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

        source_bytes = source_code.encode(
            "utf-8"
        )

        return source_bytes[
            node.start_byte:node.end_byte
        ].decode(
            "utf-8",
            errors="replace"
        )