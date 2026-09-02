import os
import logging
from typing import Dict, Any, Tuple, List, Optional, Set

from tree_sitter import Language, Parser

# ---------------------------------------------------------------------------
# Language grammar imports
# ---------------------------------------------------------------------------

import tree_sitter_java as tsjava
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts

try:
    import tree_sitter_dart as tsdart

    HAS_DART = True
except ImportError:
    tsdart = None
    HAS_DART = False

try:
    import tree_sitter_kotlin as tskotlin

    HAS_KOTLIN = True
except ImportError:
    tskotlin = None
    HAS_KOTLIN = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Initialize languages
# ---------------------------------------------------------------------------

JAVA_LANG = Language(tsjava.language())
PY_LANG = Language(tspython.language())
JS_LANG = Language(tsjs.language())
TS_LANG = Language(tsts.language_typescript())

DART_LANG = Language(tsdart.language()) if HAS_DART else None
KOTLIN_LANG = Language(tskotlin.language()) if HAS_KOTLIN else None


# ---------------------------------------------------------------------------
# Base parser
# ---------------------------------------------------------------------------

class BaseASTParser:
    """Common helpers shared by all AST parsers."""

    @staticmethod
    def get_text(node: Any, source_bytes: bytes) -> str:
        if node is None:
            return ""

        try:
            return source_bytes[
                node.start_byte:node.end_byte
            ].decode("utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def get_named_child_text(
        node: Any,
        field_names: List[str],
        source_bytes: bytes,
    ) -> str:
        """
        Try multiple Tree-sitter field names because different grammars
        represent equivalent concepts differently.
        """
        for field_name in field_names:
            try:
                child = node.child_by_field_name(field_name)
            except Exception:
                child = None

            if child is not None:
                text = BaseASTParser.get_text(
                    child,
                    source_bytes,
                ).strip()

                if text:
                    return text

        return ""

    @staticmethod
    def extract_annotations(
        node: Any,
        source_bytes: bytes,
    ) -> List[str]:

        annotations: List[str] = []

        try:
            modifiers = node.child_by_field_name("modifiers")
        except Exception:
            modifiers = None

        if modifiers:
            for child in modifiers.children:
                if child.type in (
                    "marker_annotation",
                    "annotation",
                    "decorator",
                    "annotation_entry",
                ):
                    text = BaseASTParser.get_text(
                        child,
                        source_bytes,
                    ).strip()

                    if text:
                        annotations.append(text)

        if not annotations:
            for child in getattr(node, "children", []):
                if child.type in (
                    "annotation",
                    "annotation_entry",
                    "decorator",
                ):
                    text = BaseASTParser.get_text(
                        child,
                        source_bytes,
                    ).strip()

                    if text:
                        annotations.append(text)

        return annotations

    @staticmethod
    def walk(node: Any):
        """Depth-first AST traversal generator."""

        if node is None:
            return

        stack = [node]

        while stack:
            current = stack.pop()

            yield current

            children = getattr(
                current,
                "children",
                None,
            )

            if children:
                stack.extend(
                    reversed(children)
                )


# ---------------------------------------------------------------------------
# Java parser
# ---------------------------------------------------------------------------

class JavaASTParser(BaseASTParser):
    """AST parser specialized for Java."""

    def __init__(self):
        self.parser = Parser(JAVA_LANG)

    def parse_file(
        self,
        file_path: str,
    ) -> Tuple[Any, str]:

        with open(
            file_path,
            "r",
            encoding="utf-8-sig",
            errors="ignore",
        ) as f:
            source_code = f.read()

        tree = self.parser.parse(
            source_code.encode("utf-8")
        )

        return tree, source_code

    def extract_symbols_and_relations(
        self,
        tree: Any,
        source_code: str,
    ) -> Dict[str, Any]:

        classes: List[Dict[str, Any]] = []
        methods: List[Dict[str, Any]] = []
        method_calls: List[Dict[str, Any]] = []
        fields: List[Dict[str, Any]] = []
        interfaces: List[str] = []

        if tree is None or not tree.root_node:
            return {
                "classes": [],
                "methods": [],
                "method_calls": [],
                "interfaces": [],
                "fields": [],
            }

        source_bytes = source_code.encode("utf-8")

        current_class = "Global"

        def extract_method_calls(
            body: Any,
            caller_class: str,
            caller_method: str,
        ):

            calls = []
            instantiations = []

            if body is None:
                return calls, instantiations

            for node in self.walk(body):

                if node.type == "method_invocation":

                    name_node = node.child_by_field_name(
                        "name"
                    )

                    object_node = node.child_by_field_name(
                        "object"
                    )

                    method_called = self.get_text(
                        name_node,
                        source_bytes,
                    ).strip()

                    object_expression = (
                        self.get_text(
                            object_node,
                            source_bytes,
                        ).strip()
                        if object_node
                        else None
                    )

                    if method_called:
                        calls.append(
                            {
                                "method_called": method_called,
                                "object_expression": object_expression,
                                "caller_class": caller_class,
                                "caller_method": caller_method,
                                "line": node.start_point[0] + 1,
                            }
                        )

                elif node.type == "object_creation_expression":

                    type_node = node.child_by_field_name(
                        "type"
                    )

                    if type_node:

                        type_name = self.get_text(
                            type_node,
                            source_bytes,
                        ).strip()

                        if type_name:
                            instantiations.append(
                                type_name
                            )

            return calls, instantiations

        def traverse(node: Any):

            nonlocal current_class

            # ---------------------------------------------------------------
            # Class / interface
            # ---------------------------------------------------------------

            if node.type in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
            ):

                name_node = node.child_by_field_name(
                    "name"
                )

                if name_node:

                    name = self.get_text(
                        name_node,
                        source_bytes,
                    ).strip()

                    if not name:
                        name = "Anonymous"

                    is_interface = (
                        node.type
                        == "interface_declaration"
                    )

                    if is_interface:
                        interfaces.append(name)

                    classes.append(
                        {
                            "name": name,
                            "type": (
                                "INTERFACE"
                                if is_interface
                                else "CLASS"
                            ),
                            "annotations": self.extract_annotations(
                                node,
                                source_bytes,
                            ),
                        }
                    )

                    previous_class = current_class

                    current_class = name

                    for child in node.children:
                        traverse(child)

                    current_class = previous_class

                    return

            # ---------------------------------------------------------------
            # Fields
            # ---------------------------------------------------------------

            if node.type == "field_declaration":

                type_node = node.child_by_field_name(
                    "type"
                )

                field_type = (
                    self.get_text(
                        type_node,
                        source_bytes,
                    ).strip()
                    if type_node
                    else ""
                )

                annotations = self.extract_annotations(
                    node,
                    source_bytes,
                )

                for child in node.children:

                    if child.type == "variable_declarator":

                        name_node = child.child_by_field_name(
                            "name"
                        )

                        if name_node:

                            fields.append(
                                {
                                    "name": self.get_text(
                                        name_node,
                                        source_bytes,
                                    ).strip(),
                                    "type": field_type,
                                    "enclosing_class": current_class,
                                    "annotations": annotations,
                                }
                            )

            # ---------------------------------------------------------------
            # Methods
            # ---------------------------------------------------------------

            if node.type == "method_declaration":

                name_node = node.child_by_field_name(
                    "name"
                )

                method_name = (
                    self.get_text(
                        name_node,
                        source_bytes,
                    ).strip()
                    if name_node
                    else "unknown"
                )

                method_code = self.get_text(
                    node,
                    source_bytes,
                )

                body = node.child_by_field_name(
                    "body"
                )

                calls = []
                instantiations = []

                if body:

                    calls, instantiations = (
                        extract_method_calls(
                            body,
                            current_class,
                            method_name,
                        )
                    )

                methods.append(
                    {
                        "name": method_name,
                        "enclosing_class": current_class,
                        "annotations": self.extract_annotations(
                            node,
                            source_bytes,
                        ),
                        "calls": calls,
                        "method_calls": calls,
                        "instantiations": instantiations,
                        "source_code": method_code,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "signature": f"{method_name}()",
                    }
                )

                for call in calls:

                    method_calls.append(
                        {
                            "caller_class": current_class,
                            "caller_method": method_name,
                            "method_called": call.get(
                                "method_called"
                            ),
                            "object_expression": call.get(
                                "object_expression"
                            ),
                            "line": call.get("line"),
                        }
                    )

                return

            for child in node.children:
                traverse(child)

        traverse(
            tree.root_node
        )

        return {
            "classes": classes,
            "methods": methods,
            "method_calls": method_calls,
            "interfaces": interfaces,
            "fields": fields,
        }


# ---------------------------------------------------------------------------
# Generic parser
# ---------------------------------------------------------------------------

class GenericASTParser(BaseASTParser):
    """
    Multi-language AST parser.

    Handles language-specific Tree-sitter structures for:

    - Dart
    - Python
    - JavaScript
    - TypeScript
    - Kotlin
    """

    COMMON_CLASS_TYPES: Set[str] = {
        "class_declaration",
        "class_definition",
        "interface_declaration",
        "interface_definition",
        "mixin_declaration",
        "enum_declaration",
        "enum_definition",
    }

    COMMON_FUNCTION_TYPES: Set[str] = {
        "method_declaration",
        "method_definition",
        "function_declaration",
        "function_definition",
    }

    DART_CLASS_TYPES: Set[str] = {
        "class_definition",
        "mixin_declaration",
        "enum_declaration",
        "extension_declaration",
    }

    DART_FUNCTION_TYPES: Set[str] = {
        "method_signature",
        "function_signature",
        "getter_signature",
        "setter_signature",
        "method_declaration",
        "function_declaration",
        "constructor_signature",
        "operator_signature",
    }

    CALL_TYPES: Set[str] = {
        "call_expression",
        "method_invocation",
        "function_call",
    }

    DART_CALL_TYPES: Set[str] = {
        "call_expression",
    }

    def __init__(
        self,
        language: Language,
        language_name: Optional[str] = None,
    ):

        self.parser = Parser(language)

        self.language_name = (
            language_name.lower()
            if language_name
            else "generic"
        )

    # -----------------------------------------------------------------------
    # File parsing
    # -----------------------------------------------------------------------

    def parse_file(
        self,
        file_path: str,
    ) -> Tuple[Any, str]:

        with open(
            file_path,
            "r",
            encoding="utf-8-sig",
            errors="ignore",
        ) as f:
            source_code = f.read()

        tree = self.parser.parse(
            source_code.encode("utf-8")
        )

        return tree, source_code

    # -----------------------------------------------------------------------
    # Language detection
    # -----------------------------------------------------------------------

    def _is_dart(self) -> bool:
        return self.language_name == "dart"

    # -----------------------------------------------------------------------
    # Node classification
    # -----------------------------------------------------------------------

    def _is_class_node(
        self,
        node: Any,
    ) -> bool:

        node_type = node.type

        if node_type in self.COMMON_CLASS_TYPES:
            return True

        if (
            self._is_dart()
            and node_type in self.DART_CLASS_TYPES
        ):
            return True

        return False

    def _is_function_node(
        self,
        node: Any,
    ) -> bool:

        node_type = node.type

        if node_type in self.COMMON_FUNCTION_TYPES:
            return True

        if (
            self._is_dart()
            and node_type in self.DART_FUNCTION_TYPES
        ):
            return True

        return False

    def _is_call_node(
        self,
        node: Any,
    ) -> bool:

        if node.type in self.CALL_TYPES:
            return True

        if (
            self._is_dart()
            and node.type in self.DART_CALL_TYPES
        ):
            return True

        return False

    # -----------------------------------------------------------------------
    # Name extraction
    # -----------------------------------------------------------------------

    def _extract_class_name(
        self,
        node: Any,
        source_bytes: bytes,
    ) -> str:

        name = self.get_named_child_text(
            node,
            [
                "name",
                "identifier",
            ],
            source_bytes,
        )

        if name:
            return name.strip()

        if self._is_dart():

            for child in node.children:

                if child.type in (
                    "identifier",
                    "type_identifier",
                ):

                    text = self.get_text(
                        child,
                        source_bytes,
                    ).strip()

                    if text:
                        return text

        return "Anonymous"

    def _extract_dart_method_name(
        self,
        node: Any,
        source_bytes: bytes,
    ) -> str:
        """
        Extract the actual method name from a Dart method_signature.

        Confirmed Tree-sitter Dart structure:

            method_signature
                return_type
                identifier
                formal_parameters

        Depending on the grammar version, the identifier may appear as a
        direct child or inside a small wrapper node.
        """

        # First try explicit fields.
        for field_name in (
            "name",
            "identifier",
        ):

            try:
                child = node.child_by_field_name(
                    field_name
                )
            except Exception:
                child = None

            if child is not None:

                text = self.get_text(
                    child,
                    source_bytes,
                ).strip()

                if text:
                    return self._clean_symbol_name(
                        text
                    )

        # Prefer an identifier that is immediately associated with the
        # method signature. Do not blindly select the first type identifier
        # because that can be "Future" or "Stream".
        direct_identifiers = []

        for child in node.children:

            if child.type in (
                "identifier",
                "simple_identifier",
            ):

                text = self.get_text(
                    child,
                    source_bytes,
                ).strip()

                if text:
                    direct_identifiers.append(
                        text
                    )

        if direct_identifiers:

            # In Dart method signatures such as:
            #
            # Future<void> createBooking(...)
            #
            # the actual method identifier is normally the identifier
            # immediately before formal_parameters.
            for i, child in enumerate(
                node.children
            ):

                if child.type in (
                    "formal_parameters",
                    "optional_formal_parameters",
                ):

                    for previous in reversed(
                        node.children[:i]
                    ):

                        if previous.type in (
                            "identifier",
                            "simple_identifier",
                        ):

                            text = self.get_text(
                                previous,
                                source_bytes,
                            ).strip()

                            if text:
                                return self._clean_symbol_name(
                                    text
                                )

        # Fallback: inspect the signature text directly.
        signature_text = self.get_text(
            node,
            source_bytes,
        ).strip()

        if signature_text:

            # Remove generic return type and everything before the method
            # name by looking at the text immediately preceding '('.
            paren_index = signature_text.find("(")

            if paren_index >= 0:

                prefix = signature_text[
                    :paren_index
                ].strip()

                tokens = prefix.split()

                if tokens:

                    candidate = tokens[-1]

                    if candidate not in (
                        "Future",
                        "Stream",
                        "void",
                        "dynamic",
                        "String",
                        "bool",
                        "int",
                        "double",
                        "num",
                        "Object",
                    ):

                        return self._clean_symbol_name(
                            candidate
                        )

        return "anonymous"

    def _extract_function_name(
        self,
        node: Any,
        source_bytes: bytes,
    ) -> str:

        # Dart needs special handling because method_signature is a
        # signature node and does not behave like Java-style declarations.
        if (
            self._is_dart()
            and node.type in self.DART_FUNCTION_TYPES
        ):

            return self._extract_dart_method_name(
                node,
                source_bytes,
            )

        name = self.get_named_child_text(
            node,
            [
                "name",
                "identifier",
                "function",
                "method",
            ],
            source_bytes,
        )

        if name:
            return self._clean_symbol_name(
                name
            )

        for child in node.children:

            if child.type in (
                "identifier",
                "type_identifier",
                "simple_identifier",
            ):

                text = self.get_text(
                    child,
                    source_bytes,
                ).strip()

                if text:
                    return self._clean_symbol_name(
                        text
                    )

        return "anonymous"

    @staticmethod
    def _clean_symbol_name(
        name: str,
    ) -> str:

        name = name.strip()

        if name.endswith("()"):
            name = name[:-2]

        return name.strip()

    # -----------------------------------------------------------------------
    # Function body extraction
    # -----------------------------------------------------------------------

    def _find_function_body(
        self,
        node: Any,
    ) -> Optional[Any]:

        for field_name in (
            "body",
            "function_body",
            "block",
        ):

            try:
                body = node.child_by_field_name(
                    field_name
                )
            except Exception:
                body = None

            if body is not None:
                return body

        for child in node.children:

            if child.type in (
                "function_body",
                "block",
                "statement_block",
                "expression_function_body",
                "empty_function_body",
            ):

                return child

        for child in node.children:

            for nested in getattr(
                child,
                "children",
                [],
            ):

                if nested.type in (
                    "function_body",
                    "block",
                    "statement_block",
                    "expression_function_body",
                    "empty_function_body",
                ):

                    return nested

        return None

    # -----------------------------------------------------------------------
    # Dart signature/body pairing
    # -----------------------------------------------------------------------

    def _extract_dart_methods_from_class_body(
        self,
        class_body: Any,
        source_bytes: bytes,
        current_class: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract Dart methods from the actual Tree-sitter Dart structure.

        Dart represents methods in class_body as sibling nodes:

            documentation_comment
            method_signature
            function_body
            documentation_comment
            method_signature
            function_body

        Therefore method_signature and function_body must be paired
        explicitly.
        """

        methods: List[Dict[str, Any]] = []

        if class_body is None:
            return methods

        children = class_body.children

        for index, node in enumerate(children):

            if node.type not in self.DART_FUNCTION_TYPES:
                continue

            # A Dart method_signature is followed by its function_body.
            body = None

            for next_index in range(
                index + 1,
                min(
                    index + 3,
                    len(children),
                ),
            ):

                candidate = children[
                    next_index
                ]

                if candidate.type in (
                    "function_body",
                    "expression_function_body",
                    "empty_function_body",
                ):

                    body = candidate
                    break

                # Stop if another declaration begins.
                if candidate.type in (
                    "method_signature",
                    "function_signature",
                    "getter_signature",
                    "setter_signature",
                    "constructor_signature",
                    "operator_signature",
                ):

                    break

            function_name = (
                self._extract_function_name(
                    node,
                    source_bytes,
                )
            )

            if not function_name:
                function_name = "anonymous"

            calls = self._extract_calls(
                body if body is not None else node,
                source_bytes,
                current_class,
                function_name,
            )

            signature_code = self.get_text(
                node,
                source_bytes,
            ).strip()

            full_source = signature_code

            if body is not None:

                body_code = self.get_text(
                    body,
                    source_bytes,
                )

                if body_code:
                    full_source += " " + body_code

            method_record = {
                "name": function_name,
                "enclosing_class": current_class,
                "annotations": self.extract_annotations(
                    node,
                    source_bytes,
                ),
                "calls": calls,
                "method_calls": calls,
                "instantiations": [],
                "source_code": full_source,
                "start_line": node.start_point[0] + 1,
                "end_line": (
                    body.end_point[0] + 1
                    if body is not None
                    else node.end_point[0] + 1
                ),
                "signature": (
                    signature_code
                    if signature_code
                    else f"{function_name}()"
                ),
            }

            methods.append(
                method_record
            )

        return methods

    # -----------------------------------------------------------------------
    # Call extraction
    # -----------------------------------------------------------------------
    def _extract_calls(
        self,
        node: Any,
        source_bytes: bytes,
        caller_class: str,
        caller_method: str,
    ) -> List[Dict[str, Any]]:

        calls: List[Dict[str, Any]] = []

        if node is None:
            return calls

        # ------------------------------------------------------------------
        # Dart
        #
        # Dart Tree-sitter grammar represents:
        #
        #   _db.collection('bookings').add(...)
        #
        # as selectors:
        #
        #   identifier '_db'
        #   selector '.collection'
        #   selector "('bookings')"
        #   selector '.add'
        #   selector '(...)'
        #
        # Therefore, a method call is a `.name` selector followed by a
        # selector beginning with `(`.
        # ------------------------------------------------------------------

        if self._is_dart():

            for current in self.walk(node):

                if current.type != "selector":
                    continue

                text = self.get_text(
                    current,
                    source_bytes,
                ).strip()

                # Only selectors such as ".add", ".get", ".where".
                if not text.startswith("."):
                    continue

                called_name = text[1:].strip()

                if not called_name:
                    continue

                called_name = self._clean_called_name(
                    called_name
                )

                if not called_name:
                    continue

                parent = current.parent

                if parent is None:
                    continue

                children = parent.children

                try:
                    index = children.index(current)
                except ValueError:
                    continue

                # The next selector is the argument selector:
                #
                #   selector '.add'
                #   selector '(...)'
                #
                has_arguments = False

                if index + 1 < len(children):

                    next_node = children[index + 1]

                    if next_node.type == "selector":

                        next_text = self.get_text(
                            next_node,
                            source_bytes,
                        ).strip()

                        if next_text.startswith("("):
                            has_arguments = True

                if not has_arguments:
                    continue

                # ----------------------------------------------------------
                # Build a useful receiver.
                #
                # Example:
                #
                #   _db.collection('bookings').add(...)
                #
                # becomes:
                #
                #   object_expression = "_db.collection"
                #   method_called = "add"
                # ----------------------------------------------------------

                object_expression = None

                receiver_parts: List[str] = []

                for previous in children[:index]:

                    previous_text = self.get_text(
                        previous,
                        source_bytes,
                    ).strip()

                    if previous.type == "identifier":
                        receiver_parts.append(
                            previous_text
                        )

                    elif (
                        previous.type == "selector"
                        and previous_text.startswith(".")
                        and not previous_text.startswith("..")
                    ):
                        selector_name = previous_text[1:].strip()

                        # Ignore argument selectors such as:
                        # ('bookings')
                        if not selector_name.startswith("("):
                            receiver_parts.append(
                                selector_name
                            )

                if receiver_parts:
                    object_expression = ".".join(
                        receiver_parts
                    )

                calls.append(
                    {
                        "method_called": called_name,
                        "object_expression": object_expression,
                        "caller_class": caller_class,
                        "caller_method": caller_method,
                        "line": current.start_point[0] + 1,
                    }
                )

            return calls

        # ------------------------------------------------------------------
        # Generic / Java-style call extraction.
        # ------------------------------------------------------------------

        for current in self.walk(node):

            if not self._is_call_node(current):
                continue

            called_name = self.get_named_child_text(
                current,
                [
                    "function",
                    "name",
                ],
                source_bytes,
            )

            object_expression = self.get_named_child_text(
                current,
                [
                    "object",
                    "receiver",
                    "target",
                ],
                source_bytes,
            )

            if not called_name:
                continue

            called_name = self._clean_called_name(
                called_name
            )

            if not called_name:
                continue

            calls.append(
                {
                    "method_called": called_name,
                    "object_expression": (
                        object_expression
                        if object_expression
                        else None
                    ),
                    "caller_class": caller_class,
                    "caller_method": caller_method,
                    "line": current.start_point[0] + 1,
                }
            )

        return calls

    @staticmethod
    def _clean_called_name(
        name: str,
    ) -> str:

        name = name.strip()

        if "(" in name:
            name = name.split(
                "(",
                1,
            )[0]

        if "." in name:
            name = name.split(
                "."
            )[-1]

        name = name.strip(
            " ;,:{}[]"
        )

        return name.strip()

    # -----------------------------------------------------------------------
    # Field extraction
    # -----------------------------------------------------------------------

    def _extract_fields(
        self,
        node: Any,
        source_bytes: bytes,
        current_class: str,
    ) -> List[Dict[str, Any]]:

        fields: List[Dict[str, Any]] = []

        if node.type not in (
            "field_declaration",
            "declaration",
            "field_definition",
            "initialized_variable_definition",
        ):
            return fields

        field_type = self.get_named_child_text(
            node,
            [
                "type",
                "return_type",
            ],
            source_bytes,
        )

        for child in node.children:

            if child.type in (
                "variable_declarator",
                "variable_declaration",
                "initialized_variable_definition",
                "identifier",
            ):

                name = self.get_named_child_text(
                    child,
                    [
                        "name",
                        "identifier",
                    ],
                    source_bytes,
                )

                if (
                    not name
                    and child.type == "identifier"
                ):

                    name = self.get_text(
                        child,
                        source_bytes,
                    ).strip()

                if name:

                    fields.append(
                        {
                            "name": name,
                            "type": field_type,
                            "enclosing_class": current_class,
                            "annotations": self.extract_annotations(
                                node,
                                source_bytes,
                            ),
                        }
                    )

        return fields

    # -----------------------------------------------------------------------
    # Main extraction
    # -----------------------------------------------------------------------

    def extract_symbols_and_relations(
        self,
        tree: Any,
        source_code: str,
    ) -> Dict[str, Any]:

        classes: List[Dict[str, Any]] = []
        methods: List[Dict[str, Any]] = []
        method_calls: List[Dict[str, Any]] = []
        fields: List[Dict[str, Any]] = []
        interfaces: List[str] = []

        if tree is None or not tree.root_node:

            return {
                "classes": [],
                "methods": [],
                "method_calls": [],
                "interfaces": [],
                "fields": [],
            }

        source_bytes = source_code.encode(
            "utf-8"
        )

        # -------------------------------------------------------------------
        # Dart-specific traversal
        # -------------------------------------------------------------------

        def extract_dart_class(
            node: Any,
            current_class: str = "Global",
        ):

            class_name = self._extract_class_name(
                node,
                source_bytes,
            )

            if not class_name:
                class_name = "Anonymous"

            is_interface = node.type in (
                "interface_declaration",
                "interface_definition",
            )

            if is_interface:
                interfaces.append(
                    class_name
                )

            if not any(
                c.get("name") == class_name
                for c in classes
            ):

                classes.append(
                    {
                        "name": class_name,
                        "type": (
                            "INTERFACE"
                            if is_interface
                            else "CLASS"
                        ),
                        "annotations": self.extract_annotations(
                            node,
                            source_bytes,
                        ),
                    }
                )

            class_body = None

            for child in node.children:

                if child.type == "class_body":
                    class_body = child
                    break

            if class_body is None:
                return

            # Fields.
            for child in class_body.children:

                extracted_fields = (
                    self._extract_fields(
                        child,
                        source_bytes,
                        class_name,
                    )
                )

                if extracted_fields:
                    fields.extend(
                        extracted_fields
                    )

            # Methods.
            dart_methods = (
                self._extract_dart_methods_from_class_body(
                    class_body,
                    source_bytes,
                    class_name,
                )
            )

            for method in dart_methods:

                methods.append(
                    method
                )

                for call in method.get(
                    "calls",
                    [],
                ):

                    method_calls.append(
                        {
                            "caller_class": class_name,
                            "caller_method": method.get(
                                "name"
                            ),
                            "method_called": call.get(
                                "method_called"
                            ),
                            "object_expression": call.get(
                                "object_expression"
                            ),
                            "line": call.get(
                                "line"
                            ),
                        }
                    )

        # -------------------------------------------------------------------
        # Generic recursive traversal
        # -------------------------------------------------------------------

        def traverse(
            node: Any,
            current_class: str = "Global",
        ):

            # ---------------------------------------------------------------
            # Dart class
            # ---------------------------------------------------------------

            if (
                self._is_dart()
                and node.type in self.DART_CLASS_TYPES
            ):

                extract_dart_class(
                    node,
                    current_class,
                )

                return

            # ---------------------------------------------------------------
            # Generic class
            # ---------------------------------------------------------------

            if self._is_class_node(node):

                class_name = self._extract_class_name(
                    node,
                    source_bytes,
                )

                if not class_name:
                    class_name = "Anonymous"

                is_interface = node.type in (
                    "interface_declaration",
                    "interface_definition",
                )

                if is_interface:
                    interfaces.append(
                        class_name
                    )

                if not any(
                    c.get("name") == class_name
                    for c in classes
                ):

                    classes.append(
                        {
                            "name": class_name,
                            "type": (
                                "INTERFACE"
                                if is_interface
                                else "CLASS"
                            ),
                            "annotations": self.extract_annotations(
                                node,
                                source_bytes,
                            ),
                        }
                    )

                for child in node.children:

                    traverse(
                        child,
                        class_name,
                    )

                return

            # ---------------------------------------------------------------
            # Generic function
            # ---------------------------------------------------------------

            if self._is_function_node(node):

                function_name = (
                    self._extract_function_name(
                        node,
                        source_bytes,
                    )
                )

                if not function_name:
                    function_name = "anonymous"

                function_code = self.get_text(
                    node,
                    source_bytes,
                )

                body = self._find_function_body(
                    node
                )

                calls = self._extract_calls(
                    body if body is not None else node,
                    source_bytes,
                    current_class,
                    function_name,
                )

                method_record = {
                    "name": function_name,
                    "enclosing_class": current_class,
                    "annotations": self.extract_annotations(
                        node,
                        source_bytes,
                    ),
                    "calls": calls,
                    "method_calls": calls,
                    "instantiations": [],
                    "source_code": function_code,
                    "start_line": (
                        node.start_point[0] + 1
                    ),
                    "end_line": (
                        node.end_point[0] + 1
                    ),
                    "signature": (
                        f"{function_name}()"
                    ),
                }

                duplicate = any(
                    m.get("name") == function_name
                    and m.get("enclosing_class")
                    == current_class
                    and m.get("start_line")
                    == method_record["start_line"]
                    for m in methods
                )

                if not duplicate:

                    methods.append(
                        method_record
                    )

                    for call in calls:

                        method_calls.append(
                            {
                                "caller_class": current_class,
                                "caller_method": function_name,
                                "method_called": call.get(
                                    "method_called"
                                ),
                                "object_expression": call.get(
                                    "object_expression"
                                ),
                                "line": call.get(
                                    "line"
                                ),
                            }
                        )

                return

            # ---------------------------------------------------------------
            # Fields
            # ---------------------------------------------------------------

            extracted_fields = self._extract_fields(
                node,
                source_bytes,
                current_class,
            )

            if extracted_fields:
                fields.extend(
                    extracted_fields
                )

            # ---------------------------------------------------------------
            # Continue traversal
            # ---------------------------------------------------------------

            for child in node.children:

                traverse(
                    child,
                    current_class,
                )

        traverse(
            tree.root_node,
            "Global",
        )

        return {
            "classes": classes,
            "methods": methods,
            "method_calls": method_calls,
            "interfaces": interfaces,
            "fields": fields,
        }


# ---------------------------------------------------------------------------
# Parser factory
# ---------------------------------------------------------------------------

class CodeParserFactory:
    """Factory that returns the correct AST parser for a source file."""

    @staticmethod
    def get_parser(
        file_path: str,
    ):

        ext = os.path.splitext(
            file_path
        )[1].lower()

        if ext == ".java":
            return JavaASTParser()

        if ext == ".dart":

            if DART_LANG is None:

                logger.warning(
                    "Dart parser requested but "
                    "tree_sitter_dart is not installed."
                )

                return None

            return GenericASTParser(
                DART_LANG,
                language_name="dart",
            )

        if ext in (
            ".kt",
            ".kts",
        ):

            if KOTLIN_LANG is None:

                logger.warning(
                    "Kotlin parser requested but "
                    "tree_sitter_kotlin is not installed."
                )

                return None

            return GenericASTParser(
                KOTLIN_LANG,
                language_name="kotlin",
            )

        if ext == ".py":

            return GenericASTParser(
                PY_LANG,
                language_name="python",
            )

        if ext in (
            ".js",
            ".jsx",
        ):

            return GenericASTParser(
                JS_LANG,
                language_name="javascript",
            )

        if ext in (
            ".ts",
            ".tsx",
        ):

            return GenericASTParser(
                TS_LANG,
                language_name="typescript",
            )

        return None