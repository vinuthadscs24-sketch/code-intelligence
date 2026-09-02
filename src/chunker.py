from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
import logging


logger = logging.getLogger(__name__)


class CodeChunker:
    """
    Converts parser output into retrieval-friendly code chunks.

    The chunker preserves structural information required by:

        - Hybrid retrieval
        - Code knowledge graph construction
        - Caller/callee analysis
        - Impact analysis
        - Contextual answers

    Important:
        Call dictionaries are preserved instead of reducing calls to
        plain strings. This allows the graph builder to use information
        such as:

            method_called
            object_expression
            caller_class
            caller_method
            line
    """

    def __init__(self, parser=None):
        self.parser = parser

    # ===============================================================
    # CREATE CHUNKS
    # ===============================================================

    def create_chunks(
        self,
        extracted_data: List[Tuple[Any, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:

        chunks: List[Dict[str, Any]] = []

        if not extracted_data:
            return chunks

        for file_path, data in extracted_data:

            try:
                file_chunks = self.chunk_file(
                    file_path,
                    data,
                )

                chunks.extend(file_chunks)

            except Exception as exc:
                logger.exception(
                    "Failed to create chunks for %s: %s",
                    file_path,
                    exc,
                )

        logger.info(
            "Created %d chunks from %d extracted files",
            len(chunks),
            len(extracted_data),
        )

        return chunks

    # ===============================================================
    # CHUNK FILE
    # ===============================================================

    def chunk_file(
        self,
        file_path: Union[Path, str],
        data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        chunks: List[Dict[str, Any]] = []

        path_obj = Path(file_path)
        file_name = path_obj.name

        # -----------------------------------------------------------
        # Support parser data
        # -----------------------------------------------------------

        tree = (
            data.get("tree")
            if isinstance(data, dict)
            else None
        )

        source_code = (
            data.get("source_code", "")
            if isinstance(data, dict)
            else ""
        )

        if not source_code and isinstance(data, str):
            source_code = data

        source_code = str(source_code or "")

        # -----------------------------------------------------------
        # Extract symbols
        # -----------------------------------------------------------

        symbols: Dict[str, Any] = {}

        if isinstance(data, dict):

            # Most current parser output
            if "methods" in data:
                symbols = data

            # Alternative parser structure
            elif isinstance(
                data.get("symbols"),
                dict,
            ):
                symbols = data["symbols"]

        # -----------------------------------------------------------
        # Fallback parser extraction
        # -----------------------------------------------------------

        if (
            not symbols
            and self.parser
            and hasattr(
                self.parser,
                "extract_symbols_and_relations",
            )
        ):

            try:

                extracted = (
                    self.parser
                    .extract_symbols_and_relations(
                        tree,
                        source_code,
                    )
                )

                if isinstance(extracted, dict):
                    symbols = extracted

            except Exception as exc:

                logger.warning(
                    "Could not extract symbols for %s: %s",
                    file_name,
                    exc,
                )

                symbols = {}

        # -----------------------------------------------------------
        # Make sure symbols is a dictionary
        # -----------------------------------------------------------

        if not isinstance(symbols, dict):
            symbols = {}

        methods = symbols.get("methods", [])

        if not isinstance(methods, list):
            methods = []

        # ===========================================================
        # METHOD CHUNKS
        # ===========================================================

        if methods:

            for method in methods:

                if not isinstance(method, dict):
                    continue

                # ---------------------------------------------------
                # Basic method information
                # ---------------------------------------------------

                method_name = (
                    method.get("name")
                    or method.get("method_name")
                    or "unknown"
                )

                method_name = str(method_name).strip()

                if not method_name:
                    method_name = "unknown"

                class_name = (
                    method.get("enclosing_class")
                    or method.get("class_name")
                    or "Global"
                )

                class_name = str(class_name).strip()

                if not class_name:
                    class_name = "Global"

                # ---------------------------------------------------
                # Method source
                # ---------------------------------------------------

                method_code = (
                    method.get("source_code")
                    or method.get("code")
                    or method.get("text")
                    or source_code
                    or ""
                )

                method_code = str(method_code)

                # ---------------------------------------------------
                # Annotations
                # ---------------------------------------------------

                annotations = method.get(
                    "annotations",
                    [],
                )

                annotation_list = self._normalize_annotations(
                    annotations
                )

                annotation_string = ", ".join(
                    annotation_list
                )

                # ---------------------------------------------------
                # Calls
                # ---------------------------------------------------

                raw_calls = (
                    method.get("calls")
                    if "calls" in method
                    else method.get(
                        "method_calls",
                        method.get(
                            "relationships",
                            [],
                        ),
                    )
                )

                normalized_calls = self._normalize_calls(
                    raw_calls=raw_calls,
                    caller_class=class_name,
                    caller_method=method_name,
                )

                # ---------------------------------------------------
                # Preserve fields if parser provides them
                # ---------------------------------------------------

                fields = self._normalize_fields(
                    method.get(
                        "fields",
                        [],
                    )
                )

                # ---------------------------------------------------
                # Also preserve method-level metadata
                # ---------------------------------------------------

                parameters = method.get(
                    "parameters",
                    [],
                )

                return_type = (
                    method.get("return_type")
                    or method.get("returnType")
                    or ""
                )

                signature = (
                    method.get("signature")
                    or f"{method_name}()"
                )

                start_line = self._safe_int(
                    method.get(
                        "start_line",
                        method.get(
                            "line",
                            1,
                        ),
                    ),
                    default=1,
                )

                end_line = self._safe_int(
                    method.get(
                        "end_line",
                        start_line,
                    ),
                    default=start_line,
                )

                if end_line < start_line:
                    end_line = start_line

                # ---------------------------------------------------
                # Human-readable call representation
                # ---------------------------------------------------

                call_names = []

                for call in normalized_calls:

                    method_called = str(
                        call.get(
                            "method_called",
                            "",
                        )
                        or ""
                    ).strip()

                    object_expression = (
                        call.get(
                            "object_expression"
                        )
                    )

                    if object_expression:
                        object_expression = str(
                            object_expression
                        ).strip()

                    if (
                        object_expression
                        and method_called
                    ):

                        call_names.append(
                            f"{object_expression}."
                            f"{method_called}"
                        )

                    elif method_called:

                        call_names.append(
                            method_called
                        )

                calls_string = ", ".join(
                    call_names
                )

                # ---------------------------------------------------
                # Human-readable fields
                # ---------------------------------------------------

                field_names = []

                for field in fields:

                    field_name = str(
                        field.get(
                            "name",
                            "",
                        )
                        or ""
                    ).strip()

                    field_type = str(
                        field.get(
                            "type",
                            "",
                        )
                        or ""
                    ).strip()

                    if field_name and field_type:
                        field_names.append(
                            f"{field_type} {field_name}"
                        )

                    elif field_name:
                        field_names.append(
                            field_name
                        )

                fields_string = ", ".join(
                    field_names
                )

                # ---------------------------------------------------
                # Text representation for embeddings
                # ---------------------------------------------------

                text_parts = [
                    f"File: {file_name}",
                    f"Class: {class_name}",
                    f"Method: {method_name}",
                ]

                if signature:
                    text_parts.append(
                        f"Signature: {signature}"
                    )

                if return_type:
                    text_parts.append(
                        f"Return type: {return_type}"
                    )

                if annotation_string:
                    text_parts.append(
                        f"Annotations: {annotation_string}"
                    )

                if fields_string:
                    text_parts.append(
                        f"Fields: {fields_string}"
                    )

                if calls_string:
                    text_parts.append(
                        f"Calls: {calls_string}"
                    )

                text_parts.append(
                    f"Code:\n{method_code}"
                )

                text_repr = "\n".join(
                    text_parts
                )

                # ---------------------------------------------------
                # Chunk ID
                # ---------------------------------------------------

                chunk_id = (
                    f"{file_name}::"
                    f"{class_name}::"
                    f"{method_name}"
                )

                # ---------------------------------------------------
                # Create method chunk
                # ---------------------------------------------------

                chunks.append(
                    {
                        "chunk_id": chunk_id,

                        "chunk_type": "METHOD",

                        "file_name": str(path_obj),

                        "file": str(path_obj),

                        "file_path": str(path_obj),

                        "class_name": class_name,

                        "method_name": method_name,

                        "signature": signature,

                        "return_type": return_type,

                        "parameters": parameters,

                        "start_line": start_line,

                        "end_line": end_line,

                        "annotations": annotation_list,

                        "fields": fields,

                        "code_content": method_code,

                        "source_code": method_code,

                        "text_representation": text_repr,

                        # ------------------------------------------------
                        # CRITICAL:
                        # Preserve COMPLETE call dictionaries.
                        # ------------------------------------------------

                        "calls": normalized_calls,

                        "relationships": normalized_calls,
                    }
                )

        # ===========================================================
        # FILE-LEVEL FALLBACK
        # ===========================================================

        else:

            text_repr = (
                f"File: {file_name}\n"
                f"Code:\n{source_code}"
            )

            chunks.append(
                {
                    "chunk_id":
                        f"{file_name}::FILE",

                    "chunk_type":
                        "FILE",

                    "file_name":
                        str(path_obj),

                    "file":
                        str(path_obj),

                    "file_path":
                        str(path_obj),

                    "class_name":
                        "Global",

                    "method_name":
                        "file_summary",

                    "signature":
                        "",

                    "return_type":
                        "",

                    "parameters":
                        [],

                    "start_line":
                        1,

                    "end_line":
                        (
                            len(
                                source_code.splitlines()
                            )
                            if source_code
                            else 1
                        ),

                    "annotations":
                        [],

                    "fields":
                        [],

                    "code_content":
                        source_code,

                    "source_code":
                        source_code,

                    "text_representation":
                        text_repr,

                    "calls":
                        [],

                    "relationships":
                        [],
                }
            )

        return chunks

    # ===============================================================
    # CALL NORMALIZATION
    # ===============================================================

    @staticmethod
    def _normalize_calls(
        raw_calls: Any,
        caller_class: str,
        caller_method: str,
    ) -> List[Dict[str, Any]]:

        normalized: List[Dict[str, Any]] = []

        if raw_calls is None:
            return normalized

        # -----------------------------------------------------------
        # Convert a single call into a list
        # -----------------------------------------------------------

        if isinstance(
            raw_calls,
            (str, dict),
        ):

            raw_items = [raw_calls]

        elif isinstance(
            raw_calls,
            (list, tuple, set),
        ):

            raw_items = list(raw_calls)

        else:

            raw_items = [raw_calls]

        # -----------------------------------------------------------
        # Process each call
        # -----------------------------------------------------------

        for call in raw_items:

            # =======================================================
            # STRUCTURED CALL
            # =======================================================

            if isinstance(call, dict):

                method_called = (
                    call.get("method_called")
                    or call.get("name")
                    or call.get("method")
                    or call.get("called_method")
                    or ""
                )

                method_called = str(
                    method_called
                ).strip()

                if not method_called:
                    continue

                object_expression = (
                    call.get("object_expression")
                    or call.get("object")
                    or call.get("receiver")
                    or call.get("target")
                    or call.get("caller_object")
                )

                if object_expression is not None:
                    object_expression = str(
                        object_expression
                    ).strip()

                    if not object_expression:
                        object_expression = None

                normalized.append(
                    {
                        "method_called":
                            method_called,

                        "object_expression":
                            object_expression,

                        "caller_class":
                            str(
                                call.get(
                                    "caller_class",
                                    caller_class,
                                )
                                or caller_class
                            ).strip(),

                        "caller_method":
                            str(
                                call.get(
                                    "caller_method",
                                    caller_method,
                                )
                                or caller_method
                            ).strip(),

                        "line":
                            call.get("line"),

                        # Preserve optional parser metadata.
                        "target_class":
                            call.get(
                                "target_class"
                            ),

                        "target_method":
                            call.get(
                                "target_method"
                            ),

                        "receiver_type":
                            call.get(
                                "receiver_type"
                            ),
                    }
                )

                continue

            # =======================================================
            # STRING CALL
            # =======================================================

            if isinstance(call, str):

                method_called = call.strip()

                if not method_called:
                    continue

                object_expression = None

                # ---------------------------------------------------
                # Examples:
                #
                # service.bookEquipment
                # BookingService.bookEquipment
                # this.bookEquipment
                # ---------------------------------------------------

                if "." in method_called:

                    parts = [
                        part.strip()
                        for part in method_called.split(".")
                        if part.strip()
                    ]

                    if len(parts) >= 2:

                        object_expression = ".".join(
                            parts[:-1]
                        )

                        method_called = parts[-1]

                normalized.append(
                    {
                        "method_called":
                            method_called,

                        "object_expression":
                            object_expression,

                        "caller_class":
                            caller_class,

                        "caller_method":
                            caller_method,

                        "line":
                            None,

                        "target_class":
                            None,

                        "target_method":
                            None,

                        "receiver_type":
                            None,
                    }
                )

                continue

            # =======================================================
            # UNKNOWN CALL REPRESENTATION
            # =======================================================

            if call is not None:

                try:
                    method_called = str(
                        call
                    ).strip()

                except Exception:
                    method_called = ""

                if method_called:

                    normalized.append(
                        {
                            "method_called":
                                method_called,

                            "object_expression":
                                None,

                            "caller_class":
                                caller_class,

                            "caller_method":
                                caller_method,

                            "line":
                                None,

                            "target_class":
                                None,

                            "target_method":
                                None,

                            "receiver_type":
                                None,
                        }
                    )

        # ===========================================================
        # REMOVE DUPLICATES
        # ===========================================================

        unique_calls: List[
            Dict[str, Any]
        ] = []

        seen = set()

        for call in normalized:

            key = (
                str(
                    call.get(
                        "method_called",
                        "",
                    )
                ).lower(),

                str(
                    call.get(
                        "object_expression",
                        "",
                    )
                    or ""
                ).lower(),

                str(
                    call.get(
                        "caller_class",
                        "",
                    )
                ).lower(),

                str(
                    call.get(
                        "caller_method",
                        "",
                    )
                ).lower(),

                call.get("line"),
            )

            if key in seen:
                continue

            seen.add(key)

            unique_calls.append(
                call
            )

        return unique_calls

    # ===============================================================
    # ANNOTATION NORMALIZATION
    # ===============================================================

    @staticmethod
    def _normalize_annotations(
        annotations: Any,
    ) -> List[str]:

        if annotations is None:
            return []

        if isinstance(
            annotations,
            (list, tuple, set),
        ):

            result = []

            for value in annotations:

                if value is None:
                    continue

                value = str(value).strip()

                if value:
                    result.append(value)

            return list(
                dict.fromkeys(result)
            )

        value = str(
            annotations
        ).strip()

        if not value:
            return []

        return [value]

    # ===============================================================
    # FIELD NORMALIZATION
    # ===============================================================

    @staticmethod
    def _normalize_fields(
        fields: Any,
    ) -> List[Dict[str, Any]]:

        if fields is None:
            return []

        if isinstance(
            fields,
            dict,
        ):

            fields = [fields]

        elif isinstance(
            fields,
            (str, int, float),
        ):

            fields = [
                {
                    "name": str(fields),
                    "type": "",
                }
            ]

        elif not isinstance(
            fields,
            (list, tuple, set),
        ):

            return []

        normalized = []

        for field in fields:

            if isinstance(
                field,
                dict,
            ):

                name = (
                    field.get("name")
                    or field.get("field_name")
                    or ""
                )

                field_type = (
                    field.get("type")
                    or field.get("field_type")
                    or ""
                )

                normalized.append(
                    {
                        "name":
                            str(name).strip(),

                        "type":
                            str(field_type).strip(),

                        "visibility":
                            field.get(
                                "visibility"
                            ),

                        "static":
                            field.get(
                                "static"
                            ),
                    }
                )

            elif field is not None:

                normalized.append(
                    {
                        "name":
                            str(field).strip(),

                        "type":
                            "",
                    }
                )

        return [
            field
            for field in normalized
            if field.get("name")
        ]

    # ===============================================================
    # SAFE INTEGER
    # ===============================================================

    @staticmethod
    def _safe_int(
        value: Any,
        default: int = 1,
    ) -> int:

        if value is None:
            return default

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return default