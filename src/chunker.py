from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class CodeChunker:
    def __init__(self, parser=None):
        self.parser = parser

    def create_chunks(
        self,
        extracted_data: List[Tuple[Any, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:

        chunks = []

        for file_path, data in extracted_data:
            chunks.extend(
                self.chunk_file(file_path, data)
            )

        return chunks

    def chunk_file(
        self,
        file_path: Union[Path, str],
        data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        chunks = []

        path_obj = Path(file_path)
        file_name = path_obj.name

        # ---------------------------------------------------------
        # Support both parser data formats
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Get extracted symbols
        # ---------------------------------------------------------

        symbols = {}
        if isinstance(data, dict):
            if "methods" in data:
                symbols = data
            elif isinstance(data.get("symbols"), dict):
                symbols = data["symbols"]

        if (
            not symbols
            and self.parser
            and hasattr(
                self.parser,
                "extract_symbols_and_relations"
            )
        ):
            try:
                symbols = (
                    self.parser
                    .extract_symbols_and_relations(
                        tree,
                        source_code
                    )
                )
            except Exception as e:
                logger.warning(f"Could not extract symbols for {file_name}: {e}")
                symbols = {}

        methods = (
            symbols.get("methods", [])
            if isinstance(symbols, dict)
            else []
        )

        # ---------------------------------------------------------
        # Create METHOD chunks
        # ---------------------------------------------------------

        if methods:

            for method in methods:

                method_name = method.get(
                    "name",
                    "unknown"
                )

                class_name = method.get(
                    "enclosing_class",
                    "Global"
                )

                method_code = method.get(
                    "source_code",
                    source_code
                )

                annotations = method.get(
                    "annotations",
                    []
                )

                raw_calls = method.get(
                    "calls",
                    method.get(
                        "method_calls",
                        []
                    )
                )

                # -------------------------------------------------
                # Normalize calls (Handles string or dict items)
                # -------------------------------------------------

                calls = []
                if isinstance(raw_calls, list):
                    for call in raw_calls:
                        if isinstance(call, dict):
                            c_name = call.get("method_called") or call.get("name")
                            if c_name:
                                calls.append(str(c_name))
                        elif call:
                            calls.append(str(call))
                elif raw_calls:
                    calls.append(str(raw_calls))

                # -------------------------------------------------
                # Normalize annotations
                # -------------------------------------------------

                if isinstance(
                    annotations,
                    list
                ):
                    ann_str = ", ".join(
                        map(str, annotations)
                    )
                else:
                    ann_str = str(
                        annotations
                    )

                calls_str = ", ".join(
                    calls
                )

                # -------------------------------------------------
                # Text used by VectorStore
                # -------------------------------------------------

                text_repr = (
                    f"Class: {class_name}\n"
                    f"Method: {method_name}\n"
                    f"Annotations: {ann_str}\n"
                    f"Calls: {calls_str}\n"
                    f"Code:\n{method_code}"
                )

                # -------------------------------------------------
                # IMPORTANT:
                # Store calls using the key "calls" and "relationships"
                # -------------------------------------------------

                chunks.append({

                    "chunk_id":
                        f"{file_name}::"
                        f"{class_name}::"
                        f"{method_name}",

                    "chunk_type":
                        "METHOD",

                    "file_name":
                        str(path_obj),

                    "file":
                        str(path_obj),

                    "class_name":
                        class_name,

                    "method_name":
                        method_name,

                    "signature":
                        method.get(
                            "signature",
                            f"{method_name}()"
                        ),

                    "start_line":
                        method.get(
                            "start_line",
                            1
                        ),

                    "end_line":
                        method.get(
                            "end_line",
                            1
                        ),

                    "annotations":
                        annotations,

                    "code_content":
                        method_code,

                    "source_code":
                        method_code,

                    "text_representation":
                        text_repr,

                    "calls":
                        calls,

                    "relationships":
                        calls
                })

        # ---------------------------------------------------------
        # File-level fallback
        # ---------------------------------------------------------

        else:

            text_repr = (
                f"File: {file_name}\n"
                f"Code:\n{source_code}"
            )

            chunks.append({

                "chunk_id":
                    f"{file_name}::FILE",

                "chunk_type":
                    "FILE",

                "file_name":
                    str(path_obj),

                "file":
                    str(path_obj),

                "class_name":
                    "Global",

                "method_name":
                    "file_summary",

                "signature":
                    "",

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

                "code_content":
                    source_code,

                "source_code":
                    source_code,

                "text_representation":
                    text_repr,

                "calls":
                    [],

                "relationships":
                    []
            })

        return chunks