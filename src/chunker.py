from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

class CodeChunker:
    def __init__(self, parser=None):
        self.parser = parser

    def create_chunks(self, extracted_data: List[Tuple[Any, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        chunks = []
        for file_path, data in extracted_data:
            chunks.extend(self.chunk_file(file_path, data))
        return chunks

    def chunk_file(self, file_path: Union[Path, str], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        path_obj = Path(file_path)
        file_name = path_obj.name

        # Support both data formats: dict with AST/source or raw extracted symbols dict
        tree = data.get("tree") if isinstance(data, dict) else None
        source_code = data.get("source_code", "") if isinstance(data, dict) else ""

        # Extract or fallback symbols
        symbols = data if isinstance(data, dict) and "methods" in data else data.get("symbols", {})
        if not symbols and self.parser and hasattr(self.parser, "extract_symbols_and_relations"):
            symbols = self.parser.extract_symbols_and_relations(tree, source_code)

        methods = symbols.get("methods", [])
        classes = symbols.get("classes", [])

        if methods:
            for m in methods:
                method_name = m.get("name", "unknown")
                class_name = m.get("enclosing_class", "Global")
                method_code = m.get("source_code", source_code)
                annotations = m.get("annotations", [])
                
                # Handle cases where annotations might be a list or formatted string
                ann_str = ", ".join(annotations) if isinstance(annotations, list) else str(annotations)

                # Format text_representation explicitly for VectorStore
                text_repr = (
                    f"Class: {class_name}\n"
                    f"Method: {method_name}\n"
                    f"Annotations: {ann_str}\n"
                    f"Code:\n{method_code}"
                )

                chunks.append({
                    "chunk_id": f"{file_name}::{class_name}::{method_name}",
                    "chunk_type": "METHOD",
                    "file_name": str(path_obj),
                    "file": str(path_obj),
                    "class_name": class_name,
                    "method_name": method_name,
                    "signature": m.get("signature", f"{method_name}()"),
                    "start_line": m.get("start_line", 1),
                    "end_line": m.get("end_line", 1),
                    "annotations": str(annotations),
                    "code_content": method_code,
                    "source_code": method_code,
                    "text_representation": text_repr,
                    "relationships": m.get("calls", m.get("method_calls", []))
                })
        else:
            # File-level fallback chunk if no methods extracted
            text_repr = f"File: {file_name}\nCode:\n{source_code}"
            chunks.append({
                "chunk_id": f"{file_name}::FILE",
                "chunk_type": "FILE",
                "file_name": str(path_obj),
                "file": str(path_obj),
                "class_name": "Global",
                "method_name": "file_summary",
                "signature": "",
                "start_line": 1,
                "end_line": len(source_code.splitlines()) if source_code else 1,
                "annotations": "[]",
                "code_content": source_code,
                "source_code": source_code,
                "text_representation": text_repr,
                "relationships": []
            })

        return chunks