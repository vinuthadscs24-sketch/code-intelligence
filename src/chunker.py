from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

class CodeChunker:
    def __init__(self, parser=None):
        self.parser = parser

    def create_chunks(self, extracted_data: List[Tuple[Union[Path, str], Any]]) -> List[Dict[str, Any]]:
        """Processes extracted repository data into distinct AST chunks."""
        chunks = []
        for file_path, data in extracted_data:
            file_chunks = self.chunk_file(file_path, data)
            chunks.extend(file_chunks)
        return chunks

    def chunk_file(self, file_path: Union[Path, str], data: Any = None) -> List[Dict[str, Any]]:
        """Generates code chunks for a single parsed file."""
        chunks = []
        path_obj = Path(file_path)
        file_name = path_obj.name

        # Handle dictionary input returned by JavaASTParser
        if isinstance(data, dict):
            tree = data.get("tree")
            source_code = data.get("source_code", "")
            symbols = data.get("symbols", {})
        else:
            tree = data
            source_code = ""
            symbols = {}

        # Fallback symbol extraction if parser provides it
        if not symbols and self.parser and hasattr(self.parser, "extract_symbols"):
            symbols = self.parser.extract_symbols(tree, source_code)

        classes = symbols.get("classes", [])
        methods = symbols.get("methods", [])

        # If explicit methods were extracted, chunk by method
        if methods:
            for m in methods:
                chunks.append({
                    "chunk_id": f"{file_name}::{m.get('enclosing_class', 'Global')}::{m.get('name')}",
                    "chunk_type": "METHOD",
                    "file_name": str(path_obj),
                    "class_name": m.get("enclosing_class", "Global"),
                    "method_name": m.get("name"),
                    "start_line": m.get("start_line", 1),
                    "end_line": m.get("end_line", 1),
                    "annotations": str(m.get("annotations", [])),
                    "code_content": m.get("source_code", source_code[:300]),
                    "calls": m.get("calls", [])
                })
        else:
            # File-level or module-level fallback chunk
            chunks.append({
                "chunk_id": f"{file_name}::FILE",
                "chunk_type": "FILE",
                "file_name": str(path_obj),
                "class_name": "Global",
                "method_name": "file_summary",
                "start_line": 1,
                "end_line": len(source_code.splitlines()) if source_code else 1,
                "annotations": "[]",
                "code_content": source_code[:500] if source_code else f"// File: {file_name}",
                "calls": []
            })

        return chunks