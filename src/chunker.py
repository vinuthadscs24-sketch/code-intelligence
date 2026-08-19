import javalang
from pathlib import Path
from typing import List, Dict, Any, Optional

class CodeChunker:
    def __init__(self, knowledge_graph=None):
        self.kg = knowledge_graph

    def create_chunks(self, extracted_data: List[tuple]) -> List[Dict[str, Any]]:
        """Generates class and method chunks with scoped graph contexts and source code."""
        chunks = []

        for file_path, symbols in extracted_data:
            path_obj = Path(file_path)
            try:
                code = path_obj.read_text(encoding="utf-8")
                lines = code.splitlines()
                tree = javalang.parse.parse(code)
            except Exception as e:
                print(f"[Chunker] Skipping {path_obj.name}: {e}")
                continue

            for _, class_node in tree.filter(javalang.tree.ClassDeclaration):
                class_annotations = [f"@{a.name}" for a in class_node.annotations]
                class_code = self._extract_node_code(lines, class_node)
                
                # 1. Class-Level Chunk
                class_methods = [m.name for m in class_node.methods]
                class_rels = [f"HAS_METHOD: {m}" for m in class_methods]
                if class_node.extends:
                    class_rels.append(f"EXTENDS: {class_node.extends.name}")

                chunks.append({
                    "chunk_id": f"{path_obj.name}::{class_node.name}",
                    "chunk_type": "CLASS",
                    "file_name": str(path_obj),
                    "class_name": class_node.name,
                    "method_name": None,
                    "annotations": class_annotations,
                    "relationships": class_rels,
                    "code_content": class_code,
                    "text_representation": self._format_text_representation(
                        class_node.name, None, "CLASS", class_annotations, class_rels, class_code
                    )
                })

                # 2. Method-Level Chunks (Scoped exclusively to method body)
                for method in class_node.methods:
                    method_annotations = [f"@{a.name}" for a in method.annotations]
                    method_code = self._extract_node_code(lines, method)

                    # Extract calls local only to this method
                    method_calls = set()
                    for _, call in method.filter(javalang.tree.MethodInvocation):
                        target = f"{call.qualifier}." if call.qualifier else ""
                        method_calls.add(f"CALLS: {target}{call.member}()")
                    
                    method_rels = list(method_calls)

                    chunks.append({
                        "chunk_id": f"{path_obj.name}::{class_node.name}::{method.name}",
                        "chunk_type": "METHOD",
                        "file_name": str(path_obj),
                        "class_name": class_node.name,
                        "method_name": method.name,
                        "annotations": method_annotations,
                        "relationships": method_rels,
                        "code_content": method_code,
                        "text_representation": self._format_text_representation(
                            class_node.name, method.name, "METHOD", method_annotations, method_rels, method_code
                        )
                    })

        return chunks

    def _format_text_representation(
        self, class_name: str, method_name: Optional[str], chunk_type: str, 
        annotations: List[str], relationships: List[str], code: str
    ) -> str:
        """Structured text block used for downstream embeddings."""
        ann_str = ", ".join(annotations) if annotations else "None"
        rel_str = "\n  ".join(relationships) if relationships else "None"
        return (
            f"Class: {class_name}\n"
            f"Method: {method_name or 'N/A'}\n"
            f"Type: {chunk_type}\n"
            f"Annotations: {ann_str}\n"
            f"Relationships:\n  {rel_str}\n\n"
            f"Source Code:\n{code}"
        )

    def _extract_node_code(self, lines: List[str], node) -> str:
        """Extracts exact AST node code boundaries using brace counting."""
        if not hasattr(node, "position") or not node.position:
            return ""

        start_line = node.position.line - 1
        if start_line < 0 or start_line >= len(lines):
            return ""

        # Include preceding annotations if position points directly to the declaration signature
        while start_line > 0 and lines[start_line - 1].strip().startswith("@"):
            start_line -= 1

        brace_count = 0
        started_braces = False
        end_line = start_line

        for idx in range(start_line, len(lines)):
            line = lines[idx]
            for char in line:
                if char == '{':
                    brace_count += 1
                    started_braces = True
                elif char == '}':
                    brace_count -= 1

            if started_braces and brace_count == 0:
                end_line = idx + 1
                break
        else:
            end_line = min(start_line + 30, len(lines))

        return "\n".join(lines[start_line:end_line])