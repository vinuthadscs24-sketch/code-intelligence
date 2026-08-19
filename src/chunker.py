import os
from pathlib import Path
from typing import List, Dict, Any

class CodeChunker:
    def __init__(self):
        pass

    def _get_node_text(self, node, source_bytes: bytes) -> str:
        if not node or not source_bytes:
            return ""
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

    def _extract_calls_from_node(self, node, source_bytes: bytes) -> List[str]:
        """Extracts ONLY method calls made strictly inside this node."""
        calls = set()

        def traverse(n):
            if n.type == "method_invocation":
                name_node = n.child_by_field_name("name")
                if name_node:
                    calls.add(self._get_node_text(name_node, source_bytes))
                else:
                    for child in n.children:
                        if child.type == "identifier":
                            calls.add(self._get_node_text(child, source_bytes))
                            break
            for child in n.children:
                traverse(child)

        traverse(node)
        return sorted(list(calls))

    def _extract_annotations(self, method_node, source_bytes: bytes) -> List[str]:
        annotations = []
        def traverse(n):
            if n.type in ("marker_annotation", "annotation", "single_element_annotation", "normal_annotation"):
                text = self._get_node_text(n, source_bytes)
                if text and text not in annotations:
                    annotations.append(text)
            for child in n.children:
                if child.type != "block":
                    traverse(child)

        traverse(method_node)
        return annotations

    def create_chunks(self, parsed_data: List[Any]) -> List[Dict[str, Any]]:
        chunks = []

        for item in parsed_data:
            file_path = ""
            tree = None
            source_bytes = b""

            # Unpack (file_path, (tree, source_bytes)) or (file_path, tree, source_bytes)
            if isinstance(item, tuple):
                if len(item) == 2:
                    file_path = item[0]
                    second = item[1]
                    if isinstance(second, tuple) and len(second) == 2:
                        tree, source_bytes = second
                    else:
                        tree = second
                elif len(item) == 3:
                    file_path, tree, source_bytes = item

            if not tree or not hasattr(tree, "root_node"):
                continue

            file_name = Path(file_path).name
            root = tree.root_node

            def process_node(node):
                if node.type in ("class_declaration", "interface_declaration", "enum_declaration"):
                    class_name_node = node.child_by_field_name("name")
                    class_name = self._get_node_text(class_name_node, source_bytes) or "UnknownClass"

                    def find_methods(curr):
                        if curr.type in ("method_declaration", "constructor_declaration"):
                            m_name_node = curr.child_by_field_name("name")
                            m_name = self._get_node_text(m_name_node, source_bytes) or "unknown"
                            
                            m_annotations = self._extract_annotations(curr, source_bytes)
                            
                            body_node = curr.child_by_field_name("body")
                            m_calls = self._extract_calls_from_node(body_node, source_bytes) if body_node else []
                            m_code = self._get_node_text(curr, source_bytes)

                            chunk_id = f"{file_name}::{class_name}::{m_name}"
                            text_rep = (
                                f"Class: {class_name}\n"
                                f"Method: {m_name}\n"
                                f"Annotations: {', '.join(m_annotations)}\n"
                                f"Calls: {', '.join(m_calls)}\n"
                                f"Code:\n{m_code}"
                            )

                            chunks.append({
                                "chunk_id": chunk_id,
                                "chunk_type": "METHOD",
                                "file_name": file_path,
                                "class_name": class_name,
                                "method_name": m_name,
                                "annotations": m_annotations,
                                "relationships": {"CALLS": m_calls},
                                "code_content": m_code,
                                "text_representation": text_rep
                            })
                        
                        for child in curr.children:
                            if child.type not in ("class_declaration", "interface_declaration", "enum_declaration"):
                                find_methods(child)

                    find_methods(node)

                for child in node.children:
                    process_node(child)

            process_node(root)

        return chunks