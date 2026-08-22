from typing import List, Dict, Any
from src.parser import JavaASTParser

class CodeChunker:
    def __init__(self):
        self.parser = JavaASTParser()

    def create_chunks(self, parsed_files: List[tuple]) -> List[Dict[str, Any]]:
        chunks = []

        for file_path, parsed_data in parsed_files:
            tree = parsed_data.get("tree")
            source_code = parsed_data.get("source_code", "")

            if not tree:
                continue

            extracted_methods = self.parser.extract_symbols_and_relations(tree, source_code)

            for item in extracted_methods:
                cls_name = item.get("class_name")
                m_name = item.get("method_name")
                sig = item.get("signature", m_name)
                
                chunk_id = f"{file_path}::{cls_name}::{sig}"

                chunks.append({
                    "chunk_id": chunk_id,
                    "file": file_path,
                    "class_name": cls_name,
                    "method_name": m_name,
                    "signature": sig,
                    "implements": item.get("implements", []),
                    "source_code": source_code,
                    "relationships": item.get("relationships", {"CALLS": [], "INSTANTIATES": []})
                })

        return chunks
