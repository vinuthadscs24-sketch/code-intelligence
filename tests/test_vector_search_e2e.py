import pytest
from src.parser import JavaASTParser
from src.chunker import CodeChunker
from src.vector_store import VectorStore

def test_pipeline_e2e(tmp_path):
    # 1. Create a temporary Java source file
    java_code = """
    package com.example;
    public class AuthService {
        public boolean login(String user, String pass) {
            return true;
        }
    }
    """
    java_file = tmp_path / "AuthService.java"
    java_file.write_text(java_code, encoding="utf-8")

    # 2. Parse
    parser = JavaASTParser()
    tree, source = parser.parse_file(str(java_file))
    symbols = parser.extract_symbols_and_relations(tree, source)
    data = {"tree": tree, "source_code": source, "symbols": symbols}

    # 3. Chunk
    chunker = CodeChunker(parser=parser)
    chunks = chunker.create_chunks([(str(java_file), data)])
    assert len(chunks) > 0, "Chunker should produce at least 1 chunk"

    # 4. Embed & Index into FAISS
    store = VectorStore()
    store.build_index(chunks)
    assert store.index.ntotal == len(chunks), "FAISS should contain all chunks"

    # 5. Retrieve via Semantic Search
    results = store.search("login", top_k=1)
    assert len(results) > 0, "Search should return matching results"
    score, top_chunk = results[0]
    assert top_chunk["method_name"] == "login"
