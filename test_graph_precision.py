import os
from pathlib import Path
from src.parser import JavaASTParser
from src.chunker import CodeChunker
from src.graph_builder import CodeKnowledgeGraph

def run_tests():
    # Target repository path (fallback to temp_audit_repo if spring-petclinic isn't cloned)
    repo_path = Path("workspace/spring-petclinic")
    if not repo_path.exists():
        repo_path = Path("temp_audit_repo")

    print(f"=== Running Precision Verification on: {repo_path} ===")

    parser = JavaASTParser()
    extracted = []

    for java_file in repo_path.rglob("*.java"):
        tree, source_code = parser.parse_file(str(java_file))
        if tree:
            extracted.append((str(java_file), {"tree": tree, "source_code": source_code}))

    chunker = CodeChunker()
    chunks = chunker.create_chunks(extracted)

    kg = CodeKnowledgeGraph()
    kg.build_graph_from_chunks(chunks)

    print("\n=== Knowledge Graph Query Verification ===")
    
    # 1. Methods in a class
    test_class = "OwnerController" if repo_path.name == "spring-petclinic" else "OuterRepository"
    methods = kg.get_class_methods(test_class)
    print(f"\n1. Methods in '{test_class}' ({len(methods)}):")
    print(f"   {methods}")

    # 2. Outgoing calls
    test_method = "processCreationForm" if repo_path.name == "spring-petclinic" else "OuterRepository.save"
    calls = kg.get_calls_from(test_method)
    print(f"\n2. What '{test_method}' calls:")
    print(f"   {calls}")

    # 3. Incoming callers
    callers = kg.get_callers_of("save")
    print(f"\n3. Who calls 'save':")
    print(f"   {callers}")

    # 4. Class lookup
    cls = kg.get_class_containing("save")
    print(f"\n4. Class containing 'save':")
    print(f"   {cls}")

if __name__ == "__main__":
    run_tests()