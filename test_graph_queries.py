from pathlib import Path
from src.parser import JavaASTParser
from src.chunker import CodeChunker
from src.knowledge_graph import KnowledgeGraph

def run_tests():
    repo_path = Path("workspace/spring-petclinic")
    parser = JavaASTParser()
    extracted = []

    for java_file in repo_path.rglob("*.java"):
        parsed_tree = parser.parse_file(str(java_file))
        if parsed_tree:
            extracted.append((str(java_file), parsed_tree))

    chunker = CodeChunker()
    chunks = chunker.create_chunks(extracted)

    kg = KnowledgeGraph()
    kg.build_graph(chunks)

    print("\n=== Knowledge Graph Query Verification ===")
    
    methods = kg.get_class_methods("OwnerController")
    print(f"\n1. Methods in 'OwnerController' ({len(methods)}):")
    print(f"   {methods}")

    calls = kg.get_calls_from("processCreationForm")
    print(f"\n2. What 'processCreationForm' calls:")
    print(f"   {calls}")

    callers = kg.get_callers_of("save")
    print(f"\n3. Who calls 'save':")
    print(f"   {callers}")

    cls = kg.get_class_containing("findPaginatedForOwnersLastName")
    print(f"\n4. Class containing 'findPaginatedForOwnersLastName':")
    print(f"   {cls}")

if __name__ == "__main__":
    run_tests()
