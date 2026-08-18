import sys
from scanner import RepositoryScanner
from parser import JavaASTParser
from extractor import JavaSymbolExtractor
from graph_builder import CodeKnowledgeGraph

def main():
    target_repo = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/SimpleProgramming/simple-springboot-app.git"

    # 1. Scan
    scanner = RepositoryScanner(target_repo)
    repo_path = scanner.prepare_repository()
    java_files = scanner.scan_java_files(repo_path)

    # 2. Parse & Extract
    ast_parser = JavaASTParser()
    extractor = JavaSymbolExtractor(ast_parser)

    extracted_data = []
    for file_path in java_files:
        tree, source_code = ast_parser.parse_file(file_path)
        symbols = extractor.extract_symbols(tree.root_node, source_code)
        extracted_data.append((file_path, symbols))

    # 3. Build Enhanced Knowledge Graph
    kg = CodeKnowledgeGraph()
    kg.build_graph(repo_path.name, extracted_data)

    # 4. Print Summary
    summary = kg.get_summary()

    print("\n==========================================")
    print("   Spring-Aware Knowledge Graph Built     ")
    print("==========================================")
    print(f"Target Repo : {repo_path.name}")
    print(f"Total Nodes : {summary['total_nodes']}")
    print(f"Total Edges : {summary['total_edges']}")
    
    print("\nNode Types:")
    for ntype, count in summary["node_types"].items():
        print(f"  ├── {ntype}: {count}")

    print("\nEdge Relationships:")
    for rel, count in summary["relationship_types"].items():
        print(f"  ├── {rel}: {count}")
    print("==========================================\n")

    # 5. Export GraphML
    kg.export_graphml("workspace/code_graph.graphml")

    # 6. Deep Inspection on Spring Controllers & Services
    sample_classes = ["AutowiredController", "StudentController", "StudentConfig"]
    for cls in sample_classes:
        deps = kg.inspect_class_dependencies(cls)
        if "error" not in deps:
            print(f"\n🔍 Inspection for '{cls}':")
            for rel, targets in deps["relationships"].items():
                if targets:
                    print(f"   ├── {rel}: {targets}")

if __name__ == "__main__":
    main()