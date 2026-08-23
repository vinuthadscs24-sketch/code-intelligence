import sys
import argparse
import tempfile
import shutil
import subprocess
from pathlib import Path

from src.parser import JavaASTParser
from src.chunker import CodeChunker
from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph

def clone_repo_if_url(repo_input: str) -> tuple[Path, bool]:
    """
    Checks if repo_input is a Git URL.
    If yes, clones it into a temporary directory and returns (temp_path, True).
    If no, treats it as a local path and returns (Path(repo_input), False).
    """
    if repo_input.startswith("http://") or repo_input.startswith("https://") or repo_input.endswith(".git"):
        temp_dir = tempfile.mkdtemp(prefix="repo_clone_")
        print(f"\n[Git] Cloning repository from '{repo_input}' into temporary directory...")
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo_input, temp_dir], check=True)
            return Path(temp_dir), True
        except Exception as e:
            print(f"Error cloning repository: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            sys.exit(1)
    else:
        return Path(repo_input), False

def main():
    cli_parser = argparse.ArgumentParser(description="Codebase Vector & Knowledge Graph Search Engine")
    cli_parser.add_argument(
        "repo_path", 
        nargs="?", 
        default=None, 
        help="Path or GitHub repository URL to analyze"
    )
    args = cli_parser.parse_args()

    if not args.repo_path:
        repo_input = input("Enter path or Git repository URL (default: .): ").strip()
        repo_input = repo_input if repo_input else "."
    else:
        repo_input = args.repo_path

    repo_path, is_temp = clone_repo_if_url(repo_input)

    try:
        if not repo_path.exists() or not repo_path.is_dir():
            print(f"Error: Directory '{repo_path}' does not exist.")
            sys.exit(1)

        print(f"\n[1/4] Parsing repository at '{repo_path}'...")
        parser = JavaASTParser()
        extracted_data = []

        for java_file in repo_path.rglob("*.java"):
            # Skip non-class Java metadata files directly
            if java_file.name == "module-info.java" or "package-info.java" in java_file.name:
                continue

            try:
                result = parser.parse_file(str(java_file))
                if result:
                    if isinstance(result, dict):
                        extracted_data.append((Path(java_file), result))
                    elif isinstance(result, (tuple, list)):
                        tree = result[0]
                        source_code = result[1] if len(result) > 1 else ""
                        extracted_data.append((Path(java_file), {"tree": tree, "source_code": source_code}))
            except Exception:
                # Silently skip parsing errors on complex/modern syntax
                pass

        if not extracted_data:
            print(f"No Java files found or successfully parsed in '{repo_path}'.")
            sys.exit(1)

        print(f"Parsed {len(extracted_data)} Java file(s).")

        print("\n[2/4] Generating AST chunks...")
        chunker = CodeChunker(parser=parser)
        chunks = chunker.create_chunks(extracted_data)
        print(f"Total Chunks Created: {len(chunks)}")

        print("\n[3/4] Building Code Knowledge Graph...")
        kg = CodeKnowledgeGraph()
        kg.build_graph_from_chunks(chunks)
        summary = kg.get_summary()
        print(f"Graph Built: {summary['total_nodes']} nodes, {summary['total_edges']} edges.")

        print("\n[4/4] Building FAISS Vector Index...")
        store = VectorStore()
        store.build_index(chunks)

        print("\n==================================================")
        print("      FAISS Vector & Graph Search Engine Ready    ")
        print("==================================================")
        
        while True:
            try:
                query = input("\nEnter search query (or 'exit' to quit): ").strip()
                if not query or query.lower() == 'exit':
                    break

                results = store.search(query, top_k=3)
                print(f"\nTop matches for '{query}':")
                
                for rank, (chunk, score) in enumerate(results, start=1):
                    chunk_id = chunk.get("chunk_id", "Unknown ID")
                    chunk_type = chunk.get("chunk_type", "METHOD")
                    file_name = Path(chunk.get("file_name", "unknown")).name
                    annotations = chunk.get("annotations", [])
                    code_snippet = chunk.get("code_content", "")[:200]
                    method_name = chunk.get("method_name", "")

                    print(f"\n[{rank}] Score: {score:.4f} | ID: {chunk_id}")
                    print(f"    Type: {chunk_type} | File: {file_name}")
                    print(f"    Annotations: {annotations}")
                    
                    if method_name:
                        calls = kg.get_calls_from(method_name)
                        if calls:
                            print(f"    Outgoing Calls (Graph): {calls}")
                            
                    print(f"    Code Snippet:\n{code_snippet}...\n")
                    
            except KeyboardInterrupt:
                break

    finally:
        if is_temp and repo_path.exists():
            print("\n[Git] Cleaning up temporary cloned repository...")
            shutil.rmtree(repo_path, ignore_errors=True)

if __name__ == "__main__":
    main()