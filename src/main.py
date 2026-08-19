import sys
import argparse
from pathlib import Path

from parser import JavaASTParser 
from chunker import CodeChunker
from vector_store import VectorStore

def main():
    cli_parser = argparse.ArgumentParser(description="Codebase Vector Search Engine")
    cli_parser.add_argument(
        "repo_path", 
        nargs="?", 
        default=None, 
        help="Path to the repository directory to analyze"
    )
    args = cli_parser.parse_args()

    # Prompt user for path if not provided via command line argument
    if not args.repo_path:
        repo_input = input("Enter path to Java repository: ").strip()
        repo_path = Path(repo_input)
    else:
        repo_path = Path(args.repo_path)

    if not repo_path.exists() or not repo_path.is_dir():
        print(f"Error: Directory '{repo_path}' does not exist.")
        sys.exit(1)

    print(f"[1/3] Parsing repository at '{repo_path}' and extracting code data...")
    parser = JavaASTParser()
    extracted_data = []

    for java_file in repo_path.rglob("*.java"):
        parsed_tree = parser.parse_file(str(java_file))
        if parsed_tree:
            extracted_data.append((str(java_file), parsed_tree))

    if not extracted_data:
        print(f"No Java files found in '{repo_path}'.")
        sys.exit(1)

    print(f"Parsed {len(extracted_data)} Java files.")

    print("[2/3] Generating scoped AST chunks...")
    chunker = CodeChunker()
    chunks = chunker.create_chunks(extracted_data)
    print(f"Total Chunks Created: {len(chunks)}")

    print("[3/3] Building FAISS Vector Index...")
    store = VectorStore()
    store.build_index(chunks)

    print("\n--- FAISS Vector Search Ready ---")
    while True:
        try:
            query = input("\nEnter search query (or 'exit' to quit): ").strip()
            if not query or query.lower() == 'exit':
                break

            results = store.search(query, top_k=3)
            print(f"\nTop matches for '{query}':")
            for rank, (chunk, score) in enumerate(results, start=1):
                print(f"\n[{rank}] Score: {score:.4f} | ID: {chunk['chunk_id']}")
                print(f"    Type: {chunk['chunk_type']} | File: {Path(chunk['file_name']).name}")
                print(f"    Annotations: {chunk['annotations']}")
                print(f"    Code Snippet:\n{chunk['code_content'][:200]}...\n")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()