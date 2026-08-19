import sys
import argparse
from src.repo_utils import resolve_repo_path
from src.parser import JavaASTParser
from src.chunker import CodeChunker

def verify():
    parser_cli = argparse.ArgumentParser(description="Test AST Graph Context Precision")
    parser_cli.add_argument("repo", nargs="?", help="Local path or GitHub repository URL")
    args = parser_cli.parse_args()

    repo_input = args.repo or input("Enter GitHub repository URL or local path: ").strip()
    
    try:
        repo_path = resolve_repo_path(repo_input)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not repo_path.exists():
        print(f"Error: Path '{repo_path}' does not exist.")
        sys.exit(1)

    parser = JavaASTParser()
    extracted = []

    print(f"Parsing Java files in '{repo_path}'...")
    for java_file in repo_path.rglob("*.java"):
        parsed_tree = parser.parse_file(str(java_file))
        if parsed_tree:
            extracted.append((str(java_file), parsed_tree))

    chunker = CodeChunker()
    chunks = chunker.create_chunks(extracted)

    print(f"\nTotal Chunks Extracted: {len(chunks)}")
    target_chunks = [c for c in chunks if c.get("class_name") == "OwnerController"]
    print(f"OwnerController Chunks Found: {len(target_chunks)}\n")

    for chunk in target_chunks:
        print(f"Method: {chunk['method_name']}")
        print(f"  CALLS ({len(chunk['relationships']['CALLS'])}): {chunk['relationships']['CALLS']}\n")

if __name__ == "__main__":
    verify()