import sys
import argparse
import tempfile
import shutil
import subprocess
from pathlib import Path

from src.repo_utils import clone_repo_if_url
from src.parser import JavaASTParser
from src.chunker import CodeChunker
from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph
from src.git_intelligence import GitIntelligence
from src.context_builder import CodeIntelligenceContextBuilder
from src.llm_engine import CodeIntelligenceEngine


def clone_repo_if_url(repo_input: str) -> tuple[Path, bool]:
    """
    Checks if repo_input is a Git URL.
    If yes, clones it into a temporary directory and returns (temp_path, True).
    If no, treats it as a local path and returns (Path(repo_input), False).
    """
    is_url = (
        repo_input.startswith("http://") 
        or repo_input.startswith("https://") 
        or repo_input.startswith("git@") 
        or repo_input.endswith(".git")
    )
    if is_url:
        temp_dir = tempfile.mkdtemp(prefix="repo_clone_")
        print(f"\n[Git] Cloning repository from '{repo_input}' into temporary directory...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_input, temp_dir],
                check=True,
                capture_output=True,
                text=True
            )
            return Path(temp_dir), True
        except subprocess.CalledProcessError as e:
            print(f"[Error] Failed to clone repository:\n{e.stderr}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            sys.exit(1)
        except Exception as e:
            print(f"[Error] Unexpected error cloning repository: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            sys.exit(1)
    else:
        return Path(repo_input), False


def main():
    cli_parser = argparse.ArgumentParser(
        description="Codebase Vector & Knowledge Graph Search Engine"
    )
    cli_parser.add_argument(
        "repo_path", 
        nargs="?", 
        default=None, 
        help="Path or GitHub repository URL to analyze"
    )
    cli_parser.add_argument(
        "--mode",
        choices=["interactive", "why-changed"],
        default="interactive",
        help="Mode to run: 'interactive' search or 'why-changed' method provenance query"
    )
    cli_parser.add_argument("--file", type=str, help="Relative file path for --mode why-changed")
    cli_parser.add_argument("--method", type=str, help="Method name for --mode why-changed")
    cli_parser.add_argument("--start", type=int, help="Start line number for --mode why-changed")
    cli_parser.add_argument("--end", type=int, help="End line number for --mode why-changed")
    cli_parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of FAISS index")

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

        print(f"\n[1/5] Parsing repository at '{repo_path}'...")
        parser = JavaASTParser()
        extracted_data = []

        for java_file in repo_path.rglob("*.java"):
            # Skip module and package metadata descriptors
            if java_file.name in {"module-info.java", "package-info.java"}:
                continue

            try:
                result = parser.parse_file(str(java_file))
                if result:
                    if isinstance(result, dict):
                        extracted_data.append((Path(java_file), result))
                    elif isinstance(result, (tuple, list)):
                        tree = result[0]
                        source_code = result[1] if len(result) > 1 else ""
                        symbols = parser.extract_symbols_and_relations(tree, source_code)
                        extracted_data.append((Path(java_file), {
                            "tree": tree, 
                            "source_code": source_code,
                            "symbols": symbols
                        }))
            except Exception:
                # Silently skip parsing errors on complex/non-standard syntax
                pass

        if not extracted_data:
            print(f"No Java files found or successfully parsed in '{repo_path}'.")
            sys.exit(1)

        print(f"Parsed {len(extracted_data)} Java file(s).")

        print("\n[2/5] Generating AST chunks...")
        chunker = CodeChunker(parser=parser)
        chunks = chunker.create_chunks(extracted_data)
        print(f"Total Chunks Created: {len(chunks)}")

        print("\n[3/5] Building Code Knowledge Graph...")
        kg = CodeKnowledgeGraph()
        kg.build_graph_from_chunks(chunks)
        summary = kg.get_summary()
        print(f"Graph Built: {summary['total_nodes']} nodes, {summary['total_edges']} edges.")

        print("\n[4/5] Initializing FAISS Vector Index...")
        store = VectorStore()
        
        # Load cached index or generate embeddings if missing or requested
        if args.rebuild_index or not store.load_index():
            print("Building new vector index...")
            store.build_index(chunks)
            store.save_index()
        else:
            print("FAISS vector index ready.")

        print("\n[5/5] Initializing Intelligence Engine & Context Builder...")
        git_intel = GitIntelligence(repo_path=str(repo_path))
        context_builder = CodeIntelligenceContextBuilder(git_intel=git_intel)
        
        engine = CodeIntelligenceEngine(
            repo_path=str(repo_path),
            vector_store=store,
            graph_db=kg,
            context_builder=context_builder
        )

        # Mode 1: Provenance check via CLI parameters
        if args.mode == "why-changed":
            if not all([args.file, args.method, args.start, args.end]):
                print("Error: --mode why-changed requires --file, --method, --start, and --end parameters.")
                sys.exit(1)
            
            res = engine.explain_why_changed(args.file, args.method, args.start, args.end)
            print("\n" + "="*80)
            print(f" PROVENANCE ANALYSIS: {args.method}() in {args.file}")
            print("="*80)
            print(res.get("answer", "No response generated."))
            print("="*80)
            return

        # Mode 2: Interactive CLI Search Loop
        print("\n==================================================")
        print("     FAISS Vector, Graph & LLM Engine Ready       ")
        print("==================================================")
        
        while True:
            try:
                query = input("\nEnter code query or question (or 'exit' to quit): ").strip()
                if not query or query.lower() == 'exit':
                    break

                # Query Hybrid Retriever & Synthesize via LLM
                response = engine.answer_query(query, top_k=5)
                retrieved_chunks = response.get("retrieved_chunks", [])

                print(f"\n--- Top Hybrid Matches ({len(retrieved_chunks)}) ---")
                for rank, chunk in enumerate(retrieved_chunks, start=1):
                    chunk_id = chunk.get("chunk_id", "Unknown ID")
                    chunk_type = chunk.get("chunk_type", "METHOD")
                    file_name = Path(chunk.get("file_name", "unknown")).name
                    method_name = chunk.get("method_name", "")
                    callers = chunk.get("graph_callers", [])
                    callees = chunk.get("graph_callees", [])
                    code_snippet = chunk.get("code_content", chunk.get("text_representation", ""))[:200]

                    print(f"\n[{rank}] ID: {chunk_id} | Type: {chunk_type}")
                    print(f"    File: {file_name} | Method: {method_name}")
                    if callers:
                        print(f"    Callers (Graph): {', '.join(callers)}")
                    if callees:
                        print(f"    Callees (Graph): {', '.join(callees)}")
                    print(f"    Snippet:\n{code_snippet}...\n")
                
                print("--- LLM Contextual Answer ---")
                print(response.get("answer", "No response generated."))

            except KeyboardInterrupt:
                break

    finally:
        if is_temp and repo_path.exists():
            print("\n[Git] Cleaning up temporary cloned repository...")
            shutil.rmtree(repo_path, ignore_errors=True)

if __name__ == "__main__":
    main()