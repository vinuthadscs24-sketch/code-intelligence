import os
import json
from typing import Dict, Any, List, Optional
from src.git_intelligence import GitIntelligence


class ContextAssemblyEngine:
    """
    Synthesizes structural dependencies (Graph), semantic code chunks (FAISS),
    and historical provenance (Git) into an LLM context payload.
    """
    def __init__(self, repo_path: str, vector_store=None, graph_db=None):
        self.repo_path = os.path.abspath(repo_path)
        self.vector_store = vector_store
        self.graph_db = graph_db
        self.git_intel = GitIntelligence(self.repo_path)

    def assemble_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes multi-modal retrieval and returns formatted context block + prompt.
        """
        context_items = []

        # 1. Semantic Retrieval via Vector Store (if present)
        semantic_results = []
        if self.vector_store and hasattr(self.vector_store, "search"):
            semantic_results = self.vector_store.search(query, top_k=top_k)

        # 2. Structural & Git Provenance Enrichment
        for item in semantic_results:
            symbol = item.get("symbol_name", "UnknownSymbol")
            file_path = item.get("file_path", "")
            start_line = item.get("start_line", 1)
            end_line = item.get("end_line", 10)

            # Structural relationships from NetworkX
            callers = []
            callees = []
            if self.graph_db:
                if hasattr(self.graph_db, "find_callers"):
                    callers = self.graph_db.find_callers(symbol)
                if hasattr(self.graph_db, "find_callees"):
                    callees = self.graph_db.find_callees(symbol)

            # Historical Provenance from Git Intelligence
            git_provenance = self.git_intel.get_method_provenance(
                relative_file_path=file_path,
                method_name=symbol,
                start_line=start_line,
                end_line=end_line
            )

            primary_commit = git_provenance.get("primary_commit", {})

            context_items.append({
                "symbol": symbol,
                "file": file_path,
                "lines": [start_line, end_line],
                "code_snippet": item.get("code", ""),
                "structural_context": {
                    "callers": callers,
                    "callees": callees
                },
                "git_provenance": {
                    "commit_hash": primary_commit.get("commit_hash"),
                    "author": primary_commit.get("author"),
                    "date": primary_commit.get("date"),
                    "commit_message": primary_commit.get("subject"),
                    "patch_diff": primary_commit.get("diff", "")[:500]  # Truncated patch
                }
            })

        # 3. Construct Final System Prompt
        formatted_payload = json.dumps(context_items, indent=2)
        system_prompt = self._build_prompt(query, formatted_payload)

        return {
            "query": query,
            "assembled_items_count": len(context_items),
            "raw_context_payload": context_items,
            "final_prompt": system_prompt
        }

    def _build_prompt(self, query: str, context_json: str) -> str:
        return f"""You are a Codebase Intelligence Engine.
Your task is to answer the developer's question using ONLY the provided code snippets, structural relationships, and Git provenance history.

=== RETRIEVED CODEBASE CONTEXT ===
{context_json}

=== DEVELOPER QUERY ===
{query}

=== INSTRUCTIONS ===
1. Base your answer strictly on the provided context, structural graph calls, and Git history.
2. If citing changes or history, reference the specific Commit Hash, Author, and Line Numbers.
3. Keep the response technical, direct, and actionable.

Answer:"""


# ==========================================
# Phase 8 CLI Verification Runner
# ==========================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 8: LLM Context Assembly Engine")
    parser.add_argument("--repo", type=str, default="./", help="Path to local Git repository")
    parser.add_argument("--query", type=str, required=True, help="Developer question/query")
    args = parser.parse_args()

    # Mock retrieval data for verification when FAISS is offline
    class MockVectorStore:
        def search(self, query, top_k=3):
            return [{
                "symbol_name": "scan_repository",
                "file_path": "src/scanner.py",
                "start_line": 10,
                "end_line": 35,
                "code": "def scan_repository(repo_path):\n    # Scans directory for .java files\n    pass"
            }]

    engine = ContextAssemblyEngine(
        repo_path=args.repo,
        vector_store=MockVectorStore()
    )

    result = engine.assemble_context(args.query)

    print("\n" + "="*80)
    print(" PHASE 8: ASSEMBLED LLM PROMPT PAYLOAD")
    print("="*80)
    print(result["final_prompt"])
    print("="*80 + "\n")


if __name__ == "__main__":
    main()