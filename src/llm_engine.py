import os
import json
from typing import Dict, Any, Optional
from src.git_intelligence import GitIntelligence


class CodeIntelligenceEngine:
    """
    Phase 8: Synthesizes structural graph context, semantic vector search, 
    and Git provenance into evidence-backed LLM prompt answers.
    """
    def __init__(self, repo_path: str, vector_store=None, graph_db=None):
        self.repo_path = os.path.abspath(repo_path)
        self.vector_store = vector_store
        self.graph_db = graph_db
        self.git_intel = GitIntelligence(self.repo_path)

    def explain_why_changed(self, relative_file_path: str, method_name: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """
        Gathers Git provenance + AST method code + caller/callee context 
        and formats an LLM query answering: 'Why was this method changed?'
        """
        # Fetch Git Provenance from Phase 7
        provenance = self.git_intel.get_method_provenance(
            relative_file_path=relative_file_path,
            method_name=method_name,
            start_line=start_line,
            end_line=end_line
        )

        primary_commit = provenance.get("primary_commit", {})

        # Fetch NetworkX Graph relationships if present
        callers = self.graph_db.find_callers(method_name) if self.graph_db and hasattr(self.graph_db, "find_callers") else []
        callees = self.graph_db.find_callees(method_name) if self.graph_db and hasattr(self.graph_db, "find_callees") else []

        # Construct payload for Phase 8 prompt
        context_payload = {
            "symbol": method_name,
            "file": relative_file_path,
            "line_range": [start_line, end_line],
            "structural_graph": {
                "callers": callers,
                "callees": callees
            },
            "git_evidence": {
                "commit_hash": primary_commit.get("commit_hash"),
                "author": primary_commit.get("author"),
                "date": primary_commit.get("date"),
                "commit_message": primary_commit.get("subject"),
                "patch_diff": (primary_commit.get("diff") or "")[:1000]
            }
        }

        # Build prompt for LLM execution
        prompt = f"""You are a specialized Codebase Intelligence AI.
Analyze the provided code method, its Git history, commit messages, and patch diffs to answer: "Why was this method changed?"

=== RETRIEVED CONTEXT & EVIDENCE ===
{json.dumps(context_payload, indent=2)}

=== INSTRUCTIONS ===
1. State the exact primary reason for the change based on the commit message and patch diff.
2. Cite the Commit Hash ({primary_commit.get('commit_hash')}) and Author ({primary_commit.get('author')}).
3. Explain the technical modifications made inside the line range [{start_line}-{end_line}].

Explanation:"""

        return {
            "method_name": method_name,
            "context_payload": context_payload,
            "prompt": prompt
        }


# CLI test runner
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 8 Context Builder")
    parser.add_argument("--repo", type=str, default="./")
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)

    args = parser.parse_args()
    engine = CodeIntelligenceEngine(repo_path=args.repo)
    result = engine.explain_why_changed(args.file, args.method, args.start, args.end)

    print("\n" + "="*80)
    print(f" PHASE 8 PROMPT: Why was {args.method}() changed?")
    print("="*80)
    print(result["prompt"])
    print("="*80 + "\n")