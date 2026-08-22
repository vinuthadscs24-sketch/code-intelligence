import os
import sys
import argparse
import subprocess
from typing import Dict, Any, List, Optional


class GitIntelligence:
    """
    Interfaces with Git CLI to attach commit metadata, file-targeted diffs, 
    and historical evolution tracking to AST method boundaries.
    """
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        if not os.path.exists(os.path.join(self.repo_path, ".git")):
            raise ValueError(f"Directory '{self.repo_path}' is not a valid Git repository.")

    def _run_git(self, args: List[str]) -> str:
        """Executes a git subcommand with UTF-8 encoding and fallback safety."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True
            )
            return (result.stdout or "").strip()
        except subprocess.CalledProcessError as e:
            return f"Git error: {(e.stderr or '').strip()}"

    def get_method_blame(self, relative_file_path: str, start_line: int, end_line: int) -> List[Dict[str, Any]]:
        """Runs line-bounded git blame across an AST method's line range."""
        raw_blame = self._run_git([
            "blame",
            f"-L{start_line},{end_line}",
            "--porcelain",
            "--",
            relative_file_path
        ])

        if raw_blame.startswith("Git error"):
            return []

        commits: Dict[str, Dict[str, Any]] = {}
        current_hash = None
        
        for line in raw_blame.split("\n"):
            parts = line.split(" ")
            if len(parts[0]) == 40:
                current_hash = parts[0]
                if current_hash not in commits:
                    commits[current_hash] = {
                        "commit_hash": current_hash,
                        "lines_affected": 0,
                        "author": "Unknown",
                        "summary": ""
                    }
                commits[current_hash]["lines_affected"] += 1
            elif line.startswith("author ") and current_hash:
                commits[current_hash]["author"] = line[7:]
            elif line.startswith("summary ") and current_hash:
                commits[current_hash]["summary"] = line[8:]

        return list(commits.values())

    def get_commit_details(self, commit_hash: str, relative_file_path: str) -> Dict[str, Any]:
        """
        Fetches metadata and SCOPED diff patch for a single target file,
        preventing multi-file noise in LLM context.
        """
        metadata = self._run_git([
            "show",
            "-s",
            "--format=%H|%an|%ad|%s",
            "--date=iso-strict",
            commit_hash
        ])

        # Target diff strictly to the specific file
        diff_patch = self._run_git([
            "show",
            "--patch",
            "--color=never",
            commit_hash,
            "--",
            relative_file_path
        ])

        parts = metadata.split("|") if "|" in metadata else [commit_hash, "Unknown", "Unknown", "No message"]
        
        return {
            "commit_hash": parts[0],
            "author": parts[1] if len(parts) > 1 else "Unknown",
            "date": parts[2] if len(parts) > 2 else "Unknown",
            "subject": parts[3] if len(parts) > 3 else "Unknown",
            "file_diff": diff_patch
        }

    def get_method_history(self, relative_file_path: str, start_line: int, end_line: int, max_commits: int = 5) -> List[Dict[str, Any]]:
        """
        Tracks line-range historical evolution over time via git log -L.
        """
        raw_log = self._run_git([
            "log",
            f"-L{start_line},{end_line}:{relative_file_path}",
            f"-n{max_commits}",
            "--format=COMMIT_START|%H|%an|%ad|%s",
            "--date=iso-strict",
            "--no-patch"
        ])

        if raw_log.startswith("Git error"):
            return []

        history = []
        for line in raw_log.split("\n"):
            if line.startswith("COMMIT_START|"):
                parts = line.split("|")
                if len(parts) >= 5:
                    history.append({
                        "commit_hash": parts[1],
                        "author": parts[2],
                        "date": parts[3],
                        "subject": parts[4]
                    })
        return history

    def get_method_provenance(self, relative_file_path: str, method_name: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """
        Main entrypoint: Returns fully structured JSON-ready provenance data including
        dominant commit, file-scoped diff, blame breakdown, and historical evolution.
        """
        blame_records = self.get_method_blame(relative_file_path, start_line, end_line)
        if not blame_records:
            return {"error": f"No Git blame data found for {method_name} in {relative_file_path}"}

        committed_records = [
            r for r in blame_records 
            if not r["commit_hash"].startswith("00000000")
        ]

        # Fetch chronological commit history for the method's line bounds
        method_history = self.get_method_history(relative_file_path, start_line, end_line)

        if not committed_records:
            return {
                "method_name": method_name,
                "file_path": relative_file_path,
                "line_range": [start_line, end_line],
                "blame_summary": blame_records,
                "evolution_history": [],
                "primary_commit": {
                    "commit_hash": "uncommitted",
                    "author": "Local Working Tree",
                    "date": "Now",
                    "subject": "Uncommitted local edits",
                    "file_diff": "Local uncommitted changes exist for this method."
                }
            }

        # Dominant commit (highest line impact within method)
        primary_commit = max(committed_records, key=lambda x: x["lines_affected"])
        commit_details = self.get_commit_details(primary_commit["commit_hash"], relative_file_path)

        return {
            "method_name": method_name,
            "file_path": relative_file_path,
            "line_range": [start_line, end_line],
            "blame_summary": blame_records,
            "evolution_history": method_history,
            "primary_commit": commit_details
        }


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Line-Bounded Git Intelligence CLI")
    parser.add_argument("--repo", type=str, required=True, help="Path to local target Git repository")
    parser.add_argument("--file", type=str, required=True, help="Relative file path")
    parser.add_argument("--method", type=str, default="TargetMethod", help="Target method signature or name")
    parser.add_argument("--start", type=int, required=True, help="AST start line index")
    parser.add_argument("--end", type=int, required=True, help="AST end line index")

    args = parser.parse_args()

    git_intel = GitIntelligence(args.repo)
    provenance = git_intel.get_method_provenance(
        relative_file_path=args.file,
        method_name=args.method,
        start_line=args.start,
        end_line=args.end
    )

    import json
    print("\n" + "="*80)
    print(f" STRUCTURED GIT PROVENANCE DATA: {provenance.get('method_name')}")
    print("="*80)
    print(json.dumps(provenance, indent=2))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()