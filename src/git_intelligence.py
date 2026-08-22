import os
import sys
import argparse
import subprocess
from typing import Dict, Any, List, Optional


class GitIntelligence:
    """
    Interfaces with Git CLI to attach commit metadata, diffs, 
    and author history to AST method boundaries.
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
                errors="replace",  # Prevents Windows cp1252 decode crashes
                check=True
            )
            return (result.stdout or "").strip()
        except subprocess.CalledProcessError as e:
            return f"Git error: {(e.stderr or '').strip()}"

    def get_method_blame(self, relative_file_path: str, start_line: int, end_line: int) -> List[Dict[str, Any]]:
        """
        Runs line-bounded git blame across an AST method's exact line range.
        """
        raw_blame = self._run_git([
            "blame",
            f"-L{start_line},{end_line}",
            "--porcelain",
            "--",
            relative_file_path
        ])

        if raw_blame.startswith("Git error"):
            print(f"Error executing blame: {raw_blame}")
            return []

        commits: Dict[str, Dict[str, Any]] = {}
        current_hash = None
        
        for line in raw_blame.split("\n"):
            parts = line.split(" ")
            if len(parts[0]) == 40:  # 40-char SHA-1 hash header
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

    def get_commit_details(self, commit_hash: str) -> Dict[str, Any]:
        """
        Fetches the complete commit message, author, date, and diff patch.
        """
        metadata = self._run_git([
            "show",
            "-s",
            "--format=%H|%an|%ad|%s",
            "--date=iso-strict",
            commit_hash
        ])

        diff_patch = self._run_git([
            "show",
            "--patch",
            "--color=never",
            commit_hash
        ])

        parts = metadata.split("|") if "|" in metadata else [commit_hash, "Unknown", "Unknown", "No message"]
        
        return {
            "commit_hash": parts[0],
            "author": parts[1] if len(parts) > 1 else "Unknown",
            "date": parts[2] if len(parts) > 2 else "Unknown",
            "subject": parts[3] if len(parts) > 3 else "Unknown",
            "diff": diff_patch
        }

    def get_method_provenance(self, relative_file_path: str, method_name: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """
        Main entrypoint: Maps AST method bounds to latest commit details and patch diff.
        """
        blame_records = self.get_method_blame(relative_file_path, start_line, end_line)
        if not blame_records:
            return {"error": f"No Git blame data found for {method_name} in {relative_file_path}"}

        # Filter out zero-hash (uncommitted local working tree changes)
        committed_records = [
            r for r in blame_records 
            if not r["commit_hash"].startswith("00000000")
        ]

        if not committed_records:
            return {
                "method_name": method_name,
                "file_path": relative_file_path,
                "lines": [start_line, end_line],
                "blame_summary": blame_records,
                "primary_commit": {
                    "commit_hash": "uncommitted",
                    "author": "Local Working Tree",
                    "date": "Now",
                    "subject": "Uncommitted local edits",
                    "diff": "Local uncommitted changes exist for this method."
                }
            }

        # Identify primary committed hash with max affected lines
        primary_commit = max(committed_records, key=lambda x: x["lines_affected"])
        commit_details = self.get_commit_details(primary_commit["commit_hash"])

        return {
            "method_name": method_name,
            "file_path": relative_file_path,
            "lines": [start_line, end_line],
            "blame_summary": blame_records,
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

    print("\n" + "="*80)
    print(f" GIT PROVENANCE ANALYSIS: {provenance.get('method_name')}")
    print("="*80)
    print(f" File: {provenance.get('file_path')} (Lines {args.start}-{args.end})")
    
    primary = provenance.get("primary_commit", {})
    print(f" Primary Commit Hash : {primary.get('commit_hash')}")
    print(f" Author              : {primary.get('author')}")
    print(f" Date                : {primary.get('date')}")
    print(f" Commit Message      : {primary.get('subject')}")
    
    print("\n--- LINE BLAME DISTRIBUTION ---")
    for record in provenance.get("blame_summary", []):
        print(f" * Commit {record['commit_hash'][:8]} by {record['author']} -> Modifies {record['lines_affected']} lines")

    print("\n--- COMMIT DIFF PATCH (Truncated 15 lines) ---")
    diff_lines = primary.get("diff", "").split("\n")
    for line in diff_lines[:15]:
        print(f"   {line}")
    if len(diff_lines) > 15:
        print(f"   ... ({len(diff_lines) - 15} more lines)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()