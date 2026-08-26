import os
from pathlib import Path
from typing import Dict, Any, List


class GitIntelligence:
    """
    Extracts git line provenance, method commit history, line blame, 
    and patch diffs to support Codebase Intelligence workflows.
    """
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    def _get_relative_path(self, file_path: str, repo_working_dir: str) -> str:
        clean_path_str = file_path.replace("%2F", "/").replace("\\", "/")
        full_path = (self.repo_path / clean_path_str).resolve()
        return os.path.relpath(full_path, repo_working_dir).replace("\\", "/")

    def get_file_history(self, file_path: str) -> Dict[str, Any]:
        """
        Retrieves recent commit history for a specific file.
        """
        clean_path_str = file_path.replace("%2F", "/").replace("\\", "/")
        full_path = (self.repo_path / clean_path_str).resolve()

        if not full_path.exists():
            return {
                "file": clean_path_str,
                "error": f"File '{clean_path_str}' does not exist on disk in repository.",
                "commits": []
            }

        try:
            import git
            repo = git.Repo(self.repo_path, search_parent_directories=True)
            rel_path = self._get_relative_path(clean_path_str, repo.working_dir)

            commits_data = []
            for commit in repo.iter_commits(paths=rel_path, max_count=10):
                commits_data.append({
                    "commit_hash": commit.hexsha[:8],
                    "author": commit.author.name,
                    "date": commit.committed_datetime.isoformat(),
                    "message": commit.message.strip()
                })

            return {
                "file": clean_path_str,
                "total_commits": len(commits_data),
                "commits": commits_data
            }

        except Exception as e:
            return {
                "file": clean_path_str,
                "error": f"Git tracking error: {str(e)}",
                "commits": []
            }

    def get_line_blame(self, relative_file_path: str, start_line: int, end_line: int) -> List[Dict[str, Any]]:
        """
        Extracts line-by-line git blame for a given range of lines.
        """
        clean_path_str = relative_file_path.replace("%2F", "/").replace("\\", "/")
        full_path = (self.repo_path / clean_path_str).resolve()

        try:
            import git
            repo = git.Repo(self.repo_path, search_parent_directories=True)
            rel_path = self._get_relative_path(clean_path_str, repo.working_dir)

            blame_data = repo.blame('HEAD', rel_path, L=f"{start_line},{end_line}")
            results = []

            for commit, lines in blame_data:
                results.append({
                    "commit_hash": commit.hexsha[:8],
                    "author": commit.author.name,
                    "date": commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    "subject": commit.message.strip().split('\n')[0],
                    "lines_affected": len(lines)
                })
            return results
        except Exception:
            return []

    def get_method_provenance(self, relative_file_path: str, method_name: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """
        Retrieves line-level commit provenance, multi-commit history, and patch diffs.
        """
        clean_path_str = relative_file_path.replace("%2F", "/").replace("\\", "/")

        try:
            import git
            repo = git.Repo(self.repo_path, search_parent_directories=True)
            rel_path = self._get_relative_path(clean_path_str, repo.working_dir)

            commits_list = []
            
            # 1. Attempt precise commit retrieval via file logs
            try:
                log_entries = repo.git.log(
                    "-n", "5",
                    "--pretty=format:%H|%an|%ad|%s",
                    "--date=short",
                    "--",
                    rel_path
                ).strip().split("\n")

                for entry in log_entries:
                    if not entry or "|" not in entry:
                        continue
                    parts = entry.split("|", 3)
                    if len(parts) == 4:
                        c_hash, c_author, c_date, c_subject = parts

                        try:
                            diff_output = repo.git.show(c_hash, "--", rel_path)
                        except Exception:
                            diff_output = ""

                        commits_list.append({
                            "commit_hash": c_hash[:8],
                            "author": c_author,
                            "date": c_date,
                            "subject": c_subject,
                            "diff": diff_output[:1500]  # Cap diff size for safety
                        })
            except Exception:
                commits_list = []

            primary = commits_list[0] if commits_list else {
                "commit_hash": "HEAD",
                "author": "Unknown",
                "date": "Unknown",
                "subject": "Initial revision",
                "diff": ""
            }

            return {
                "file": clean_path_str,
                "method": method_name,
                "line_range": [start_line, end_line],
                "primary_commit": primary,
                "history": commits_list
            }

        except Exception as e:
            return {
                "file": clean_path_str,
                "method": method_name,
                "line_range": [start_line, end_line],
                "primary_commit": {
                    "commit_hash": "HEAD",
                    "author": "Unknown",
                    "date": "Unknown",
                    "subject": f"Git tracking error: {str(e)}",
                    "diff": ""
                },
                "history": []
            }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Git Provenance Intelligence CLI")
    parser.add_argument("--repo", type=str, default="./")
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=20)
    args = parser.parse_args()

    git_intel = GitIntelligence(args.repo)
    history = git_intel.get_file_history(args.file)
    blame = git_intel.get_line_blame(args.file, args.start, args.end)
    
    print(f"File History Commits: {len(history.get('commits', []))}")
    print(f"Line Blame Entries: {len(blame)}")