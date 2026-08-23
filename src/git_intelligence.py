import os
from pathlib import Path
from typing import Dict, Any, List

class GitIntelligence:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)

    def get_file_history(self, file_path: str) -> Dict[str, Any]:
        # 1. Clean and normalize path separators (handles %2F and Windows backslashes)
        clean_path_str = file_path.replace("%2F", "/").replace("\\", "/")
        full_path = (self.repo_path / clean_path_str).resolve()

        # 2. Verify file existence within the repository
        if not full_path.exists():
            return {
                "file": clean_path_str,
                "error": f"File '{clean_path_str}' does not exist on disk in repository.",
                "commits": []
            }

        try:
            import git
            repo = git.Repo(self.repo_path, search_parent_directories=True)
            
            # Get relative path for Git command execution
            rel_path = os.path.relpath(full_path, repo.working_dir)

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