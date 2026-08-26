import os
from pathlib import Path
from typing import Dict, Any, List, Optional


class GitIntelligence:
    """
    Extracts Git line provenance, method commit history, line blame,
    patch diffs, and commit-to-commit differences for Codebase
    Intelligence workflows.
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    def _get_relative_path(
        self,
        file_path: str,
        repo_working_dir: str
    ) -> str:
        """Normalizes and extracts repository-relative paths across OS boundaries."""
        clean_path_str = (
            file_path
            .replace("%2F", "/")
            .replace("\\", "/")
        )

        path_obj = Path(clean_path_str)

        if path_obj.is_absolute():
            full_path = path_obj.resolve()
        else:
            full_path = (
                self.repo_path / clean_path_str
            ).resolve()

        return os.path.relpath(
            full_path,
            repo_working_dir
        ).replace("\\", "/")

    def get_file_history(
        self,
        file_path: str,
        max_count: int = 10
    ) -> Dict[str, Any]:
        """Retrieves recent commit history for a specific file."""

        clean_path_str = (
            file_path
            .replace("%2F", "/")
            .replace("\\", "/")
        )

        full_path = (
            self.repo_path / clean_path_str
        ).resolve()

        if not full_path.exists():
            return {
                "file": clean_path_str,
                "error": (
                    f"File '{clean_path_str}' does not exist "
                    "on disk in repository."
                ),
                "commits": []
            }

        try:
            import git

            repo = git.Repo(
                self.repo_path,
                search_parent_directories=True
            )

            rel_path = self._get_relative_path(
                clean_path_str,
                repo.working_dir
            )

            commits_data = []

            for commit in repo.iter_commits(
                paths=rel_path,
                max_count=max_count
            ):
                commits_data.append({
                    "commit_hash": commit.hexsha[:8],
                    "author": commit.author.name,
                    "date": commit.committed_datetime.isoformat(),
                    "message": (
                        commit.message
                        .strip()
                        .split("\n")[0]
                    )
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

    def get_line_blame(
        self,
        relative_file_path: str,
        start_line: int,
        end_line: int
    ) -> List[Dict[str, Any]]:
        """Extracts Git blame information for a range of lines."""

        clean_path_str = (
            relative_file_path
            .replace("%2F", "/")
            .replace("\\", "/")
        )

        full_path = (
            self.repo_path / clean_path_str
        ).resolve()

        if (
            not full_path.exists()
            or start_line <= 0
            or end_line < start_line
        ):
            return []

        try:
            import git

            repo = git.Repo(
                self.repo_path,
                search_parent_directories=True
            )

            rel_path = self._get_relative_path(
                clean_path_str,
                repo.working_dir
            )

            blame_data = repo.blame(
                "HEAD",
                rel_path,
                L=(start_line, end_line)
            )

            results = []

            for commit, lines in blame_data:
                results.append({
                    "commit_hash": commit.hexsha[:8],
                    "author": commit.author.name,
                    "date": commit.committed_datetime.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "subject": (
                        commit.message
                        .strip()
                        .split("\n")[0]
                    ),
                    "lines_affected": len(lines)
                })

            return results

        except Exception:
            return []

    def get_commit_diff(
        self,
        old_commit: str,
        new_commit: str = "HEAD",
        file_path: Optional[str] = None,
        max_length: int = 10000
    ) -> Dict[str, Any]:
        """
        Returns the diff between two commits.

        Optionally restricts the diff to a specific file.
        """

        try:
            import git

            repo = git.Repo(
                self.repo_path,
                search_parent_directories=True
            )

            diff_args = [
                old_commit,
                new_commit
            ]

            clean_path_str = None

            if file_path:
                clean_path_str = (
                    file_path
                    .replace("%2F", "/")
                    .replace("\\", "/")
                )

                rel_path = self._get_relative_path(
                    clean_path_str,
                    repo.working_dir
                )

                diff_args.extend([
                    "--",
                    rel_path
                ])

            diff_output = repo.git.diff(
                *diff_args
            )

            truncated = len(diff_output) > max_length

            if truncated:
                diff_output = diff_output[:max_length]

            return {
                "old_commit": old_commit,
                "new_commit": new_commit,
                "file": clean_path_str,
                "diff": diff_output,
                "changed": bool(diff_output.strip()),
                "truncated": truncated
            }

        except Exception as e:
            return {
                "old_commit": old_commit,
                "new_commit": new_commit,
                "file": file_path,
                "diff": "",
                "changed": False,
                "truncated": False,
                "error": str(e)
            }

    def get_method_provenance(
        self,
        relative_file_path: str,
        method_name: str,
        start_line: int,
        end_line: int,
        max_commits: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieves line-level commit provenance, targeted method
        history, and patch diffs.
        """

        clean_path_str = (
            relative_file_path
            .replace("%2F", "/")
            .replace("\\", "/")
        )

        try:
            import git

            repo = git.Repo(
                self.repo_path,
                search_parent_directories=True
            )

            rel_path = self._get_relative_path(
                clean_path_str,
                repo.working_dir
            )

            commits_list = []

            record_sep = "<--COMMIT_END-->"

            try:
                log_raw = repo.git.log(
                    f"-n{max_commits}",
                    f"-L{start_line},{end_line}:{rel_path}",
                    (
                        "--pretty="
                        "format:%H|%an|%ad|%s"
                        f"{record_sep}"
                    ),
                    "--date=short"
                )

                raw_entries = [
                    entry.strip()
                    for entry in log_raw.split(record_sep)
                    if entry.strip()
                ]

                for entry in raw_entries:

                    lines = entry.splitlines()

                    header = (
                        lines[0]
                        if lines
                        else ""
                    )

                    if "|" not in header:
                        continue

                    parts = header.split(
                        "|",
                        3
                    )

                    if len(parts) != 4:
                        continue

                    (
                        commit_hash,
                        author,
                        commit_date,
                        subject
                    ) = parts

                    diff_lines = (
                        lines[1:]
                        if len(lines) > 1
                        else []
                    )

                    diff_output = (
                        "\n".join(diff_lines)
                        .strip()
                    )

                    if not diff_output:
                        try:
                            diff_output = repo.git.show(
                                commit_hash,
                                "--",
                                rel_path
                            )
                        except Exception:
                            diff_output = ""

                    commits_list.append({
                        "commit_hash": commit_hash[:8],
                        "author": author,
                        "date": commit_date,
                        "subject": subject,
                        "diff": diff_output[:1500]
                    })

            except Exception:

                fallback_history = self.get_file_history(
                    clean_path_str,
                    max_count=max_commits
                )

                for commit in fallback_history.get(
                    "commits",
                    []
                ):
                    commits_list.append({
                        "commit_hash": commit[
                            "commit_hash"
                        ],
                        "author": commit["author"],
                        "date": commit["date"][:10],
                        "subject": commit["message"],
                        "diff": ""
                    })

            primary = (
                commits_list[0]
                if commits_list
                else {
                    "commit_hash": "HEAD",
                    "author": "Unknown",
                    "date": "Unknown",
                    "subject": "Initial revision",
                    "diff": ""
                }
            )

            return {
                "file": clean_path_str,
                "method": method_name,
                "line_range": [
                    start_line,
                    end_line
                ],
                "primary_commit": primary,
                "history": commits_list
            }

        except Exception as e:

            return {
                "file": clean_path_str,
                "method": method_name,
                "line_range": [
                    start_line,
                    end_line
                ],
                "primary_commit": {
                    "commit_hash": "HEAD",
                    "author": "Unknown",
                    "date": "Unknown",
                    "subject": (
                        f"Git tracking error: {str(e)}"
                    ),
                    "diff": ""
                },
                "history": []
            }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Git Provenance Intelligence CLI"
    )

    parser.add_argument(
        "--repo",
        type=str,
        default="./"
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True
    )

    parser.add_argument(
        "--start",
        type=int,
        default=1
    )

    parser.add_argument(
        "--end",
        type=int,
        default=20
    )

    args = parser.parse_args()

    git_intel = GitIntelligence(
        args.repo
    )

    history = git_intel.get_file_history(
        args.file
    )

    blame = git_intel.get_line_blame(
        args.file,
        args.start,
        args.end
    )

    print(
        f"File History Commits: "
        f"{len(history.get('commits', []))}"
    )

    print(
        f"Line Blame Entries: "
        f"{len(blame)}"
    )