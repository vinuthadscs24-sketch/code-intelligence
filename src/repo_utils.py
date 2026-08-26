import sys
import tempfile
import shutil
import subprocess
from pathlib import Path


def clone_repo_if_url(repo_input: str) -> tuple[Path, bool]:
    """
    Checks if repo_input is a Git URL.
    If yes, clones it into a temporary directory.
    If no, treats it as a local path.
    """
    is_url = (
        repo_input.startswith("http://")
        or repo_input.startswith("https://")
        or repo_input.startswith("git@")
        or repo_input.endswith(".git")
    )

    if is_url:
        temp_dir = tempfile.mkdtemp(prefix="repo_clone_")

        print(
            f"\n[Git] Cloning repository from "
            f"'{repo_input}' into temporary directory..."
        )

        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    repo_input,
                    temp_dir,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            return Path(temp_dir), True

        except subprocess.CalledProcessError as e:
            print(
                f"[Error] Failed to clone repository:\n{e.stderr}"
            )
            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )
            sys.exit(1)

        except Exception as e:
            print(
                f"[Error] Unexpected error cloning repository: {e}"
            )
            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )
            sys.exit(1)

    else:
        return Path(repo_input), False