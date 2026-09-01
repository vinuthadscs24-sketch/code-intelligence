import sys
import tempfile
import shutil
import subprocess
from pathlib import Path


def clone_repo_if_url(repo_input: str) -> tuple[Path, bool]:
    """
    Clone a Git repository when repo_input is a URL.

    If repo_input is not a URL, treat it as a local repository path.

    Returns:
        (repository_path, is_temporary)
    """

    if not repo_input or not repo_input.strip():
        raise ValueError("Repository path or URL cannot be empty.")

    repo_input = repo_input.strip()

    is_url = (
        repo_input.startswith("http://")
        or repo_input.startswith("https://")
        or repo_input.startswith("git@")
        or repo_input.endswith(".git")
    )

    # ---------------------------------------------------------
    # Local repository
    # ---------------------------------------------------------

    if not is_url:
        repo_path = Path(repo_input).expanduser().resolve()

        if not repo_path.exists():
            raise FileNotFoundError(
                f"Repository path does not exist: {repo_path}"
            )

        if not repo_path.is_dir():
            raise NotADirectoryError(
                f"Repository path is not a directory: {repo_path}"
            )

        return repo_path, False

    # ---------------------------------------------------------
    # Remote repository
    # ---------------------------------------------------------

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="repo_clone_"
        )
    )

    print(
        f"\n[Git] Cloning repository from "
        f"'{repo_input}' into temporary directory..."
    )

    try:

        result = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                repo_input,
                str(temp_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        if result.stdout:
            print(
                result.stdout.strip()
            )

        print(
            f"[Git] Repository cloned successfully: "
            f"{temp_dir}"
        )

        return temp_dir, True

    except subprocess.CalledProcessError as e:

        error_message = (
            e.stderr.strip()
            if e.stderr
            else str(e)
        )

        print(
            "[Error] Failed to clone repository:\n"
            f"{error_message}"
        )

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise RuntimeError(
            f"Failed to clone repository: "
            f"{error_message}"
        ) from e

    except FileNotFoundError as e:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise RuntimeError(
            "Git was not found on the system. "
            "Make sure Git is installed and available "
            "in PATH."
        ) from e

    except Exception as e:

        print(
            f"[Error] Unexpected error cloning repository: {e}"
        )

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise RuntimeError(
            f"Unexpected error cloning repository: {e}"
        ) from e