import os
import subprocess
from pathlib import Path

def resolve_repo_path(repo_input: str) -> Path:
    repo_input = repo_input.strip()

    if repo_input.startswith("http://") or repo_input.startswith("https://") or repo_input.endswith(".git"):
        repo_name = repo_input.rstrip("/").split("/")[-1].replace(".git", "")
        target_dir = Path("workspace") / repo_name

        if not target_dir.exists():
            print(f"[Git] Cloning repository '{repo_input}' into '{target_dir}'...")
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", repo_input, str(target_dir)], check=True)
        else:
            print(f"[Git] Using existing cloned repository at '{target_dir}'.")

        return target_dir

    return Path(repo_input)