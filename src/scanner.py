import os
import shutil
from pathlib import Path
from git import Repo

class RepositoryScanner:
    def __init__(self, repo_input: str, workspace_dir: str = "./workspace"):
        self.repo_input = repo_input
        self.workspace_dir = Path(workspace_dir)

    def prepare_repository(self) -> Path:
        """Clones a remote GitHub URL or returns a validated local directory path."""
        if self.repo_input.startswith("http://") or self.repo_input.startswith("https://"):
            repo_name = self.repo_input.rstrip("/").split("/")[-1].replace(".git", "")
            target_path = self.workspace_dir / repo_name
            
            if target_path.exists():
                print(f"[Scanner] Repository already cached at: {target_path}")
                return target_path

            print(f"[Scanner] Cloning remote repository: {self.repo_input} ...")
            Repo.clone_from(self.repo_input, target_path, depth=50)
            return target_path
        
        local_path = Path(self.repo_input)
        if not local_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_path}")
        return local_path

    def scan_java_files(self, target_path: Path) -> list[Path]:
        """Finds all .java source files while ignoring build and test directories."""
        java_files = [
            p for p in target_path.rglob("*.java")
            if "test" not in p.parts and "target" not in p.parts and "build" not in p.parts
        ]
        return java_files