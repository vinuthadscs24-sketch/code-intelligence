import os
import re
from pathlib import Path
from urllib.parse import urlparse
from git import Repo

class RepositoryScanner:
    def __init__(self, repo_input: str, workspace_dir: str = "workspace"):
        # 1. Strip Markdown link syntax [text](url) or <url>
        clean_str = re.sub(r'\[.*?\]\((.*?)\)', r'\1', repo_input.strip())
        clean_str = clean_str.strip('<> ')

        # 2. Parse URL and strip tracking query strings (?utm_source=...)
        parsed = urlparse(clean_str)
        if parsed.scheme in ["http", "https"]:
            self.repo_input = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        else:
            self.repo_input = clean_str

        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def prepare_repository(self) -> Path:
        if os.path.exists(self.repo_input):
            return Path(self.repo_input)

        if not self.repo_input.startswith(("http://", "https://")):
            raise FileNotFoundError(f"Invalid local path or URL: {self.repo_input}")

        repo_name = self.repo_input.rstrip("/").split("/")[-1].replace(".git", "")
        target_path = self.workspace_dir / repo_name

        if target_path.exists():
            print(f"[Scanner] Repository already cached at: {target_path}")
            return target_path

        print(f"[Scanner] Cloning remote repository: {self.repo_input} ...")
        Repo.clone_from(self.repo_input, target_path, depth=50)
        return target_path

    def scan_java_files(self, repo_path: Path) -> list[Path]:
        java_files = []
        for root, _, files in os.walk(repo_path):
            if any(ignore in root for ignore in ["test", "target", "build", ".git"]):
                continue
            for file in files:
                if file.endswith(".java"):
                    java_files.append(Path(root) / file)
        return java_files