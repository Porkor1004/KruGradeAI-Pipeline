"""Shared output-path guard: generated artifacts never belong in data/raw/."""
from pathlib import Path


def output_directory(path: str | Path) -> Path:
    directory = Path(path).resolve()
    reserved = Path(__file__).resolve().parent.parent / "data" / "raw"
    if directory == reserved or reserved in directory.parents:
        raise ValueError("Generated files cannot be placed inside data/raw/")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def claim_live_directory(path):
    """Atomically claim a new Git-ignored experiment directory."""
    import subprocess
    directory = Path(path).resolve()
    repo = Path(__file__).resolve().parents[1]
    if repo/'artifacts' not in directory.parents:
        raise ValueError("Live experiment outputs must be inside the private artifacts directory")
    check = subprocess.run(["git", "check-ignore", "-q", "--", str(directory/"status.json")],
                           cwd=repo, capture_output=True)
    if check.returncode:
        raise ValueError("Live experiment directory must be ignored by Git")
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    return directory
