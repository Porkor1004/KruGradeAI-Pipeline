"""Shared output-path guard: generated artifacts never belong in data/raw/."""
from pathlib import Path


def output_directory(path: str | Path) -> Path:
    directory = Path(path).resolve()
    reserved = Path(__file__).resolve().parent.parent / "data" / "raw"
    if directory == reserved or reserved in directory.parents:
        raise ValueError("Generated files cannot be placed inside data/raw/")
    directory.mkdir(parents=True, exist_ok=True)
    return directory
