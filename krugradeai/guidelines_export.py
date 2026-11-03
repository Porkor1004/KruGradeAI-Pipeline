"""Versioned selected-rule exports, with rubric identity rather than rubric edits."""
import hashlib
import re
from pathlib import Path
from krugradeai.artifact_paths import output_directory
from krugradeai.experiment_models import ExperimentResult, GuidelinesArtifact


def _filename_token(text: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]", "-", text).strip(".-")[:40] or "scope"
    return readable + "-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def export_guidelines(result: ExperimentResult, directory: str | Path) -> tuple[Path, ...]:
    result = ExperimentResult.model_validate(result)
    root = output_directory(directory)
    scopes = {scope.rubric_id: scope for scope in result.rubric_scopes}
    final = {item.rubric_id: item for item in result.heldout}
    artifacts = []
    for path in result.optimization.paths:
        selection = path.validation
        if selection is None or selection.best_version is None:
            continue
        selected = selection.evaluations[selection.best_version]
        heldout = final[path.rubric_id]
        artifact = GuidelinesArtifact(
            experiment_id=result.experiment_id, guideline_version=selection.best_version,
            rubric_scope=scopes[path.rubric_id], rules=selection.best_rules,
            validation_metrics=selected.metrics, validation_comparison_metrics=selected.comparison_metrics,
            heldout_baseline_metrics=heldout.baseline.metrics, heldout_selected_metrics=heldout.selected.metrics,
            heldout_status=heldout.status, configuration=result.config, provenance=result.provenance,
        )
        name = f"{_filename_token(result.experiment_id)}__{_filename_token(path.rubric_id)}__g{selection.best_version}.json"
        artifacts.append((root / name, artifact.model_dump_json(indent=2)))
    # Refuse all known collisions before writing; never overwrite a prior version.
    if any(path.exists() for path, _ in artifacts):
        raise FileExistsError("A versioned guidelines artifact already exists")
    written = []
    for path, content in artifacts:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content + "\n")
        written.append(path)
    return tuple(written)
