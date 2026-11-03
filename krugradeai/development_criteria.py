"""Opt-in extraction of actual source criterion labels for Development only.

Does not import agents/providers, access Validation/Test files, redistribute totals,
or overwrite the prepared split. Enrichment is a separately versioned artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import re
import sys
import zipfile

from krugradeai.artifact_paths import output_directory
from krugradeai.models import CriterionLabelProvenance, TAReference, TACriterionScore
from krugradeai.optimization_models import DevelopmentRecord
from krugradeai.real_data import canonical, digest, load_prepared_partition, _decimal, _human_criteria, _answer, _definition


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate source JSON key; no silent repair")
        result[key] = value
    return result


def _validate_preservation(base, enriched):
    if len(base) != len(enriched) or [r.id for r in base] != [r.id for r in enriched]:
        raise ValueError("Development identity/order changes are forbidden")
    for original, updated in zip(base, enriched):
        left, right = original.model_dump(), updated.model_dump()
        left["ta_output"].pop("criterion_scores")
        right["ta_output"].pop("criterion_scores")
        if left != right:
            raise ValueError("Enrichment may only add source-provided criterion labels")
        if original.ta_output.criterion_scores:
            raise ValueError("Base Development already has criterion labels; do not silently replace them")
        for score in updated.ta_output.criterion_scores:
            p = score.provenance
            if p is None or p.original_criterion_id != score.criterion_id or p.label_origin != original.provenance.label_origin:
                raise ValueError("Missing or inconsistent per-score provenance")


def enrich_development(data_directory="artifacts/real-mixed-v1", raw_directory="data/raw", *,
                       evidence_audit="artifacts/development-evidence-audit-v1/evidence_availability.json",
                       enable_source_criteria=False):
    if enable_source_criteria is not True:
        raise ValueError("Source criterion labels require explicit opt-in, including AI pseudo-ground truth")
    directory, raw = Path(data_directory), Path(raw_directory)
    audit_path = Path(evidence_audit)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    base = load_prepared_partition(directory, "development")
    if digest(manifest_bytes) != audit["split_manifest_sha256"] or digest((directory / "development.jsonl").read_bytes()) != audit["development_sha256"]:
        raise ValueError("Evidence audit does not match the immutable prepared Development snapshot")
    records = {r.id: r for r in base}
    lineage = {r["id"]: r for r in manifest["lineage"] if r["partition"] == "development"}
    if set(records) != set(lineage):
        raise ValueError("Development lineage mismatch")
    labels, decisions = {}, []
    checked_files = {}

    def source_file(name):
        # Locators are evidence metadata, never arbitrary filesystem access.
        if Path(name).name != name or name not in manifest["input_hashes"]:
            raise ValueError("Unknown or unsafe source carrier")
        path = raw / name
        expected = manifest["input_hashes"][name]
        if audit["source_hashes"].get(name) != expected or digest(path.read_bytes()) != expected:
            raise ValueError("Source carrier differs from audited input")
        checked_files[name] = expected
        return path

    def target(rid, origin):
        if rid not in records or records[rid].provenance.label_origin != origin:
            raise ValueError("Criterion evidence must belong to a Development record with matching origin")
        if rid in labels:
            raise ValueError("Duplicate record evidence; no silent merging")
        return records[rid], lineage[rid]

    ai_by_member = {}
    zip_paths = {}
    for evidence in audit["ai_only_criterion_evidence"]:
        rid = evidence["record_id"]
        record, item = target(rid, "ai_derived")
        locator = evidence["source_locator"]
        if locator not in item["locators"]:
            raise ValueError("AI evidence locator is not original Development lineage")
        name, member_line = locator.split("!", 1)
        member, number = member_line.rsplit(":", 1)
        if not member.endswith("/training.jsonl") or int(number) < 1:
            raise ValueError("Unsupported source representation")
        if name not in zip_paths:
            zip_paths[name] = source_file(name)
        key = name, member
        mapping = ai_by_member.setdefault(key, {})
        if int(number) in mapping:
            raise ValueError("Duplicate source locator")
        mapping[int(number)] = evidence
    for (name, member), wanted in ai_by_member.items():
        with zipfile.ZipFile(zip_paths[name]) as archive:
            found = set()
            for number, line in enumerate(archive.read(member).splitlines(), 1):
                if number not in wanted:
                    continue  # Never parse non-Development examples.
                evidence = wanted[number]
                rid = evidence["record_id"]
                record, item = records[rid], lineage[rid]
                row = json.loads(line, parse_float=Decimal, object_pairs_hook=_unique_object)
                points = row["target"]["point_scores"]
                if not isinstance(points, dict) or not points or set(points) != set(evidence["criterion_ids"]):
                    raise ValueError("Source criterion-map mismatch")
                if row["response_id"] not in item["response_ids"] or row["label_source"] != "ai_generated" or row["generation_kind"] not in item["generation_kinds"]:
                    raise ValueError("AI source identity/origin mismatch")
                if row["reference_origin"] != "teacher_supplied" or row["input"]["question_id"] != record.question.id:
                    raise ValueError("AI rubric scope mismatch")
                # Use exactly the existing adapter's Pydantic Text normalization
                # for question/reference edges; never normalize student answers.
                source_question, source_rubric = _definition(row["input"])
                if source_question != record.question or source_rubric != record.rubric or row["input"]["answer_text"] != record.response.text:
                    raise ValueError("AI source context differs from normalized Development")
                source_criteria = row["input"]["rubric"]
                actual = [(c["criterion_id"], c["criterion"], _decimal(c["score"])) for c in source_criteria]
                expected = [(c.id, c.description, c.max_score) for c in record.rubric.criteria]
                if actual != expected or _decimal(row["input"]["max_score"]) != record.rubric.total_score:
                    raise ValueError("Teacher rubric integrity mismatch")
                if _decimal(row["target"]["score"]) != record.ta_output.total_score:
                    raise ValueError("Source total differs from unchanged target")
                labels[rid] = tuple(TACriterionScore(criterion_id=cid, score=_decimal(value), provenance=CriterionLabelProvenance(
                    source=item["source"], label_origin="ai_derived", operational_role="pseudo_ground_truth",
                    original_criterion_id=cid, source_response_id=row["response_id"], source_question_id=record.question.id,
                    source_locator=evidence["source_locator"], source_sha256=checked_files[name],
                    extraction_evidence=(f"target.point_scores[{cid}]", "label_source=ai_generated", "generation_kind=" + row["generation_kind"])))
                    for cid, value in points.items())
                found.add(number)
            if found != set(wanted):
                raise ValueError("Missing source rows")

    human = audit["recoverable_human_criterion_evidence"]
    if human:
        import openpyxl
        from openpyxl.utils import get_column_letter
        workbooks = {}

        def read_cell_row(locator, sheet_name, field):
            match = re.fullmatch(r"([^!]+)!([^!]+):([A-Z]+)([1-9][0-9]*)", locator)
            if match is None or match[2] != sheet_name:
                raise ValueError("Unsupported worksheet evidence locator")
            name, _, column, number = match.groups()
            if name not in workbooks:
                workbooks[name] = openpyxl.load_workbook(source_file(name), read_only=True, data_only=True)
            sheet = workbooks[name][sheet_name]
            header = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
            if field not in header or get_column_letter(header.index(field) + 1) != column:
                raise ValueError("Worksheet score-column mismatch")
            values = next(sheet.iter_rows(min_row=int(number), max_row=int(number), values_only=True))
            return name, number, dict(zip(header, values))

        try:
            for evidence in human:
                rid = evidence["record_id"]
                record, item = target(rid, "human_derived")
                scores, seen_source = [], set()
                for entry in evidence["evidence"]:
                    name, number, row = read_cell_row(entry["score_cell"], "usable-319", "expected_score")
                    rn, review_number, review = read_cell_row(entry["review_score_cell"], "krugrade-human-review success31", "w")
                    if rn != name or f"{name}!usable-319:{number}" not in item["locators"] or f"{name}!krugrade-human-review success31:{review_number}" not in item["review_locators"]:
                        raise ValueError("Human evidence does not belong to Development lineage")
                    response_id, cid = entry["source_response_id"], entry["criterion_id"]
                    if response_id in seen_source or row["response_id"] != response_id or review["response_id"] != response_id or response_id not in item["response_ids"]:
                        raise ValueError("Human source identity mismatch or duplicate")
                    seen_source.add(response_id)
                    if review["review_status"] not in ("approved", "corrected") or review["stale"] is not False:
                        raise ValueError("Human source score is unreviewed/stale")
                    if review["source_pdf"] not in item["source_files"] or row["attempt_id"] not in item["source_attempt_ids"]:
                        raise ValueError("Human source attempt mismatch")
                    value = _decimal(row["expected_score"])
                    if value != _decimal(review["w"]):
                        raise ValueError("Human review score mismatch")
                    scope = row["question_id"]
                    if scope != entry["source_question_id"] or scope == record.question.id or not scope.startswith(record.question.id + "."):
                        raise ValueError("No named subquestion source; never allocate a whole-question total")
                    criteria = _human_criteria(row)
                    known = {c.id: c for c in record.rubric.criteria}
                    if len(criteria) != 1 or cid not in known or criteria[0] != dict(criterion_id=cid, criterion=known[cid].description, score=known[cid].max_score):
                        raise ValueError("Source unit is not exactly one unchanged rubric criterion")
                    if _decimal(row["rubric_max_score"]) != known[cid].max_score:
                        raise ValueError("Source unit maximum mismatch")
                    if review["verified_max_score"] is not None and _decimal(review["verified_max_score"]) != known[cid].max_score:
                        raise ValueError("Reviewed maximum mismatch")
                    # Verify the source's named unit rather than deducing it from a number.
                    label = str(review.get("corrected_question_label") if review.get("corrected_question_label") is not None else review["question_label"]).strip().rstrip(")")
                    if label != scope.rsplit("-p", 1)[-1]:
                        raise ValueError("Review entry does not explicitly identify this graded subquestion")
                    reviewed_answer = review["corrected_answer"] if review["corrected_answer"] is not None else review["original_answer"]
                    if _answer(row["answer_text"]) != _answer(reviewed_answer) or _answer(row["answer_text"]) not in record.response.text:
                        raise ValueError("Human source answer mismatch")
                    scores.append(TACriterionScore(criterion_id=cid, score=value, provenance=CriterionLabelProvenance(
                        source=item["source"], label_origin="human_derived", operational_role="human_reference",
                        original_criterion_id=cid, source_response_id=response_id, source_question_id=scope,
                        source_locator=entry["score_cell"], source_sha256=checked_files[name],
                        extraction_evidence=("named single-criterion reviewed subquestion", entry["review_score_cell"],
                                             "review_status=" + review["review_status"], "expected_score equals reviewed w; reviewer identity not independently authenticated"))))
                labels[rid] = tuple(scores)
        finally:
            for workbook in workbooks.values():
                workbook.close()

    enriched = []
    for record in base:
        scores = labels.get(record.id, ())
        data = record.model_dump()
        data["ta_output"] = TAReference(total_score=record.ta_output.total_score, criterion_scores=scores)
        # Reuse all contextual bounds, unique-ID and full/partial consistency checks.
        enriched.append(DevelopmentRecord.model_validate(data))
        decisions.append(dict(record_id=record.id, label_origin=record.provenance.label_origin,
                              decision="source_criteria_recovered" if scores else "unchanged_total_only",
                              criterion_ids=[s.criterion_id for s in scores]))
    enriched = tuple(enriched)
    _validate_preservation(base, enriched)
    content = "".join(r.model_dump_json() + "\n" for r in enriched).encode("utf-8")
    metadata = dict(schema_version=1, policy="explicit-source-criteria-mixed-development-v1", enabled=True,
        base_development_sha256=audit["development_sha256"], split_manifest_sha256=digest(manifest_bytes),
        evidence_audit_sha256=digest(audit_path.read_bytes()), source_hashes=checked_files,
        enriched_development_sha256=digest(content), decisions=decisions,
        summary=dict(records=len(enriched), enriched_records=len(labels), total_only_records=len(base)-len(labels),
                     criterion_scores=sum(len(r.ta_output.criterion_scores) for r in enriched),
                     criterion_scores_by_origin=dict(Counter(s.provenance.label_origin for r in enriched for s in r.ta_output.criterion_scores))),
        warnings=["AI criterion labels are operational pseudo-ground truth, never human-verified.",
                  "Human source-unit labels are review-linked; reviewer identities and original PDFs remain unauthenticated.",
                  "Only Development is enriched; no criterion allocations are inferred or repaired.",
                  "Supported findings remain relative to supplied targets; uncertainty or suspected inconsistencies still require review."])
    return enriched, metadata


def write_enrichment(records, metadata, destination, *, data_directory="artifacts/real-mixed-v1", raw_directory="data/raw"):
    destination = Path(destination).resolve()
    for protected in (Path(data_directory).resolve(), Path(raw_directory).resolve()):
        if destination == protected or protected in destination.parents or destination in protected.parents:
            raise ValueError("Enrichment output must be separate from raw/prepared data")
    if destination.exists():
        raise FileExistsError("Enrichment destination already exists; nothing overwritten")
    content = "".join(r.model_dump_json() + "\n" for r in records).encode("utf-8")
    if digest(content) != metadata["enriched_development_sha256"]:
        raise ValueError("Enrichment output hash mismatch")
    output_directory(destination.parent)
    destination.mkdir(exist_ok=False)
    with (destination / "development.jsonl").open("xb") as stream:
        stream.write(content)
    with (destination / "criterion_manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(canonical(metadata))


def load_enriched_development(data_directory, criteria_directory, *, enable_source_criteria=False):
    if enable_source_criteria is not True:
        raise ValueError("Explicit source criterion/pseudo-ground-truth opt-in required")
    base = load_prepared_partition(data_directory, "development")
    source = Path(data_directory)
    overlay = Path(criteria_directory)
    meta = json.loads((overlay / "criterion_manifest.json").read_text(encoding="utf-8"))
    if meta["policy"] != "explicit-source-criteria-mixed-development-v1" or meta["enabled"] is not True:
        raise ValueError("Unknown enrichment policy")
    if digest((source / "manifest.json").read_bytes()) != meta["split_manifest_sha256"] or digest((source / "development.jsonl").read_bytes()) != meta["base_development_sha256"]:
        raise ValueError("Enrichment belongs to a different prepared split")
    content = (overlay / "development.jsonl").read_bytes()
    if digest(content) != meta["enriched_development_sha256"]:
        raise ValueError("Enriched Development hash mismatch")
    records = tuple(DevelopmentRecord.model_validate_json(line) for line in content.decode("utf-8").splitlines() if line.strip())
    _validate_preservation(base, records)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description="Enrich Development only with explicit source scores; no agents")
    parser.add_argument("--enable-source-criteria", action="store_true")
    parser.add_argument("--data-dir", default="artifacts/real-mixed-v1")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--evidence-audit", default="artifacts/development-evidence-audit-v1/evidence_availability.json")
    parser.add_argument("--output-dir", default="artifacts/development-criteria-v1")
    args = parser.parse_args(argv)
    if not args.enable_source_criteria:
        parser.error("Explicit --enable-source-criteria is required; no sources read")
    if Path(args.output_dir).exists():
        parser.error("Output already exists; no artifact overwritten")
    try:
        records, meta = enrich_development(args.data_dir, args.raw_dir, evidence_audit=args.evidence_audit, enable_source_criteria=True)
        write_enrichment(records, meta, args.output_dir, data_directory=args.data_dir, raw_directory=args.raw_dir)
    except Exception as error:
        # Pydantic exception strings may contain source values; never print them.
        print(f"Development criterion enrichment failed ({type(error).__name__}); inspect source/audit consistency. No automatic repair or retry was made; base data is unchanged.", file=sys.stderr)
        return 1
    print(canonical(meta["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
