"""Read-only real-data adaptation and deterministic, group-aware preparation.

No agent/provider imports. Source-specific evidence belongs here, not in agents.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

from krugradeai.artifact_paths import output_directory
from krugradeai.experiment_models import DatasetProvenance, ExperimentDataset, HeldoutRecord
from krugradeai.models import Question, Rubric, RubricCriterion, StudentResponse, TAReference
from krugradeai.optimization_models import DevelopmentRecord, RecordProvenance, ValidationRecord

SEED = "krugradeai-mixed-signal-v1"
PREFIX = "com-eng-math-ii-2026-1-p"
MARKERS = ("unclear", "illegible", "ไม่พบคำตอบ", "อ่านไม่ชัด", "ไม่ชัด")
FLAGS = {
    "36dbbbd4-dd47-5309-aca2-0136b959d281": "unresolved_score_rubric_conflict",
    "706aa417-6b4c-5613-b19e-e858e99a0f50": "unresolved_score_rubric_conflict",
    "e610c8da-3ce0-54f2-9bee-db81590006fe": "ambiguous_answer",
}
AUDITED_HASHES = {
    "zip": "af2a5efbc152294fe96dc7dfcece346b6800d39f27745d641f2c940ad05c8288",
    "human": "5a5c8688816ac9afa6d00edcf58200f61c8641202111f6a2b44d402f5b714f2d",
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _rank(seed, value):
    return digest((seed + "\0" + value).encode("utf-8"))


def _decimal(value):
    if value is None or isinstance(value, bool):
        raise ValueError("Missing or Boolean score")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Non-finite score")
    return result


def _answer(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing answer")
    return value


def _marked(value):
    return any(marker in value.casefold() for marker in MARKERS)


def _sheet(workbook, name):
    rows = iter(workbook[name].values)
    header = next(rows)
    return [(number, dict(zip(header, row))) for number, row in enumerate(rows, 2)
            if any(value is not None for value in row)]


def _human_criteria(row):
    return [{"criterion_id": c["criterion_id"],
             "criterion": c["description"] + "\npartial_credit: " + c["partial_credit"],
             "score": c["points"]} for c in json.loads(row["rubric_criteria"])]


def _definition(item):
    criteria = tuple(RubricCriterion(id=c["criterion_id"], description=c["criterion"],
                                     max_score=_decimal(c["score"])) for c in item["rubric"])
    definition = {"question_id": item["question_id"], "criteria": [c.model_dump(mode="json") for c in criteria],
                  "total": str(_decimal(item["max_score"]))}
    identity = item["question_id"] + "@source-865c178123a2@" + digest(canonical(definition).encode())[:16]
    return (Question(id=item["question_id"], text=item["question_text"], reference_solution=item["reference_answer"]),
            Rubric(id=identity, criteria=criteria, total_score=_decimal(item["max_score"])))


def load_sources(raw_directory: str | Path):
    """Discover carriers, verify their declarations, and return audited candidates.

    Returned candidates are not split records. Quarantine retains source locators,
    without copying raw student answers into reports.
    """
    import openpyxl  # Optional data-preparation dependency; never opens in write mode.

    raw = Path(raw_directory).resolve()
    files = sorted(p for p in raw.iterdir() if p.is_file())
    hashes = {p.name: digest(p.read_bytes()) for p in files}
    archives = []
    for path in files:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                roots = [n.removesuffix("manifest.json") for n in archive.namelist()
                         if n.endswith("/manifest.json") and "reference-manifest.json" not in n]
                for root in roots:
                    if root + "01_source_reviewed/training.jsonl" in archive.namelist():
                        archives.append((path, root))
    carriers = []
    for path in files:
        if path.suffix.lower() == ".xlsx":
            with_workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
            try:
                if {"usable-319", "krugrade-human-review success31"} <= set(with_workbook.sheetnames):
                    carriers.append(path)
            finally:
                with_workbook.close()
    if len(archives) != 1 or len(carriers) != 1:
        raise ValueError("Expected one recognized ZIP and one human-review carrier")
    archive_path, root = archives[0]
    if hashes[archive_path.name] != AUDITED_HASHES["zip"] or hashes[carriers[0].name] != AUDITED_HASHES["human"]:
        raise ValueError("Source snapshot differs from the approved scope audit; re-audit before importing")
    quarantine, candidates, decisions, definitions = [], [], [], {}

    def reject(source, locator, reason, **metadata):
        quarantine.append(dict(source=source, locator=locator, reason=reason, **metadata))

    def consider(candidate):
        candidates.append(candidate)
        for locator in candidate["locators"]:
            decisions.append(dict(source=candidate["source"], locator=locator,
                                  decision="eligible", record_id=candidate["id"], group_id=candidate["group_id"]))

    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(root + "manifest.json"))
        for name, expected in manifest["files_sha256"].items():
            if digest(archive.read(root + name)) != expected:
                raise ValueError("ZIP member hash mismatch: " + name)
        reference_manifest = json.loads(archive.read(root + "reference-manifest.json"))
        official = [r for r in reference_manifest["references"] if r["role"] == "official_rubric_1_8"]
        if len(official) != 1 or official[0]["sha256"] != "865c178123a2d14342de72087f8fd19dd2e5a8312ae660c18da92a604b4dd136":
            raise ValueError("Unrecognized official rubric carrier; repeat scope audit")
        rows = []
        for group in ("01_source_reviewed", "02_ai_augmented"):
            provenance = {p["response_id"]: p for p in json.loads(archive.read(root + group + "/provenance.json"))}
            for number, line in enumerate(archive.read(root + group + "/training.jsonl").splitlines(), 1):
                if line.strip():
                    row = json.loads(line, parse_float=Decimal)
                    rows.append((group, number, row, provenance[row["response_id"]]))
        for group, number, row, prov in rows:
            item = row["input"]
            if row["reference_origin"] == "teacher_supplied" and item["question_id"] in {PREFIX + str(i) for i in range(1, 9)}:
                definition = _definition(item)
                if item["question_id"] in definitions and definitions[item["question_id"]] != definition:
                    raise ValueError("Conflicting immutable ZIP question/rubric/reference definitions")
                definitions[item["question_id"]] = definition
        if len(definitions) != 8:
            raise ValueError("Missing audited parent definitions")
        for group, number, row, prov in rows:
            locator = f"{archive_path.name}!{root}{group}/training.jsonl:{number}"
            item, rid = row["input"], row["response_id"]
            metadata = dict(response_id=rid, generation_kind=row["generation_kind"], reference_origin=row["reference_origin"])
            if row["reference_origin"] != "teacher_supplied" or item["question_id"] not in definitions:
                reject("zip320", locator, "unsupported_ai_reference_scope", **metadata)
                continue
            try:
                text = _answer(item["answer_text"])
                if _marked(text):
                    raise ValueError("ambiguous_answer_marker")
                if row["split_group"] != prov["source_attempt_id"] or row["source_response_id"] != prov["source_response_id"]:
                    raise ValueError("inconsistent_source_lineage")
                if prov["question_id"] != item["question_id"] or prov["reference_origin"] != row["reference_origin"]:
                    raise ValueError("inconsistent_source_scope")
                score = _decimal(row["target"]["score"])
                question, rubric = definitions[item["question_id"]]
                if not 0 <= score <= rubric.total_score:
                    raise ValueError("score_out_of_bounds")
                if row["generation_kind"] == "source_reviewed" and row["label_source"] != "teacher_derived":
                    raise ValueError("inconsistent_label_provenance")
                if row["generation_kind"] != "source_reviewed" and row["label_source"] != "ai_generated":
                    raise ValueError("inconsistent_label_provenance")
            except (ValueError, KeyError, TypeError) as error:
                reject("zip320", locator, str(error), **metadata)
                continue
            consider(dict(id="zip320:" + rid, source="zip320", label_origin="ai_derived",
                          group_id="zip320:" + row["split_group"], question_id=item["question_id"],
                          text=text, score=str(score), locators=[locator], response_ids=[rid],
                          lineage=sorted(set([rid, row["source_response_id"], *prov["source_response_ids"]])),
                          source_attempt_ids=[prov["source_attempt_id"]], source_files=[prov["source_file"]],
                          generation_kinds=[row["generation_kind"]], label_sources=[row["label_source"]],
                          reference_origin=row["reference_origin"], reference_verified_by="source_manifest",
                          verification="AI source review or AI generation; not human-verified"))

    carrier = carriers[0]
    workbook = openpyxl.load_workbook(carrier, read_only=True, data_only=True)
    try:
        review_rows = _sheet(workbook, "krugrade-human-review success31")
        review = {r["response_id"]: (n, r) for n, r in review_rows}
        if len(review) != len(review_rows):
            raise ValueError("Duplicate human review identities require adjudication")
        human = _sheet(workbook, "usable-319")
    finally:
        workbook.close()
    by_parent = defaultdict(list)
    for number, row in human:
        rid = row.get("response_id")
        locator = f"{carrier.name}!usable-319:{number}"
        try:
            if rid in FLAGS:
                raise ValueError(FLAGS[rid])
            review_number, verified = review[rid]
            if verified["review_status"] not in ("approved", "corrected") or verified["stale"] is not False:
                raise ValueError("unverified_or_stale_human_label")
            score = _decimal(row["expected_score"])
            if score != _decimal(verified["w"]):
                raise ValueError("human_score_review_mismatch")
            text = _answer(row["answer_text"])
            original = _answer(verified["corrected_answer"] if verified["corrected_answer"] is not None else verified["original_answer"])
            if "".join(text.split()).casefold() != "".join(original.split()).casefold():
                raise ValueError("answer_review_mismatch")
            if _marked(text):
                raise ValueError("ambiguous_answer_marker")
            scope = row["question_id"]
            if not scope.startswith(PREFIX):
                raise ValueError("unknown_question_scope")
            parent = PREFIX + scope[len(PREFIX):].split(".")[0]
            question, rubric = definitions[parent]
            parts = _human_criteria(row)
            expected = {c.id: (c.description, c.max_score) for c in rubric.criteria}
            ids = [c["criterion_id"] for c in parts]
            if len(ids) != len(set(ids)) or any(expected.get(c["criterion_id"]) != (c["criterion"], _decimal(c["score"])) for c in parts):
                raise ValueError("rubric_subset_mismatch")
            maximum = _decimal(row["rubric_max_score"])
            if sum((_decimal(c["score"]) for c in parts), Decimal(0)) != maximum or not 0 <= score <= maximum:
                raise ValueError("invalid_score_or_maximum")
            if verified["verified_max_score"] is not None and maximum != _decimal(verified["verified_max_score"]):
                raise ValueError("review_maximum_mismatch")
            # Freeze the audited wording, context, references and subset scope:
            # this is an import of the audited carrier, not a generic ID guesser.
            if row["rubric_version"] not in ("source-865c178123a2", "source-865c178123a2-subset-7.1-7.2"):
                raise ValueError("unrecognized_rubric_version")
            if scope == parent and row["question_text"] != question.text:
                raise ValueError("full_question_prompt_mismatch")
            if str(row["answer"]) not in question.reference_solution:
                raise ValueError("reference_answer_mismatch")
            if scope.endswith("7.1-7.2"):
                if verified["question_label"] != "7.1)-7.2)" or maximum != 4:
                    raise ValueError("unconfirmed_combined_subset")
            elif str(row["solution"]) not in question.reference_solution:
                raise ValueError("reference_solution_mismatch")
            by_parent[(row["attempt_id"], parent)].append(dict(row=row, rid=rid, locator=locator,
                review_locator=f"{carrier.name}!krugrade-human-review success31:{review_number}",
                text=text, score=score, ids=ids, pdf=verified["source_pdf"]))
        except (ValueError, KeyError, TypeError) as error:
            reject("human319", locator, str(error), response_id=rid)

    for (attempt, parent), rows in sorted(by_parent.items()):
        expected = {c.id for c in definitions[parent][1].criteria}
        covers = []
        for size in range(1, len(rows) + 1):
            for subset in itertools.combinations(rows, size):
                ids = [cid for part in subset for cid in part["ids"]]
                if len(ids) == len(set(ids)) and set(ids) == expected:
                    covers.append(subset)
        if len(covers) > 1:
            raise ValueError("Ambiguous human parent cover; do not choose scores automatically")
        if not covers:
            for part in rows:
                reject("human319", part["locator"], "incomplete_parent_criterion_coverage", response_id=part["rid"])
            continue
        selected = covers[0]
        if len({p["pdf"] for p in selected}) != 1:
            raise ValueError("Human attempt combines different source PDFs")
        # Exact disjoint criterion cover authorizes addition; no allocation of
        # a full score to constituent criteria or missing parts is performed.
        score = sum((p["score"] for p in selected), Decimal(0))
        text = selected[0]["text"] if len(selected) == 1 else "\n\n".join(
            "[Source scope " + p["row"]["question_id"] + "]\n" + p["text"] for p in selected)
        ids = sorted(p["rid"] for p in selected)
        rid = "human319:" + digest((attempt + "|" + parent + "|" + "|".join(ids)).encode())[:24]
        consider(dict(id=rid, source="human319", label_origin="human_derived", group_id="human319:" + attempt,
                      question_id=parent, text=text, score=str(score), locators=[p["locator"] for p in selected],
                      review_locators=[p["review_locator"] for p in selected], response_ids=ids, lineage=ids,
                      source_attempt_ids=[attempt], source_files=[selected[0]["pdf"]],
                      generation_kinds=["reviewed_full_question" if len(selected) == 1 else "disjoint_reviewed_subparts"],
                      label_sources=["approved_or_corrected_review_log"], reference_origin="teacher_supplied",
                      reference_verified_by="shared_criterion_definitions",
                      constituent_scores=[dict(response_id=p["rid"], score=str(p["score"]),
                                               criterion_ids=p["ids"]) for p in selected],
                      verification="Review-linked workbook scores; reviewer identities and source PDFs not independently authenticated"))
    # Reviewed source rows not present in usable are not silently promoted.
    usable_ids = {row["response_id"] for _, row in human}
    for number, row in review_rows:
        if row["response_id"] not in usable_ids:
            reject("human319_review_only", f"{carrier.name}!krugrade-human-review success31:{number}",
                   "not_in_audited_usable_sheet", response_id=row["response_id"], review_status=row["review_status"])
    return dict(input_hashes=hashes, definitions=definitions, candidates=candidates,
                quarantine=quarantine, decisions=decisions, reference_manifest=reference_manifest)


def _connect_groups(candidates):
    """Connect explicit attempt, source-file and response ancestry, across carriers."""
    parent = {c["group_id"]: c["group_id"] for c in candidates}

    def root(key):
        while parent[key] != key:
            key = parent[key]
        return key

    owner = {}
    for c in candidates:
        keys = (["response:" + i for i in c["lineage"]] + ["attempt:" + i for i in c["source_attempt_ids"]]
                + ["file:" + i.casefold().replace("\\", "/") for i in c["source_files"]])
        for key in keys:
            here = root(c["group_id"])
            if key in owner:
                other = root(owner[key])
                parent[max(here, other)] = min(here, other)
            owner[key] = here
    return {g: root(g) for g in parent}


def assign_groups(candidates, seed=SEED, trials=20000):
    """Choose by coverage/source counts only, never achieved scores or feedback.

    Whole eligible groups are selected. Unused ZIP groups are a reserve, not bad
    examples. Validation uses human-only connected components.
    """
    if len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate identities require source review")
    connected = _connect_groups(candidates)
    groups = defaultdict(list)
    for c in candidates:
        groups[connected[c["group_id"]]].append(c)
    human = [g for g, rows in groups.items() if all(c["label_origin"] == "human_derived" for c in rows)]
    ai = [g for g, rows in groups.items() if all(c["label_origin"] == "ai_derived" for c in rows)]
    if len(human) + len(ai) != len(groups):
        raise ValueError("Cross-source linked component requires manual split review")
    if len(ai) < 2:
        raise ValueError("Need distinct AI groups for mixed development and held-out")
    scopes = {c["question_id"] for c in candidates}
    human.sort()
    if len(human) < 6:
        raise ValueError("Insufficient independent human groups for three covered partitions")
    size = len(human) // 3
    best = None
    for trial in range(trials):
        order = sorted(human, key=lambda g: _rank(f"{seed}:human:{trial}", g))
        parts = [order[:size], order[size:2 * size], order[2 * size:]]
        counts = [Counter(c["question_id"] for g in part for c in groups[g]) for part in parts]
        if any(set(c) != scopes for c in counts):
            continue
        # Prefer >=2 validation observations per scope, then balanced human rows.
        objective = (sum(max(0, 2 - counts[1][q]) for q in scopes),
                     max(sum(c.values()) for c in counts) - min(sum(c.values()) for c in counts), trial)
        if best is None or objective < best[0]:
            best = objective, parts
    if best is None:
        raise ValueError("No group-aware human split covers all scopes; do not relax identity guards")
    _, (dev_h, val_h, test_h) = best
    assignments = {g: split for split, part in [("development", dev_h), ("validation", val_h), ("test", test_h)] for g in part}
    # One intact ZIP group closest to 20% of mixed test; human coverage wins.
    test_h_count = sum(len(groups[g]) for g in test_h)
    test_ai = min(ai, key=lambda g: (abs(4 * len(groups[g]) - test_h_count), _rank(seed + ":test-ai", g)))
    assignments[test_ai] = "test"
    # Deterministic subset-sum gives the closest attainable ~80% AI development.
    target = 4 * sum(len(groups[g]) for g in dev_h)
    reachable = {0: ()}
    for g in sorted((g for g in ai if g != test_ai), key=lambda g: _rank(seed + ":dev-ai", g)):
        for total, selected in list(reachable.items()):
            value = total + len(groups[g])
            if value not in reachable:
                reachable[value] = (*selected, g)
    chosen_total = min((n for n in reachable if n), key=lambda n: (abs(n - target), n))
    for g in reachable[chosen_total]:
        assignments[g] = "development"
    for g in groups:
        assignments.setdefault(g, "reserve")
    return {c["id"]: assignments[connected[c["group_id"]]] for c in candidates}, connected


def make_record(candidate, definition, split):
    cls = {"development": DevelopmentRecord, "validation": ValidationRecord, "test": HeldoutRecord}[split]
    question, rubric = definition
    return cls(id=candidate["id"], question=question, rubric=rubric,
               response=StudentResponse(id=candidate["id"], question_id=question.id, text=candidate["text"]),
               ta_output=TAReference(total_score=Decimal(candidate["score"])),
               provenance=RecordProvenance(source=candidate["source"], item_id=candidate["id"],
                                           label_origin=candidate["label_origin"]))


def heldout_metrics_by_origin(evaluation, records, *, qwk_categories=None):
    """Report an already completed final pass; never invokes an agent.

    All failed/review records remain in each cohort's expected count. This helper
    is deliberately separate from candidate selection and optimization.
    """
    from krugradeai.metrics import ScorePair, calculate_metrics
    records = tuple(HeldoutRecord.model_validate(r) for r in records)
    outcomes = {r.record_id: r for r in evaluation.records}
    if len(outcomes) != len(evaluation.records) or set(outcomes) != {r.id for r in records}:
        raise ValueError("Final outcomes must cover exactly the declared held-out identities")
    groups = {"mixed_source": records,
              "human_only": tuple(r for r in records if r.provenance and r.provenance.label_origin == "human_derived"),
              "ai_labeled_only": tuple(r for r in records if r.provenance and r.provenance.label_origin == "ai_derived")}
    if len(groups["human_only"]) + len(groups["ai_labeled_only"]) != len(records):
        raise ValueError("Explicit label provenance is required for every held-out record")
    result = {}
    for name, cohort in groups.items():
        pairs = []
        for record in cohort:
            outcome = outcomes[record.id]
            if outcome.grading is not None:
                grading = outcome.grading
                request = grading.grader_input
                if request.question != record.question or request.rubric != record.rubric or request.response != record.response:
                    raise ValueError("Held-out grading context does not match the source record")
                pairs.append(ScorePair(predicted=grading.grader_output.total_score,
                                       reference=record.ta_output.total_score, maximum=record.rubric.total_score))
        result[name] = calculate_metrics(pairs, expected_count=len(cohort), qwk_categories=qwk_categories)
    return result


def prepare(raw_directory, *, seed=SEED):
    imported = load_sources(raw_directory)
    assignments, connected = assign_groups(imported["candidates"], seed)
    partitions = {s: [] for s in ("development", "validation", "test")}
    lineage = []
    for c in sorted(imported["candidates"], key=lambda c: c["id"]):
        split = assignments[c["id"]]
        if split != "reserve":
            partitions[split].append(make_record(c, imported["definitions"][c["question_id"]], split))
        lineage.append({k: v for k, v in c.items() if k not in ("text", "score", "constituent_scores")} | {
            "constituent_criteria": [{k: v for k, v in part.items() if k != "score"}
                                     for part in c.get("constituent_scores", ())],
            "partition": split, "connected_group_id": connected[c["group_id"]],
            "decision": "eligible_reserve_for_source_balance" if split == "reserve" else "active"})
    dataset = ExperimentDataset(provenance=DatasetProvenance(kind="real",
        description="Mixed Signal 320 AI-derived and Signal 319 review-linked human-derived supervision; mixed metrics are not human grading accuracy.",
        sources=("zip320", "human319")), development=tuple(partitions["development"]),
        validation=tuple(partitions["validation"]), heldout=tuple(partitions["test"]))
    summary = {}
    for split in (*partitions, "reserve"):
        rows = [c for c in imported["candidates"] if assignments[c["id"]] == split]
        summary[split] = dict(records=len(rows), groups=len({connected[c["group_id"]] for c in rows}),
            label_origins=dict(Counter(c["label_origin"] for c in rows)),
            per_question={q: dict(Counter(c["label_origin"] for c in rows if c["question_id"] == q))
                          for q in sorted(imported["definitions"])},
            proportions={origin: str(Decimal(sum(c["label_origin"] == origin for c in rows)) / len(rows)) if rows else "0"
                         for origin in ("ai_derived", "human_derived")})
    manifest = dict(schema_version=1, seed=seed, policy_version="mixed-signal-v1", input_hashes=imported["input_hashes"],
        policy=dict(human_groups="deterministic coverage search; approximately one third per partition",
                    human_search_trials=20000, ai_development_target="4 times human development records, whole groups",
                    ai_test_target="one group closest to one quarter of human test records",
                    ambiguity_markers=MARKERS, ai_generated_references="excluded", synthetic_answers="allowed with provenance",
                    heldout_reporting=["mixed_source", "human_only", "ai_labeled_only"],
                    score_policy="disjoint complete human criterion covers summed; no score allocation or imputation"),
        summary=summary, rubric_definitions={q: r.model_dump(mode="json") for q, (_, r) in imported["definitions"].items()},
        lineage=lineage, eligibility_decisions=imported["decisions"],
        caveats=["ZIP labels are AI-derived, including teacher-derived totals read by AI; never verified human ground truth.",
                 "Excel labels accepted from non-stale approved/corrected review joins per user authorization; reviewer identities and PDFs are absent.",
                 "ta_output is a legacy numeric-reference field; interpret it with provenance.label_origin, not as certification of a TA label.",
                 "Do not pass validation or held-out records to Reflector/Refiner; only development belongs in OptimizationLoop.run.",
                 "No real-data optimization or evaluation has been performed. Future held-out reports must separate mixed/human/AI metrics."])
    manifest["partition_sha256"] = {split: digest("".join(row.model_dump_json() + "\n" for row in rows).encode("utf-8"))
                                    for split, rows in partitions.items()}
    manifest["quarantine_sha256"] = digest(canonical(imported["quarantine"]).encode("utf-8"))
    return dataset, manifest, imported["quarantine"]


def load_prepared_partition(directory, split):
    """Read one explicit partition; loading development never reads held-out labels."""
    cls = {"development": DevelopmentRecord, "validation": ValidationRecord, "test": HeldoutRecord}.get(split)
    if cls is None:
        raise ValueError("Unknown prepared partition")
    path = Path(directory) / (split + ".jsonl")
    manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    content = path.read_bytes()
    if digest(content) != manifest["partition_sha256"][split]:
        raise ValueError("Prepared partition hash mismatch; do not use modified labels")
    records = tuple(cls.model_validate_json(line) for line in content.decode("utf-8").splitlines() if line.strip())
    if any(r.provenance is None or r.provenance.label_origin is None for r in records):
        raise ValueError("Real-data partitions require explicit label provenance")
    if split == "validation" and any(r.provenance.label_origin != "human_derived" for r in records):
        raise ValueError("Validation must contain human-derived labels only")
    return records


def write_prepared(dataset, manifest, quarantine, destination):
    destination = Path(destination).resolve()
    # Refuse before any artifact is written; never overwrite a manifest/partition.
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Existing preparation output at {destination}; inspect its manifest before using a new destination")
    output_directory(destination)
    for split, rows in (("development", dataset.development), ("validation", dataset.validation), ("test", dataset.heldout)):
        with (destination / (split + ".jsonl")).open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(row.model_dump_json() + "\n")
    for name, content in (("manifest.json", manifest), ("quarantine.json", quarantine),
                          ("dataset_provenance.json", dataset.provenance.model_dump(mode="json"))):
        with (destination / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(content))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare mixed-source data only; never runs an agent")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="artifacts/real-mixed-v1")
    parser.add_argument("--seed", default=SEED)
    args = parser.parse_args(argv)
    # Identify existing outputs before even reading source material.
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error(f"Existing output directory: {output.resolve()}; no files overwritten")
    dataset, manifest, quarantine = prepare(args.raw_dir, seed=args.seed)
    write_prepared(dataset, manifest, quarantine, args.output_dir)
    print(canonical(manifest["summary"]))
    print(f"Quarantine decisions: {len(quarantine)}; no grading or evaluation performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
