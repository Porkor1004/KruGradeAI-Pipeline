"""Synthetic import fixtures only; no real student data or LLM calls in tests."""
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import openpyxl

from krugradeai.experiment_models import ExperimentDataset
from krugradeai.models import CriterionScore, GraderInput, GraderOutput, GradingRecord
from krugradeai.optimization_models import RecordHistory, ValidationConfig
from krugradeai.real_data import (
    FLAGS, PREFIX, assign_groups, canonical, heldout_metrics_by_origin, load_sources,
    load_prepared_partition, make_record, prepare, write_prepared,
)
from krugradeai.validation import prepare_validation


OFFICIAL = "865c178123a2d14342de72087f8fd19dd2e5a8312ae660c18da92a604b4dd136"


def fixture(folder, *, flagged=False, incomplete=False, bad_criterion=False, stale=False,
            ambiguous=False, ai_reference=False):
    """Eight made-up scopes, eight ZIP attempts and six human attempts.

    Names exercise the audited mapping, but content and scores are synthetic.
    Parent 2 is represented by disjoint separately reviewed human parts.
    """
    archive_path = folder / "synthetic.zip"
    members = {}
    for group in ("01_source_reviewed", "02_ai_augmented"):
        training, provenance = [], []
        start = 0 if group.startswith("01") else 4
        for attempt in range(start, start + 4):
            for number in range(1, 9):
                rid = f"zip-{attempt}-{number}"
                criteria = [{"criterion_id": f"P{number}-{i}", "criterion": f"Part {i}\npartial_credit: Test only", "score": 1} for i in (1, 2)]
                reference_origin = "ai_generated_from_question" if ai_reference and attempt == 0 and number == 1 else "teacher_supplied"
                training.append(dict(response_id=rid, input=dict(question_id=PREFIX + str(number),
                    question_text=f"Synthetic question {number}", reference_answer=f"Reference {number}\nSolution {number}",
                    rubric=criteria, max_score=2, answer_text="[unclear]" if ambiguous and attempt == 0 and number == 1 else f"Synthetic ZIP answer {attempt}/{number}"),
                    target=dict(score=1), label_source="teacher_derived" if start == 0 else "ai_generated",
                    generation_kind="source_reviewed" if start == 0 else "synthetic_full_response",
                    source_response_id="source-" + rid, split_group=f"zip-attempt-{attempt}",
                    reference_origin=reference_origin))
                provenance.append(dict(response_id=rid, source_response_id="source-" + rid,
                    source_response_ids=["source-" + rid], source_attempt_id=f"zip-attempt-{attempt}",
                    source_file=f"zip-{attempt}.csv", question_id=PREFIX + str(number), reference_origin=reference_origin))
        members[group + "/training.jsonl"] = "\n".join(json.dumps(x) for x in training).encode()
        members[group + "/provenance.json"] = json.dumps(provenance).encode()
    members["reference-manifest.json"] = json.dumps(dict(references=[dict(role="official_rubric_1_8", sha256=OFFICIAL)])).encode()
    members["manifest.json"] = json.dumps(dict(files_sha256={n: hashlib.sha256(v).hexdigest() for n, v in members.items()})).encode()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, value in members.items():
            archive.writestr("synthetic/" + name, value)

    workbook = openpyxl.Workbook()
    usable = workbook.active
    usable.title = "usable-319"
    review = workbook.create_sheet("krugrade-human-review success31")
    header = ["response_id", "attempt_id", "question_id", "question_text", "answer", "solution", "answer_text",
              "expected_score", "rubric_criteria", "rubric_max_score", "rubric_version"]
    review_header = ["response_id", "review_status", "stale", "w", "original_answer", "corrected_answer",
                     "verified_max_score", "source_pdf", "question_label"]
    usable.append(header)
    review.append(review_header)
    for attempt in range(6):
        for number in range(1, 9):
            parts = (1, 2) if number == 2 else (None,)
            for part in parts:
                if incomplete and attempt == 0 and number == 2 and part == 2:
                    continue
                rid = f"human-{attempt}-{number}-{part}"
                if flagged and attempt == 0 and number == 3:
                    rid = next(iter(FLAGS))
                ids = [part] if part is not None else [1, 2]
                criteria = [dict(criterion_id=f"P{number}-{i}", description=f"Part {i}", points=1, partial_credit="Test only") for i in ids]
                if bad_criterion and attempt == 0 and number == 1:
                    criteria[0]["points"] = 9
                text = attempt % 2 == 0 if number == 3 else f"Synthetic human answer {attempt}/{number}/{part}"
                scope = str(number) if part is None else f"{number}.{part}"
                row = dict(response_id=rid, attempt_id=f"human-attempt-{attempt}", question_id=PREFIX + scope,
                    question_text=f"Synthetic question {number}", answer=f"Reference {number}", solution=f"Solution {number}",
                    answer_text=text, expected_score=1 if part is None or part == 1 else 0,
                    rubric_criteria=json.dumps(criteria), rubric_max_score=len(ids), rubric_version="source-865c178123a2")
                usable.append([row[k] for k in header])
                r = dict(response_id=rid, review_status="corrected", stale=stale and attempt == 0 and number == 1,
                    w=row["expected_score"], original_answer=text, corrected_answer=None, verified_max_score=len(ids),
                    source_pdf=f"human-{attempt}.pdf", question_label=scope)
                review.append([r[k] for k in review_header])
    workbook.save(folder / "synthetic.xlsx")
    workbook.close()
    return {"zip": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "human": hashlib.sha256((folder / "synthetic.xlsx").read_bytes()).hexdigest()}


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.raw = Path(cls.temporary.name) / "raw"
        cls.raw.mkdir()
        cls.hashes = fixture(cls.raw)
        with patch("krugradeai.real_data.AUDITED_HASHES", cls.hashes):
            cls.imported = load_sources(cls.raw)
            cls.dataset, cls.manifest, cls.quarantine = prepare(cls.raw)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_repeat_preparation_is_identical_and_raw_unchanged(self):
        before = {p.name: p.read_bytes() for p in self.raw.iterdir()}
        with patch("krugradeai.real_data.AUDITED_HASHES", self.hashes):
            dataset, manifest, quarantine = prepare(self.raw)
        self.assertEqual(dataset, self.dataset)
        self.assertEqual(canonical(manifest), canonical(self.manifest))
        self.assertEqual(quarantine, self.quarantine)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.raw.iterdir()})

    def test_existing_pipeline_schemas_and_validation_accept_prepared_data(self):
        reconstructed = ExperimentDataset.model_validate_json(self.dataset.model_dump_json())
        self.assertEqual(reconstructed, self.dataset)
        groups = prepare_validation(self.dataset.development, self.dataset.validation, ValidationConfig())
        self.assertEqual(len(groups), 8)

    def test_label_origins_human_only_validation_and_no_label_in_grader_input(self):
        self.assertEqual({r.provenance.label_origin for r in self.dataset.validation}, {"human_derived"})
        for partition in (self.dataset.development, self.dataset.heldout):
            self.assertEqual({r.provenance.label_origin for r in partition}, {"human_derived", "ai_derived"})
        for r in self.dataset.development:
            request = GraderInput(question=r.question, rubric=r.rubric, response=r.response)
            self.assertNotIn("ta_output", request.model_dump())
            self.assertNotIn("expected_score", request.model_dump_json())
            self.assertEqual(r.provenance.label_origin, "ai_derived" if r.provenance.source == "zip320" else "human_derived")

    def test_every_partition_covers_all_eight_immutable_rubrics(self):
        definitions = self.imported["definitions"]
        for partition in (self.dataset.development, self.dataset.validation, self.dataset.heldout):
            self.assertEqual({r.question.id for r in partition}, set(definitions))
            for r in partition:
                self.assertEqual((r.question, r.rubric), definitions[r.question.id])
                with self.assertRaises(Exception):
                    r.rubric.total_score = Decimal(999)

    def test_attempts_and_source_lineage_do_not_cross_partitions(self):
        owners = {}
        for row in self.manifest["lineage"]:
            for token in ["group:" + row["connected_group_id"], *["response:" + r for r in row["lineage"]]]:
                self.assertEqual(owners.setdefault(token, row["partition"]), row["partition"])

    def test_related_answers_with_new_ids_stay_connected(self):
        rows = deepcopy(self.imported["candidates"])
        first = next(c for c in rows if c["source"] == "zip320")
        derived = deepcopy(first)
        derived.update(id="derived", group_id="newly-regenerated-group", source_files=["other.csv"], source_attempt_ids=["other-attempt"])
        rows.append(derived)
        assigned, connected = assign_groups(rows, trials=20)
        self.assertEqual(connected[first["group_id"]], connected[derived["group_id"]])
        self.assertEqual(assigned[first["id"]], assigned[derived["id"]])

    def test_split_choices_ignore_reference_score_values(self):
        rows = deepcopy(self.imported["candidates"])
        before = assign_groups(rows, trials=20)[0]
        for row in rows:
            row["score"] = "0"
        self.assertEqual(assign_groups(rows, trials=20)[0], before)

    def test_disjoint_human_parts_sum_only_the_supplied_scores(self):
        c = next(c for c in self.imported["candidates"] if c["source"] == "human319" and c["question_id"] == PREFIX + "2")
        self.assertEqual(len(c["response_ids"]), 2)
        self.assertEqual(Decimal(c["score"]), sum(Decimal(p["score"]) for p in c["constituent_scores"]))
        r = make_record(c, self.imported["definitions"][c["question_id"]], "development")
        self.assertEqual(r.ta_output.criterion_scores, ())
        self.assertIn("[Source scope", r.response.text)

    def test_boolean_answer_is_preserved_as_text(self):
        rows = [c for c in self.imported["candidates"] if c["source"] == "human319" and c["question_id"] == PREFIX + "3"]
        self.assertEqual({c["text"] for c in rows}, {"True", "False"})

    def test_quarantine_incomplete_conflicting_stale_and_changed_criteria(self):
        for options, reason in [({"incomplete": True}, "incomplete_parent_criterion_coverage"),
                                ({"flagged": True}, "unresolved_score_rubric_conflict"),
                                ({"stale": True}, "unverified_or_stale_human_label"),
                                ({"bad_criterion": True}, "rubric_subset_mismatch")]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                raw = Path(temporary)
                hashes = fixture(raw, **options)
                with patch("krugradeai.real_data.AUDITED_HASHES", hashes):
                    result = load_sources(raw)
                self.assertIn(reason, {r["reason"] for r in result["quarantine"]})

    def test_ai_references_and_ambiguous_zip_content_are_quarantined(self):
        for options, reason in [({"ai_reference": True}, "unsupported_ai_reference_scope"),
                                ({"ambiguous": True}, "ambiguous_answer_marker")]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                raw = Path(temporary)
                hashes = fixture(raw, **options)
                with patch("krugradeai.real_data.AUDITED_HASHES", hashes):
                    result = load_sources(raw)
                self.assertIn(reason, {r["reason"] for r in result["quarantine"]})
                self.assertNotIn("zip320:zip-0-1", {c["id"] for c in result["candidates"]})

    def test_duplicate_identity_is_rejected_before_partitioning(self):
        rows = self.imported["candidates"]
        with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
            assign_groups([*rows, rows[0]])

    def test_cross_source_ancestry_never_leaks_into_human_validation(self):
        rows = deepcopy(self.imported["candidates"])
        human = next(c for c in rows if c["source"] == "human319")
        ai = next(c for c in rows if c["source"] == "zip320")
        ai["lineage"].append(human["lineage"][0])
        with self.assertRaisesRegex(ValueError, "Cross-source linked"):
            assign_groups(rows)

    def test_changed_unapproved_snapshot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "snapshot differs"):
            load_sources(self.raw)

    def test_output_is_exclusive_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prepared"
            write_prepared(self.dataset, self.manifest, self.quarantine, output)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                write_prepared(self.dataset, self.manifest, self.quarantine, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
            self.assertEqual(json.loads((output / "manifest.json").read_text(encoding="utf-8")), json.loads(canonical(self.manifest)))

    def test_development_loading_never_opens_validation_or_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prepared"
            write_prepared(self.dataset, self.manifest, self.quarantine, output)
            # Removing other partitions proves this boundary needs neither.
            (output / "validation.jsonl").unlink()
            (output / "test.jsonl").unlink()
            self.assertEqual(load_prepared_partition(output, "development"), self.dataset.development)

    def test_manifest_does_not_disclose_student_answers_or_reference_labels(self):
        for row in self.manifest["lineage"]:
            self.assertNotIn("text", row)
            self.assertNotIn("score", row)
            self.assertNotIn("constituent_scores", row)
            for part in row["constituent_criteria"]:
                self.assertNotIn("score", part)

    def test_partition_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prepared"
            write_prepared(self.dataset, self.manifest, self.quarantine, output)
            path = output / "development.jsonl"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_prepared_partition(output, "development")

    def test_synthetic_answer_provenance_is_never_promoted_to_human(self):
        rows = [c for c in self.imported["candidates"] if "synthetic_full_response" in c["generation_kinds"]]
        self.assertTrue(rows)
        self.assertTrue(all(c["label_origin"] == "ai_derived" for c in rows))

    def test_reporting_stratifies_final_metrics_and_counts_failures(self):
        # Mock completed output, not an agent or evaluation run.
        from types import SimpleNamespace
        outcomes = []
        for r in self.dataset.heldout:
            if r.provenance.label_origin == "ai_derived":
                outcomes.append(RecordHistory(record_id=r.id))
            else:
                grading = GradingRecord(grader_input=GraderInput(question=r.question, rubric=r.rubric, response=r.response),
                    grader_output=GraderOutput(total_score=0, criterion_scores=tuple(
                        CriterionScore(criterion_id=c.id, score=0, rationale="Synthetic mock only") for c in r.rubric.criteria)))
                outcomes.append(RecordHistory(record_id=r.id, grading=grading))
        metrics = heldout_metrics_by_origin(SimpleNamespace(records=outcomes), self.dataset.heldout)
        counts = Counter(r.provenance.label_origin for r in self.dataset.heldout)
        self.assertEqual(metrics["mixed_source"].expected_count, len(self.dataset.heldout))
        self.assertEqual(metrics["human_only"].sample_count, counts["human_derived"])
        self.assertEqual(metrics["ai_labeled_only"].sample_count, 0)
        self.assertEqual(metrics["ai_labeled_only"].expected_count, counts["ai_derived"])

    def test_reporting_rejects_missing_outcomes(self):
        from types import SimpleNamespace
        with self.assertRaisesRegex(ValueError, "identities"):
            heldout_metrics_by_origin(SimpleNamespace(records=[]), self.dataset.heldout)


if __name__ == "__main__":
    unittest.main()
