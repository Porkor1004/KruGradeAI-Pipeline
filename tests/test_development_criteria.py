"""Synthetic source extraction and operational pseudo-reference regressions."""
from copy import deepcopy
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from pydantic import ValidationError
from krugradeai.development_criteria import (enrich_development, load_enriched_development,
    main, write_enrichment, _validate_preservation, _unique_object)
from krugradeai.models import (CriterionLabelProvenance, TACriterionScore, TAReference,
                              AdaptationRules, RefinerInput)
from krugradeai.optimization_models import DevelopmentRecord, RecordProvenance
from krugradeai.real_data import prepare, write_prepared
from krugradeai.refiner import validate_candidate_rules
from krugradeai.reflector_prompt import build_reflector_prompt
from test_real_data import fixture
from test_reflector import reflection_input, reflection_json, finding, run_reflection
from test_total_only_refinement import candidate


def synthetic_sources(raw):
    fixture(raw)
    archive_path = raw / "synthetic.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    for name in list(members):
        if name.endswith("02_ai_augmented/training.jsonl"):
            rows = [json.loads(line) for line in members[name].splitlines()]
            for row in rows:
                criteria = row["input"]["rubric"]
                row["target"]["point_scores"] = {criteria[0]["criterion_id"]: 1, criteria[1]["criterion_id"]: 0}
                # Existing Question Text normalization must not be confused with a
                # forbidden score repair or change in student content.
                row["input"]["reference_answer"] += "\n"
            members[name] = "\n".join(json.dumps(row) for row in rows).encode()
    manifest_name = "synthetic/manifest.json"
    metadata = json.loads(members[manifest_name])
    for name, value in members.items():
        if name != manifest_name:
            metadata["files_sha256"][name.removeprefix("synthetic/")] = hashlib.sha256(value).hexdigest()
    members[manifest_name] = json.dumps(metadata).encode()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return {"zip": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "human": hashlib.sha256((raw / "synthetic.xlsx").read_bytes()).hexdigest()}


class DevelopmentCriterionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.raw, cls.base = cls.root / "raw", cls.root / "prepared"
        cls.raw.mkdir()
        hashes = synthetic_sources(cls.raw)
        with patch("krugradeai.real_data.AUDITED_HASHES", hashes):
            dataset, manifest, quarantine = prepare(cls.raw)
        write_prepared(dataset, manifest, quarantine, cls.base)
        cls.original = dataset.development
        # Derive a synthetic audit allowlist, never from held-out content.
        audit = dict(split_manifest_sha256=hashlib.sha256((cls.base / "manifest.json").read_bytes()).hexdigest(),
            development_sha256=hashlib.sha256((cls.base / "development.jsonl").read_bytes()).hexdigest(),
            source_hashes=manifest["input_hashes"], ai_only_criterion_evidence=[], recoverable_human_criterion_evidence=[])
        for item in manifest["lineage"]:
            if item["partition"] != "development":
                continue
            r = next(r for r in cls.original if r.id == item["id"])
            if item["label_sources"] == ["ai_generated"]:
                audit["ai_only_criterion_evidence"].append(dict(record_id=r.id, source_locator=item["locators"][0],
                    criterion_ids=[c.id for c in r.rubric.criteria]))
            elif item["source"] == "human319" and r.question.id.endswith("p2"):
                entries = []
                for part, loc, reviewloc in zip(item["constituent_criteria"], item["locators"], item["review_locators"]):
                    entries.append(dict(criterion_id=part["criterion_ids"][0], source_response_id=part["response_id"],
                        source_question_id=r.question.id + "." + part["criterion_ids"][0].rsplit("-", 1)[1],
                        score_cell=loc.replace(":", ":H"), review_score_cell=reviewloc.replace(":", ":D")))
                audit["recoverable_human_criterion_evidence"].append(dict(record_id=r.id, evidence=entries))
        cls.audit_path = cls.root / "evidence.json"
        cls.audit_path.write_text(json.dumps(audit), encoding="utf-8")
        cls.enriched, cls.metadata = enrich_development(cls.base, cls.raw, evidence_audit=cls.audit_path, enable_source_criteria=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def extract(self, audit):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps(audit), encoding="utf-8")
            return enrich_development(self.base, self.raw, evidence_audit=path, enable_source_criteria=True)

    def test_no_opt_in_reads_no_files(self):
        with patch.object(Path, "read_bytes") as read:
            with self.assertRaises(ValueError):
                enrich_development()
            with self.assertRaises(ValueError):
                load_enriched_development("missing", "missing")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([])
        read.assert_not_called()

    def test_cli_does_not_print_private_source_validation_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            messages = io.StringIO()
            with patch("krugradeai.development_criteria.enrich_development", side_effect=ValueError("PRIVATE SOURCE ANSWER")), redirect_stderr(messages):
                status = main(["--enable-source-criteria", "--output-dir", str(Path(temporary) / "fresh")])
            self.assertEqual(status, 1)
            self.assertNotIn("PRIVATE", messages.getvalue())
            self.assertIn("ValueError", messages.getvalue())

    def test_only_criterion_labels_are_added_with_complete_provenance(self):
        _validate_preservation(self.original, self.enriched)
        origins = set()
        for record in self.enriched:
            for score in record.ta_output.criterion_scores:
                p = score.provenance
                self.assertEqual(p.original_criterion_id, score.criterion_id)
                self.assertTrue(p.source_response_id and p.source_question_id and p.source_locator)
                self.assertEqual(len(p.source_sha256), 64)
                self.assertTrue(p.extraction_evidence)
                origins.add(p.label_origin)
        self.assertEqual(origins, {"human_derived", "ai_derived"})
        self.assertTrue(any(not r.ta_output.criterion_scores for r in self.enriched))

    def test_extraction_repeats_without_changes(self):
        before = {p.name: p.read_bytes() for p in self.raw.iterdir()}
        rows, metadata = enrich_development(self.base, self.raw, evidence_audit=self.audit_path, enable_source_criteria=True)
        self.assertEqual(rows, self.enriched)
        self.assertEqual(metadata, self.metadata)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.raw.iterdir()})

    def test_loader_requires_same_split_and_preserves_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "criteria"
            write_enrichment(self.enriched, self.metadata, output, data_directory=self.base, raw_directory=self.raw)
            self.assertEqual(load_enriched_development(self.base, output, enable_source_criteria=True), self.enriched)
            with self.assertRaises(FileExistsError):
                write_enrichment(self.enriched, self.metadata, output, data_directory=self.base, raw_directory=self.raw)
            # The loader needs neither partition's contents.
            with patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected")):
                with self.assertRaises(ValueError):
                    load_enriched_development(self.base, output)

    def test_non_development_audit_identity_is_rejected(self):
        audit = json.loads(self.audit_path.read_text())
        audit["ai_only_criterion_evidence"][0]["record_id"] = "not-in-development"
        with self.assertRaisesRegex(ValueError, "Development record"):
            self.extract(audit)

    def test_foreign_locators_cannot_read_validation_or_external_files(self):
        audit = json.loads(self.audit_path.read_text())
        audit["ai_only_criterion_evidence"][0]["source_locator"] = "../private.zip!other:1"
        with self.assertRaisesRegex(ValueError, "locator"):
            self.extract(audit)

    def test_unknown_source_criterion_and_duplicate_evidence_rejected(self):
        audit = json.loads(self.audit_path.read_text())
        audit["ai_only_criterion_evidence"][0]["criterion_ids"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "criterion-map"):
            self.extract(audit)
        audit = json.loads(self.audit_path.read_text())
        audit["recoverable_human_criterion_evidence"][0]["evidence"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.extract(audit)

    def test_missing_keys_or_duplicate_json_keys_are_not_silently_repaired(self):
        with self.assertRaises(ValueError):
            json.loads('{"criterion": 1, "criterion": 0}', object_pairs_hook=_unique_object)

    def test_pseudo_ground_truth_cannot_claim_human_reference_role(self):
        ai = next(s for r in self.enriched for s in r.ta_output.criterion_scores if s.provenance.label_origin == "ai_derived")
        data = ai.provenance.model_dump()
        data["operational_role"] = "human_reference"
        with self.assertRaises(ValidationError):
            CriterionLabelProvenance.model_validate(data)

    def test_bounds_and_total_consistency_stay_strict(self):
        record = next(r for r in self.enriched if r.ta_output.criterion_scores)
        for value in ("-1", "3", "0.25"):
            data = record.model_dump()
            data["ta_output"]["criterion_scores"][0]["score"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                DevelopmentRecord.model_validate(data)

    def test_changed_question_or_total_is_not_an_enrichment(self):
        data = self.enriched[0].model_dump()
        data["question"]["text"] = "Different question"
        changed = DevelopmentRecord.model_validate(data)
        with self.assertRaisesRegex(ValueError, "only add"):
            _validate_preservation(self.original, (changed, *self.enriched[1:]))

    def test_ai_operational_label_is_eligible_for_supported_refinement(self):
        request = reflection_input(criteria=({"criterion_id": "c", "score": "1"},))
        p = next(s.provenance for r in self.enriched for s in r.ta_output.criterion_scores if s.provenance.label_origin == "ai_derived")
        data = p.model_dump()
        data["original_criterion_id"] = "c"
        request = type(request).model_validate(dict(record=request.record, ta_output=TAReference(total_score=1,
            criterion_scores=(TACriterionScore(criterion_id="c", score=1, provenance=CriterionLabelProvenance.model_validate(data)),))))
        output = run_reflection(reflection_json(findings=[finding(status="supported")],
            criterion_differences=[dict(criterion_id="c", difference="1")], human_review_required=False, review_reasons=[]), request)
        refinement = RefinerInput(current_rules=AdaptationRules(), reflection=output, rubric=request.record.grader_input.rubric)
        self.assertEqual(validate_candidate_rules(refinement, candidate()), candidate())
        prompt = build_reflector_prompt(request)
        self.assertIn('"operational_role": "pseudo_ground_truth"', prompt)
        self.assertIn("AI origin alone does not mandate human review", prompt)
        self.assertFalse(output.human_review_required)

    def test_opted_in_ai_labels_do_not_override_human_review(self):
        request = reflection_input(criteria=({"criterion_id": "c", "score": "1"},))
        output = run_reflection(reflection_json(criterion_differences=[dict(criterion_id="c", difference="1")]), request)
        refinement = RefinerInput(current_rules=AdaptationRules(), reflection=output, rubric=request.record.grader_input.rubric)
        with self.assertRaises(ValidationError):
            validate_candidate_rules(refinement, candidate())

    def test_pilot_explicit_overlay_opt_in_reports_label_counts_with_mocked_agents(self):
        from krugradeai.pilot_real import run_pilot
        from krugradeai.mock_provider import MockLLMProvider
        from krugradeai.grader import LLMGrader
        from krugradeai.reflector import LLMReflector
        from krugradeai.refiner import LLMRefiner

        class AgreeingSyntheticProvider(MockLLMProvider):
            def generate(self, prompt):
                request = json.loads(prompt.split("\nINPUT_JSON\n")[1])
                return json.dumps(dict(total_score=1, overall_feedback="Synthetic feedback", criterion_scores=[
                    dict(criterion_id=c["id"], score=1 if i == 0 else 0, rationale="Synthetic rationale")
                    for i, c in enumerate(request["rubric"]["criteria"])]))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            overlay = root / "criteria"
            write_enrichment(self.enriched, self.metadata, overlay, data_directory=self.base, raw_directory=self.raw)
            agents = (LLMGrader(AgreeingSyntheticProvider("")), LLMReflector(MockLLMProvider("")), LLMRefiner(MockLLMProvider("")))
            with patch("krugradeai.pilot_real.build_agents", return_value=agents):
                report = run_pilot(live=True, data_directory=self.base, source_criteria_directory=overlay,
                    output_directory_path=root / "pilot", sample_size=4)
            self.assertEqual(report.status, "completed")
            self.assertTrue(report.source_criteria_enabled)
            self.assertEqual(report.source_criteria_sha256, self.metadata["enriched_development_sha256"])
            self.assertEqual(report.criterion_label_summary["records_with_criteria"], self.metadata["summary"]["enriched_records"])
            self.assertEqual(len(report.invocations), 4)
            self.assertFalse(report.candidates_promoted)

    def test_source_enrichment_loads_without_opening_validation_test_or_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "criteria"
            write_enrichment(self.enriched, self.metadata, output, data_directory=self.base, raw_directory=self.raw)
            original = Path.open
            opened = []

            def observe(path, *args, **kwargs):
                opened.append(path)
                if path.name in ("validation.jsonl", "test.jsonl") or self.raw in path.parents:
                    raise AssertionError("Forbidden source partition access")
                return original(path, *args, **kwargs)

            with patch.object(Path, "open", observe):
                self.assertEqual(load_enriched_development(self.base, output, enable_source_criteria=True), self.enriched)
            self.assertTrue(opened)


if __name__ == "__main__":
    unittest.main()
