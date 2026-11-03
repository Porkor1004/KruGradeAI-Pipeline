"""Sanitized possible failure patterns; rejected live payloads were not retained."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from krugradeai.codex_provider import CodexCLIProvider, CodexValidationError
from krugradeai.models import ReflectorOutput
from krugradeai.reflector_prompt import build_reflector_prompt
from test_reflector import finding, reflection_input, reflection_json, run_reflection


class ReflectorDiagnosticTests(unittest.TestCase):
    def provider_error(self, raw):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "codex.exe"
            executable.touch()
            provider = CodexCLIProvider(executable=executable, output_model=ReflectorOutput)

            def run(args, **kwargs):
                self.assertEqual(args[args.index("--sandbox") + 1], "read-only")
                Path(args[args.index("--output-last-message") + 1]).write_text(raw, encoding="utf-8")
                return subprocess.CompletedProcess(args, 0, stderr="PRIVATE DIAGNOSTIC")

            with patch("krugradeai.codex_provider.subprocess.run", side_effect=run) as invoke:
                with self.assertRaises(CodexValidationError) as raised:
                    provider.generate("SANITIZED SYNTHETIC REQUEST")
            self.assertEqual(invoke.call_count, 1)
            return raised.exception

    def test_provider_reports_three_distinct_cross_field_patterns(self):
        patterns = [
            (reflection_json(direction="undergrading"), "direction_sign_mismatch"),
            (reflection_json(review_reasons=[]), "review_flag_reasons_mismatch"),
            (reflection_json(findings=[finding(category="insufficient_evidence", status="supported")]),
             "insufficient_evidence_requires_hypothesis"),
        ]
        for raw, code in patterns:
            with self.subTest(code=code):
                error = self.provider_error(raw)
                self.assertEqual(error.safe_diagnostics[0]["code"], code)
                self.assertEqual(error.safe_diagnostics[0]["stage"], "provider_output_contract")

    def test_unknown_extra_keys_and_private_values_are_redacted(self):
        data = json.loads(reflection_json())
        data["PRIVATE_STUDENT_ANSWER"] = "SECRET_TOKEN PRIVATE QUOTE"
        data["findings"][0]["status"] = "PRIVATE_STATUS"
        error = self.provider_error(json.dumps(data))
        safe = str(error) + json.dumps(error.safe_diagnostics)
        self.assertNotIn("PRIVATE", safe)
        self.assertNotIn("SECRET_TOKEN", safe)
        self.assertIn("unknown_field", safe)
        self.assertIn("literal_error", safe)
        self.assertIn("extra_forbidden", safe)

    def test_uncertain_findings_without_review_are_still_rejected(self):
        error = self.provider_error(reflection_json(human_review_required=False, review_reasons=[]))
        self.assertEqual(error.safe_diagnostics[0]["code"], "uncertainty_requires_review")

    def test_missing_supported_error_suggestion_is_still_rejected(self):
        error = self.provider_error(reflection_json(findings=[finding(status="supported", adaptation_suggestion=None)]))
        self.assertEqual(error.safe_diagnostics[0]["code"], "supported_error_requires_suggestion")

    def test_valid_total_only_review_passes_without_fabricated_criterion_scores(self):
        output = run_reflection()
        self.assertTrue(output.human_review_required)
        self.assertEqual(output.criterion_differences, ())
        self.assertEqual(output.findings[0].status, "hypothesis")

    def test_context_checks_have_distinct_safe_constraint_codes(self):
        patterns = [
            (reflection_json("2"), "total_difference_mismatch"),
            (reflection_json(criterion_differences=[dict(criterion_id="c", difference="1")]), "criterion_difference_label_mismatch"),
            (reflection_json(findings=[finding(criterion_id="unknown")]), "unknown_finding_criterion"),
            (reflection_json(findings=[finding(evidence=[dict(source="student_answer", quote="PRIVATE INVENTED QUOTE")])]), "evidence_quote_not_found"),
            (reflection_json(findings=[finding(status="supported")]), "supported_error_requires_criterion_label"),
        ]
        for raw, code in patterns:
            with self.subTest(code=code), self.assertRaises(ValidationError) as raised:
                run_reflection(raw)
            details = raised.exception.safe_diagnostics
            self.assertEqual(details[0]["stage"], "reflector_context")
            self.assertEqual(details[0]["code"], code)
            self.assertNotIn("PRIVATE", json.dumps(details))

    def test_local_contract_validation_retains_legacy_exception_type(self):
        with self.assertRaises(ValidationError) as raised:
            run_reflection(reflection_json(direction="undergrading"))
        self.assertEqual(raised.exception.safe_diagnostics[0]["stage"], "reflector_output_contract")

    def test_prompt_computed_context_matches_request_and_does_not_change_it(self):
        request = reflection_input()
        before = request.model_dump_json()
        prompt = build_reflector_prompt(request)
        context = json.loads(prompt.split("\nCOMPUTED_CONTRACT_CONTEXT\n")[1].split("\nINPUT_JSON\n")[0])
        self.assertTrue(context["total_only"])
        self.assertEqual(context["overall_score_difference"], "1")
        self.assertEqual(context["direction"], "overgrading")
        self.assertEqual(context["criterion_differences"], [])
        self.assertIn('insufficient_evidence ALWAYS uses status="hypothesis"', prompt)
        self.assertEqual(request.model_dump_json(), before)

    def test_pilot_persists_safe_provider_and_agent_diagnostics(self):
        from test_pilot_real import RealPilotTests
        case = RealPilotTests()
        case.setUp()
        try:
            from krugradeai.pilot_real import run_pilot
            from test_pilot_real import mocked_agents
            agents = mocked_agents("discrepancy")
            failure = self.provider_error(reflection_json(direction="undergrading"))
            with patch.object(agents[1].provider, "generate", side_effect=failure), patch("krugradeai.pilot_real.build_agents", return_value=agents):
                report = run_pilot(live=True, data_directory=case.data, output_directory_path=case.root / "pilot")
            failed = next(i for i in report.invocations if i.agent == "reflector")
            outcome = next(o for o in report.agent_outcomes if o.agent == "reflector")
            self.assertEqual(failed.validation_diagnostics[0]["code"], "direction_sign_mismatch")
            self.assertEqual(outcome.validation_diagnostics, failed.validation_diagnostics)
            self.assertNotIn("PRIVATE", report.model_dump_json())
        finally:
            case.doCleanups()


if __name__ == "__main__":
    unittest.main()
