"""Regression for the observed invalid_json_schema/lookaround failure."""
import json
import re
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from krugradeai import codex_provider as transport
from krugradeai.models import GraderOutput
import test_codex_provider as fixtures


def patterns(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "pattern":
                yield child
            yield from patterns(child)
    elif isinstance(value, list):
        for child in value:
            yield from patterns(child)


class CodexSchemaRegressionTests(unittest.TestCase):
    setUp = fixtures.CodexProviderTests.setUp
    provider = fixtures.CodexProviderTests.provider
    runner = fixtures.CodexProviderTests.runner

    def test_smoke_schema_survives_cli_regex_restriction(self):
        # Emulate the observed server rejection, not an actual inference call.
        def restricted_cli(args, **options):
            schema = json.loads(Path(args[args.index("--output-schema") + 1]).read_text(encoding="utf-8"))
            if any("(?=" in p or "(?!" in p or "(?<=" in p or "(?<!" in p for p in patterns(schema)):
                return subprocess.CompletedProcess(args, 1, stderr="invalid_json_schema: look-around assertion is not supported")
            return self.runner('{"criterion_scores":[{"criterion_id":"calculation","score":"1","rationale":"Correct addition"}],"total_score":"1","overall_feedback":"Correct"}')(args, **options)
        with patch("krugradeai.codex_provider.subprocess.run", side_effect=restricted_cli) as run:
            output = self.provider(output_model=GraderOutput).generate("Harmless synthetic grading request")
        self.assertEqual(json.loads(output)["total_score"], "1")
        self.assertEqual(run.call_count, 1)

    def test_decimal_wire_pattern_and_local_score_validation(self):
        schema = transport.generate_output_schema(GraderOutput)
        pattern = schema["properties"]["total_score"]["anyOf"][1]["pattern"]
        self.assertNotIn("(?!", pattern)
        for value in ("0", "0.5", "-1.5", "+1", "1.", ".5", "0001.20"):
            with self.subTest(valid=value):
                self.assertIsNotNone(re.fullmatch(pattern, value))
        for value in ("", "+", ".", "NaN", "Infinity", "1.2.3", "++1"):
            with self.subTest(invalid=value):
                self.assertIsNone(re.fullmatch(pattern, value))
        with patch("krugradeai.codex_provider.subprocess.run", side_effect=self.runner('{"criterion_scores":[{"criterion_id":"calculation","score":"-1","rationale":"x"}],"total_score":"-1","overall_feedback":"x"}')):
            with self.assertRaises(transport.CodexOutputError):
                self.provider(output_model=GraderOutput).generate("Synthetic request")

    def test_schema_rejection_has_safe_actionable_diagnostic(self):
        raw = "invalid_json_schema: regex lookahead is not supported. PRIVATE_STUDENT SECRET_TOKEN 401"
        error = transport._process_error(1, raw)
        self.assertEqual(error.stage, "schema")
        self.assertIn("lookaround", str(error))
        self.assertNotIn("PRIVATE_STUDENT", str(error))
        self.assertNotIn("SECRET_TOKEN", str(error))
        self.assertNotIsInstance(error, transport.CodexAuthenticationError)

    def test_invocation_and_configuration_errors_are_distinct(self):
        cases = (("error: unexpected argument '--private-flag' found", "invocation"),
                 ("Error loading config: SECRET_VALUE", "configuration"),
                 ("The selected model is not supported with ChatGPT", "configuration"))
        for diagnostic, stage in cases:
            with self.subTest(stage=stage):
                error = transport._process_error(1, diagnostic)
                self.assertEqual(error.stage, stage)
                self.assertNotIn("SECRET_VALUE", str(error))
                self.assertNotIn("--private-flag", str(error))

    def test_json_and_pydantic_output_failures_are_distinct_without_retry(self):
        for response, stage in (("not JSON PRIVATE_STUDENT", "output_json"), ('{"rubric": "PRIVATE_STUDENT"}', "output_validation")):
            with self.subTest(stage=stage), patch("krugradeai.codex_provider.subprocess.run", side_effect=self.runner(response)) as run:
                with self.assertRaises(transport.CodexOutputError) as caught:
                    self.provider(output_model=GraderOutput).generate("PRIVATE_STUDENT")
                self.assertEqual(caught.exception.stage, stage)
                self.assertNotIn("PRIVATE_STUDENT", str(caught.exception))
                self.assertEqual(run.call_count, 1)

    def test_unknown_failure_does_not_suggest_authentication_as_cause(self):
        error = transport._process_error(1, "unknown private diagnostic")
        self.assertNotIn("login", str(error).lower())
        self.assertIn("unclassified", str(error).lower())
