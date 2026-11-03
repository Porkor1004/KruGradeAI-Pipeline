"""Codex transport tests. Every inference subprocess is mocked."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel
from krugradeai.codex_provider import (
    CodexCLIProvider, CodexCLIError, CodexNotFoundError, CodexTimeoutError,
    CodexProcessError, CodexOutputError, generate_output_schema, resolve_codex_command,
)
from krugradeai.models import AdaptationRules


class CodexProviderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="krugradeai-test-")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.exe = self.root / "codex.exe"
        self.exe.touch()
        self.seen = []

    def provider(self, **kwargs):
        return CodexCLIProvider(executable=self.exe, temp_root=self.root, **kwargs)

    def runner(self, response='{"rules": []}', returncode=0):
        def run(args, **kwargs):
            self.seen.append((args, kwargs))
            output = Path(args[args.index("--output-last-message") + 1])
            output.write_text(response, encoding="utf-8")
            return subprocess.CompletedProcess(args, returncode, stdout="PROGRESS IS NOT THE RESPONSE", stderr="progress")
        return run

    def test_final_file_returned_prompt_stdin_and_safe_flags(self):
        prompt = 'Private answer ไทย & echo "unsafe" $(anything)'
        with patch("krugradeai.codex_provider.subprocess.run", side_effect=self.runner()):
            output = self.provider().generate(prompt)
        self.assertEqual(output, '{"rules": []}')
        args, options = self.seen[0]
        self.assertEqual(json.loads(options["input"].split("\nTASK_REQUEST_JSON\n", 1)[1]), {"task_request": prompt})
        self.assertNotIn(prompt, args)
        self.assertFalse(options["shell"])
        self.assertEqual(options["encoding"], "utf-8")
        self.assertEqual(args[args.index("--sandbox") + 1], "read-only")
        self.assertEqual(args[args.index("--ask-for-approval") + 1], "never")
        self.assertIn("exec", args)
        self.assertIn("--ephemeral", args)
        self.assertIn("--ignore-user-config", args)
        self.assertIn("--no-daemon", args)
        self.assertIn("--skip-git-repo-check", args)
        self.assertEqual(args[-1], "-")
        for banned in ("--approve-for-me", "--dangerously-bypass-approvals-and-sandbox", "--add-dir", "--full-auto"):
            self.assertNotIn(banned, args)
        self.assertEqual(Path(options["cwd"]), Path(args[args.index("--cd") + 1]))
        self.assertFalse(Path(options["cwd"]).exists())

    def test_schema_file_generated_and_removed(self):
        def run(args, **kwargs):
            schema_path = Path(args[args.index("--output-schema") + 1])
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            self.assertEqual(schema["type"], "object")
            self.assertEqual(schema["required"], ["rules"])
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(list(Path(kwargs["cwd"]).iterdir()), [])
            self.assertNotIn("INPUT", schema_path.read_text(encoding="utf-8"))
            return self.runner()(args, **kwargs)
        with patch("krugradeai.codex_provider.subprocess.run", side_effect=run):
            self.assertEqual(self.provider(output_model=AdaptationRules).generate("INPUT"), '{"rules": []}')
        args, _ = self.seen[0]
        self.assertFalse(Path(args[args.index("--output-schema") + 1]).exists())

    def test_no_schema_mode_supports_grader_review_protocol(self):
        from krugradeai.grader import LLMGrader, HumanReviewRequired
        from test_grader import request
        raw = '{"human_review_required": true, "review_reason": "Unreadable work"}'
        with patch("krugradeai.codex_provider.subprocess.run", side_effect=self.runner(raw)):
            with self.assertRaises(HumanReviewRequired):
                LLMGrader(self.provider()).grade(request("[unclear]"))
        self.assertNotIn("--output-schema", self.seen[0][0])

    def test_existing_agents_can_use_provider_without_interface_changes(self):
        from krugradeai.grader import LLMGrader
        from test_grader import request, result
        with patch("krugradeai.codex_provider.subprocess.run", side_effect=self.runner(result("1.5"))):
            output = LLMGrader(self.provider()).grade(request())
        self.assertEqual(str(output.total_score), "1.5")

    def test_process_failure_no_response_fallback_or_progress_leak(self):
        with patch("krugradeai.codex_provider.subprocess.run", side_effect=self.runner(returncode=1)):
            with self.assertRaises(CodexProcessError) as caught:
                self.provider().generate("private answer")
        self.assertEqual(caught.exception.returncode, 1)
        self.assertNotIn("private answer", str(caught.exception))
        self.assertNotIn("PROGRESS", str(caught.exception))
        self.assertEqual(len(self.seen), 1)
        self.assertFalse(Path(self.seen[0][1]["cwd"]).exists())

    def test_timeout_translated_and_cleanup(self):
        def timeout(args, **kwargs):
            self.seen.append((args, kwargs))
            raise subprocess.TimeoutExpired(args, kwargs["timeout"], output="PRIVATE", stderr="PRIVATE")
        with patch("krugradeai.codex_provider.subprocess.run", side_effect=timeout):
            with self.assertRaises(CodexTimeoutError) as caught:
                self.provider(timeout_seconds=0.5).generate("private")
        self.assertNotIn("PRIVATE", str(caught.exception))
        self.assertFalse(Path(self.seen[0][1]["cwd"]).exists())

    def test_missing_empty_and_malformed_response(self):
        cases = (None, "", "   ", "not JSON", "[]", '{} trailing', '{"rules": [], "rules": []}', '{"x": NaN}')
        for raw in cases:
            with self.subTest(raw=raw):
                def run(args, **kwargs):
                    self.seen.append((args, kwargs))
                    if raw is not None:
                        Path(args[args.index("--output-last-message") + 1]).write_text(raw, encoding="utf-8")
                    return subprocess.CompletedProcess(args, 0)
                with patch("krugradeai.codex_provider.subprocess.run", side_effect=run), self.assertRaises(CodexOutputError):
                    self.provider().generate("private")
                self.assertFalse(Path(self.seen[-1][1]["cwd"]).exists())

    def test_invalid_model_output_rejected(self):
        with patch("krugradeai.codex_provider.subprocess.run", side_effect=self.runner('{"rubric": {}}')):
            with self.assertRaises(CodexOutputError):
                self.provider(output_model=AdaptationRules).generate("private")

    def test_executable_disappears_and_launch_os_error(self):
        for error, expected in ((FileNotFoundError("PRIVATE"), CodexNotFoundError), (PermissionError("PRIVATE"), CodexCLIError)):
            with self.subTest(error=error), patch("krugradeai.codex_provider.subprocess.run", side_effect=error), self.assertRaises(expected) as caught:
                self.provider().generate("private")
            self.assertNotIn("PRIVATE", str(caught.exception))

    def test_invalid_configuration_and_empty_prompt_do_not_execute(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.provider(timeout_seconds=timeout)
        with patch("krugradeai.codex_provider.subprocess.run") as run:
            with self.assertRaises(ValueError):
                self.provider().generate(" \n")
            run.assert_not_called()


class CodexResolutionTests(unittest.TestCase):
    def test_native_path_and_missing_executable(self):
        with tempfile.TemporaryDirectory() as root:
            exe = Path(root) / "codex.exe"
            exe.touch()
            self.assertEqual(resolve_codex_command(exe), [str(exe.resolve())])
        with patch("krugradeai.codex_provider.shutil.which", return_value=None):
            with self.assertRaises(CodexNotFoundError):
                resolve_codex_command()

    def test_npm_shim_keeps_path_installation_instead_of_other_native_cli(self):
        with tempfile.TemporaryDirectory() as root:
            shim = Path(root) / "codex.cmd"
            shim.touch()
            native = Path(root) / "node_modules/@openai/codex-win32-x64/vendor/x86_64-pc-windows-msvc/bin/codex.exe"
            native.parent.mkdir(parents=True)
            native.touch()
            with patch("krugradeai.codex_provider.shutil.which", return_value=str(shim)), patch("krugradeai.codex_provider.platform.machine", return_value="AMD64"):
                self.assertEqual(resolve_codex_command(), [str(native.resolve())])

    def test_unknown_shim_is_rejected_without_invoking_shell(self):
        with tempfile.TemporaryDirectory() as root:
            for extension in ("cmd", "bat", "ps1"):
                shim = Path(root) / ("codex." + extension)
                shim.touch()
                with self.subTest(extension=extension), self.assertRaises(CodexNotFoundError):
                    resolve_codex_command(shim)


class CodexSchemaTests(unittest.TestCase):
    def test_nested_objects_required_and_no_defaults(self):
        schema = generate_output_schema(AdaptationRules)
        rule = schema["$defs"]["AdaptationRule"]
        self.assertFalse(rule["additionalProperties"])
        self.assertEqual(set(rule["required"]), {"id", "instruction", "criterion_ids", "rationale"})
        self.assertNotIn("default", rule["properties"]["criterion_ids"])
        self.assertEqual(schema["required"], ["rules"])

    def test_property_named_default_is_preserved(self):
        class Example(BaseModel):
            default: str = "x"
        schema = generate_output_schema(Example)
        self.assertIn("default", schema["properties"])

    def test_unsupported_dynamic_object_schema_rejected(self):
        class Example(BaseModel):
            values: dict[str, str]
        with self.assertRaises(ValueError):
            generate_output_schema(Example)

class CodexAgentCompatibilityTests(unittest.TestCase):
    setUp = CodexProviderTests.setUp
    provider = CodexProviderTests.provider
    runner = CodexProviderTests.runner
    def test_reflector_with_schema(self):
        from krugradeai.reflector import LLMReflector
        from krugradeai.models import ReflectorOutput
        from test_reflector import reflection_input, reflection_json
        with patch("krugradeai.codex_provider.subprocess.run", side_effect=self.runner(reflection_json())):
            output = LLMReflector(self.provider(output_model=ReflectorOutput)).reflect(reflection_input())
        self.assertEqual(output.overall_score_difference, 1)
        self.assertIn("--output-schema", self.seen[0][0])

    def test_refiner_with_schema(self):
        from krugradeai.refiner import LLMRefiner
        from test_refiner import input_request, rule
        raw = json.dumps({"rules": [rule()]})
        with patch("krugradeai.codex_provider.subprocess.run", side_effect=self.runner(raw)):
            output = LLMRefiner(self.provider(output_model=AdaptationRules)).refine(input_request())
        self.assertEqual(output.rules[0].id, "equivalence")
        self.assertIn("--output-schema", self.seen[0][0])
