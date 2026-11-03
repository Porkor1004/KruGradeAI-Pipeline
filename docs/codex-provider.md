# Codex CLI provider (Step 5)

`CodexCLIProvider` implements the unchanged `LLMProvider.generate(str) -> str`
interface. It is an opt-in transport for the existing Grader, Reflector and
Refiner. No agent logic, prompts, contracts or optimization loop were changed.
The existing deterministic mock remains available for offline use.

## Verified local runtime

The installed PATH npm CLI was inspected with `--version`, `--help` and
`exec --help`: **codex-cli 0.166.0**. The user has verified ChatGPT login. No API key
is required by this adapter, and it does not log in, install packages, update
Codex or perform automatic version substitution.

This machine also has a separate older desktop-app CLI binary. Resolution keeps
PATH precedence instead of preferring any file named codex.exe. For a Windows
npm .cmd/.ps1/.bat shim, it locates the native executable in that same installation's
platform package (including package-local and older vendor layouts). It never
invokes the shim through cmd.exe or PowerShell. The resolved native npm binary
was separately checked and reports 0.166.0. Unknown wrapper layouts fail with
CodexNotFoundError; an explicit native `executable=` path can be supplied.

## Invocation boundary

Each call creates a new TemporaryDirectory containing a generated schema when
configured and a response-file path. By default, subprocess `cwd` and `--cd` point
to its empty `work/` directory. An explicit `working_directory` uses that existing
directory without creating, deleting or modifying its contents. Prefer a dedicated
empty directory to avoid unnecessary project context. Project files and data/raw/
are not read by this adapter.
The command is an argument list with shell=False, using this form:

```text
<resolved-codex> --ask-for-approval never --no-daemon exec
  --sandbox read-only --ignore-user-config --ephemeral --skip-git-repo-check
  --cd <temporary-work> --output-last-message <temporary-response>
  --color never [--model <model>] [--output-schema <generated-schema>] -
```

The complete prompt is sent only through stdin using UTF-8. A provider-level
runtime policy wraps a JSON-encoded task_request and tells the model to ignore
runtime-control instructions in student answers, reference solutions, rubrics,
rules and evidence. It explicitly forbids tools, filesystem inspection, commands,
credential access and file modification. Existing agent prompts are unchanged.
This instruction is defense in depth, not a proof against prompt injection.
The prompt is never a command-line argument or a prompt file.
`--ask-for-approval never` denies an
escalation path rather than approving requests; read-only sandboxing remains in
force. No unrestricted-execution, automatic-review/approval, extra writable
directory or web-search flags are enabled. `--ignore-user-config` avoids inheriting
personal profiles/tools/hooks from that config, while Codex still uses its existing
authentication. `--no-daemon` avoids creating a shared background-server session.

CLI stdout progress is discarded. Stderr is captured only in process memory for
sanitized error classification, never logged or used as the response. Only
--output-last-message is read for the final response; a
successful exit without a valid response file fails. No progress-text fallback,
JSON repair, retries or default grade is provided. Error messages exclude prompts,
raw responses and CLI diagnostics, which may contain student content.

Temporary files are closed before invoking Codex (no Windows NamedTemporaryFile
sharing lock). Context-manager cleanup runs after success, subprocess failure,
timeout and output validation failures. subprocess.run kills/waits for the direct
native process on timeout; no Node/shell launcher is left wrapping it. Windows
processes use CREATE_NO_WINDOW. OS cleanup failures are surfaced rather than
silently ignored. There is no cross-process filesystem/security sandbox beyond
Codex's sandbox policy; read-only is not a guarantee against reads elsewhere, and
third-party detached descendants or abrupt interpreter termination are outside
TemporaryDirectory's cleanup guarantee.

## JSON Schema and validation

Supply `output_model=<Pydantic model class>` to generate the per-call schema.
The adapter requires a single object root, makes every property required on the
wire, forbids additional properties, preserves nullable types, removes defaults
and converts const to enum. The known Pydantic Decimal regex is translated to an
equivalent decimal-syntax pattern without regex lookaround, which Codex's schema
compiler rejects. Only that specific pattern is translated; patterns are not
generally removed, and Pydantic numeric bounds/finite-value validation remains
enabled. Dynamic mapping schemas and root union/composition
schemas are rejected rather than silently altered or retried without a schema.
Pydantic's input models themselves are not modified. Nested rule objects, Decimal
score representations and references remain in the generated schema.

ReflectorOutput and AdaptationRules are suitable schema sources. Agent-specific
contextual validation remains mandatory: a schema does not establish rubric
compliance, quote accuracy, supported claims or safe adaptation instructions.
Provider-side validation also checks output against the configured Pydantic model.

Use explicit no-schema mode (`output_model=None`, the default) for the Grader's
existing scored-or-human-review protocol. Forcing the scored-only model schema
would prevent the valid alternative review object; the provider does not rewrite
that protocol or wrap/unpack the response. It still requires a valid JSON object,
and LLMGrader applies the original strict scored/review and rubric validation.
The initial live smoke failure was reproduced once during debugging and traced
to the Decimal lookaround pattern. Post-fix server acceptance has not been tested
with another live call. Other backend schema restrictions can still produce a
schema failure and will not trigger a fallback.

## Usage

```python
from krugradeai.codex_provider import CodexCLIProvider
from krugradeai.grader import LLMGrader
from krugradeai.reflector import LLMReflector
from krugradeai.refiner import LLMRefiner
from krugradeai.models import ReflectorOutput, AdaptationRules

# Constructors resolve the installation; they do not start an LLM call.
grader = LLMGrader(CodexCLIProvider(output_model=None))
reflector = LLMReflector(CodexCLIProvider(output_model=ReflectorOutput))
refiner = LLMRefiner(CodexCLIProvider(output_model=AdaptationRules))

# Calling grade/reflect/refine later explicitly invokes the installed Codex CLI.
```

Options are `executable`, `output_model`, `model` (optional CLI model identifier),
a finite positive `timeout_seconds` (default 120), `working_directory` (optional
existing directory), and an existing `temp_root` for temporary control files
(default OS temporary root). These keyword arguments are the provider configuration;
no configuration file or environment secrets are required.
There is intentionally no arbitrary extra-arguments option. Each request is an
ephemeral session; provider instances retain no conversational history. Login and
account usage limits belong to the installed CLI, not this Python provider.

Errors derive from CodexCLIError: CodexNotFoundError, CodexTimeoutError,
CodexProcessError (with returncode), and CodexOutputError. Recognized missing/expired
authentication failures use CodexAuthenticationError; rate/usage-limit failures
use CodexUsageLimitError. Both are CodexProcessError subclasses. Classification
uses diagnostic patterns and cannot recognize every future CLI message; unknown
failures remain generic CodexProcessError. Filesystem/startup
failures use CodexCLIError. Malformed JSON, duplicate keys, non-finite constants,
non-object responses, missing/empty/unreadable files and configured-model
validation failures are rejected. Raw response text is returned unchanged after
validation; consuming agents still perform their own checks.

Failures also expose a safe `stage` attribute. CLI argument rejection uses
CodexInvocationError (`invocation`), model/configuration/sandbox-setup errors use
CodexConfigurationError (`configuration`), and invalid_json_schema errors use
CodexSchemaError (`schema`). Output file failures use `output_read`; malformed
JSON uses CodexJSONError (`output_json`), and Pydantic failures use
CodexValidationError (`output_validation`). The latter two remain CodexOutputError
subclasses. Messages contain fixed actionable hints and may identify unsupported
regex lookaround, never raw stderr, prompts, responses or credential values.
Unclassified failures no longer suggest authentication as their presumed cause.
The smoke command includes the safe stage in its error message.

## Setup, authentication and troubleshooting

Install the Python project and verify the already installed CLI in a normal terminal:

```console
python -m pip install -e .
codex --version
codex login
codex login status
```

Use ChatGPT login when prompted. The adapter does not launch login automatically,
read credential files, or accept API-key parameters. OPENAI_API_KEY, CODEX_API_KEY
and AZURE_OPENAI_API_KEY are excluded from the child environment; existing CLI
authentication and CODEX_HOME remain available without printing their contents.

| Failure | Action |
| --- | --- |
| CodexNotFoundError | Check PATH or supply the correct native executable/npm shim. |
| CodexAuthenticationError | Run `codex login` with ChatGPT, then `codex login status`. |
| CodexUsageLimitError | Check account limits and wait for availability; requests are not retried. |
| CodexTimeoutError | Check connectivity or deliberately increase timeout_seconds. |
| CodexOutputError | Check the selected output model and agent protocol; invalid responses are not repaired. |
| CodexSchemaError | Check schema compatibility, particularly unsupported regex syntax; retain Pydantic validation. |
| CodexInvocationError | Compare the selected binary's `exec --help` with the provider flags. |
| CodexConfigurationError | Check model availability/configuration and read-only sandbox setup. |
| Other CodexProcessError | Verify CLI version/options and local setup; raw diagnostics are intentionally omitted. |
| Temporary workspace error | Choose a writable temp_root with normal Windows permissions. |

## Optional live smoke test

This command is separate from unit tests and sends exactly one synthetic addition
grading request through the real provider. It may use account quota:

```console
python -m krugradeai.smoke_codex --live
```

Optional arguments are `--executable`, `--model`, `--timeout` and
`--working-directory`. Without --live the command exits 2 without constructing a
provider or making a call. Failure exits 1; a validated grade exits 0 and prints
only the synthetic grading result. This unambiguous demo uses the scored
GraderOutput schema; normal scored-or-review grading can retain no-schema mode.
No real dataset or TA labels are included. Initial implementation tested the live
branch only with mocks. One synthetic live reproduction was subsequently used
to diagnose the schema rejection; no live call was repeated after the fix.

## Tests

```console
python -m unittest discover -s tests -v
```

Provider tests only:

```console
python -m unittest discover -s tests -p test_codex_provider.py -v
```

Every Codex inference subprocess in automated tests is mocked. Tests create empty
executable/shim fixtures and temporary response files, check flags/stdin/working
directories/schema files, simulate failures/timeouts and verify cleanup. Agent
compatibility tests exercise all three agents through this transport with mocked
CLI output. No network, login, installation or live inference occurs in tests.
Windows execution policies may require running tests in a normal terminal rather
than an AppContainer that prohibits creating temporary executable fixtures.
All automated tests use mocks. The separate diagnostic reproduction described
below is the only live call made during the debugging session.
No dataset integration, orchestration, evaluation metrics, early stopping or
splitting is part of Step 5.

## Diagnosed smoke failure

The successful direct command differs from the smoke path in these ways:

| Boundary | Direct CLI | Provider smoke | Finding |
| --- | --- | --- | --- |
| Executable | PATH Codex 0.166.0 | Matching npm native binary | Same installed version. |
| Arguments/configuration | Minimal exec flags | Additional noninteractive/ephemeral/config flags | Exact flag combination parses successfully with --help. |
| Permissions | read-only | read-only, approval never | Retained unchanged. |
| Environment | Terminal environment | API-key variables excluded | No authentication failure established by the reproduction. |
| Working directory | Terminal directory | Isolated temporary work directory | Retained unchanged. |
| Prompt | Simple positional text | UTF-8 stdin with runtime policy | Retained unchanged. |
| Schema | None | Generated GraderOutput schema | Server returned invalid_json_schema and a regex-lookaround error. |
| Output | CLI final text | Separate --output-last-message file | Rejection occurred before a valid final result could be accepted. |

Pydantic generated `^(?!^[-+.]*$)[+-]?0*\d*\.?\d*$` for Decimal strings. Its
negative lookahead caused the schema compiler failure. The wire pattern is now
`^[+-]?(\d+(\.\d*)?|\.\d+)$`, retaining decimal syntax without lookaround.
The domain models, score boundaries, rubric validation, sandbox settings and
agent behaviors were not changed. Regression tests emulate the observed schema
compiler restriction, exercise diagnostic stages/redaction, and confirm that
negative score strings still fail local validation. No successful post-fix live
inference is claimed.
