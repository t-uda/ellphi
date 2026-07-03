This file provides instructions for AI agents working on this repository. Read it fully before starting any task so you understand the required local checks and CI expectations.

## CI Awareness & Required Local Checks

1. **Mirror CI locally.** Review `.github/workflows/python-app.yml` at the beginning of every task so you know exactly which commands CI will run. Your local workflow must include those same commands (and in the same order when relevant).
2. **Track executed commands.** Keep a short checklist while you work. Before you hand off or open a PR, verify that every command below has been run in the current branch and note any intentional omissions in your PR summary.
3. **No-skip policy.** Even for small edits, formatting and linting are mandatory. If a command cannot be run (e.g., tooling unavailable), stop and communicate the blocker rather than submitting unverified code.

## Development Workflow with pre-commit

### 1. One-Time Setup

Before you start working, set up your environment by running the following command. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This command installs all Python dependencies (including the `dev` dependency group) and also compiles the C++ backend for the tangency solver. If the C++ code (`src/ellphi/_tangency_cpp_impl.cpp`) is modified, the backend will be automatically rebuilt the next time you run `uv sync`.

Optional Eigen build (C++ linear algebra):
- Install Eigen headers (Ubuntu: `apt-get install libeigen3-dev`, macOS: `brew install eigen`).
- Rebuild with `ELLPHI_USE_EIGEN=1` and `ELLPHI_EIGEN_INCLUDE` set to the Eigen include path, forcing a reinstall of the project: `ELLPHI_USE_EIGEN=1 ELLPHI_EIGEN_INCLUDE=/usr/include/eigen3 uv sync --reinstall-package ellphi`
  - Linux default: `/usr/include/eigen3`
  - macOS default: `/opt/homebrew/include/eigen3` (Apple Silicon) or `/usr/local/include/eigen3` (Intel)
- Verify with `uv run python -m ellphi --build-info` and check `cpp_linalg_kind`.

### A Note on Managing Dependencies

If you need to add, remove, or update dependencies in `pyproject.toml`, you must also update the `uv.lock` file to reflect these changes. After modifying `pyproject.toml`, run the following command:

```bash
uv lock
```

This ensures that the project's dependencies remain consistent and reproducible. After running the command, remember to commit both the `pyproject.toml` and `uv.lock` files. CI verifies lockfile freshness with `uv lock --check`, so a stale `uv.lock` will fail the build.

### 2. Run Test and Lint

In this repository, all CI tests must pass before PRs are merged. Therefore, before pushing you must have run all relevant tests. Use the following checklist and do not skip any step:

```bash
# Format and verify formatting
uv run black src tests scripts
uv run black --check src tests scripts

# Lint and static analysis
uv run flake8 src tests scripts
uv run mypy src tests

# Type Stub Validation
MYPYPATH=src uv run stubtest ellphi --allowlist stubtest-allowlist.txt

# Tests
uv run pytest

# Lockfile freshness (if pyproject.toml changed)
uv lock --check
```

If CI introduces new tools, immediately add them to this checklist.

#### A Note on `stubtest`

`stubtest` is a tool that verifies the consistency between your Python type stubs (`.pyi` files) and the actual runtime implementation. It helps catch discrepancies like missing or mismatched function signatures, ensuring that your type hints are accurate.

**Command:**

```bash
MYPYPATH=src uv run stubtest ellphi --allowlist stubtest-allowlist.txt
```

**Purpose:**

*   To validate that the type stubs accurately reflect the runtime code.
*   To prevent type-related errors in projects that consume this library.

**Handling Failures:**

When `stubtest` reports a failure, it means there is a mismatch between the implementation and the type stub. You have two options:

1.  **Fix the Type Stub:** If the `.pyi` file is incorrect or outdated, update it to match the runtime implementation. This is the preferred solution in most cases.
2.  **Update the Allowlist:** If the reported mismatch is intentional or cannot be resolved (e.g., due to dynamic attributes or limitations in the type system), you can add the specific symbol to the `stubtest-allowlist.txt` file. When doing so, you must document the reason in the "Allowlist Justification" section below.

#### Allowlist Justification

This section documents the reasons for each entry in the `stubtest-allowlist.txt` file.

*   `ellphi._solver_python.MethodName`

**Reason:** This entry is necessary because the `MethodName` type alias is defined as a `typing.Literal` in the implementation, but it is simplified to `str` in the corresponding stub file (`.pyi`). This simplification is intentional to avoid duplicating the literal values in the stub, which would make it harder to maintain. (An equivalent entry for the `ellphi.solver` re-export was needed with older mypy versions but is reported as unused since mypy 1.20.)

#### Managing the Allowlist

When adding a new entry to the `stubtest-allowlist.txt` file, you must also update the "Allowlist Justification" section in this document to include a clear and concise explanation for why the entry is needed. This ensures that the allowlist remains transparent and easy to manage.

### 3. Pre-PR Handoff

Before requesting review or handing work back to the user:

* Confirm the above commands were run successfully in the current branch.
* Mention the exact commands (and their status) in the PR description or handoff note.
* If any command was skipped, clearly explain why and what follow-up is required.

## Language and Style

*   **Coding Style:** All Python code should adhere to the [PEP 8 style guide](https://peps.python.org/pep-0008/).
*   **Docstrings:** Docstrings must be written in **English**. They can include simple mathematical expressions where appropriate.
*   **Comments:** Code comments should be short and simple. Do not add verbose trivial comments.

## Documentation-only Tasks

Docs-only changes (files under `docs/`, `mkdocs.yml`, `docs/javascripts/`)
do **not** require the full Python CI checklist (black/flake8/mypy/pytest).
Instead, run:

```bash
uv sync --group docs
uv run mkdocs build
```

and confirm it exits without errors before committing.
