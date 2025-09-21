This file provides instructions for AI agents working on this repository.

## Development Workflow with pre-commit

### 1. One-Time Setup

Before you start working, set up your environment by running the following command.

```bash
poetry install
```

### A Note on Managing Dependencies

If you need to add, remove, or update dependencies in `pyproject.toml`, you must also update the `poetry.lock` file to reflect these changes. After modifying `pyproject.toml`, run the following command:

```bash
poetry lock
```

This ensures that the project's dependencies remain consistent and reproducible. After running the command, remember to commit both the `pyproject.toml` and `poetry.lock` files.

### 2. Build the C++ tangency backend

The Python tests depend on the compiled shared library generated from `src/ellphi/_tangency_cpp_impl.cpp`. After installing the
Poetry environment, always build (or rebuild when the C++ source changes) the backend by running:

```bash
poetry run python -m pip install --upgrade pip setuptools
poetry run python build_tangency_cpp.py
```

If `setuptools` is already present the first command is a no-op, but including it guarantees that the build script can import the
required tooling. The build must succeed before executing pytest locally or in CI; otherwise, all tangency-related tests will be
skipped and the results will be invalid.

### 3. Run Test and Lint

In this repository, all CI tests must pass before PRs are merged. Therefore, before pushing you must have run all relevant tests.

```bash
poetry run pytest
poetry run flake8
poetry run black src tests
poetry run mypy src tests
```

## Language and Style

*   **Coding Style:** All Python code should adhere to the [PEP 8 style guide](https://peps.python.org/pep-0008/).
*   **Docstrings:** Docstrings must be written in **English**. They can include simple mathematical expressions where appropriate.
*   **Comments:** Code comments should be short and simple. Do not add verbose trivial comments.
