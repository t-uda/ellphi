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

### 2. Run Test and Lint

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
