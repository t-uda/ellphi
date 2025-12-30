# Contributing

This document describes a simple, low-overhead workflow for developing and
releasing ellphi. It keeps `main` as a released version and uses a dedicated
release branch for ongoing work.

## Before You Start

Always check the project-wide guidelines and CI expectations:

- `AGENTS.md`
- `.github/workflows/python-app.yml`

They define the required local checks and the order CI will run them.

## Branching Model

- `main` always points to the latest released version.
- Each release is developed on its own branch (e.g., `ellphi-0.1.2`).
- Feature PRs are merged into the release branch, not into `main`.

## Build System Tools

This project leverages `cibuildwheel` and `auditwheel` to produce robust Python wheels, particularly for `manylinux` compatibility. Developers should be aware of their roles:

-   **`cibuildwheel`**: Orchestrates the building of wheels across different Python versions and platforms within CI/CD pipelines.
-   **`auditwheel`**: Analyzes and modifies Linux wheels to ensure adherence to `manylinux` standards, including bundling shared library dependencies (such as OpenBLAS) directly into the wheel.

## Versioning

We only bump versions at two moments:

1. Start of a new release cycle: set a dev version (e.g., `0.1.2.dev0`).
2. Release: set the final version (e.g., `0.1.2`).

Use the version sync script so that metadata and runtime version stay aligned:

```bash
poetry run python scripts/update_version.py 0.1.2.dev0
```

## Release Notes

Create an "Unreleased" section at the start of the cycle and update it as
features land:

```text
## 0.1.2 - Unreleased
```

Finalize the content right before release.

## Standard Workflow (Simple)

1. Create a release branch from `main`.
2. Bump to a dev version once (e.g., `0.1.2.dev0`) and commit it.
3. Merge feature PRs into the release branch.
4. When ready:
   - finalize `RELEASE_NOTES.md`
   - bump to the release version (e.g., `0.1.2`)
   - run all required local checks
   - merge into `main`
   - tag the release and publish

## Required Local Checks

Mirror CI locally. Run the exact commands listed in `AGENTS.md` and
`.github/workflows/python-app.yml`, in the same order when relevant. Do not
skip steps. If something cannot run, stop and document the blocker.
