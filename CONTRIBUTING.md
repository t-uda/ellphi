# Contributing

This document describes a simple, low-overhead workflow for developing and
releasing ellphi. It keeps `main` as a released version and uses a dedicated
release branch for ongoing work.

## Before You Start

Always check the project-wide guidelines and CI expectations:

- `AGENTS.md`
- `.github/workflows/python-app.yml`
- `.github/workflows/docs.yml` (for docs / release-note changes)

They define the required local checks and the order CI will run them.

## Branching Model

- `main` always points to the latest released version.
- Each release is developed on its own branch (e.g., `ellphi-0.1.2`).
- Feature PRs are merged into the release branch, not into `main`.

## Versioning

We only bump versions at two moments:

1. Start of a new release cycle: set a dev version (e.g., `0.1.2.dev0`).
2. Release: set the final version (e.g., `0.1.2`).

Use the version sync script so that metadata and runtime version stay aligned:

```bash
uv run python scripts/update_version.py 0.1.2.dev0
```

## Release Notes

Create an "Unreleased" section at the start of the cycle and update it as
features land:

```text
## Unreleased
```

Right before the release, replace it with a dated heading such as:

```text
## 0.1.2 - YYYY-MM-DD
```

## Standard Workflow (Simple)

1. Create a release branch from `main`.
2. Bump to a dev version once (e.g., `0.1.2.dev0`) and commit it.
3. Merge feature PRs into the release branch.
4. When ready for the final release, create a short-lived version-bump branch
   from the release branch.
5. On that version-bump branch:
   - finalize `RELEASE_NOTES.md`
   - run `uv run python scripts/update_version.py 0.1.2`
   - run all required local checks
   - run `uv run mkdocs build`
6. Open a PR from the version-bump branch back into the release branch and
   merge it after CI passes.
7. Open a PR from the release branch into `main` and merge it.
8. Create the final release tag (e.g., `v0.1.2`) from the commit now on
   `main`, then push the tag.

Do not tag a side branch or an unmerged release candidate commit. Tag the exact
commit that now represents the released state on `main`.

## Publishing

Releases are published automatically by the CI release workflow
(`.github/workflows/release.yml`).

| Tag | Destination |
|-----|-------------|
| `v0.1.2.dev1` | TestPyPI |
| `v0.1.2` | PyPI |
| `v0.1.2.post1` | PyPI |
| `v0.1.2a1`, `v0.1.2rc1` | Blocked by environment rule — manual decision |

Additional `v0.1.2.devN` tags are optional. Create another dev release only if
you want a fresh TestPyPI snapshot for validation before the final release.

```bash
# Example: publish a dev build to TestPyPI
git tag v0.1.2.dev1
git push origin v0.1.2.dev1

# Example: after the release branch has been merged into main,
# tag the main commit that now contains version 0.1.2
git tag v0.1.2
git push origin v0.1.2
```

The workflow uses PyPI Trusted Publishing (no API tokens needed).
See the comments at the top of `release.yml` for one-time setup steps.

## Required Local Checks

Mirror CI locally. Run the exact commands listed in `AGENTS.md` and
`.github/workflows/python-app.yml`, in the same order when relevant. For docs
and release-note changes, also mirror `.github/workflows/docs.yml` by running
`uv run mkdocs build` (after `uv sync --group docs`). Do not skip steps. If something cannot run, stop
and document the blocker.

## Optional Eigen Build (C++ linear algebra)

If you want to test the Eigen build locally:

1. Install Eigen headers:
   - Ubuntu: `apt-get install libeigen3-dev`
   - macOS (Homebrew): `brew install eigen`
2. Rebuild the extension with:

```bash
ELLPHI_USE_EIGEN=1 ELLPHI_EIGEN_INCLUDE=/usr/include/eigen3 uv sync --reinstall-package ellphi
```

On macOS, use `/opt/homebrew/include/eigen3` (Apple Silicon) or
`/usr/local/include/eigen3` (Intel).

Verify via:

```bash
uv run python -m ellphi --build-info
```

## Docs CI and Deployment

- `.github/workflows/docs.yml` runs `mkdocs build` on pull requests, on pushes
  to `ellphi-*`, and via `workflow_dispatch`.
- It deploys to `gh-pages` only on pushes to `main`.
- `.github/workflows/python-app.yml` contains a `demo-install` job that checks
  a plain install from the sdist; it is not the docs deployment workflow.

## Documentation

### Building and previewing

```bash
# Install the docs dependency group once
uv sync --group docs

# One-off build
uv run mkdocs build

# Live-reload server (http://127.0.0.1:8000)
uv run mkdocs serve
```

Run `uv run mkdocs build` before committing any docs change and confirm
the build completes without errors.

### English style

All documentation is written in **British English**
(`optimisation`, `normalise`, `characterisation`, `centred`, etc.).

### Mathematical notation

| Element | Convention | Example |
|---|---|---|
| Inline math | `\(...\)` | `\(t \ge 0\)` |
| Display math | `\[...\]` or `$$...$$` (block only) | — |
| Vectors | `\bm{·}` (bold italic via `\boldsymbol`) | `\(\bm{x}_c\)` |
| Matrices | Plain capital letter | `\(A\)`, `\(\Sigma\)` |
| Scalars | Plain italic | `\(t\)`, `\(\mu\)` |

`\bm` is defined as a MathJax macro in `docs/javascripts/mathjax.js`.
Inline math must not span multiple source lines.

### Assets

Logo assets live in `docs/assets/`:

- `logo.png` — full logo (icon + wordmark), transparent background
- `logo-icon.png` — icon only, used for navbar and favicon

The root `ellphi-logo.png` is kept in sync with `docs/assets/logo.png`
(used by the README).
