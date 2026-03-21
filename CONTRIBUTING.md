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
   - tag the release and push the tag

## Publishing

Releases are published automatically by the CI release workflow
(`.github/workflows/release.yml`).

| Tag | Destination |
|-----|-------------|
| `v0.1.2.dev1` | TestPyPI |
| `v0.1.2` | PyPI |
| `v0.1.2.post1` | PyPI |
| `v0.1.2a1`, `v0.1.2rc1` | Blocked by environment rule — manual decision |

```bash
# Example: publish a dev build to TestPyPI
git tag v0.1.2.dev1
git push origin v0.1.2.dev1

# Example: publish a release to PyPI
git tag v0.1.2
git push origin v0.1.2
```

The workflow uses PyPI Trusted Publishing (no API tokens needed).
See the comments at the top of `release.yml` for one-time setup steps.

## Required Local Checks

Mirror CI locally. Run the exact commands listed in `AGENTS.md` and
`.github/workflows/python-app.yml`, in the same order when relevant. Do not
skip steps. If something cannot run, stop and document the blocker.

## Optional Eigen Build (C++ linear algebra)

If you want to test the Eigen build locally:

1. Install Eigen headers:
   - Ubuntu: `apt-get install libeigen3-dev`
   - macOS (Homebrew): `brew install eigen`
2. Rebuild the extension with:

```bash
ELLPHI_USE_EIGEN=1 ELLPHI_EIGEN_INCLUDE=/usr/include/eigen3 poetry install
```

On macOS, use `/opt/homebrew/include/eigen3` (Apple Silicon) or
`/usr/local/include/eigen3` (Intel).

Verify via:

```bash
python -m ellphi --build-info
```

## Documentation

### Building and previewing

```bash
# One-off build
poetry run mkdocs build

# Live-reload server (http://127.0.0.1:8000)
poetry run mkdocs serve
```

Run `poetry run mkdocs build` before committing any docs change and confirm
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
