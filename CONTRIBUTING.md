# Contributing

Thanks for your interest in contributing to **KANIT**.

## Development setup

```bash
git clone https://github.com/iumobg/KANIT.git
cd KANIT
conda env create -f environment.yml && conda activate amr-prediction   # or: pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Branch from `main` (or the active feature branch): `git checkout -b feat/my-change`.
2. Keep changes **incremental, organism-scoped, and tested**. Do not rewrite working code without reason.
3. Run the checks locally before pushing:
   ```bash
   make lint        # ruff
   make typecheck   # mypy on scripts/lib
   make test        # unit + smoke (seconds)
   make stub        # the workflow with stub tasks (needs Nextflow 25.04 or later)
   ```
4. Open a pull request. CI (GitHub Actions) runs ruff, mypy, the unit and smoke tests on
   Python 3.10–3.12, and the workflow with stub tasks.

## Conventions

- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`).
- **Single sources of truth:** shared helpers live in `scripts/lib/`; organism/antibiotic definitions in `config/registry/`; all tunables in `config/config.yaml`.
- **Paths:** anchor to `PROJECT_ROOT`; resolve organism/antibiotic paths via `lib.config.resolve_path`; resolve external tools via `lib.config.resolve_tool` (never hardcode binary paths).
- **Cross-environment:** code must run on macOS, Linux and HPC. No absolute paths, no CWD assumptions.
- **Tests:** add a unit test for every new pure function; the smoke list must include every new step script.
- **Reproducibility:** never commit generated artifacts (matrices, models, results, logs, run metadata, generated experiment configs). They are reproduced by running the pipeline.

## Adding an organism or antibiotic

Add a block to `config/registry/organisms.yaml` (or `antibiotics.yaml`) and check it with
`make registry`. The `DOWNLOAD` entry of the workflow then acquires the organism's data; no
code changes are required.
