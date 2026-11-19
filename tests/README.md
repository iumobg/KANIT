# Tests

The tests check the workflow in seconds or minutes, without rerunning the analysis.

## Layers

| Layer | Marker | Speed | What it catches |
|-------|--------|-------|-----------------|
| **Smoke** | `smoke` | seconds | Syntax, import, configuration and path errors in every step script |
| **Unit** | `unit` | seconds | Regressions in the pure functions: phenotype counting, lineage-aware folds, the evidence layers, CARD concordance, grading, the knowledge base and its package, the registries and path resolution |
| **Workflow stub** | — | minutes | The wiring of every workflow entry with stub tasks (`tests/test_workflow_stub.py`; opt-in, needs Nextflow 25.04 or later) |

## How to run

```bash
pytest                 # unit and smoke tests (default)
pytest -m unit         # one layer
pytest -m smoke
make stub              # the workflow with stub tasks
```

`pytest` defaults to `-m "not integration and not slow"` (see `pytest.ini`), so heavy tests
are opt-in.

## Environments

In an environment without `xgboost` or `optuna`, the tests that need them are skipped; the
smoke tests and the remaining unit tests still run.
