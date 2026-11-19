# ============================================================================
# Makefile — KANIT development shortcuts
# ============================================================================
# `make help` lists the targets. The analysis itself is the Nextflow workflow
# (main.nf); see README.md for how to run it. PYTHON can be overridden, e.g.
#     make test PYTHON=/path/to/env/bin/python
# ============================================================================
PYTHON   ?= python
NEXTFLOW ?= nextflow

.DEFAULT_GOAL := help
.PHONY: help setup dev-install lint format typecheck registry test test-all stub clean-pyc

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the conda environment (external tools + Python dependencies)
	conda env create -f environment.yml

dev-install:  ## Install the development toolchain (ruff, mypy, pre-commit, pytest)
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

lint:  ## Lint with ruff
	ruff check .

format:  ## Format with ruff
	ruff format .

typecheck:  ## Type-check scripts/lib with mypy
	mypy

registry:  ## Check the organism and antibiotic registries
	$(PYTHON) scripts/validate_registry.py

test:  ## Run the unit and smoke tests (seconds)
	$(PYTHON) -m pytest -ra

test-all:  ## Run every test, including the opt-in integration tests
	$(PYTHON) -m pytest -m "unit or smoke or integration" -ra

stub:  ## Run the whole workflow with stub tasks (wiring check; needs Nextflow)
	KANIT_RUN_NEXTFLOW=1 NEXTFLOW=$(NEXTFLOW) $(PYTHON) -m pytest -ra tests/test_workflow_stub.py

clean-pyc:  ## Remove Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} + ; \
	find . -type f -name '*.py[co]' -delete
