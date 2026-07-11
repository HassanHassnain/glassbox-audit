PYTHON ?= python3
PYTHONPATH := src

.PHONY: install test lint compile toy validate report figures reproduce-cleanroom reproduce-external reproduce-hardening release-check smoke

install:
	$(PYTHON) -m pip install -e ".[dev,ui,paper]"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

lint:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ruff check .

compile:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q src tests

toy:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m glassbox_audit.cli validate-data --data data/fixtures/refusal_pairs_tiny.jsonl
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m glassbox_audit.cli run --config configs/toy.yaml --output artifacts/demo

smoke: toy test

validate: test lint compile toy
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m glassbox_audit.cli workbench --help

report: toy
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m glassbox_audit.cli report --artifacts artifacts/demo

figures:
	$(PYTHON) scripts/generate_readme_figures.py
	$(PYTHON) scripts/generate_hardening_figures.py

reproduce-cleanroom:
	bash scripts/reproduce_cleanroom.sh

reproduce-external:
	bash scripts/reproduce_external.sh

reproduce-hardening:
	bash scripts/reproduce_hardening.sh

release-check:
	bash scripts/release_check.sh
