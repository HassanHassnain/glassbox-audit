PYTHON ?= python3
PYTHONPATH := src

.PHONY: install test lint compile toy validate report reproduce-cleanroom reproduce-external release-check smoke

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

reproduce-cleanroom:
	bash scripts/reproduce_cleanroom.sh

reproduce-external:
	bash scripts/reproduce_external.sh

release-check:
	bash scripts/release_check.sh
