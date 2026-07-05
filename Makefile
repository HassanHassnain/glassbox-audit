.PHONY: install smoke test lint compile toy sycophancy-toy workbench qwen qwen-full real-audit model-sweep head-path paper reproduce-core reproduce-cleanroom reproduce-external report final-report validate release-check

install:
	pip install -e ".[dev,ui]"

smoke:
	bash scripts/00_smoke.sh

test:
	python -m pytest

lint:
	ruff check .

compile:
	python -m compileall -q glassbox_audit tests

toy:
	python -m glassbox_audit.cli run --config configs/toy.yaml --output artifacts/demo

sycophancy-toy:
	python -m glassbox_audit.cli build-sycophancy-data --output data/sycophancy_controlled_v1.jsonl
	python -m glassbox_audit.cli run --config configs/sycophancy-toy.yaml --output artifacts/sycophancy-toy

workbench:
	python -m glassbox_audit.cli workbench --artifacts artifacts/demo

qwen:
	bash scripts/10_run_qwen.sh

qwen-full:
	bash scripts/11_run_qwen_full.sh

real-audit:
	bash scripts/12_run_real_audit.sh

model-sweep:
	bash scripts/sweep_models.sh

head-path:
	python -m glassbox_audit.cli head-path --artifacts artifacts/qwen-real-audit

paper:
	python -m glassbox_audit.cli paper --artifacts artifacts/qwen-real-audit --docs docs

reproduce-core:
	bash scripts/20_phase5_reproduce_core.sh

reproduce-cleanroom:
	bash scripts/30_phase6_reproduce_cleanroom.sh

reproduce-external:
	bash scripts/21_phase5_reproduce_external.sh

report:
	bash scripts/22_phase5_reports.sh

final-report:
	bash scripts/31_phase6_final_report.sh

validate: test lint compile toy
	python -m glassbox_audit.cli workbench --help

release-check: validate final-report
	python -m glassbox_audit.cli workbench --help
