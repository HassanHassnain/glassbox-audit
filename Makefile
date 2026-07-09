.PHONY: install smoke test lint compile toy sycophancy-toy workbench qwen qwen-full real-audit model-sweep head-path certify audit-card-gallery certify-demo drift leaderboard evidence-explorer docs-build docs-serve paper reproduce-core reproduce-cleanroom reproduce-external report final-report validate release-check

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

certify:
	python -m glassbox_audit.cli certify \
		--artifacts artifacts/qwen-real-audit \
		--output-json results/certification/qwen-real-audit.card.json \
		--output-html results/certification/qwen-real-audit.card.html \
		--output-pdf results/certification/qwen-real-audit.card.pdf \
		--sign

audit-card-gallery:
	python scripts/build_audit_card_gallery.py

certify-demo:
	python scripts/build_certify_demo.py

drift:
	python -m glassbox_audit.cli drift \
		--source-artifacts artifacts/qwen-real-audit \
		--new-artifacts artifacts/qwen2.5-3b-replication-preregistered \
		--output-json results/drift/qwen15b_to_qwen3b.json \
		--output-doc results/drift/qwen15b_to_qwen3b.md

leaderboard:
	python scripts/build_leaderboard.py \
		--artifacts-root results \
		--output-json results/leaderboard/leaderboard.json \
		--output-html docs/leaderboard/index.html

evidence-explorer:
	python scripts/build_evidence_explorer.py \
		--artifacts-root results \
		--output-json results/evidence_explorer/explorer.json \
		--output-html docs/evidence/index.html

docs-build:
	mkdocs build --strict

docs-serve:
	mkdocs serve

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

release-check: validate final-report audit-card-gallery certify-demo leaderboard evidence-explorer docs-build
	python -m glassbox_audit.cli workbench --help
