from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .analysis import refresh_failure_analysis
from .analysis.component_analysis import run_component_localization
from .analysis.hardening import run_paper_hardening
from .analysis.path_component import run_path_component_analysis
from .analysis.release_hardening import (
    write_data_leakage_audit,
    write_dose_response,
    write_reproducibility_diff,
    write_scorer_robustness,
    write_statistical_tests,
)
from .config import load_config
from .data import dataset_summary, load_records
from .data.dataset_builder import build_paired_dataset
from .data.external_data import (
    external_to_glassbox_pairs,
    load_external_refusal_records,
    load_huggingface_refusal_records,
    write_external_normalized,
)
from .data.real_dataset import build_controlled_real_audit_dataset
from .evaluation.capability import run_wikitext_capability_eval
from .evaluation.external_causal import run_external_causal_transfer
from .evaluation.external_eval import run_external_behavior_eval
from .interventions.control_refresh import refresh_random_sae_controls
from .pipeline import run_audit
from .reporting.publication import generate_publication_bundle
from .reporting.report import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glassbox", description="Mechanistic behavior audit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run discovery, intervention, and baseline experiments")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)

    validate = subparsers.add_parser("validate-data", help="Validate and summarize a dataset")
    validate.add_argument("--data", required=True)

    build = subparsers.add_parser("build-pairs", help="Build paired JSONL from matched source rows")
    build.add_argument("--source", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--seed", type=int, default=17)

    external = subparsers.add_parser(
        "normalize-external-data", help="Normalize external refusal/safety datasets"
    )
    external.add_argument("--source", required=True)
    external.add_argument("--output", required=True)
    external.add_argument(
        "--glassbox-output",
        default=None,
        help="Optional paired PromptRecord JSONL output for direct audit runs",
    )
    external.add_argument("--prompt-field", default="prompt")
    external.add_argument("--label-field", default="behavior_label")
    external.add_argument("--harmful-field", default="harmful")
    external.add_argument("--split-field", default="split")
    external.add_argument("--source-field", default="source")
    external.add_argument("--category-field", default="category")
    external.add_argument("--family-field", default="family_id")
    external.add_argument("--pair-field", default="pair_id")
    external.add_argument(
        "--default-harmful",
        choices=["true", "false"],
        default=None,
        help="Use when a whole external subset is harmful or benign and rows omit a boolean flag",
    )

    hf_external = subparsers.add_parser(
        "download-hf-external-data",
        help="Download/normalize a Hugging Face refusal/safety dataset subset",
    )
    hf_external.add_argument("--dataset", required=True)
    hf_external.add_argument("--subset", default=None)
    hf_external.add_argument("--split", default="train")
    hf_external.add_argument("--output", required=True)
    hf_external.add_argument("--prompt-field", default="prompt")
    hf_external.add_argument("--label-field", default="category")
    hf_external.add_argument("--harmful-field", default="harmful")
    hf_external.add_argument("--source-name", default=None)
    hf_external.add_argument(
        "--default-harmful",
        choices=["true", "false"],
        default=None,
        help="Use when a whole HF subset is harmful or benign and rows omit a boolean flag",
    )

    external_eval = subparsers.add_parser(
        "external-evaluate",
        help="Run behavior-only evaluation on normalized external refusal/safety rows",
    )
    external_eval.add_argument("--artifacts", required=True)
    external_eval.add_argument("--inputs", nargs="+", required=True)
    external_eval.add_argument("--output", required=True)
    external_eval.add_argument("--max-records-per-label", type=int, default=None)
    external_eval.add_argument("--seed", type=int, default=17)
    external_eval.add_argument("--device", default=None)

    external_causal = subparsers.add_parser(
        "external-causal",
        help="Run frozen controlled-artifact interventions on normalized external rows",
    )
    external_causal.add_argument("--artifacts", required=True)
    external_causal.add_argument("--inputs", nargs="+", required=True)
    external_causal.add_argument("--output", required=True)
    external_causal.add_argument("--max-records-per-label", type=int, default=None)
    external_causal.add_argument("--seed", type=int, default=None)
    external_causal.add_argument("--device", default=None)
    external_causal.add_argument("--random-controls", type=int, default=None)

    capability = subparsers.add_parser(
        "capability-evaluate",
        help="Evaluate frozen baseline/mean/SAE conditions on WikiText-2 perplexity",
    )
    capability.add_argument("--artifacts", required=True)
    capability.add_argument("--output", required=True)
    capability.add_argument("--dataset", default="Salesforce/wikitext")
    capability.add_argument("--subset", default="wikitext-2-raw-v1")
    capability.add_argument("--split", default="test")
    capability.add_argument("--revision", default=None)
    capability.add_argument("--max-blocks", type=int, default=128)
    capability.add_argument("--block-size", type=int, default=256)
    capability.add_argument("--batch-size", type=int, default=4)
    capability.add_argument("--bootstrap-samples", type=int, default=None)
    capability.add_argument("--confidence", type=float, default=None)
    capability.add_argument("--seed", type=int, default=None)
    capability.add_argument("--device", default=None)
    capability.add_argument("--expected-subset-sha256", default=None)

    real_data = subparsers.add_parser(
        "build-real-audit-data", help="Build the versioned controlled real-audit corpus"
    )
    real_data.add_argument("--output", default="data/refusal_controlled_v1.jsonl")
    real_data.add_argument("--pairs-per-family", type=int, default=3)
    real_data.add_argument("--source-name", default="glassbox_controlled_refusal_v1")

    report = subparsers.add_parser("report", help="Rebuild REPORT.md from artifacts")
    report.add_argument("--artifacts", required=True)

    paper = subparsers.add_parser("paper", help="Generate paper-style docs, figures, and tables")
    paper.add_argument("--artifacts", required=True)
    paper.add_argument("--docs", default="docs")

    analyze = subparsers.add_parser("analyze", help="Refresh derived failure-analysis artifacts")
    analyze.add_argument("--artifacts", required=True)

    controls = subparsers.add_parser(
        "refresh-controls", help="Refresh frozen control evaluations from existing artifacts"
    )
    controls.add_argument("--artifacts", required=True)
    controls.add_argument(
        "--control",
        choices=["random-sae"],
        default="random-sae",
        help="Control family to refresh without retuning discovery artifacts",
    )
    controls.add_argument("--device", default=None, help="Optional model/SAE device override")

    component = subparsers.add_parser(
        "component-localize", help="Run component-level localization from frozen artifacts"
    )
    component.add_argument("--artifacts", required=True)
    component.add_argument("--components", default="residual,attention,mlp")
    component.add_argument("--layers", default=None, help="Optional comma-separated layer list")
    component.add_argument("--direction", default="mean_difference")
    component.add_argument("--device", default=None)
    component.add_argument("--max-records", type=int, default=None)
    component.add_argument("--output-name", default="component_localization.json")

    path_component = subparsers.add_parser(
        "path-component",
        help="Run residual projection-patching and component-output localization from artifacts",
    )
    path_component.add_argument("--artifacts", required=True)
    path_component.add_argument("--layers", default=None, help="Optional comma-separated layer list")
    path_component.add_argument("--components", default="residual,attention,mlp")
    path_component.add_argument("--device", default=None)
    path_component.add_argument("--max-records", type=int, default=None)
    path_component.add_argument("--output-name", default="path_component_analysis.json")

    repro = subparsers.add_parser(
        "reproducibility-diff", help="Compare reference and clean-room rerun artifacts"
    )
    repro.add_argument("--reference-artifacts", required=True)
    repro.add_argument("--cleanroom-artifacts", required=True)
    repro.add_argument("--reference-external", required=True)
    repro.add_argument("--cleanroom-external", required=True)
    repro.add_argument("--reference-component", required=True)
    repro.add_argument("--cleanroom-component", required=True)
    repro.add_argument("--output-json", default="results/final/reproducibility_diff.json")
    repro.add_argument("--output-doc", default="docs/REPRODUCIBILITY_REPORT.md")

    stats = subparsers.add_parser(
        "statistical-tests", help="Build release paired bootstrap/permutation summaries"
    )
    stats.add_argument("--reference-artifacts", required=True)
    stats.add_argument("--cleanroom-artifacts", default=None)
    stats.add_argument(
        "--stability-summary",
        default="results/sae-stability/stability_grid.json",
    )
    stats.add_argument("--external-artifacts", nargs="*", default=[])
    stats.add_argument("--component-artifacts", nargs="*", default=[])
    stats.add_argument("--output-json", default="results/final/statistical_tests.json")
    stats.add_argument("--output-doc", default="docs/STATISTICAL_TESTS.md")
    stats.add_argument("--output-table", default="docs/tables/statistical_summary.md")
    stats.add_argument("--permutation-samples", type=int, default=5000)
    stats.add_argument("--bootstrap-samples", type=int, default=3000)

    scorer = subparsers.add_parser(
        "scorer-robustness", help="Audit refusal-score threshold and keyword-scorer robustness"
    )
    scorer.add_argument("--artifacts", required=True)
    scorer.add_argument("--output-json", default="results/final/scorer_robustness.json")
    scorer.add_argument("--output-doc", default="docs/SCORER_ROBUSTNESS.md")
    scorer.add_argument(
        "--disagreement-csv",
        default="data/manual_review/refusal_scorer_disagreements.csv",
    )

    leakage = subparsers.add_parser(
        "leakage-audit", help="Run controlled/external data leakage and split-integrity checks"
    )
    leakage.add_argument("--controlled", required=True)
    leakage.add_argument("--external", nargs="*", default=[])
    leakage.add_argument("--output-json", default="results/final/data_leakage_audit.json")
    leakage.add_argument("--output-doc", default="docs/DATA_LEAKAGE_AUDIT.md")

    dose = subparsers.add_parser(
        "dose-response", help="Summarize fixed intervention-scale sweeps from artifacts"
    )
    dose.add_argument("--artifacts", required=True)
    dose.add_argument("--output-json", default="results/final/dose_response.json")
    dose.add_argument("--output-doc", default="docs/DOSE_RESPONSE.md")
    dose.add_argument("--figure-prefix", default="docs/figures/dose_response")

    workbench = subparsers.add_parser("workbench", help="Launch the Streamlit feature browser")
    workbench.add_argument("--artifacts", required=True)

    hardening = subparsers.add_parser(
        "paper-hardening",
        help="Run pair-aware inference, intervention comparability, and the SAE budget frontier",
    )
    hardening.add_argument("--artifacts", required=True)
    hardening.add_argument("--dataset", required=True)
    hardening.add_argument("--config", required=True)
    hardening.add_argument("--output-dir", required=True)
    hardening.add_argument("--result", required=True)
    hardening.add_argument("--device", default="cuda:0")
    hardening.add_argument("--cache-dir", default=None)
    hardening.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        summary = run_audit(load_config(args.config), args.output)
        print(json.dumps(summary, indent=2))
    elif args.command == "paper-hardening":
        result = run_paper_hardening(
            artifacts=args.artifacts,
            dataset=args.dataset,
            config_path=args.config,
            output_dir=args.output_dir,
            result_path=args.result,
            device=args.device,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
        )
        print(json.dumps({
            "result": args.result,
            "analysis_status": result["analysis_status"],
            "baseline_batch_crosscheck": result["baseline_batch_crosscheck"],
        }, indent=2))
    elif args.command == "validate-data":
        print(json.dumps(dataset_summary(load_records(args.data)), indent=2))
    elif args.command == "build-pairs":
        count = build_paired_dataset(args.source, args.output, args.seed)
        print(f"Wrote {count} records to {args.output}")
    elif args.command == "normalize-external-data":
        records = load_external_refusal_records(
            args.source,
            prompt_field=args.prompt_field,
            label_field=args.label_field,
            harmful_field=args.harmful_field,
            split_field=args.split_field,
            source_field=args.source_field,
            category_field=args.category_field,
            family_field=args.family_field,
            pair_field=args.pair_field,
            default_harmful=(
                None if args.default_harmful is None else args.default_harmful == "true"
            ),
        )
        write_external_normalized(records, args.output)
        result = {"normalized_records": len(records), "output": args.output}
        if args.glassbox_output:
            result["glassbox_records"] = external_to_glassbox_pairs(records, args.glassbox_output)
            result["glassbox_output"] = args.glassbox_output
        print(json.dumps(result, indent=2))
    elif args.command == "download-hf-external-data":
        records = load_huggingface_refusal_records(
            args.dataset,
            subset=args.subset,
            split=args.split,
            source_name=args.source_name,
            prompt_field=args.prompt_field,
            label_field=args.label_field,
            harmful_field=args.harmful_field,
            default_harmful=(
                None if args.default_harmful is None else args.default_harmful == "true"
            ),
        )
        write_external_normalized(records, args.output)
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "subset": args.subset,
                    "split": args.split,
                    "normalized_records": len(records),
                    "output": args.output,
                    "harmful_labels": {
                        "true": sum(record.harmful is True for record in records),
                        "false": sum(record.harmful is False for record in records),
                        "missing": sum(record.harmful is None for record in records),
                    },
                },
                indent=2,
            )
        )
    elif args.command == "external-evaluate":
        artifact = run_external_behavior_eval(
            artifact_dir=args.artifacts,
            normalized_paths=args.inputs,
            output_path=args.output,
            max_records_per_label=args.max_records_per_label,
            seed=args.seed,
            device=args.device,
        )
        compact = {key: value for key, value in artifact.items() if key != "records"}
        print(json.dumps(compact, indent=2))
    elif args.command == "external-causal":
        artifact = run_external_causal_transfer(
            artifact_dir=args.artifacts,
            normalized_paths=args.inputs,
            output_path=args.output,
            max_records_per_label=args.max_records_per_label,
            seed=args.seed,
            device=args.device,
            random_controls=args.random_controls,
        )
        compact = {
            key: value
            for key, value in artifact.items()
            if key not in {"baseline_outputs", "causal_tests", "random_controls"}
        }
        print(json.dumps(compact, indent=2))
    elif args.command == "capability-evaluate":
        artifact = run_wikitext_capability_eval(
            artifact_dir=args.artifacts,
            output_path=args.output,
            dataset_name=args.dataset,
            dataset_config=args.subset,
            split=args.split,
            revision=args.revision,
            max_blocks=args.max_blocks,
            block_size=args.block_size,
            batch_size=args.batch_size,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed,
            device=args.device,
            expected_subset_sha256=args.expected_subset_sha256,
        )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in artifact.items()
                    if key != "per_block"
                },
                indent=2,
            )
        )
    elif args.command == "build-real-audit-data":
        print(
            json.dumps(
                build_controlled_real_audit_dataset(
                    args.output,
                    pairs_per_family=args.pairs_per_family,
                    source_name=args.source_name,
                ),
                indent=2,
            )
        )
    elif args.command == "report":
        print(write_report(args.artifacts))
    elif args.command == "paper":
        print(json.dumps(generate_publication_bundle(args.artifacts, args.docs), indent=2))
    elif args.command == "analyze":
        print(json.dumps(refresh_failure_analysis(args.artifacts), indent=2))
    elif args.command == "refresh-controls":
        if args.control != "random-sae":
            raise ValueError(f"Unsupported control family: {args.control}")
        print(
            json.dumps(
                refresh_random_sae_controls(args.artifacts, device=args.device),
                indent=2,
            )
        )
    elif args.command == "component-localize":
        artifact = run_component_localization(
            args.artifacts,
            components=[item.strip() for item in args.components.split(",") if item.strip()],
            layers=(
                [int(item.strip()) for item in args.layers.split(",") if item.strip()]
                if args.layers
                else None
            ),
            direction_name=args.direction,
            device=args.device,
            max_records=args.max_records,
            output_name=args.output_name,
        )
        compact = {
            key: value for key, value in artifact.items() if key != "components"
        }
        compact["components"] = [
            {
                key: value
                for key, value in row.items()
                if key not in {"outputs"}
            }
            for row in artifact["components"]
        ]
        print(
            json.dumps(
                compact,
                indent=2,
            )
        )
    elif args.command == "path-component":
        artifact = run_path_component_analysis(
            args.artifacts,
            layers=(
                [int(item.strip()) for item in args.layers.split(",") if item.strip()]
                if args.layers
                else None
            ),
            components=[item.strip() for item in args.components.split(",") if item.strip()],
            device=args.device,
            max_records=args.max_records,
            output_name=args.output_name,
        )
        compact = {
            key: value
            for key, value in artifact.items()
            if key not in {"component_outputs", "residual_ablation", "residual_projection_patching"}
        }
        print(json.dumps(compact, indent=2))
    elif args.command == "reproducibility-diff":
        artifact = write_reproducibility_diff(
            reference_artifact=args.reference_artifacts,
            cleanroom_artifact=args.cleanroom_artifacts,
            reference_external=args.reference_external,
            cleanroom_external=args.cleanroom_external,
            reference_component=args.reference_component,
            cleanroom_component=args.cleanroom_component,
            output_json=args.output_json,
            output_doc=args.output_doc,
        )
        print(
            json.dumps(
                {
                    "output": args.output_json,
                    "cleanroom_reproduces_reference": artifact["cleanroom_reproduces_reference"],
                },
                indent=2,
            )
        )
    elif args.command == "statistical-tests":
        artifact = write_statistical_tests(
            reference_artifact=args.reference_artifacts,
            cleanroom_artifact=args.cleanroom_artifacts,
            stability_summary=args.stability_summary,
            external_artifacts=args.external_artifacts,
            component_artifacts=args.component_artifacts,
            output_json=args.output_json,
            output_doc=args.output_doc,
            output_table=args.output_table,
            permutation_samples=args.permutation_samples,
            bootstrap_samples=args.bootstrap_samples,
        )
        print(
            json.dumps(
                {
                    "output": args.output_json,
                    "internal_families": len(artifact["internal"]),
                    "external_families": len(artifact["external_causal"]),
                    "component_families": len(artifact["component_path"]),
                },
                indent=2,
            )
        )
    elif args.command == "scorer-robustness":
        artifact = write_scorer_robustness(
            artifact_dir=args.artifacts,
            output_json=args.output_json,
            output_doc=args.output_doc,
            disagreement_csv=args.disagreement_csv,
        )
        print(
            json.dumps(
                {
                    "output": args.output_json,
                    "headline_survives": artifact[
                        "headline_survives_fixed_score_threshold_variants"
                    ],
                },
                indent=2,
            )
        )
    elif args.command == "leakage-audit":
        artifact = write_data_leakage_audit(
            controlled_path=args.controlled,
            external_paths=args.external,
            output_json=args.output_json,
            output_doc=args.output_doc,
        )
        print(
            json.dumps(
                {
                    "output": args.output_json,
                    "leakage_pass": artifact["leakage_pass"],
                    "paraphrase_warning": artifact["paraphrase_warning"],
                },
                indent=2,
            )
        )
    elif args.command == "dose-response":
        artifact = write_dose_response(
            artifact_dir=args.artifacts,
            output_json=args.output_json,
            output_doc=args.output_doc,
            figure_prefix=args.figure_prefix,
        )
        print(
            json.dumps(
                {
                    "output": args.output_json,
                    "sae_more_specific_regime": artifact[
                        "sae_has_more_specific_regime_than_mean_probe"
                    ],
                },
                indent=2,
            )
        )
    elif args.command == "workbench":
        app = Path(__file__).parent / "reporting" / "workbench.py"
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", str(app), "--", "--artifacts", args.artifacts]
            )
        )


if __name__ == "__main__":
    main()
