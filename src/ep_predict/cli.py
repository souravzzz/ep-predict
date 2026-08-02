from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ep_predict.config import load_toml


def _inspect(args: argparse.Namespace) -> int:
    from ep_predict.modeling import (
        inspect_loaded_model,
        load_model_and_tokenizer,
        print_model_summary,
    )
    from ep_predict.tracing.storage import write_json

    config = load_toml(args.config)
    model, _tokenizer = load_model_and_tokenizer(config)
    report, _routers = inspect_loaded_model(
        model,
        router_name_contains=config.get("router_name_contains"),
    )
    if args.output:
        write_json(args.output, report)
    print_model_summary(report)
    return 0


def _prepare_dataset(args: argparse.Namespace) -> int:
    from ep_predict.data.standard import materialize_standard_workload

    manifest = materialize_standard_workload(load_toml(args.config))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _collect(args: argparse.Namespace) -> int:
    from ep_predict.collect import collect_run

    manifest = collect_run(
        load_toml(args.model_config),
        load_toml(args.experiment_config),
        limit=args.limit,
    )
    print(json.dumps({"run_id": manifest["run_id"], "state": manifest["state"]}))
    return 0


def _analyze_h1(args: argparse.Namespace) -> int:
    from ep_predict.analysis.h1 import analyze_h1

    summary = analyze_h1(args.run, load_toml(args.config))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


def _plot_h1(args: argparse.Namespace) -> int:
    from ep_predict.visualize.h1 import plot_h1

    manifest = plot_h1(
        args.run,
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _analyze_h2(args: argparse.Namespace) -> int:
    from ep_predict.analysis.h2 import analyze_h2

    summary = analyze_h2(args.run, load_toml(args.config))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


def _plot_h2(args: argparse.Namespace) -> int:
    from ep_predict.visualize.h2 import plot_h2

    manifest = plot_h2(
        args.run,
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _analyze_h3(args: argparse.Namespace) -> int:
    from ep_predict.analysis.h3 import analyze_h3

    summary = analyze_h3(args.run, load_toml(args.config))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


def _plot_h3(args: argparse.Namespace) -> int:
    from ep_predict.visualize.h3 import plot_h3

    manifest = plot_h3(
        args.run,
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _plot_extended_horizon(args: argparse.Namespace) -> int:
    from ep_predict.visualize.extended_horizon import plot_extended_horizon

    manifest = plot_extended_horizon(
        args.run,
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _analyze_checkpoint_trajectories(args: argparse.Namespace) -> int:
    from ep_predict.analysis.checkpoint import analyze_checkpoint_trajectories

    summary = analyze_checkpoint_trajectories(load_toml(args.config))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


def _plot_checkpoint_trajectories(args: argparse.Namespace) -> int:
    from ep_predict.visualize.checkpoint import plot_checkpoint_trajectories

    manifest = plot_checkpoint_trajectories(
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _measure_h4(args: argparse.Namespace) -> int:
    from ep_predict.hardware.h4 import measure_h4

    result = measure_h4(
        load_toml(args.model_config),
        load_toml(args.experiment_config),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _analyze_h4(args: argparse.Namespace) -> int:
    from ep_predict.analysis.h4 import analyze_h4

    summary = analyze_h4(args.run, load_toml(args.config))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


def _plot_h4(args: argparse.Namespace) -> int:
    from ep_predict.visualize.h4 import plot_h4

    manifest = plot_h4(
        args.run,
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _analyze_codesign_map(args: argparse.Namespace) -> int:
    from ep_predict.analysis.codesign import analyze_codesign_map

    summary = analyze_codesign_map(load_toml(args.config))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _plot_codesign_map(args: argparse.Namespace) -> int:
    from ep_predict.visualize.codesign import plot_codesign_map

    manifest = plot_codesign_map(
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _analyze_h5(args: argparse.Namespace) -> int:
    from ep_predict.analysis.h5 import analyze_h5

    summary = analyze_h5(load_toml(args.config))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


def _plot_h5(args: argparse.Namespace) -> int:
    from ep_predict.visualize.h5 import plot_h5

    manifest = plot_h5(
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _analyze_h5_admission(args: argparse.Namespace) -> int:
    from ep_predict.analysis.admission import analyze_admission

    summary = analyze_admission(load_toml(args.config))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _plot_h5_admission(args: argparse.Namespace) -> int:
    from ep_predict.visualize.admission import plot_admission

    manifest = plot_admission(
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _analyze_h6(args: argparse.Namespace) -> int:
    from ep_predict.analysis.h6 import analyze_h6

    summary = analyze_h6(load_toml(args.config))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


def _plot_h6(args: argparse.Namespace) -> int:
    from ep_predict.visualize.h6 import plot_h6

    manifest = plot_h6(
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _analyze_architecture(args: argparse.Namespace) -> int:
    from ep_predict.analysis.architecture import analyze_architecture

    summary = analyze_architecture(load_toml(args.config))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _plot_architecture(args: argparse.Namespace) -> int:
    from ep_predict.visualize.architecture import plot_architecture

    manifest = plot_architecture(
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _analyze_deadline_degradation(args: argparse.Namespace) -> int:
    from ep_predict.analysis.degradation import analyze_deadline_degradation

    summary = analyze_deadline_degradation(load_toml(args.config))
    print(
        json.dumps(
            {
                "formal_gate_passed": summary["formal_gate_passed"],
                "oracle_stop_triggered": summary["oracle_stop_triggered"],
                "headline_fcfs_candidate": summary["headline_fcfs_candidate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _plot_deadline_degradation(args: argparse.Namespace) -> int:
    from ep_predict.visualize.degradation import plot_deadline_degradation

    manifest = plot_deadline_degradation(
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _audit_artifacts(args: argparse.Namespace) -> int:
    from ep_predict.artifacts import audit_artifacts

    report = audit_artifacts(
        args.repo,
        require_tracked=args.require_tracked,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["state"] == "complete" else 1


def _measure_q1(args: argparse.Namespace) -> int:
    from ep_predict.hardware.q1 import measure_q1

    result = measure_q1(
        load_toml(args.model_config),
        load_toml(args.experiment_config),
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _analyze_q1(args: argparse.Namespace) -> int:
    from ep_predict.analysis.q1 import analyze_q1

    summary = analyze_q1(load_toml(args.config))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


def _plot_q1(args: argparse.Namespace) -> int:
    from ep_predict.visualize.q1 import plot_q1

    manifest = plot_q1(
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _measure_q1_tail(args: argparse.Namespace) -> int:
    from ep_predict.hardware.q1 import measure_q1_tail

    result = measure_q1_tail(
        load_toml(args.model_config),
        load_toml(args.experiment_config),
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _analyze_q1_tail(args: argparse.Namespace) -> int:
    from ep_predict.analysis.q1 import analyze_q1_tail

    summary = analyze_q1_tail(load_toml(args.config))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


def _plot_q1_tail(args: argparse.Namespace) -> int:
    from ep_predict.visualize.q1 import plot_q1_tail

    manifest = plot_q1_tail(
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _measure_q1b(args: argparse.Namespace) -> int:
    from ep_predict.hardware.q1 import measure_q1b

    result = measure_q1b(
        load_toml(args.model_config),
        load_toml(args.experiment_config),
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _analyze_q1b(args: argparse.Namespace) -> int:
    from ep_predict.analysis.q1 import analyze_q1b

    summary = analyze_q1b(load_toml(args.config))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


def _plot_q1b(args: argparse.Namespace) -> int:
    from ep_predict.visualize.q1 import plot_q1b

    manifest = plot_q1b(
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _measure_q2(args: argparse.Namespace) -> int:
    from ep_predict.hardware.q2 import measure_q2

    result = measure_q2(
        load_toml(args.model_config),
        load_toml(args.experiment_config),
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _analyze_q2(args: argparse.Namespace) -> int:
    from ep_predict.analysis.q2 import analyze_q2

    summary = analyze_q2(load_toml(args.config))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _plot_q2(args: argparse.Namespace) -> int:
    from ep_predict.visualize.q2 import plot_q2

    manifest = plot_q2(
        load_toml(args.config),
        output_dir=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ep-predict",
        description="Hook-based MoE routing research tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="load a model and report router/expert structure"
    )
    inspect_parser.add_argument("--config", required=True, type=Path)
    inspect_parser.add_argument("--output", type=Path)
    inspect_parser.set_defaults(function=_inspect)

    dataset_parser = subparsers.add_parser(
        "prepare-dataset",
        help="materialize a revision-pinned standard trace workload",
    )
    dataset_parser.add_argument("--config", required=True, type=Path)
    dataset_parser.set_defaults(function=_prepare_dataset)

    collect_parser = subparsers.add_parser(
        "collect", help="collect resumable request-level routing traces"
    )
    collect_parser.add_argument("--model-config", required=True, type=Path)
    collect_parser.add_argument("--experiment-config", required=True, type=Path)
    collect_parser.add_argument(
        "--limit", type=int, help="run only the first N prompts for a smoke test"
    )
    collect_parser.set_defaults(function=_collect)

    analyze_parser = subparsers.add_parser(
        "analyze-h1", help="compute H1 skew and hot-set stability metrics"
    )
    analyze_parser.add_argument("--run", required=True, type=Path)
    analyze_parser.add_argument("--config", required=True, type=Path)
    analyze_parser.set_defaults(function=_analyze_h1)

    plot_parser = subparsers.add_parser(
        "plot-h1", help="generate publication-style H1 figures"
    )
    plot_parser.add_argument("--run", required=True, type=Path)
    plot_parser.add_argument("--config", required=True, type=Path)
    plot_parser.add_argument(
        "--output",
        type=Path,
        help="figure directory (default: RUN/analysis/h1/figures)",
    )
    plot_parser.set_defaults(function=_plot_h1)

    h2_parser = subparsers.add_parser(
        "analyze-h2",
        help="evaluate held-out routing-conditioned future-expert baselines",
    )
    h2_parser.add_argument("--run", required=True, type=Path)
    h2_parser.add_argument("--config", required=True, type=Path)
    h2_parser.set_defaults(function=_analyze_h2)

    h2_plot_parser = subparsers.add_parser(
        "plot-h2", help="generate publication-style H2 decision figures"
    )
    h2_plot_parser.add_argument("--run", required=True, type=Path)
    h2_plot_parser.add_argument("--config", required=True, type=Path)
    h2_plot_parser.add_argument(
        "--output",
        type=Path,
        help="figure directory (default: RUN/analysis/h2/figures)",
    )
    h2_plot_parser.set_defaults(function=_plot_h2)

    h3_parser = subparsers.add_parser(
        "analyze-h3",
        help="train and evaluate the held-out linear hidden-state predictor",
    )
    h3_parser.add_argument("--run", required=True, type=Path)
    h3_parser.add_argument("--config", required=True, type=Path)
    h3_parser.set_defaults(function=_analyze_h3)

    h3_plot_parser = subparsers.add_parser(
        "plot-h3", help="generate publication-style H3 decision figures"
    )
    h3_plot_parser.add_argument("--run", required=True, type=Path)
    h3_plot_parser.add_argument("--config", required=True, type=Path)
    h3_plot_parser.add_argument(
        "--output",
        type=Path,
        help="figure directory (default: RUN/analysis/h3/figures)",
    )
    h3_plot_parser.set_defaults(function=_plot_h3)

    horizon_plot_parser = subparsers.add_parser(
        "plot-extended-horizon",
        help="plot post-hoc H2/H3 coverage through the final MoE layer",
    )
    horizon_plot_parser.add_argument("--run", required=True, type=Path)
    horizon_plot_parser.add_argument("--config", required=True, type=Path)
    horizon_plot_parser.add_argument("--output", type=Path)
    horizon_plot_parser.set_defaults(function=_plot_extended_horizon)

    checkpoint_parser = subparsers.add_parser(
        "analyze-checkpoint-trajectories",
        help="compare matched Base and Instruct routing trajectories",
    )
    checkpoint_parser.add_argument("--config", required=True, type=Path)
    checkpoint_parser.set_defaults(function=_analyze_checkpoint_trajectories)

    checkpoint_plot_parser = subparsers.add_parser(
        "plot-checkpoint-trajectories",
        help="plot matched Base versus Instruct trajectory evidence",
    )
    checkpoint_plot_parser.add_argument("--config", required=True, type=Path)
    checkpoint_plot_parser.add_argument("--output", type=Path)
    checkpoint_plot_parser.set_defaults(function=_plot_checkpoint_trajectories)

    h4_measure_parser = subparsers.add_parser(
        "measure-h4",
        help="measure hook-free decode timing and pinned-host transfers",
    )
    h4_measure_parser.add_argument("--model-config", required=True, type=Path)
    h4_measure_parser.add_argument(
        "--experiment-config", required=True, type=Path
    )
    h4_measure_parser.set_defaults(function=_measure_h4)

    h4_parser = subparsers.add_parser(
        "analyze-h4",
        help="replay exact expert demand with an oracle prefetcher",
    )
    h4_parser.add_argument("--run", required=True, type=Path)
    h4_parser.add_argument("--config", required=True, type=Path)
    h4_parser.set_defaults(function=_analyze_h4)

    h4_plot_parser = subparsers.add_parser(
        "plot-h4", help="generate the two H4 oracle decision figures"
    )
    h4_plot_parser.add_argument("--run", required=True, type=Path)
    h4_plot_parser.add_argument("--config", required=True, type=Path)
    h4_plot_parser.add_argument("--output", type=Path)
    h4_plot_parser.set_defaults(function=_plot_h4)

    codesign_parser = subparsers.add_parser(
        "analyze-codesign-map",
        help="combine H4 physical headroom with H2/H3 complete coverage",
    )
    codesign_parser.add_argument("--config", required=True, type=Path)
    codesign_parser.set_defaults(function=_analyze_codesign_map)

    codesign_plot_parser = subparsers.add_parser(
        "plot-codesign-map",
        help="plot categorical physical/prediction co-design regions",
    )
    codesign_plot_parser.add_argument("--config", required=True, type=Path)
    codesign_plot_parser.add_argument("--output", type=Path)
    codesign_plot_parser.set_defaults(function=_plot_codesign_map)

    h5_parser = subparsers.add_parser(
        "analyze-h5",
        help="run the first-order H5 requirements sweep and policy placement",
    )
    h5_parser.add_argument("--config", required=True, type=Path)
    h5_parser.set_defaults(function=_analyze_h5)

    h5_plot_parser = subparsers.add_parser(
        "plot-h5", help="plot the H5 profitability and inverse-design figures"
    )
    h5_plot_parser.add_argument("--config", required=True, type=Path)
    h5_plot_parser.add_argument("--output", type=Path)
    h5_plot_parser.set_defaults(function=_plot_h5)

    admission_parser = subparsers.add_parser(
        "analyze-h5-admission",
        help="sweep cost-sensitive admission over frozen expert scores",
    )
    admission_parser.add_argument("--config", required=True, type=Path)
    admission_parser.set_defaults(function=_analyze_h5_admission)

    admission_plot_parser = subparsers.add_parser(
        "plot-h5-admission",
        help="plot admission frontiers and useful/useless expert scores",
    )
    admission_plot_parser.add_argument("--config", required=True, type=Path)
    admission_plot_parser.add_argument("--output", type=Path)
    admission_plot_parser.set_defaults(function=_plot_h5_admission)

    h6_parser = subparsers.add_parser(
        "analyze-h6",
        help="replay prediction-guided expert residency on existing traces",
    )
    h6_parser.add_argument("--config", required=True, type=Path)
    h6_parser.set_defaults(function=_analyze_h6)

    h6_plot_parser = subparsers.add_parser(
        "plot-h6", help="plot the H6 layer/lookahead residency gain heatmap"
    )
    h6_plot_parser.add_argument("--config", required=True, type=Path)
    h6_plot_parser.add_argument("--output", type=Path)
    h6_plot_parser.set_defaults(function=_plot_h6)

    architecture_parser = subparsers.add_parser(
        "analyze-architecture",
        help="run the AX1-AX3 trace-calibrated architecture exploration",
    )
    architecture_parser.add_argument("--config", required=True, type=Path)
    architecture_parser.set_defaults(function=_analyze_architecture)

    architecture_plot_parser = subparsers.add_parser(
        "plot-architecture",
        help="plot the AX profitability, Pareto, and inverse-design figures",
    )
    architecture_plot_parser.add_argument("--config", required=True, type=Path)
    architecture_plot_parser.add_argument("--output", type=Path)
    architecture_plot_parser.set_defaults(function=_plot_architecture)

    degradation_parser = subparsers.add_parser(
        "analyze-deadline-degradation",
        help="run the AX4 weighted hard-deadline expert-erasure replay",
    )
    degradation_parser.add_argument("--config", required=True, type=Path)
    degradation_parser.set_defaults(function=_analyze_deadline_degradation)

    degradation_plot_parser = subparsers.add_parser(
        "plot-deadline-degradation",
        help="plot the AX4 latency-quality, capacity, and hardware figures",
    )
    degradation_plot_parser.add_argument("--config", required=True, type=Path)
    degradation_plot_parser.add_argument("--output", type=Path)
    degradation_plot_parser.set_defaults(function=_plot_deadline_degradation)

    q1_measure_parser = subparsers.add_parser(
        "measure-q1",
        help="run Q1 paired clean-vs-erased forward passes on the base model",
    )
    q1_measure_parser.add_argument("--model-config", required=True, type=Path)
    q1_measure_parser.add_argument(
        "--experiment-config", required=True, type=Path
    )
    q1_measure_parser.add_argument(
        "--limit",
        type=int,
        help="process only the first N token chunks for a smoke test",
    )
    q1_measure_parser.set_defaults(function=_measure_q1)

    q1_parser = subparsers.add_parser(
        "analyze-q1",
        help="apply the frozen Q1 gate and aggregate erasure scans",
    )
    q1_parser.add_argument("--config", required=True, type=Path)
    q1_parser.set_defaults(function=_analyze_q1)

    q1_plot_parser = subparsers.add_parser(
        "plot-q1",
        help="generate the Q1 quality-vs-mass and positioning figures",
    )
    q1_plot_parser.add_argument("--config", required=True, type=Path)
    q1_plot_parser.add_argument("--output", type=Path)
    q1_plot_parser.set_defaults(function=_plot_q1)

    q1_tail_measure_parser = subparsers.add_parser(
        "measure-q1-tail",
        help="run the AX4-faithful Q1 tail-erasure sweep on the base model",
    )
    q1_tail_measure_parser.add_argument("--model-config", required=True, type=Path)
    q1_tail_measure_parser.add_argument(
        "--experiment-config", required=True, type=Path
    )
    q1_tail_measure_parser.add_argument(
        "--limit",
        type=int,
        help="process only the first N token chunks for a smoke test",
    )
    q1_tail_measure_parser.set_defaults(function=_measure_q1_tail)

    q1_tail_parser = subparsers.add_parser(
        "analyze-q1-tail",
        help="apply the frozen Q1 tail gate and aggregate the tail sweep",
    )
    q1_tail_parser.add_argument("--config", required=True, type=Path)
    q1_tail_parser.set_defaults(function=_analyze_q1_tail)

    q1_tail_plot_parser = subparsers.add_parser(
        "plot-q1-tail",
        help="generate the Q1 tail-event incidence/compounding figures",
    )
    q1_tail_plot_parser.add_argument("--config", required=True, type=Path)
    q1_tail_plot_parser.add_argument("--output", type=Path)
    q1_tail_plot_parser.set_defaults(function=_plot_q1_tail)

    q1b_measure_parser = subparsers.add_parser(
        "measure-q1b",
        help="run the Q1B null-drop mechanism sweep on the base model",
    )
    q1b_measure_parser.add_argument("--model-config", required=True, type=Path)
    q1b_measure_parser.add_argument(
        "--experiment-config", required=True, type=Path
    )
    q1b_measure_parser.add_argument(
        "--limit",
        type=int,
        help="process only the first N token chunks for a smoke test",
    )
    q1b_measure_parser.set_defaults(function=_measure_q1b)

    q1b_parser = subparsers.add_parser(
        "analyze-q1b",
        help="apply the frozen Q1B null-drop gate and aggregate the scans",
    )
    q1b_parser.add_argument("--config", required=True, type=Path)
    q1b_parser.set_defaults(function=_analyze_q1b)

    q1b_plot_parser = subparsers.add_parser(
        "plot-q1b",
        help="generate the Q1B depth-additivity and mechanism figures",
    )
    q1b_plot_parser.add_argument("--config", required=True, type=Path)
    q1b_plot_parser.add_argument("--output", type=Path)
    q1b_plot_parser.set_defaults(function=_plot_q1b)

    q2_measure_parser = subparsers.add_parser(
        "measure-q2",
        help="run the Q2 stress arms (cross-domain, decode, cliff) on the base model",
    )
    q2_measure_parser.add_argument("--model-config", required=True, type=Path)
    q2_measure_parser.add_argument(
        "--experiment-config", required=True, type=Path
    )
    q2_measure_parser.add_argument(
        "--limit",
        type=int,
        help="process only the first N token chunks per arm for a smoke test",
    )
    q2_measure_parser.set_defaults(function=_measure_q2)

    q2_parser = subparsers.add_parser(
        "analyze-q2",
        help="apply the frozen Q2 stress-arm gates",
    )
    q2_parser.add_argument("--config", required=True, type=Path)
    q2_parser.set_defaults(function=_analyze_q2)

    q2_plot_parser = subparsers.add_parser(
        "plot-q2",
        help="generate the Q2 cross-domain, decode, and cliff figures",
    )
    q2_plot_parser.add_argument("--config", required=True, type=Path)
    q2_plot_parser.add_argument("--output", type=Path)
    q2_plot_parser.set_defaults(function=_plot_q2)

    audit_parser = subparsers.add_parser(
        "audit-artifacts",
        help="validate durable artifacts, hashes, references, and Git retention",
    )
    audit_parser.add_argument(
        "--repo",
        type=Path,
        default=Path("."),
        help="repository root (default: current directory)",
    )
    audit_parser.add_argument(
        "--require-tracked",
        action="store_true",
        help="fail if any durable artifact is not staged or already tracked",
    )
    audit_parser.set_defaults(function=_audit_artifacts)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
