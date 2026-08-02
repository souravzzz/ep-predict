"""Q1 analysis: read the measured erasure tables, apply the frozen stop/go gate,
and aggregate the exploratory positioning/correlation scans.

Analysis only — no inference. The gate is applied unchanged to the primary
headline cell (renormalize + mass-omission at m=0.125), then the mass sweep and
the positioning/correlation scans are surfaced as explicitly non-gating.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_by(
    rows: list[dict[str, str]],
    *,
    policy: str,
    positioning: str,
    target_m: float,
    topology: str = "spread",
) -> dict[str, str]:
    matches = [
        r
        for r in rows
        if r["policy"] == policy
        and r["positioning"] == positioning
        and abs(float(r["target_m"]) - target_m) < 1e-9
        and r["topology"] == topology
    ]
    if len(matches) != 1:
        raise ValueError(f"Q1 cell not unique: {policy}/{positioning}/{target_m}/{topology}")
    return matches[0]


def _monotone_non_decreasing(values: list[float], slack: float) -> bool:
    for previous, current in zip(values, values[1:], strict=False):
        if current < previous - slack:
            return False
    return True


def analyze_q1(experiment_config: dict[str, Any]) -> dict[str, Any]:
    analysis_dir = Path(experiment_config["output_dir"])
    aggregates_path = analysis_dir / "quality_aggregates.csv"
    if not aggregates_path.is_file():
        raise FileNotFoundError(f"run measure-q1 before analysis: {aggregates_path}")
    rows = _read_csv(aggregates_path)

    gate = experiment_config["decision_gate"]
    headline_policy = str(gate["headline_policy"])
    headline_positioning = str(gate["headline_positioning"])
    headline_m = float(gate["headline_m"])
    max_kl = float(gate["max_mean_forward_kl"])
    min_agreement = float(gate["min_top1_agreement"])
    max_ppl_ratio = float(gate["max_ppl_ratio"])
    large_kl = float(gate["large_divergence_kl"])
    max_large_fraction = float(gate["large_divergence_max_fraction"])
    monotone_slack = float(gate.get("monotone_slack", 1e-6))

    headline_row = _row_by(
        rows,
        policy=headline_policy,
        positioning=headline_positioning,
        target_m=headline_m,
    )

    # Monotonicity of the headline policy over the mass sweep (by realized
    # mass, the honest x-axis of the `Delta Q vs m_missing` curve).
    primary_mass_rows = [
        _row_by(
            rows,
            policy=headline_policy,
            positioning=headline_positioning,
            target_m=float(target_m),
        )
        for target_m in experiment_config["probe"]["mass_targets"]
    ]
    primary_mass_rows.sort(key=lambda r: float(r["realized_missing_mass_mean"]))
    kl_sequence = [float(r["mean_forward_kl"]) for r in primary_mass_rows]
    monotone = _monotone_non_decreasing(kl_sequence, monotone_slack)

    # Kill / stop signal: "sub-1% mass causes frequent large divergence" or a
    # jumpy (non-monotone) curve.
    lowest_mass_row = primary_mass_rows[0]
    frequent_large_divergence = (
        float(lowest_mass_row["large_divergence_fraction"]) > max_large_fraction
    )
    stop_triggered = (not monotone) or frequent_large_divergence

    passed = (
        float(headline_row["mean_forward_kl"]) <= max_kl
        and float(headline_row["top1_agreement"]) >= min_agreement
        and float(headline_row["perplexity_ratio"]) <= max_ppl_ratio
        and monotone
    )
    decision = "GO" if passed else "STOP"

    gate_verdict = {
        "hypothesis": "Q1",
        "decision": decision,
        "stop_triggered": stop_triggered,
        "primary_scope": {
            "policy": headline_policy,
            "positioning": headline_positioning,
            "m_missing": headline_m,
            "topology": "spread",
        },
        "thresholds": {
            "max_mean_forward_kl": max_kl,
            "min_top1_agreement": min_agreement,
            "max_ppl_ratio": max_ppl_ratio,
            "monotone_required": True,
            "large_divergence_kl": large_kl,
            "large_divergence_max_fraction": max_large_fraction,
        },
        "primary": {
            "mean_forward_kl": float(headline_row["mean_forward_kl"]),
            "top1_agreement": float(headline_row["top1_agreement"]),
            "perplexity_ratio": float(headline_row["perplexity_ratio"]),
            "realized_missing_mass": float(
                headline_row["realized_missing_mass_mean"]
            ),
            "tokens": int(headline_row["tokens"]),
        },
        "monotonicity": {
            "monotone_non_decreasing": monotone,
            "slack": monotone_slack,
            "kl_by_realized_mass": [
                {
                    "realized_missing_mass": float(r["realized_missing_mass_mean"]),
                    "mean_forward_kl": float(r["mean_forward_kl"]),
                    "top1_agreement": float(r["top1_agreement"]),
                    "perplexity_ratio": float(r["perplexity_ratio"]),
                    "target_m": float(r["target_m"]),
                }
                for r in primary_mass_rows
            ],
        },
        "kill_signal": {
            "frequent_large_divergence_at_lowest_mass": frequent_large_divergence,
            "lowest_mass": float(lowest_mass_row["realized_missing_mass_mean"]),
            "lowest_mass_large_divergence_fraction": float(
                lowest_mass_row["large_divergence_fraction"]
            ),
        },
    }
    write_json(analysis_dir / "gate.json", gate_verdict)

    # Positioning comparison (renormalize) across the mass sweep, non-gating.
    positioning_rows: list[dict[str, Any]] = []
    for policy in (headline_policy, "null"):
        for positioning in (
            "mass_omission",
            "random_within_route",
            "mass_adversarial",
        ):
            for target_m in experiment_config["probe"]["mass_targets"]:
                r = _row_by(rows, policy=policy, positioning=positioning, target_m=float(target_m))
                positioning_rows.append(
                    {
                        "policy": policy,
                        "positioning": positioning,
                        "target_m": float(target_m),
                        "realized_missing_mass_mean": float(
                            r["realized_missing_mass_mean"]
                        ),
                        "mean_forward_kl": float(r["mean_forward_kl"]),
                        "top1_agreement": float(r["top1_agreement"]),
                        "perplexity_ratio": float(r["perplexity_ratio"]),
                    }
                )
    _write_compact_csv(
        analysis_dir / "positioning_scan.csv",
        positioning_rows,
    )

    # Correlation scan (headline m): spread vs layer-burst vs block vs scattered.
    correlation_rows = [
        {
            "topology": "spread",
            "mean_forward_kl": float(headline_row["mean_forward_kl"]),
            "top1_agreement": float(headline_row["top1_agreement"]),
            "perplexity_ratio": float(headline_row["perplexity_ratio"]),
        }
    ]
    for topology in ("layer_burst", "consecutive_block", "scattered"):
        r = _row_by(rows, policy=headline_policy, positioning=headline_positioning, target_m=headline_m, topology=topology)
        correlation_rows.append(
            {
                "topology": topology,
                "mean_forward_kl": float(r["mean_forward_kl"]),
                "top1_agreement": float(r["top1_agreement"]),
                "perplexity_ratio": float(r["perplexity_ratio"]),
            }
        )
    _write_compact_csv(analysis_dir / "correlation_scan.csv", correlation_rows)

    summary = {
        "hypothesis": "Q1",
        "decision": decision,
        "gate": gate_verdict,
        "headline_row": {k: v for k, v in headline_row.items()},
        "analysis_dir": str(analysis_dir),
    }
    write_json(analysis_dir / "summary.json", summary)

    _write_report(analysis_dir, summary, primary_mass_rows, correlation_rows)
    return summary


def _write_compact_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    analysis_dir: Path,
    summary: dict[str, Any],
    primary_mass_rows: list[dict[str, str]],
    correlation_rows: list[dict[str, Any]],
) -> None:
    g = summary["gate"]
    h = g["primary"]
    lines = [
        "# Q1 expert-erasure quality probe result",
        "",
        f"**Decision:** `{g['decision']}` (stop signal: `{g['stop_triggered']}`)",
        "",
        "## Frozen headline gate",
        "",
        f"- Primary cell: {g['primary_scope']['policy']} + "
        f"{g['primary_scope']['positioning']} at m = "
        f"{g['primary_scope']['m_missing']} (realized "
        f"{h['realized_missing_mass']:.4f}) over {h['tokens']} tokens.",
        f"- Mean forward-KL: **{h['mean_forward_kl']:.4f}** "
        f"(gate ≤ {g['thresholds']['max_mean_forward_kl']})",
        f"- Top-1 agreement: **{100 * h['top1_agreement']:.2f}%** "
        f"(gate ≥ {100 * g['thresholds']['min_top1_agreement']:.0f}%)",
        f"- Perplexity ratio: **{h['perplexity_ratio']:.4f}** "
        f"(gate ≤ {g['thresholds']['max_ppl_ratio']})",
        f"- Monotone in m: **{g['monotonicity']['monotone_non_decreasing']}**",
        "",
        "| realized m | mean KL | top-1 | PPL ratio |",
        "|---:|---:|---:|---:|",
    ]
    for r in primary_mass_rows:
        lines.append(
            f"| {float(r['realized_missing_mass_mean']):.4f} | "
            f"{float(r['mean_forward_kl']):.4f} | "
            f"{100 * float(r['top1_agreement']):.2f}% | "
            f"{float(r['perplexity_ratio']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Non-gating scans",
            "",
            "### Erase positioning (renormalize), mean forward-KL by mass",
            "",
            "| m (target) | mass-omission | random-in-route | mass-adversarial |",
            "|---:|---:|---:|---:|",
        ]
    )
    pos = _read_csv(analysis_dir / "positioning_scan.csv")
    mass_targets = sorted(
        {
            float(x["target_m"])
            for x in pos
            if x["policy"] == "renormalize"
        }
    )
    for target_m in mass_targets:
        def _kl(policy: str, positioning: str) -> float:
            for row in pos:
                if (
                    row["policy"] == policy
                    and row["positioning"] == positioning
                    and abs(float(row["target_m"]) - float(target_m)) < 1e-9
                ):
                    return float(row["mean_forward_kl"])
            return float("nan")

        lines.append(
            f"| {float(target_m):.3f} | {_kl('renormalize', 'mass_omission'):.4f} | "
            f"{_kl('renormalize', 'random_within_route'):.4f} | "
            f"{_kl('renormalize', 'mass_adversarial'):.4f} |"
        )
    lines.extend(
        [
            "",
            "### Correlation topology (headline m), mean forward-KL",
            "",
            "| topology | mean KL | top-1 | PPL ratio |",
            "|---:|---:|---:|---:|",
        ]
    )
    for r in correlation_rows:
        lines.append(
            f"| {r['topology']} | {r['mean_forward_kl']:.4f} | "
            f"{100 * r['top1_agreement']:.2f}% | {r['perplexity_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "Everything is a measured paired forward pass on the frozen "
            "OLMoE-1B-7B-0125 base checkpoint over WikiText-2 validation "
            "(prefill scope). It measures quality, not latency or capacity. "
            "Missing mass is normalized within the routed top-8 matching AX4's "
            "primary semantics. Layer/domain sensitivity is aggregated across "
            "the 16 layers and a single domain in this pilot.",
            "",
        ]
    )
    (analysis_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_q1_tail(experiment_config: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen Q1 tail gate to the tail_sweep.csv table.

    The primary cell is the AX4-faithful tail event: one expert in one layer,
    conditional on the tokens that actually suffered erasure, at the
    AX4-anchored incidence. GO requires controlled, monotone quality there;
    STOP if even a sub-1% tail produces frequent large divergence.
    """
    analysis_dir = Path(experiment_config["tail_output_dir"])
    sweep_path = analysis_dir / "tail_sweep.csv"
    if not sweep_path.is_file():
        raise FileNotFoundError(f"run measure-q1-tail before analysis: {sweep_path}")
    rows = _read_csv(sweep_path)

    gate = experiment_config["tail_gate"]
    policy = str(gate["headline_policy"])
    positioning = str(gate["headline_positioning"])
    incidence = float(gate["headline_incidence"])
    run_length = int(gate["headline_run_length"])
    experts = int(gate["experts_per_drop"])
    max_kl = float(gate["max_mean_forward_kl"])
    min_agreement = float(gate["min_top1_agreement"])
    max_ppl = float(gate["max_ppl_ratio"])
    large_kl = float(gate["large_divergence_kl"])
    max_large_fraction = float(gate["large_divergence_max_fraction"])
    monotone_slack = float(gate.get("monotone_slack", 1e-6))

    def _cell(p: str, inc: float, rl: int) -> dict[str, str]:
        matches = [
            r
            for r in rows
            if r["policy"] == p
            and r["positioning"] == positioning
            and abs(float(r["incidence"]) - inc) < 1e-9
            and int(r["run_length"]) == rl
        ]
        if len(matches) != 1:
            raise ValueError(f"tail cell not unique: {p}/{positioning}/{inc}/{rl}")
        return matches[0]

    headline = _cell(policy, incidence, run_length)

    # Incidence sweep at one layer: overall diluted view should worsen
    # (or stay flat) monotonically as more tokens are exposed.
    inc_rows = [
        _cell(policy, float(inc), 1) for inc in experiment_config["tail_probe"]["incidence"]
    ]
    inc_rows.sort(key=lambda r: float(r["incidence"]))
    overall_kl_seq = [float(r["overall_mean_forward_kl"]) for r in inc_rows]
    monotone_overall = _monotone_non_decreasing(overall_kl_seq, monotone_slack)

    # Run-length compounding at the AX4 anchor.
    run_rows = [
        _cell(policy, incidence, int(rl)) for rl in experiment_config["tail_probe"]["run_lengths"]
    ]
    run_rows.sort(key=lambda r: int(r["run_length"]))
    run_kl_seq = [float(r["affected_mean_forward_kl"]) for r in run_rows]

    # Kill signal: at the lowest swept incidence, a large-divergence tail.
    lowest_inc_row = inc_rows[0]
    frequent_large_divergence = (
        float(lowest_inc_row["affected_large_divergence_fraction"]) > max_large_fraction
        and float(lowest_inc_row["affected_mean_forward_kl"]) > large_kl
    )
    stop_triggered = (not monotone_overall) or frequent_large_divergence

    akl = float(headline["affected_mean_forward_kl"])
    atop = float(headline["affected_top1_agreement"])
    appr = float(headline["affected_perplexity_ratio"])
    passed = (
        akl <= max_kl
        and atop >= min_agreement
        and appr <= max_ppl
        and monotone_overall
    )
    decision = "GO" if passed else "STOP"

    gate_json = {
        "hypothesis": "Q1-tail",
        "decision": decision,
        "stage": "tail_event",
        "stop_triggered": stop_triggered,
        "primary_scope": {
            "policy": policy,
            "positioning": positioning,
            "incidence": incidence,
            "run_length": run_length,
            "experts_per_drop": experts,
            "conditional_on": "affected_tokens",
        },
        "thresholds": {
            "max_mean_forward_kl_affected": max_kl,
            "min_top1_agreement_affected": min_agreement,
            "max_ppl_ratio_affected": max_ppl,
            "monotone_required_incidence": True,
            "large_divergence_kl": large_kl,
            "large_divergence_max_fraction": max_large_fraction,
        },
        "primary": {
            "condition": "affected",
            "mean_forward_kl": akl,
            "top1_agreement": atop,
            "perplexity_ratio": appr,
            "realized_incidence": float(headline["realized_incidence"]),
            "tokens_total": int(headline["tokens_total"]),
            "tokens_affected": int(headline["tokens_affected"]),
        },
        "monotonicity": {
            "overall_non_decreasing_in_incidence": monotone_overall,
            "slack": monotone_slack,
            "overall_kl_by_incidence": [
                {"incidence": float(r["incidence"]),
                 "overall_mean_forward_kl": float(r["overall_mean_forward_kl"]),
                 "affected_mean_forward_kl": float(r["affected_mean_forward_kl"])}
                for r in inc_rows
            ],
        },
        "run_length_compounding": {
            "run_kl_by_run_length": [
                {"run_length": int(r["run_length"]),
                 "affected_mean_forward_kl": float(r["affected_mean_forward_kl"]),
                 "affected_top1_agreement": float(r["affected_top1_agreement"])}
                for r in run_rows
            ]
        },
        "kill_signal": {
            "frequent_large_divergence_at_lowest_mass": frequent_large_divergence,
            "lowest_incidence": float(lowest_inc_row["incidence"]),
            "lowest_incidence_affected_large_frac": float(
                lowest_inc_row["affected_large_divergence_fraction"]
            ),
        },
    }
    write_json(analysis_dir / "tail_gate.json", gate_json)

    summary = {
        "hypothesis": "Q1-tail",
        "decision": decision,
        "stage": "tail_event",
        "gate": gate_json,
        "headline_row": {k: v for k, v in headline.items()},
        "analysis_dir": str(analysis_dir),
    }
    write_json(analysis_dir / "tail_summary.json", summary)
    _write_tail_report(analysis_dir, summary, inc_rows, run_rows)
    return summary


def _write_tail_report(
    analysis_dir: Path,
    summary: dict[str, Any],
    inc_rows: list[dict[str, str]],
    run_rows: list[dict[str, str]],
) -> None:
    g = summary["gate"]
    h = g["primary"]
    lines = [
        "# Q1 tail-event (AX4-faithful) erasure probe result",
        "",
        f"**Decision:** `{g['decision']}` (stop signal: `{g['stop_triggered']}`)",
        "",
        "## Frozen tail headline gate",
        "",
        f"- Primary cell: {g['primary_scope']['policy']} + "
        f"{g['primary_scope']['positioning']}, one expert in "
        f"{g['primary_scope']['run_length']} consecutive layer(s), at incidence "
        f"{g['primary_scope']['incidence']} (AX4 anchor), conditional on the "
        f"{h['tokens_affected']}/{h['tokens_total']} affected tokens "
        f"(realized incidence {h['realized_incidence']:.4f}).",
        f"- Conditional-on-affected mean forward-KL: **{h['mean_forward_kl']:.4f}** "
        f"(gate ≤ {g['thresholds']['max_mean_forward_kl_affected']})",
        f"- Conditional-on-affected top-1 agreement: **{100 * h['top1_agreement']:.2f}%** "
        f"(gate ≥ {100 * g['thresholds']['min_top1_agreement_affected']:.0f}%)",
        f"- Conditional-on-affected perplexity ratio: **{h['perplexity_ratio']:.4f}** "
        f"(gate ≤ {g['thresholds']['max_ppl_ratio_affected']})",
        f"- Overall diluted mean KL grows monotonically in incidence: "
        f"**{g['monotonicity']['overall_non_decreasing_in_incidence']}**",
        "",
        "### Incidence sweep (one layer), conditional-on-affected vs diluted",
        "",
        "| incidence | affected mean KL | affected top-1 | diluted mean KL |",
        "|---:|---:|---:|---:|",
    ]
    for r in inc_rows:
        lines.append(
            f"| {float(r['incidence']):.4f} | {float(r['affected_mean_forward_kl']):.4f} | "
            f"{100 * float(r['affected_top1_agreement']):.2f}% | "
            f"{float(r['overall_mean_forward_kl']):.4f} |"
        )
    lines.extend(
        [
            "",
            "### Run-length compounding (AX4 anchor incidence), affected KL",
            "",
            "| run length | affected mean KL | affected top-1 |",
            "|---:|---:|---:|",
        ]
    )
    for r in run_rows:
        lines.append(
            f"| {float(r['run_length']):.0f} | {float(r['affected_mean_forward_kl']):.4f} | "
            f"{100 * float(r['affected_top1_agreement']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "Measured paired forward passes on the frozen OLMoE-1B-7B-0125 base "
            "checkpoint over WikiText-2 validation (prefill scope). Tail mode "
            "erases an exact expert count for a bounded fraction of tokens "
            "(swept by incidence) in a bounded run of consecutive layers, "
            "matching AX4's ~0.9-1% degraded-wave regime. Conditional metrics "
            "are over only the tokens that suffered erasure; diluted metrics "
            "include the untouched majority.",
            "",
        ]
    )
    (analysis_dir / "TAIL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
