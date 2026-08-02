"""Q2 analysis: read the Q2 stress tables, apply the frozen per-arm gates, and
write a per-arm report. Analysis only — no inference.

Three arms:
- Q2-A (cross-domain): repeat the Q1-B additive depth gate per domain. GO per
  domain iff the depth curve is monotone, has no super-linear per-layer
  marginal blow-up, and keeps <=1% large divergence at the headline run length.
  Any domain that breaks is a candidate robustness target (non-gating for the
  overall decision but reported explicitly).
- Q2-B (decode): GO iff the erased continuation stays coherent: high
  clean-vs-erased token agreement, bounded mean step KL, no runaway
  late/early divergence growth.
- Q2-C (cliff): locate where conditional-on-affected KL leaves the near-free
  band and report whether it is comfortably beyond AX4's nominal bound.
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


def _monotone_non_decreasing(values: list[float], slack: float) -> bool:
    for previous, current in zip(values, values[1:], strict=False):
        if current < previous - slack:
            return False
    return True


def _per_layer_marginal(k_lo: float, k_hi: float, delta_l: int) -> float:
    if delta_l <= 0:
        return 0.0
    return (k_hi - k_lo) / delta_l


# ---------------------------------------------------------------------------
# Q2-A: cross-domain
# ---------------------------------------------------------------------------


def analyze_q2_cross_domain(analysis_dir: Path, cfg: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    table = _read_csv(analysis_dir / "depth_by_domain.csv")
    domains = sorted({r["domain"] for r in table})
    worst_l = int(gate.get("headline_run_length", 8))
    max_super_ratio = float(gate["max_superlinear_ratio"])
    large_kl = float(gate["large_divergence_kl"])
    max_large_frac = float(gate["large_divergence_max_fraction"])
    monotone_slack = float(gate.get("monotone_slack", 1e-6))

    per_domain: dict[str, Any] = {}
    for domain in domains:
        dr = [r for r in table if r["domain"] == domain]
        dr.sort(key=lambda r: int(r["run_length"]))
        kl_seq = [float(r["affected_mean_forward_kl"]) for r in dr]
        monotone = _monotone_non_decreasing(kl_seq, monotone_slack)

        marginals: list[dict[str, Any]] = []
        for lo, hi in zip(dr, dr[1:], strict=False):
            marginals.append(
                {
                    "from_l": int(lo["run_length"]),
                    "to_l": int(hi["run_length"]),
                    "per_layer_marginal_kl": _per_layer_marginal(
                        float(lo["affected_mean_forward_kl"]),
                        float(hi["affected_mean_forward_kl"]),
                        int(hi["run_length"]) - int(lo["run_length"]),
                    ),
                }
            )
        total_growth = kl_seq[-1] - kl_seq[0]
        if len(marginals) >= 2 and total_growth > 1e-3:
            first = marginals[0]["per_layer_marginal_kl"]
            last = marginals[-1]["per_layer_marginal_kl"]
            super_ratio = last / first if first > 1e-9 else float("inf")
            superlinear = super_ratio > max_super_ratio
        else:
            super_ratio = 1.0
            superlinear = False

        worst = [r for r in dr if int(r["run_length"]) == worst_l]
        worst_frac = float(worst[0]["affected_large_divergence_fraction"]) if worst else float("nan")
        large_divergence = worst_frac > max_large_frac
        per_domain[domain] = {
            "monotone_in_l": monotone,
            "superlinear_marginal_blowup": superlinear,
            "superlinear_marginal_ratio": super_ratio,
            "large_divergence_at_headline_l": large_divergence,
            f"headline_l={worst_l}_large_divergence_fraction": worst_frac,
            "headline_kl": float(dr[-1]["affected_mean_forward_kl"]),
            "tokens_affected": int(dr[-1]["tokens_affected"]),
            "depth": [
                {
                    "run_length": int(r["run_length"]),
                    "affected_mean_forward_kl": float(r["affected_mean_forward_kl"]),
                    "affected_top1_agreement": float(r["affected_top1_agreement"]),
                }
                for r in dr
            ],
        }

    broken = [
        domain
        for domain, v in per_domain.items()
        if not (v["monotone_in_l"] and not v["superlinear_marginal_blowup"]
                and not v["large_divergence_at_headline_l"])
    ]
    arm_pass = len(broken) == 0
    decision = "GO" if arm_pass else "CANDIDATE_ROBUSTNESS_TARGET"

    gate_json = {
        "hypothesis": "Q2-A",
        "decision": decision,
        "stage": "cross_domain",
        "policy": gate.get("policy", "null"),
        "positioning": gate.get("positioning", "mass_omission"),
        "primary_scope": {
            "domains": domains,
            "incidence": float(cfg["incidence"]),
            "headline_run_length": worst_l,
            "experts_per_drop": int(cfg["experts_per_drop"]),
            "conditional_on": "affected_tokens",
        },
        "thresholds": {
            "monotone_required_in_l": True,
            "max_superlinear_marginal_ratio": max_super_ratio,
            "large_divergence_kl": large_kl,
            "large_divergence_max_fraction": max_large_frac,
        },
        "per_domain": per_domain,
        "broken_domains": broken,
        "decision_note": (
            "A broken domain is a candidate robustness target, reported "
            "explicitly and non-gating for the overall Q2 decision."
        ),
    }
    write_json(analysis_dir / "gate.json", gate_json)
    _write_cross_report(analysis_dir, gate_json, table)
    return gate_json


def _write_cross_report(analysis_dir: Path, g: dict[str, Any], table: list[dict[str, str]]) -> None:
    lines = [
        "# Q2-A cross-domain tolerance result",
        "",
        f"**Decision:** `{g['decision']}`",
        "",
        "## Frozen per-domain additive-depth gate",
        "",
        f"- Cell: {g['policy']} + {g['positioning']}, one expert per degraded "
        f"layer at incidence {g['primary_scope']['incidence']} (AX4 anchor), "
        f"domain text from local parquet only (no download).",
        f"- Headline run length L = {g['primary_scope']['headline_run_length']}, "
        f"conditional on affected tokens.",
        "",
        "| domain | monotone | super-linear | large-div frac (L8) | L8 KL | n affected |",
        "|---:|:---:|:---:|---:|---:|---:|",
    ]
    for domain, v in g["per_domain"].items():
        lines.append(
            f"| {domain} | {v['monotone_in_l']} | "
            f"{v['superlinear_marginal_blowup']} "
            f"(ratio {v['superlinear_marginal_ratio']:.2f}) | "
            f"{v[f'headline_l={g['primary_scope']['headline_run_length']}_large_divergence_fraction']:.5f} | "
            f"{v['headline_kl']:.4f} | {v['tokens_affected']} |"
        )
    lines.append("")
    lines.append("### Depth sweep, conditional-on-affected KL by domain")
    lines.append("")
    lines.append("| domain | L | affected mean KL | affected top-1 |")
    lines.append("|---:|---:|---:|---:|")
    for r in sorted(table, key=lambda x: (x["domain"], int(x["run_length"]))):
        lines.append(
            f"| {r['domain']} | {int(r['run_length'])} | "
            f"{float(r['affected_mean_forward_kl']):.4f} | "
            f"{100 * float(r['affected_top1_agreement']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "Measured forward passes on the frozen OLMoE-1B-7B-0125 base "
            "checkpoint, null-drop only. The reference domain is WikiText-2 "
            "(Q1-B in-family); math is gsm8k word problems from local parquet. "
            "Code was not locally materialized, so it is not in this arm. "
            "Quality only; no latency or capacity claim.",
            "",
        ]
    )
    (analysis_dir / "CROSS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Q2-B: decode compounding
# ---------------------------------------------------------------------------


def analyze_q2_decode(analysis_dir: Path, cfg: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    cont = _read_csv(analysis_dir / "continuation.csv")
    summary_path = analysis_dir / "decode_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"run measure-q2 before analysis: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    min_agree = float(gate["min_token_agreement"])
    max_kl = float(gate["max_mean_step_kl"])
    late_early_ratio_max = float(gate["runaway_late_early_ratio"])
    run_lengths = sorted({int(r["run_length"]) for r in cont})
    eps = 1e-6

    per_rl: dict[str, Any] = {}
    for rl in run_lengths:
        rows = [r for r in cont if int(r["run_length"]) == rl]
        rows.sort(key=lambda r: int(r["step"]))
        n = len(rows)
        step_kl = [float(r["step_kl"]) for r in rows]
        agree = sum(int(r["token_agree"]) for r in rows) / n
        mean_kl = sum(step_kl) / n
        half = max(1, n // 4)
        early = sum(step_kl[:half]) / half
        late = sum(step_kl[-half:]) / half
        runaway_ratio = late / (early + eps)
        runaway = (late > max_kl) and (runaway_ratio > late_early_ratio_max)
        per_rl[str(rl)] = {
            "token_agreement": agree,
            "mean_step_kl": mean_kl,
            "final_cumulative_kl": float(rows[-1]["cumulative_kl"]),
            "early_window_mean_kl": early,
            "late_window_mean_kl": late,
            "late_early_ratio": runaway_ratio,
            "runaway": runaway,
            "pass": (agree >= min_agree) and (mean_kl <= max_kl) and (not runaway),
        }

    passed = all(v["pass"] for v in per_rl.values())
    decision = "GO" if passed else "CANDIDATE_ROBUSTNESS_TARGET"

    gate_json = {
        "hypothesis": "Q2-B",
        "decision": decision,
        "stage": "decode_compounding",
        "primary_scope": {
            "incidence": float(cfg["incidence"]),
            "run_lengths": run_lengths,
            "experts_per_drop": int(cfg["experts_per_drop"]),
            "max_new_tokens": int(cfg["max_new_tokens"]),
        },
        "thresholds": {
            "min_token_agreement": min_agree,
            "max_mean_step_kl": max_kl,
            "runaway_late_early_ratio": late_early_ratio_max,
        },
        "per_run_length": per_rl,
        "runaway_detected": any(v["runaway"] for v in per_rl.values()),
    }
    write_json(analysis_dir / "gate.json", gate_json)
    _write_decode_report(analysis_dir, gate_json, cont)
    return gate_json


def _write_decode_report(analysis_dir: Path, g: dict[str, Any], cont: list[dict[str, str]]) -> None:
    lines = [
        "# Q2-B decode compounding result",
        "",
        f"**Decision:** `{g['decision']}`",
        "",
        "## Frozen decode-coherence gate",
        "",
        f"- Clean vs erased autoregressive continuation from a shared "
        f"{g['primary_scope']['run_lengths']} prefix over "
        f"{g['primary_scope']['max_new_tokens']} steps, AX4 tail incidence "
        f"{g['primary_scope']['incidence']}, null-drop.",
        "- GO requires high token agreement, bounded mean step KL, and no "
        "runaway late-window divergence at both L=1 and L=8.",
        "",
        "| L | token agreement | mean step KL | final cum KL | late/early | runaway | pass |",
        "|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for rl, v in sorted(g["per_run_length"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| {rl} | {100 * v['token_agreement']:.1f}% | "
            f"{v['mean_step_kl']:.5f} | {v['final_cumulative_kl']:.5f} | "
            f"{v['late_early_ratio']:.2f} | {v['runaway']} | {v['pass']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "Measured two-stream generation on the frozen base checkpoint over "
            "WikiText-2, null-drop only, AX4 anchor incidence. Step KL is the "
            "clean-vs-erased next-token KL at each generation step; the streams "
            "advance by their own argmax so a changed token propagates. This "
            "detects compounding the prefill cannot see; it is not a "
            "throughput or latency claim.",
            "",
        ]
    )
    (analysis_dir / "DECODE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Q2-C: cliff mapping
# ---------------------------------------------------------------------------


def analyze_q2_cliff(analysis_dir: Path, cfg: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    table = _read_csv(analysis_dir / "cliff_surface.csv")
    free_band = float(gate["free_band_max_kl"])
    ax4_inc = float(gate.get("ax4_nominal_incidence", cfg["incidence"]))
    ax4_rl = int(gate.get("ax4_nominal_run_length", 8))
    ax4_exp = int(gate.get("ax4_nominal_experts", 1))

    def _first_cross(axis: str) -> dict[str, Any] | None:
        rows = [r for r in table if r["axis"] == axis]
        rows.sort(key=lambda r: float(r["incidence"]) if axis == "incidence"
                  else (int(r["run_length"]) if axis == "run_length" else int(r["experts_per_drop"])))
        key = "incidence" if axis == "incidence" else (
            "run_length" if axis == "run_length" else "experts_per_drop")
        for r in rows:
            if float(r["affected_mean_forward_kl"]) > free_band:
                return {
                    "axis": axis,
                    "crossing_at": float(r[key]),
                    "affected_mean_forward_kl": float(r["affected_mean_forward_kl"]),
                    "incidence": float(r["incidence"]),
                    "run_length": int(r["run_length"]),
                    "experts_per_drop": int(r["experts_per_drop"]),
                }
        return None

    def _nominal(inc, rl, exp):
        for r in table:
            if (abs(float(r["incidence"]) - inc) < 1e-9
                    and int(r["run_length"]) == rl
                    and int(r["experts_per_drop"]) == exp):
                return float(r["affected_mean_forward_kl"])
        return None

    nominal_kl = _nominal(ax4_inc, ax4_rl, ax4_exp)
    nominal_free = nominal_kl is not None and nominal_kl <= free_band

    inc_cliff = _first_cross("incidence")
    rl_cliff = _first_cross("run_length")
    exp_cliff = _first_cross("experts_per_layer")

    # Margin: how far each cliff sits from its AX4 nominal value.
    def _margin(cliff, nominal) -> float | None:
        if cliff is None:
            return None  # no crossing within the swept range
        v = cliff["crossing_at"]
        return v / nominal if nominal else None

    inc_margin = _margin(inc_cliff, ax4_inc)
    rl_margin = _margin(rl_cliff, ax4_rl) if rl_cliff else None
    exp_margin = _margin(exp_cliff, ax4_exp)

    # Interpretation.
    if not nominal_free:
        arm_reading = "INSIDE"
        interpretation = (
            "null-drop leaves the near-free band at/near the AX4 nominal "
            "bound, so a robustness target is justified to soften it."
        )
    else:
        arm_reading = "WITH_MARGIN"
        interpretation = (
            "the free band holds at the AX4 nominal cells; cliffs sit "
            "beyond them, so the measured contract stands with margin."
        )

    gate_json = {
        "hypothesis": "Q2-C",
        "stage": "cliff_mapping",
        "reading": arm_reading,
        "free_band_max_kl": free_band,
        "ax4_nominal": {
            "incidence": ax4_inc,
            "run_length": ax4_rl,
            "experts_per_drop": ax4_exp,
            "affected_mean_forward_kl": nominal_kl,
            "free": nominal_free,
        },
        "cliffs": {
            "incidence": inc_cliff,
            "run_length": rl_cliff,
            "experts_per_layer": exp_cliff,
        },
        "margins_over_ax4_nominal": {
            "incidence": inc_margin,
            "run_length": rl_margin,
            "experts_per_layer": exp_margin,
        },
        "interpretation": interpretation,
    }
    write_json(analysis_dir / "gate.json", gate_json)
    _write_cliff_report(analysis_dir, gate_json, table)
    return gate_json


def _write_cliff_report(analysis_dir: Path, g: dict[str, Any], table: list[dict[str, str]]) -> None:
    lines = [
        "# Q2-C cliff mapping result",
        "",
        f"**Reading:** `{g['reading']}`",
        "",
        "## Where null-drop stops being free",
        "",
        f"- Free band: conditional-on-affected mean KL ≤ {g['free_band_max_kl']} nats.",
        f"- AX4 nominal cell (incidence {g['ax4_nominal']['incidence']}, "
        f"L={g['ax4_nominal']['run_length']}, "
        f"{g['ax4_nominal']['experts_per_drop']} expert/layer): "
        f"affected KL = {g['ax4_nominal']['affected_mean_forward_kl']}, "
        f"free = **{g['ax4_nominal']['free']}**.",
        "",
        "| axis | first crossing | margin over AX4 nominal |",
        "|---|---:|---:|",
    ]
    axis_names = {"incidence": "incidence", "run_length": "run length", "experts_per_layer": "experts/layer"}
    for axis, label in axis_names.items():
        c = g["cliffs"][axis]
        if c is None:
            lines.append(f"| {label} | none within swept range | ∞ |")
        else:
            lines.append(
                f"| {label} | {c['crossing_at']} "
                f"(KL {c['affected_mean_forward_kl']:.3f}) | "
                f"{g['margins_over_ax4_nominal'][axis]} |"
            )
    lines.extend(
        [
            "",
            "### Cliff surface cells (conditional-on-affected)",
            "",
            "| axis | incidence | run length | experts/layer | affected KL | top-1 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in sorted(table, key=lambda x: (x["axis"], float(x["incidence"]), int(x["run_length"]))):
        lines.append(
            f"| {r['axis']} | {float(r['incidence']):.4f} | "
            f"{int(r['run_length'])} | {int(r['experts_per_drop'])} | "
            f"{float(r['affected_mean_forward_kl']):.4f} | "
            f"{100 * float(r['affected_top1_agreement']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            g["interpretation"],
            "",
            "## Evidence boundary",
            "",
            "Measured forward passes on the frozen base checkpoint, null-drop "
            "only, WikiText-2. Pushes erasure past AX4's nominal bound "
            "(incidence, run length, experts per layer) to locate the cliff. "
            "Quality only; AX4's deadline regime remains a separate contract.",
            "",
        ]
    )
    (analysis_dir / "CLIFF_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def analyze_q2(experiment_config: dict[str, Any]) -> dict[str, Any]:
    """Analyze any Q2 arms whose measurement tables exist in their configured
    output dirs. Runs each section's gate if its table is present."""
    verdicts: dict[str, Any] = {}
    if "q2_cross_domain_probe" in experiment_config:
        cfg = experiment_config["q2_cross_domain_probe"]
        analysis = Path(cfg["output_dir"])
        if (analysis / "depth_by_domain.csv").is_file():
            gate = experiment_config.get("q2_cross_domain_gate", {})
            verdicts["cross_domain"] = analyze_q2_cross_domain(analysis, cfg, gate)
    if "q2_decode_probe" in experiment_config:
        cfg = experiment_config["q2_decode_probe"]
        analysis = Path(cfg["output_dir"])
        if (analysis / "continuation.csv").is_file():
            gate = experiment_config.get("q2_decode_gate", {})
            verdicts["decode"] = analyze_q2_decode(analysis, cfg, gate)
    if "q2_cliff_probe" in experiment_config:
        cfg = experiment_config["q2_cliff_probe"]
        analysis = Path(cfg["output_dir"])
        if (analysis / "cliff_surface.csv").is_file():
            gate = experiment_config.get("q2_cliff_gate", {})
            verdicts["cliff"] = analyze_q2_cliff(analysis, cfg, gate)
    if not verdicts:
        raise FileNotFoundError(
            "no Q2 measurement tables found; run measure-q2 before analyze-q2"
        )
    return {"hypothesis": "Q2", "verdicts": verdicts}
