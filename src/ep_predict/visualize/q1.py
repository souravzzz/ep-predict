"""Q1 visualization: the `Delta Q vs m_missing` headline curve and the
positioning/correlation panel, read from the measured analysis tables."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ep_predict.tracing.storage import write_json

BLUE = "#3266A8"
ORANGE = "#D97732"
GREEN = "#2A8C72"
RED = "#B03A3A"
PURPLE = "#7A53A8"
GRID = "#D9DEE7"
TEXT = "#20242B"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#7A828E",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "legend.frameon": False,
        }
    )


def _save(figure: Any, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=450, bbox_inches="tight", facecolor="white")
    figure.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "ep-predict Q1 visualization"},
    )
    return [png, pdf]


def _plot_quality_curve(
    rows: list[dict[str, str]],
    mass_targets: list[float],
    gate: dict[str, Any],
    output: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharex=True)
    figure.subplots_adjust(left=0.09, right=0.96, bottom=0.2, top=0.74, wspace=0.28)
    for position, axis in enumerate(axes):
        for policy, color, marker in (
            ("renormalize", BLUE, "o"),
            ("null", ORANGE, "s"),
        ):
            xs = [
                float(
                    _row_by(
                        rows,
                        policy=policy,
                        positioning="mass_omission",
                        target_m=m,
                    )["realized_missing_mass_mean"]
                )
                for m in mass_targets
            ]
            if position == 0:
                ys = [
                    float(
                        _row_by(
                            rows,
                            policy=policy,
                            positioning="mass_omission",
                            target_m=m,
                        )["mean_forward_kl"]
                    )
                    for m in mass_targets
                ]
                ylabel = "Mean forward KL (clean ‖ erased), nats"
                title = "Forward-KL cost vs missing routed mass"
            else:
                ys = [
                    100
                    * (
                        1.0
                        - float(
                            _row_by(
                                rows,
                                policy=policy,
                                positioning="mass_omission",
                                target_m=m,
                            )["top1_agreement"]
                        )
                    )
                    for m in mass_targets
                ]
                ylabel = "Top-1 token disagreement (%)"
                title = "Top-1 drift vs missing routed mass"
            axis.plot(
                xs,
                ys,
                color=color,
                marker=marker,
                linewidth=2.1,
                markersize=5,
                label="mass-omission" if position == 0 else None,
            )
            if position == 0 and policy == "renormalize":
                axis.axvline(
                    float(
                        _row_by(
                            rows,
                            policy="renormalize",
                            positioning="mass_omission",
                            target_m=0.125,
                        )["realized_missing_mass_mean"]
                    ),
                    color=RED,
                    linestyle="--",
                    linewidth=1.1,
                    label="headline m",
                )
        axis.set_xlabel("Realized missing routed mass m_missing")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
    axes[0].legend(loc="upper left")
    gate_decision = gate["decision"]
    figure.suptitle(
        f"Quality cost of deadline expert-erasure is controlled by mass — "
        f"Q1 {gate_decision}",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.88,
        "Renormalize rescales survivors to sum 1; null drops mass with no "
        "rescaling. Paired prefill forward passes, WikiText-2 validation, "
        "frozen base checkpoint, mass-omission ordering.",
        fontsize=9,
        color="#555E6B",
    )
    return _save(figure, output / "fig1_q1_quality_vs_missing_mass")


def _plot_positioning_panel(
    rows: list[dict[str, str]],
    mass_targets: list[float],
    gate: dict[str, Any],
    output: Path,
    analysis: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    figure.subplots_adjust(left=0.09, right=0.96, bottom=0.2, top=0.74, wspace=0.28)

    # (a) Positioning within the route (renormalize), KL vs mass.
    axis = axes[0]
    for positioning, color, marker in (
        ("mass_omission", GREEN, "o"),
        ("random_within_route", PURPLE, "s"),
        ("mass_adversarial", RED, "^"),
    ):
        xs = [
            float(
                _row_by(rows, policy="renormalize", positioning=positioning, target_m=m)[
                    "realized_missing_mass_mean"
                ]
            )
            for m in mass_targets
        ]
        ys = [
            float(
                _row_by(rows, policy="renormalize", positioning=positioning, target_m=m)[
                    "mean_forward_kl"
                ]
            )
            for m in mass_targets
        ]
        axis.plot(xs, ys, color=color, marker=marker, linewidth=2.0, markersize=5, label=positioning)
    axis.set_xlabel("Realized missing routed mass")
    axis.set_ylabel("Mean forward KL (nats)")
    axis.set_title("Which experts are lost matters")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.legend(loc="upper left")

    # (b) Correlation topology at headline m (renormalize, mass-omission).
    axis = axes[1]
    correlation = {r["topology"]: r for r in _read_csv(correlation_path(analysis))}
    topologies = ["spread", "layer_burst", "consecutive_block", "scattered"]
    labels = ["spread (all layers)", "layer-burst", "token-block", "scattered"]
    kl_vals = [float(correlation[t]["mean_forward_kl"]) for t in topologies]
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    axis.bar(range(len(topologies)), kl_vals, color=colors, alpha=0.9)
    axis.set_xticks(range(len(topologies)), labels, rotation=12, ha="right")
    axis.set_ylabel("Mean forward KL (nats)")
    axis.set_title("Does depth/temporal concentration change cost?")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    for i, value in enumerate(kl_vals):
        axis.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    gate_decision = gate["decision"]
    figure.suptitle(
        f"Positioning and concentration, Q1 {gate_decision}",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.88,
        "Non-gating. (a) mass-omission vs random vs adversarial ordering under "
        "renormalize. (b) same headline m delivered per layer (spread) vs all "
        "in one layer (burst) vs temporally clumped vs scattered tokens.",
        fontsize=9,
        color="#555E6B",
    )
    return _save(figure, output / "fig2_q1_positioning_correlation")


def correlation_path(analysis: Path) -> Path:
    return analysis / "correlation_scan.csv"


def plot_q1(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    _style()
    analysis = Path(experiment_config["output_dir"])
    aggregates_path = analysis / "quality_aggregates.csv"
    gate_path = analysis / "gate.json"
    for path in (aggregates_path, gate_path, correlation_path(analysis)):
        if not path.is_file():
            raise FileNotFoundError(f"run analyze-q1 before plotting: {path}")
    destination = (
        Path(output_dir) if output_dir is not None else analysis / "figures"
    )
    destination.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(aggregates_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    mass_targets = [float(m) for m in experiment_config["probe"]["mass_targets"]]

    outputs: list[Path] = []
    outputs.extend(_plot_quality_curve(rows, mass_targets, gate, destination))
    plt.close("all")
    outputs.extend(
        _plot_positioning_panel(rows, mass_targets, gate, destination, analysis)
    )
    plt.close("all")

    note = destination / "FIGURES.md"
    note.write_text(
        "\n".join(
            [
                "# Q1 figure review",
                "",
                "## Automated headline",
                "",
                f"Formal decision: `{gate['decision']}` (stop signal "
                f"`{gate['stop_triggered']}`). Headline renormalize + "
                "mass-omission at m=0.125 gives mean forward-KL "
                f"{gate['primary']['mean_forward_kl']:.4f}, top-1 agreement "
                f"{100 * gate['primary']['top1_agreement']:.2f}%, and "
                f"perplexity ratio {gate['primary']['perplexity_ratio']:.4f} "
                f"over {gate['primary']['tokens']} tokens.",
                "",
                "## Human review checklist",
                "",
                "- [ ] The renormalize-vs-null gap (intrinsic vs recoverable "
                "cost) is visible on fig 1.",
                "- [ ] The headline m=0.125 marker and the gate thresholds are "
                "correct.",
                "- [ ] Fig 2(a) ordering (omission low, adversarial high), "
                "2(b) concentration, and `gate.json` agree.",
                "- [ ] Single-domain, layer-aggregated pilot limitation is "
                "accepted.",
                "- [ ] One next action is recorded before Q2 / AX4 hand-off.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(note)
    manifest = {
        "analysis": "q1",
        "decision": gate["decision"],
        "inputs": {
            str(aggregates_path): _sha256(aggregates_path),
            str(gate_path): _sha256(gate_path),
            str(correlation_path(analysis)): _sha256(correlation_path(analysis)),
        },
        "outputs": {str(path): _sha256(path) for path in outputs},
        "human_review_complete": False,
    }
    write_json(destination / "figure_manifest.json", manifest)
    return manifest


def plot_q1_tail(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate the tail-event sweep figures: diluted vs conditional quality
    across incidence, and run-length compounding at the AX4 anchor."""
    import matplotlib.pyplot as plt

    _style()
    analysis = Path(experiment_config["tail_output_dir"])
    sweep = analysis / "tail_sweep.csv"
    gate_path = analysis / "tail_gate.json"
    for path in (sweep, gate_path):
        if not path.is_file():
            raise FileNotFoundError(f"run analyze-q1-tail before plotting: {path}")
    destination = (
        Path(output_dir) if output_dir is not None else analysis / "figures_tail"
    )
    destination.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(sweep)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    incidences = [float(i) for i in experiment_config["tail_probe"]["incidence"]]
    run_lengths = [int(r) for r in experiment_config["tail_probe"]["run_lengths"]]
    policy = str(gate["primary_scope"]["policy"])

    def _pfx(x: str) -> str:
        return x

    # (a) Incidence sweep: conditional-on-affected vs diluted mean KL.
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    figure.subplots_adjust(left=0.09, right=0.96, bottom=0.2, top=0.78, wspace=0.3)
    axis = axes[0]
    for key, color, marker, label in (
        ("affected_mean_forward_kl", BLUE, "o", "conditional-on-affected"),
        ("overall_mean_forward_kl", ORANGE, "s", "diluted (all tokens)"),
    ):
        ys = [
            float(
                _row_tail(rows, policy, positioning=gate["primary_scope"]["positioning"], incidence=i, run_length=1)[key]
            )
            for i in incidences
        ]
        axis.plot(incidences, ys, color=color, marker=marker, linewidth=2.0, markersize=5, label=label)
    axis.axvline(float(gate["primary_scope"]["incidence"]), color=RED, linestyle="--", linewidth=1.1, label="AX4 anchor")
    axis.set_xscale("log")
    axis.set_xlabel("Incidence (fraction of tokens erased)")
    axis.set_ylabel("Mean forward KL (nats)")
    axis.set_title("Tail: diluted vs conditional quality vs incidence")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.legend(loc="upper left")

    # (b) Run-length compounding at the anchor.
    axis = axes[1]
    rs = [
        _row_tail(rows, policy, positioning=gate["primary_scope"]["positioning"], incidence=gate["primary_scope"]["incidence"], run_length=r)
        for r in run_lengths
    ]
    axis.plot(
        run_lengths,
        [float(r["affected_mean_forward_kl"]) for r in rs],
        color=GREEN, marker="^", linewidth=2.0, markersize=6,
    )
    axis.set_xlabel("Consecutive degraded layers (run length)")
    axis.set_ylabel("Conditional-on-affected mean KL (nats)")
    axis.set_title("Does compounding a drop across layers raise cost?")
    axis.grid(axis="y", color=GRID, linewidth=0.7)

    gate_decision = gate["decision"]
    figure.suptitle(
        f"Q1 tail-event erasure sweep — {gate_decision}",
        x=0.02, y=0.98, ha="left", fontsize=14, fontweight="bold",
    )
    figure.text(
        0.02, 0.88,
        "Each affected token drops exactly one expert in one layer; incidence "
        f"swept around AX4's ~0.9% degraded-wave anchor. {policy} policy, "
        "mass-omission ordering, frozen base checkpoint, WikiText-2.",
        fontsize=9, color="#555E6B",
    )
    outputs = _save(figure, destination / "fig_tail_q1_incidence_compounding")
    plt.close("all")

    note = destination / "FIGURES.md"
    note.write_text(
        "\n".join(
            [
                "# Q1 tail-event figure review",
                "",
                "## Automated headline",
                "",
                f"Formal decision: `{gate['decision']}`. One-expert one-layer "
                f"drop at incidence {gate['primary_scope']['incidence']} gives "
                f"conditional-on-affected mean KL "
                f"{gate['primary']['mean_forward_kl']:.4f}, top-1 "
                f"{100 * gate['primary']['top1_agreement']:.2f}%, PPL ratio "
                f"{gate['primary']['perplexity_ratio']:.4f} over the affected tail.",
                "",
                "## Human review checklist",
                "",
                "- [ ] The diluted-vs-conditional gap and AX4 anchor are "
                "clearly separated on (a).",
                "- [ ] Run-length compounding trend on (b) and `tail_gate.json` agree.",
                "- [ ] The tail regime is the AX4-faithful one; the mass-budget "
                "headline (universal erasure) is kept separate.",
                "- [ ] Single-domain, layer-band sampling limitation accepted.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(note)
    manifest = {
        "analysis": "q1_tail",
        "decision": gate["decision"],
        "inputs": {
            str(sweep): _sha256(sweep),
            str(gate_path): _sha256(gate_path),
        },
        "outputs": {str(path): _sha256(path) for path in outputs},
        "human_review_complete": False,
    }
    write_json(destination / "figure_manifest.json", manifest)
    return manifest


def _row_tail(rows, policy, *, positioning, incidence, run_length):
    matches = [
        r
        for r in rows
        if r["policy"] == policy
        and r["positioning"] == positioning
        and abs(float(r["incidence"]) - float(incidence)) < 1e-9
        and int(r["run_length"]) == int(run_length)
    ]
    if len(matches) != 1:
        raise ValueError(f"tail row not unique: {policy}/{positioning}/{incidence}/{run_length}")
    return matches[0]


def _read_csv_vis(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require(analysis: Path, *names: str) -> None:
    for name in names:
        if not (analysis / name).is_file():
            raise FileNotFoundError(f"run analyze-q1b before plotting: {analysis / name}")


def _plot_q1b_depth(depth_csv: Path, gate: dict[str, Any], output: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    rows = sorted(_read_csv_vis(depth_csv), key=lambda r: int(r["run_length"]))
    Ls = [int(r["run_length"]) for r in rows]
    kl = [float(r["affected_mean_forward_kl"]) for r in rows]
    top1 = [100 * float(r["affected_top1_agreement"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.18, top=0.72, wspace=0.3)

    # (a) conditional-on-affected KL vs depth, with additive reference.
    ax = axes[0]
    ax.plot(Ls, kl, color=BLUE, marker="o", linewidth=2.2, markersize=6, label="null-drop, affected")
    # Additive reference: linear from the L=1 point (cost ~ per degraded layer).
    if len(Ls) >= 2 and Ls[0] != 0:
        slope = (kl[-1] - kl[0]) / (Ls[-1] - Ls[0])
        ref = [kl[0] + slope * (L - Ls[0]) for L in Ls]
        ax.plot(Ls, ref, color="#9AA3AF", linestyle="--", linewidth=1.4, label="additive (linear) reference")
    ax.set_xlabel("Consecutive degraded layers (run length L)")
    ax.set_ylabel("Conditional-on-affected mean KL (nats)")
    ax.set_title("Is null-drop cost additive in depth?")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(loc="lower left")

    # (b) top-1 agreement vs depth.
    ax = axes[1]
    ax.plot(Ls, top1, color=GREEN, marker="s", linewidth=2.2, markersize=6)
    ax.set_xlabel("Consecutive degraded layers (run length L)")
    ax.set_ylabel("Conditional-on-affected top-1 agreement (%)")
    ax.set_title("Routing/top-1 stability vs depth")
    ax.grid(axis="y", color=GRID, linewidth=0.7)

    v = gate["verdicts"]
    fig.suptitle(
        f"null-drop depth additivity — {gate['decision']} ",
        x=0.02, y=0.98, ha="left", fontsize=14, fontweight="bold",
    )
    fig.text(
        0.02, 0.86,
        "Each affected token drops one expert (lowest mass) per layer, no "
        "renormalization; same affected sample across L. Additive-residual "
        f"hypothesis predicts KL ~ L. Monotone: {v['monotone_in_l']}; "
        f"super-linear: {v['superlinear_marginal_blowup']}.",
        fontsize=9, color="#555E6B",
    )
    return _save(fig, output / "fig1_q1b_depth_additivity")


def _plot_q1b_mechanism(
    layer_csv: Path | None,
    spacing_csv: Path | None,
    cross_csv: Path | None,
    gate: dict[str, Any],
    output: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    has_layer = layer_csv is not None and layer_csv.is_file()
    has_spacing = spacing_csv is not None and spacing_csv.is_file()
    has_cross = cross_csv is not None and cross_csv.is_file()
    panels = [has_layer, has_spacing, has_cross]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.18, top=0.74, wspace=0.45)

    if has_layer:
        rows = sorted(_read_csv_vis(layer_csv), key=lambda r: float(r["affected_mean_forward_kl"]), reverse=True)
        Ls = [int(r["layer"]) for r in rows]
        kl = [float(r["affected_mean_forward_kl"]) for r in rows]
        axes[0].bar(range(len(rows)), kl, color=[BLUE if k > max(kl) * 0.5 else "#AAB4C2" for k in kl], alpha=0.9)
        axes[0].set_xticks(range(len(rows)), Ls, rotation=0, fontsize=7)
        axes[0].set_xlabel("MoE layer (highest → lowest sensitivity)")
        axes[0].set_ylabel("One-expert one-layer affected KL (nats)")
        axes[0].set_title("Which layers are most sensitive?")
        axes[0].grid(axis="y", color=GRID, linewidth=0.7)
    else:
        axes[0].text(0.5, 0.5, "layer-order scan off", ha="center", va="center")
        axes[0].axis("off")

    if has_spacing:
        rows = sorted(_read_csv_vis(spacing_csv), key=lambda r: (int(r["run_length"]), int(r["gap"])))
        seen: set[int] = set()
        for r in rows:
            L = int(r["run_length"])
            if L in seen:
                continue
            seen.add(L)
            xs = [int(x["gap"]) for x in rows if int(x["run_length"]) == L]
            ys = [float(x["affected_mean_forward_kl"]) for x in rows if int(x["run_length"]) == L]
            axes[1].plot(xs, ys, marker="o", linewidth=2.0, markersize=5, label=f"L={L}")
        axes[1].set_xlabel("Degraded-layer gap (1 = contiguous)")
        axes[1].set_ylabel("Affected mean KL (nats)")
        axes[1].set_title("Does spacing (reconstruction) help?")
        axes[1].grid(axis="y", color=GRID, linewidth=0.7)
        axes[1].legend(loc="best")
        axes[1].set_xticks([1, 2, 4, 8])
    else:
        axes[1].text(0.5, 0.5, "spacing scan off", ha="center", va="center")
        axes[1].axis("off")

    if has_cross:
        rows = sorted(_read_csv_vis(cross_csv), key=lambda r: int(r["offset"]))
        far = next((r for r in rows if r["bucket"] == "far_control"), None)
        d_rows = [r for r in rows if int(r["offset"]) >= 1]
        xs = [int(r["offset"]) for r in d_rows]
        ys = [float(r["mean_forward_kl"]) for r in d_rows]
        axes[2].plot(xs, ys, color=PURPLE, marker="^", linewidth=2.0, markersize=5, label="leak at +d")
        if far is not None:
            axes[2].axhline(float(far["mean_forward_kl"]), color=GREEN, linestyle="--", linewidth=1.2, label="far control")
        axes[2].set_xlabel("Downstream offset from affected token (d)")
        axes[2].set_ylabel("Clean-vs-erased KL (nats)")
        axes[2].set_title("Does damage leak to other tokens?")
        axes[2].grid(axis="y", color=GRID, linewidth=0.7)
        axes[2].legend(loc="best")
    else:
        axes[2].text(0.5, 0.5, "cross-token scan off", ha="center", va="center")
        axes[2].axis("off")

    fig.suptitle(
        f"null-drop mechanism: placement and locality — {gate['decision']}",
        x=0.02, y=0.98, ha="left", fontsize=14, fontweight="bold",
    )
    fig.text(
        0.02, 0.86,
        "Non-gating. Layer order ranks which single layers must not be dropped; "
        "spacing tests reconstruction headroom; cross-token leak separates local "
        "damage from sequence-wide propagation.",
        fontsize=9, color="#555E6B",
    )
    return _save(fig, output / "fig2_q1b_mechanism")


def plot_q1b(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    _style()
    analysis = Path(experiment_config["q1b_output_dir"])
    _require(analysis, "null_gate.json", "null_depth_scan.csv")
    destination = (
        Path(output_dir) if output_dir is not None else analysis / "figures_q1b"
    )
    destination.mkdir(parents=True, exist_ok=True)

    gate = json.loads((analysis / "null_gate.json").read_text(encoding="utf-8"))
    depth_csv = analysis / "null_depth_scan.csv"
    layer_csv = analysis / "null_layer_order.csv"
    spacing_csv = analysis / "null_spacing_scan.csv"
    cross_csv = analysis / "null_cross_token.csv"

    outputs: list[Path] = []
    outputs.extend(_plot_q1b_depth(depth_csv, gate, destination))
    plt.close("all")
    outputs.extend(
        _plot_q1b_mechanism(layer_csv, spacing_csv, cross_csv, gate, destination)
    )
    plt.close("all")

    v = gate["verdicts"]
    note = destination / "FIGURES.md"
    note.write_text(
        "\n".join(
            [
                "# Q1-B null-drop figure review",
                "",
                "## Automated headline",
                "",
                f"Formal decision: `{gate['decision']}`. Depth sweep L="
                f"{gate['primary_scope']['run_lengths'][0]}→"
                f"{gate['primary_scope']['run_lengths'][-1]} at AX4 anchor "
                f"incidence {gate['primary_scope']['incidence']}: monotone "
                f"`{v['monotone_in_l']}`, super-linear blow-up "
                f"`{v['superlinear_marginal_blowup']}` "
                f"(ratio {v['superlinear_marginal_ratio']:.2f}), large-divergence "
                f"at L={gate['primary_scope']['headline_run_length']} frac "
                f"{v['headline_l_large_divergence_fraction']:.5f}.",
                "",
                "## Human review checklist",
                "",
                "- [ ] Fig 1(a): affected KL vs L and the additive (linear) "
                "reference agree with `null_gate.json`.",
                "- [ ] The monotone / super-linear verdict matches the plotted "
                "shape.",
                "- [ ] Fig 2: layer sensitivity ranking, spacing benefit, and "
                "cross-token leak are internally consistent.",
                "- [ ] Single-domain, layer-banded pilot limitation accepted.",
                "- [ ] One next action recorded before Q2 / AX4 hand-off.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(note)

    inputs = {
        str(depth_csv): _sha256(depth_csv),
        str(analysis / "null_gate.json"): _sha256(analysis / "null_gate.json"),
    }
    for name, path in (
        ("layer", layer_csv), ("spacing", spacing_csv), ("cross", cross_csv),
    ):
        if path.is_file():
            inputs[str(path)] = _sha256(path)

    manifest = {
        "analysis": "q1b_null",
        "decision": gate["decision"],
        "inputs": inputs,
        "outputs": {str(path): _sha256(path) for path in outputs},
        "human_review_complete": False,
    }
    write_json(destination / "figure_manifest.json", manifest)
    return manifest
