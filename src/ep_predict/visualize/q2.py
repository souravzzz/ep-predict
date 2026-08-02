"""Q2 visualization: one primary figure per stress arm, read from the measured
analysis tables and per-arm gates."""

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
FREEBAND = "#B03A3A"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        metadata={"Creator": "ep-predict Q2 visualization"},
    )
    return [png, pdf]


def _plot_cross_domain(analysis: Path, gate: dict[str, Any], output: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    table = _read_csv(analysis / "depth_by_domain.csv")
    domains = sorted({r["domain"] for r in table})
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.18, top=0.74, wspace=0.32)

    ax = axes[0]
    for i, domain in enumerate(domains):
        dr = sorted(
            [r for r in table if r["domain"] == domain],
            key=lambda r: int(r["run_length"]),
        )
        Ls = [int(r["run_length"]) for r in dr]
        kl = [float(r["affected_mean_forward_kl"]) for r in dr]
        ax.plot(Ls, kl, color=colors[i % len(colors)], marker="o", linewidth=2.0,
                markersize=5, label=domain)
    ax.axhline(0.02, color=FREEBAND, linestyle="--", linewidth=1.1, label="free band (0.02)")
    ax.set_xlabel("Consecutive degraded layers (run length L)")
    ax.set_ylabel("Conditional-on-affected mean KL (nats)")
    ax.set_title("Is null-drop cost domain-dependent?")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(loc="upper left")

    ax = axes[1]
    for i, domain in enumerate(domains):
        dr = sorted(
            [r for r in table if r["domain"] == domain],
            key=lambda r: int(r["run_length"]),
        )
        Ls = [int(r["run_length"]) for r in dr]
        top1 = [100 * float(r["affected_top1_agreement"]) for r in dr]
        ax.plot(Ls, top1, color=colors[i % len(colors)], marker="s", linewidth=2.0,
                markersize=5, label=domain)
    ax.set_xlabel("Consecutive degraded layers (run length L)")
    ax.set_ylabel("Conditional-on-affected top-1 agreement (%)")
    ax.set_title("Top-1 stability by domain")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(loc="lower left")

    fig.suptitle(
        f"Q2-A cross-domain null-drop tolerance — {gate['decision']}",
        x=0.02, y=0.98, ha="left", fontsize=14, fontweight="bold",
    )
    fig.text(
        0.02, 0.86,
        "Repeat of the Q1-B depth sweep (L=1..8, incidence 0.009, one "
        "low-mass expert per layer, null-drop) on WikiText-2 and gsm8k-math, "
        "both from local parquet. Free band at KL 0.02 nats.",
        fontsize=9, color="#555E6B",
    )
    return _save(fig, output / "fig1_q2_cross_domain")


def _plot_decode(analysis: Path, gate: dict[str, Any], output: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    cont = _read_csv(analysis / "continuation.csv")
    run_lengths = sorted({int(r["run_length"]) for r in cont})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.18, top=0.74, wspace=0.32)

    ax = axes[0]
    for rl in run_lengths:
        rows = sorted(
            [r for r in cont if int(r["run_length"]) == rl],
            key=lambda r: int(r["step"]),
        )
        steps = [int(r["step"]) for r in rows]
        cum = [float(r["cumulative_kl"]) for r in rows]
        ax.plot(steps, cum, color=BLUE if rl == 1 else ORANGE, marker="o",
                linewidth=2.0, markersize=3, label=f"L={rl} cumulative")
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Cumulative mean clean-vs-erased KL (nats)")
    ax.set_title("Does erasure compound with decode depth?")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(loc="upper left")

    ax = axes[1]
    for rl in run_lengths:
        rows = sorted(
            [r for r in cont if int(r["run_length"]) == rl],
            key=lambda r: int(r["step"]),
        )
        steps = [int(r["step"]) for r in rows]
        skl = [float(r["step_kl"]) for r in rows]
        ax.plot(steps, skl, color=BLUE if rl == 1 else ORANGE, marker="s",
                linewidth=1.8, markersize=3, label=f"L={rl} step KL")
    ax.axhline(0.05, color=FREEBAND, linestyle="--", linewidth=1.1, label="bounded KL gate (0.05)")
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Per-step clean-vs-erased KL (nats)")
    ax.set_title("Step-wise divergence")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(loc="upper left")

    fig.suptitle(
        f"Q2-B decode compounding — {gate['decision']}",
        x=0.02, y=0.98, ha="left", fontsize=14, fontweight="bold",
    )
    fig.text(
        0.02, 0.86,
        "Clean vs erased two-stream generation from one WikiText-2 prefix, "
        "AX4 tail incidence 0.009, null-drop. Streams advance by their own "
        "argmax, so a changed token propagates — exposing compounding prefill "
        "cannot see.",
        fontsize=9, color="#555E6B",
    )
    return _save(fig, output / "fig1_q2_decode")


def _plot_cliff(analysis: Path, gate: dict[str, Any], output: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    table = [r for r in _read_csv(analysis / "cliff_surface.csv")]
    free_band = float(gate.get("free_band_max_kl", 0.02))
    axes_spec = [
        ("incidence", "Incidence (fraction of tokens erased)", "incidence", True),
        ("run_length", "Consecutive degraded layers (run length)", "run_length", False),
        ("experts_per_layer", "Experts dropped per layer", "experts_per_drop", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.18, top=0.74, wspace=0.4)

    for ax, (axis, xlabel, key, is_inc) in zip(axes, axes_spec):
        rows = [r for r in table if r["axis"] == axis]
        rows.sort(key=lambda r: int(r[key]) if not is_inc else float(r[key]))
        xs = [float(r[key]) if is_inc else int(r[key]) for r in rows]
        kl = [float(r["affected_mean_forward_kl"]) for r in rows]
        ax.plot(xs, kl, color=RED, marker="o", linewidth=2.0, markersize=5)
        ax.axhline(free_band, color=GREEN, linestyle="--", linewidth=1.1,
                   label=f"free band ({free_band})")
        if is_inc:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Conditional-on-affected mean KL (nats)")
        ax.set_title(axis)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.legend(loc="upper left")

    fig.suptitle(
        f"Q2-C cliff mapping: where null-drop stops being free — {gate['reading']}",
        x=0.02, y=0.98, ha="left", fontsize=14, fontweight="bold",
    )
    fig.text(
        0.02, 0.86,
        "Pushes erasure past AX4's nominal bound along three axes (incidence, "
        "run length, experts per layer). The free band is "
        f"conditional-on-affected KL ≤ {free_band} nats.",
        fontsize=9, color="#555E6B",
    )
    return _save(fig, output / "fig1_q2_cliff")


def _write_note(path: Path, headline: str, checks: list[str]) -> None:
    path.write_text(
        "\n".join(["# Q2 figure review", "", "## Automated headline", "",
                   headline, "", "## Human review checklist", ""]
                  + [f"- [ ] {c}" for c in checks] + [""]),
        encoding="utf-8",
    )


def plot_q2(
    experiment_config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    _style()
    outputs: list[Path] = []
    inputs: dict[str, str] = {}
    manifests: list[dict[str, Any]] = []

    if "q2_cross_domain_probe" in experiment_config:
        cfg = experiment_config["q2_cross_domain_probe"]
        analysis = Path(cfg["output_dir"])
        if (analysis / "depth_by_domain.csv").is_file() and (analysis / "gate.json").is_file():
            destination = Path(output_dir) if output_dir is not None else analysis / "figures_q2a"
            destination.mkdir(parents=True, exist_ok=True)
            gate = json.loads((analysis / "gate.json").read_text(encoding="utf-8"))
            outs = _plot_cross_domain(analysis, gate, destination)
            plt.close("all")
            _write_note(
                destination / "FIGURES.md",
                f"Q2-A decision: `{gate['decision']}`.",
                [
                    "Depth curves for each domain agree with `gate.json` per-domain verdicts.",
                    "The xi/free-band line (KL 0.02) is correct.",
                    "Single additional-domain (math) limitation accepted.",
                ],
            )
            outs.append(destination / "FIGURES.md")
            inputs[str(analysis / "depth_by_domain.csv")] = _sha256(analysis / "depth_by_domain.csv")
            inputs[str(analysis / "gate.json")] = _sha256(analysis / "gate.json")
            outputs.extend(outs)
            manifests.append({"analysis": "q2_cross_domain", "decision": gate["decision"],
                              "outputs": {str(p): _sha256(p) for p in outs}})

    if "q2_decode_probe" in experiment_config:
        cfg = experiment_config["q2_decode_probe"]
        analysis = Path(cfg["output_dir"])
        if (analysis / "continuation.csv").is_file() and (analysis / "gate.json").is_file():
            destination = Path(output_dir) if output_dir is not None else analysis / "figures_q2b"
            destination.mkdir(parents=True, exist_ok=True)
            gate = json.loads((analysis / "gate.json").read_text(encoding="utf-8"))
            outs = _plot_decode(analysis, gate, destination)
            plt.close("all")
            _write_note(
                destination / "FIGURES.md",
                f"Q2-B decision: `{gate['decision']}`.",
                [
                    "Compounding curve and step-KL agree with `gate.json` per-run_length verdicts.",
                    "The bounded-KL line (0.05) is correct.",
                    "Two-stream argmax propagation semantics accepted.",
                ],
            )
            outs.append(destination / "FIGURES.md")
            inputs[str(analysis / "continuation.csv")] = _sha256(analysis / "continuation.csv")
            inputs[str(analysis / "gate.json")] = _sha256(analysis / "gate.json")
            outputs.extend(outs)
            manifests.append({"analysis": "q2_decode", "decision": gate["decision"],
                              "outputs": {str(p): _sha256(p) for p in outs}})

    if "q2_cliff_probe" in experiment_config:
        cfg = experiment_config["q2_cliff_probe"]
        analysis = Path(cfg["output_dir"])
        if (analysis / "cliff_surface.csv").is_file() and (analysis / "gate.json").is_file():
            destination = Path(output_dir) if output_dir is not None else analysis / "figures_q2c"
            destination.mkdir(parents=True, exist_ok=True)
            gate = json.loads((analysis / "gate.json").read_text(encoding="utf-8"))
            outs = _plot_cliff(analysis, gate, destination)
            plt.close("all")
            _write_note(
                destination / "FIGURES.md",
                f"Q2-C reading: `{gate['reading']}`. AX4 nominal cell free: "
                f"`{gate['ax4_nominal']['free']}`.",
                [
                    "Cliff crossing points on each panel agree with `gate.json` cliffs.",
                    "The free-band line and log incidence axis are correct.",
                    "The margin interpretation is accepted.",
                ],
            )
            outs.append(destination / "FIGURES.md")
            inputs[str(analysis / "cliff_surface.csv")] = _sha256(analysis / "cliff_surface.csv")
            inputs[str(analysis / "gate.json")] = _sha256(analysis / "gate.json")
            outputs.extend(outs)
            manifests.append({"analysis": "q2_cliff", "reading": gate["reading"],
                              "outputs": {str(p): _sha256(p) for p in outs}})

    if not manifests:
        raise FileNotFoundError(
            "no Q2 analysis outputs found; run analyze-q2 before plot-q2"
        )
    # The cross-arm manifest belongs in the analysis base dir that holds the
    # per-arm q2_* output dirs, not in a relative per-arm subdir.
    if output_dir is not None:
        destination = Path(output_dir)
    else:
        probe_key = next(
            (
                k
                for k in ("q2_cross_domain_probe", "q2_decode_probe", "q2_cliff_probe")
                if k in experiment_config
            ),
            None,
        )
        destination = (
            Path(experiment_config[probe_key]["output_dir"]).parent
            if probe_key is not None
            else Path.cwd()
        )
    manifest = {
        "analysis": "q2_stress",
        "inputs": inputs,
        "outputs": {str(path): _sha256(path) for path in outputs},
        "arms": manifests,
        "human_review_complete": False,
    }
    write_json(destination / "figure_manifest.json", manifest)
    return manifest
