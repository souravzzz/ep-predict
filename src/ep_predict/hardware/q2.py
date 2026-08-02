"""Q2: availability-conditioned robustness — stress and cliff mapping.

Q1/Q1-B measured that the frozen base model tolerates AX4's null-drop tail
essentially for free (monotone-additive in depth, worst case L=8 conditional
KL ~0.013, layer-uniform, no leak) — but only in the regime actually measured
(prefill, one domain, bounded erasure). Q2 verifies that measured free
tolerance across the untested axes and locates the quality cliff. It runs three
measured, forward-pass-only stress arms that reuse the Q1-B machinery:

- Q2-A cross-domain: repeat the depth sweep on additional domains (math);
- Q2-B decode compounding: autoregressive continuation under the AX4 tail;
- Q2-C cliff mapping: push erasure past AX4's bound (incidence, run length,
  experts per layer) to locate where null-drop stops being free.

Everything is a measured forward pass on the frozen OLMoE-1B-7B-0125 base
checkpoint. No training, no model download, no second checkpoint. Null-drop
only; renormalize remains dropped as a strategy.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import torch

from ep_predict.config import config_fingerprint
from ep_predict.modeling import (
    environment_report,
    inspect_loaded_model,
    load_model_and_tokenizer,
)
from ep_predict.tracing.storage import write_json

from ep_predict.hardware.q1 import (
    MASS_OMISSION,
    NULL,
    ErasureController,
    _TailCellAccumulator,
    _load_wikitext_ids,
    _paired_metrics,
    _token_chunks,
    _write_csv,
)

_RUN_DIR = Path("artifacts/runs/q1-quality-erasure")
_DEFAULT_WIKITEXT_PARQUET = (
    "~/.cache/huggingface/hub/datasets--wikitext/snapshots/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/"
    "wikitext-2-raw-v1/validation-00000-of-00001.parquet"
)


def _load_domain_ids(
    domain: str,
    tokenizer: Any,
    token_budget: int,
    domain_sources: dict[str, Any] | None,
) -> list[int]:
    """Load a text-token domain from local (cached) data only.

    ``ref_wikitext2`` is the Q1-B in-family reference (WikiText-2 raw
    validation). Any other domain name must resolve to a local parquet path in
    ``domain_sources`` (no download). Each row's text is tokenized and
    EOS-terminated, mirroring ``_load_wikitext_ids``.
    """
    if domain == "ref_wikitext2":
        ids = _load_wikitext_ids(
            str(domain_sources.get("ref_wikitext2", _DEFAULT_WIKITEXT_PARQUET))
            if domain_sources
            else _DEFAULT_WIKITEXT_PARQUET,
            tokenizer,
        )
    else:
        sources = domain_sources or {}
        path = sources.get(domain)
        if not path:
            raise ValueError(
                f"no local parquet source for domain {domain!r}; Q2 is "
                "offline and will not download. Add its path under "
                "[domains_sources] in the Q2 config."
            )
        from datasets import Dataset

        eos = int(tokenizer.eos_token_id or tokenizer.pad_token_id or 0)
        dataset = Dataset.from_parquet(os.path.expanduser(str(path)))
        ids: list[int] = []
        for row in dataset:
            text = _domain_row_text(domain, row)
            if not text:
                continue
            ids.extend(
                int(v) for v in tokenizer(text, add_special_tokens=False)["input_ids"]
            )
            ids.append(eos)
    if token_budget:
        ids = ids[:token_budget]
    if not ids:
        raise ValueError(f"domain {domain!r} produced no tokens")
    return ids


def _domain_row_text(domain: str, row: dict[str, Any]) -> str:
    """Turn a domain-specific dataset row into displayable text."""
    if domain == "math":  # gsm8k word problems (local parquet)
        q = row.get("question")
        a = row.get("answer")
        parts = [str(q) if q is not None else "", str(a) if a is not None else ""]
        return "\n".join(p for p in parts if p)
    # Generic fallback: join all string columns in order.
    return "\n".join(
        str(v) for v in row.values() if isinstance(v, str) and v.strip()
    )


# ---------------------------------------------------------------------------
# Q2-A: cross-domain depth sweep
# ---------------------------------------------------------------------------


def _measure_cross_domain(
    model: torch.nn.Module,
    controller: ErasureController,
    tokenizer: Any,
    cfg: dict[str, Any],
    gate: dict[str, Any],
    num_layers: int,
    seed: int,
    fingerprint: str,
    limit: int | None,
    domain_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    analysis_dir = Path(cfg["output_dir"])
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _check_fingerprint(analysis_dir, fingerprint)

    policy = str(gate.get("policy", NULL))
    positioning = str(gate.get("positioning", MASS_OMISSION))
    incidence = float(cfg["incidence"])
    experts = int(cfg["experts_per_drop"])
    depth_run_lengths = [int(v) for v in cfg["depth_run_lengths"]]
    domains = [str(d) for d in cfg["domains"]]
    large_kl = float(gate["large_divergence_kl"])
    sources = dict(domain_sources) if domain_sources else {}

    rows_out: list[dict[str, Any]] = []
    summary_per_domain: dict[str, Any] = {}
    DEPTH_SEED = 1000

    # Deterministic per-domain offset (hash() is random per process, so use
    # crc32) so each domain's affected-token sample is reproducible.
    def _domain_seed(domain: str) -> int:
        import zlib

        return DEPTH_SEED + zlib.crc32(domain.encode("utf-8")) % 1000

    for domain in domains:
        ids = _load_domain_ids(domain, tokenizer, int(cfg.get("token_budget", 0)), sources)
        chunks = _token_chunks(ids, int(cfg.get("chunk_size", 512)), limit)
        depth_accs = {
            L: _TailCellAccumulator(
                policy=policy, positioning=positioning, incidence=incidence,
                run_length=L, experts_per_drop=experts,
            )
            for L in depth_run_lengths
        }
        for acc in depth_accs.values():
            acc.set_large_kl(large_kl)

        for chunk_index, chunk in enumerate(chunks):
            input_ids = chunk[:-1].unsqueeze(0).to(model.device)
            targets = chunk[1:]
            num_tokens = int(input_ids.shape[1])
            with torch.inference_mode():
                logits_c = model(input_ids=input_ids).logits.float().cpu()
            for L in depth_run_lengths:
                cell_stats = {"n_active": 0, "sum_realized_mass": 0.0, "n_erased": 0}
                controller.set_tail_cell(
                    policy=policy, positioning=positioning, incidence=incidence,
                    run_length=L, experts_per_drop=experts,
                    cell_stats=cell_stats, num_tokens=num_tokens,
                    stream_id=chunk_index, cell_seed=_domain_seed(domain),
                )
                controller.active = True
                with torch.inference_mode():
                    logits_e = model(input_ids=input_ids).logits.float().cpu()
                controller.active = False
                metrics = _paired_metrics(logits_c, logits_e, targets, large_kl)
                depth_accs[L].add(metrics, controller.affected_mask)
                depth_accs[L].add_cell_stats(cell_stats)
            del logits_c
            print(
                f"[Q2-A:{domain}] chunk {chunk_index + 1}/{len(chunks)} "
                f"({num_tokens} tokens, {len(depth_run_lengths)} L values)"
            )

        for L in depth_run_lengths:
            rows_out.append(
                {"domain": domain, **depth_accs[L].row()}
            )

        depth = [depth_accs[L].row() for L in sorted(depth_run_lengths)]
        depth.sort(key=lambda r: int(r["run_length"]))
        summary_per_domain[domain] = {
            "tokens_affected": int(depth[-1]["tokens_affected"]),
            "l8_affected_mean_forward_kl": float(depth[-1]["affected_mean_forward_kl"]),
            "l8_affected_top1_agreement": float(depth[-1]["affected_top1_agreement"]),
            "l8_affected_large_divergence_fraction": float(
                depth[-1]["affected_large_divergence_fraction"]
            ),
        }

    rows_out.sort(key=lambda r: (r["domain"], int(r["run_length"])))
    _write_csv(analysis_dir / "depth_by_domain.csv", rows_out)
    write_json(
        analysis_dir / "depth_by_domain_summary.json",
        {"domains": summary_per_domain},
    )
    return {
        "output_dir": str(analysis_dir),
        "domains": domains,
        "summary_per_domain": summary_per_domain,
    }


# ---------------------------------------------------------------------------
# Q2-B: decode compounding
# ---------------------------------------------------------------------------


def _run_q2_decode_leg(
    model: torch.nn.Module,
    controller: ErasureController,
    prefix_ids: torch.Tensor,
    *,
    incidence: float,
    run_length: int,
    experts: int,
    max_new_tokens: int,
    cell_seed: int,
) -> dict[str, Any]:
    """Autoregressive continuation comparing a clean and an erased stream from
    one shared prefix, and measuring how their predictive divergence grows with
    generation depth (compounding the prefill cannot see).

    At each step both streams advance by their own argmax, so a small erasure
    that changes one token can propagate. We record per-step clean-vs-erased
    next-token KL, top-1 token agreement, and the cumulative mean KL.
    """
    device = model.device
    clean_ids = prefix_ids.clone().to(device)
    erased_ids = prefix_ids.clone().to(device)

    def _apply(ids: torch.Tensor) -> None:
        controller.set_tail_cell(
            policy=NULL, positioning=MASS_OMISSION, incidence=incidence,
            run_length=run_length, experts_per_drop=experts,
            num_tokens=int(ids.shape[1]), cell_seed=cell_seed,
        )

    step_kl: list[float] = []
    agree: list[int] = []
    cumulative: list[float] = []
    with torch.inference_mode():
        for _step in range(max_new_tokens):
            logits_c = model(input_ids=clean_ids).logits[:, -1, :].float()
            _apply(erased_ids)
            controller.active = True
            logits_e = model(input_ids=erased_ids).logits[:, -1, :].float()
            controller.active = False

            pc = logits_c.softmax(dim=-1)
            lpc = logits_c.log_softmax(dim=-1)
            lpe = logits_e.log_softmax(dim=-1)
            kl = float((pc * (lpc - lpe)).sum(dim=-1).item())
            a = int((logits_c.argmax(dim=-1) == logits_e.argmax(dim=-1)).item())
            step_kl.append(kl)
            agree.append(a)
            cumulative.append(sum(step_kl) / len(step_kl))

            nt_c = logits_c.argmax(dim=-1).unsqueeze(-1)
            nt_e = logits_e.argmax(dim=-1).unsqueeze(-1)
            clean_ids = torch.cat([clean_ids, nt_c], dim=1)
            erased_ids = torch.cat([erased_ids, nt_e], dim=1)
    controller.active = False
    return {
        "run_length": run_length,
        "step_kl": step_kl,
        "token_agree": agree,
        "cumulative_kl": cumulative,
        # Generated token IDs for both streams (prefix + max_new_tokens), kept
        # so a human can inspect the actual output (paraphrase vs degradation)
        # without re-running. SOP: never throw away the decoded sequences.
        "clean_ids": clean_ids[0].tolist(),
        "erased_ids": erased_ids[0].tolist(),
        "prefix_ids": prefix_ids[0].tolist(),
    }


def _measure_decode(
    model: torch.nn.Module,
    controller: ErasureController,
    tokenizer: Any,
    cfg: dict[str, Any],
    gate: dict[str, Any],
    num_layers: int,
    seed: int,
    fingerprint: str,
    limit: int | None,
) -> dict[str, Any]:
    analysis_dir = Path(cfg["output_dir"])
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _check_fingerprint(analysis_dir, fingerprint)

    incidence = float(cfg["incidence"])
    experts = int(cfg["experts_per_drop"])
    run_lengths = [int(v) for v in cfg["run_lengths"]]
    max_new_tokens = int(cfg["max_new_tokens"])
    prefix_tokens = int(cfg.get("prefix_tokens", 64))
    cell_seed = int(cfg.get("cell_seed", seed + 5000))

    # Reference prefix domain: match the Q1/Q1-B WikiText-2 in-family text.
    ids = _load_domain_ids(
        "ref_wikitext2", tokenizer, int(cfg.get("token_budget", 4096)),
        dict(cfg.get("domain_sources", {})),
    )
    chunks = _token_chunks(ids, 512, limit=1)
    prefix = chunks[0][:prefix_tokens].unsqueeze(0).to(model.device)

    legs: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for run_length in run_lengths:
        leg = _run_q2_decode_leg(
            model, controller, prefix,
            incidence=incidence, run_length=run_length, experts=experts,
            max_new_tokens=max_new_tokens, cell_seed=cell_seed,
        )
        legs[run_length] = leg
        for step in range(max_new_tokens):
            rows.append(
                {
                    "run_length": run_length,
                    "step": step,
                    "step_kl": leg["step_kl"][step],
                    "token_agree": int(leg["token_agree"][step]),
                    "cumulative_kl": leg["cumulative_kl"][step],
                }
            )
        print(
            f"[Q2-B] decode leg L={run_length} complete "
            f"({max_new_tokens} steps)"
        )

    _write_csv(analysis_dir / "continuation.csv", rows)

    # SOP: persist the actual generated streams (disposable trace, gitignored)
    # so post-hoc human inspection never requires a re-run. Written under
    # <run_dir>/trace/ like other raw replay inputs.
    trace_dir = analysis_dir.parents[1] / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_lines: list[str] = []
    for rl in run_lengths:
        leg = legs[rl]
        trace_lines.append("=" * 78)
        trace_lines.append(f"RUN LENGTH L={rl}  (incidence {incidence}, {experts} expert/layer)")
        trace_lines.append("")
        trace_lines.append("SHARED PREFIX:")
        trace_lines.append(tokenizer.decode(leg["prefix_ids"], skip_special_tokens=True))
        trace_lines.append("")
        trace_lines.append("CLEAN stream (prefix + generated):")
        trace_lines.append(tokenizer.decode(leg["clean_ids"], skip_special_tokens=True))
        trace_lines.append("")
        trace_lines.append("ERASED stream (prefix + generated):")
        trace_lines.append(tokenizer.decode(leg["erased_ids"], skip_special_tokens=True))
        trace_lines.append("")
    (trace_dir / f"q2_decode_streams_{cell_seed}.txt").write_text(
        "\n".join(trace_lines), encoding="utf-8"
    )

    summary = {
        "incidence": incidence,
        "max_new_tokens": max_new_tokens,
        "prefix_tokens": prefix_tokens,
        "per_run_length": {
            str(rl): {
                "mean_step_kl": float(sum(leg["step_kl"]) / max_new_tokens),
                "final_cumulative_kl": float(leg["cumulative_kl"][-1]),
                "token_agreement": float(sum(leg["token_agree"]) / max_new_tokens),
            }
            for rl, leg in legs.items()
        },
    }
    write_json(analysis_dir / "decode_summary.json", summary)
    return {
        "output_dir": str(analysis_dir),
        "run_lengths": run_lengths,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Q2-C: cliff mapping
# ---------------------------------------------------------------------------


def _measure_cliff(
    model: torch.nn.Module,
    controller: ErasureController,
    tokenizer: Any,
    cfg: dict[str, Any],
    gate: dict[str, Any],
    num_layers: int,
    seed: int,
    fingerprint: str,
    limit: int | None,
) -> dict[str, Any]:
    analysis_dir = Path(cfg["output_dir"])
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _check_fingerprint(analysis_dir, fingerprint)

    policy = NULL
    positioning = MASS_OMISSION
    incidence = float(cfg["incidence"])
    experts = int(cfg["experts_per_drop"])
    incidence_sweep = [float(v) for v in cfg["incidence_sweep"]]
    run_length_sweep = [int(v) for v in cfg["run_length_sweep"]]
    experts_sweep = [int(v) for v in cfg["experts_per_layer_sweep"]]
    large_kl = float(gate["large_divergence_kl"])

    ids = _load_domain_ids(
        "ref_wikitext2", tokenizer, int(cfg.get("token_budget", 16384)),
        dict(cfg.get("domain_sources", {})),
    )
    chunks = _token_chunks(ids, int(cfg.get("chunk_size", 512)), limit)

    # (axis, value) -> (incidence, run_length, experts_per_drop)
    # Each axis isolates one erasure dimension at the config defaults.
    cells: list[dict[str, Any]] = []

    def _add_cells(inc, rl, exp):
        if rl < 1 or rl > num_layers:
            return
        cells.append(
            {
                "incidence": inc, "run_length": rl, "experts_per_drop": exp,
            }
        )

    base_inc, base_rl, base_exp = incidence, 8, 1
    # Incidence axis: push exposure at the AX4 worst-case run.
    for inc in incidence_sweep:
        _add_cells(inc, base_rl, base_exp)
    # Run-length axis: push past the AX4 8-layer cap at the anchor incidence.
    for rl in run_length_sweep:
        _add_cells(base_inc, rl, base_exp)
    # Experts-per-layer axis: multiple experts per layer at the worst-case run.
    for exp in experts_sweep:
        _add_cells(base_inc, base_rl, exp)

    # De-duplicate, preserving order.
    seen: set[tuple[float, int, int]] = set()
    unique_cells: list[dict[str, Any]] = []
    for c in cells:
        key = (c["incidence"], c["run_length"], c["experts_per_drop"])
        if key in seen:
            continue
        seen.add(key)
        unique_cells.append(c)

    accumulators: dict[tuple[float, int, int], _TailCellAccumulator] = {}
    for c in unique_cells:
        acc = _TailCellAccumulator(
            policy=policy, positioning=positioning, incidence=c["incidence"],
            run_length=c["run_length"], experts_per_drop=c["experts_per_drop"],
        )
        acc.set_large_kl(large_kl)
        accumulators[(c["incidence"], c["run_length"], c["experts_per_drop"])] = acc

    for chunk_index, chunk in enumerate(chunks):
        input_ids = chunk[:-1].unsqueeze(0).to(model.device)
        targets = chunk[1:]
        num_tokens = int(input_ids.shape[1])
        with torch.inference_mode():
            logits_c = model(input_ids=input_ids).logits.float().cpu()
        for c in unique_cells:
            acc = accumulators[(c["incidence"], c["run_length"], c["experts_per_drop"])]
            cell_stats = {"n_active": 0, "sum_realized_mass": 0.0, "n_erased": 0}
            controller.set_tail_cell(
                policy=policy, positioning=positioning, incidence=c["incidence"],
                run_length=c["run_length"], experts_per_drop=c["experts_per_drop"],
                cell_stats=cell_stats, num_tokens=num_tokens,
                stream_id=chunk_index, cell_seed=7000,
            )
            controller.active = True
            with torch.inference_mode():
                logits_e = model(input_ids=input_ids).logits.float().cpu()
            controller.active = False
            metrics = _paired_metrics(logits_c, logits_e, targets, large_kl)
            acc.add(metrics, controller.affected_mask)
            acc.add_cell_stats(cell_stats)
        del logits_c
        print(
            f"[Q2-C] chunk {chunk_index + 1}/{len(chunks)} "
            f"({num_tokens} tokens, {len(unique_cells)} cells)"
        )

    rows: list[dict[str, Any]] = []
    for c in unique_cells:
        r = accumulators[(c["incidence"], c["run_length"], c["experts_per_drop"])].row()
        r["axis"] = _axis_label(c["incidence"], c["run_length"], c["experts_per_drop"], base_inc, base_rl, base_exp)
        rows.append(r)
    rows.sort(
        key=lambda r: (r["axis"], r["incidence"], r["run_length"], r["experts_per_drop"])
    )
    _write_csv(analysis_dir / "cliff_surface.csv", rows)
    return {
        "output_dir": str(analysis_dir),
        "cells": len(rows),
    }


def _axis_label(inc, rl, exp, base_inc, base_rl, base_exp) -> str:
    if rl == base_rl and exp == base_exp:
        return "incidence"
    if inc == base_inc and exp == base_exp:
        return "run_length"
    if inc == base_inc and rl == base_rl:
        return "experts_per_layer"
    return "mixed"


def _check_fingerprint(analysis_dir: Path, fingerprint: str) -> None:
    probe_definition = analysis_dir / "probe_definition.json"
    if probe_definition.exists():
        existing = json.loads(probe_definition.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise RuntimeError("Q2 probe directory has a different configuration")
    else:
        write_json(
            probe_definition,
            {
                "config_fingerprint": fingerprint,
                "semantics": "null_drop_runtime_patch_exact_expert_no_renormalize",
            },
        )


def measure_q2(
    model_config: dict[str, Any],
    experiment_config: dict[str, Any],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the Q2 stress arms on the frozen base checkpoint."""
    run_dir = Path(experiment_config.get("run_dir", _RUN_DIR))
    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(model_config, {"experiment": experiment_config})

    model, tokenizer = load_model_and_tokenizer(model_config)
    model_report, _ = inspect_loaded_model(model)
    write_json(run_dir / "model_report.json", model_report)
    num_layers = int(model_report["router_count"])
    seed = int(experiment_config.get("seed", 7))
    large_kl = float(experiment_config["quality_readouts"]["large_divergence_kl"])
    gate_defaults = {
        "large_divergence_kl": large_kl,
        "large_divergence_max_fraction": float(
            experiment_config["quality_readouts"]["large_divergence_max_fraction"]
        ),
        "policy": NULL,
        "positioning": MASS_OMISSION,
    }

    results: dict[str, Any] = {}
    controller = ErasureController(model, num_layers=num_layers, seed=seed)
    with controller:
        if "q2_cross_domain_probe" in experiment_config:
            cfg = experiment_config["q2_cross_domain_probe"]
            g = {**gate_defaults, **experiment_config.get("q2_cross_domain_gate", {})}
            results["cross_domain"] = _measure_cross_domain(
                model, controller, tokenizer, cfg, g, num_layers, seed, fingerprint, limit,
                domain_sources=experiment_config.get("domains_sources"),
            )
        if "q2_decode_probe" in experiment_config:
            cfg = experiment_config["q2_decode_probe"]
            g = {**gate_defaults, **experiment_config.get("q2_decode_gate", {})}
            results["decode"] = _measure_decode(
                model, controller, tokenizer, cfg, g, num_layers, seed, fingerprint, limit,
            )
        if "q2_cliff_probe" in experiment_config:
            cfg = experiment_config["q2_cliff_probe"]
            g = {**gate_defaults, **experiment_config.get("q2_cliff_gate", {})}
            results["cliff"] = _measure_cliff(
                model, controller, tokenizer, cfg, g, num_layers, seed, fingerprint, limit,
            )

    # N.B. writes a q2-specific manifest so Q1's `run_manifest.json` (same
    # run_dir) provenance is never clobbered.
    write_json(
        run_dir / "q2_run_manifest.json",
        {
            "run_id": str(experiment_config.get("run_id", "q1-quality-erasure")),
            "stage": "q2_stress",
            "state": "complete",
            "config_fingerprint": fingerprint,
            "environment": environment_report(),
            "arms": list(results.keys()),
            "analysis_dirs": {
                arm: res.get("output_dir", "") for arm, res in results.items()
            },
        },
    )
    return {
        "state": "complete",
        "run_id": str(experiment_config.get("run_id", "q1-quality-erasure")),
        "arms": list(results.keys()),
        "results": results,
    }
