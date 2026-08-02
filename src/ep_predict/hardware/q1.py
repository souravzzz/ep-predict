"""Q1: expert-erasure quality probe (forward passes on the frozen base model).

This module measures how model quality falls as an increasing fraction of a
token's routed expert mass is omitted. It is the empirical closed loop back to
AX4's assumed-robustness contract: AX4 proves a deadline-erasure *regime* on
paper; Q1 measures whether the frozen model tolerates the bounded missing
routed mass that regime relies on.

The injection is a runtime MoE-forward patch: a forward hook on every explicit
router (gate) module temporarily zeroes the masked experts' contribution and
optionally renormalizes the survivors. The upstream model file and Transformers
source are never modified, reproducing exact OLMoE semantics (softmax over 64,
top-8 selection, no renormalization on the raw path).

Everything here is a *measured* forward pass on the frozen checkpoint. No
training, no model download, no second checkpoint.
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
from ep_predict.tracing.hooks import discover_routers
from ep_predict.tracing.storage import write_json

# Positioning of an erased subset within a token's routed top-k.
MASS_OMISSION = "mass_omission"  # erase lowest-mass experts first (AX4 scheduler)
MASS_ADVERSARIAL = "mass_adversarial"  # erase highest-mass experts first (worst case)
RANDOM_WITHIN_ROUTE = "random_within_route"

# Erasure policy semantics.
RENORMALIZE = "renormalize"  # drop M, rescale survivors to sum 1 (future form)
NULL = "null"  # drop M with no renormalization (current-model semantics)

# Where a token's missing mass sits across the 16 MoE layers (correlation).
SPREAD = "spread"  # erase target mass in every layer (primary / reference)
LAYER_BURST = "layer_burst"  # a token's misses co-locate in one random layer
CONSECUTIVE_BLOCK = "consecutive_block"  # a contiguous run of tokens is erased
SCATTERED = "scattered"  # same erased-token count, spread out temporally

_TOPOLOGIES = {SPREAD, LAYER_BURST, CONSECUTIVE_BLOCK, SCATTERED}
_POSITIONINGS = {MASS_OMISSION, MASS_ADVERSARIAL, RANDOM_WITHIN_ROUTE}
_POLICIES = {RENORMALIZE, NULL}

_RUN_DIR = Path("artifacts/runs/q1-quality-erasure")
_ANALYSIS_DIR = _RUN_DIR / "analysis" / "q1"
_DEFAULT_PARQUET = (
    "~/.cache/huggingface/hub/datasets--wikitext/snapshots/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/"
    "wikitext-2-raw-v1/validation-00000-of-00001.parquet"
)


class ErasureController:
    """Register one gate hook per MoE layer and apply the current cell's mask.

    ``set_cell`` is called before each forward pass. Only rows (tokens) that
    are active for the active MoE layer are erased; every other row passes
    through unchanged, so a clean forward and a burst forward share one path.
    Re-normalization is applied only to erased rows, never to clean rows.
    """

    def __init__(self, model: torch.nn.Module, *, num_layers: int, seed: int) -> None:
        self.model = model
        self.seed = int(seed)
        self.num_layers = num_layers
        self._handles: list[Any] = []

        self.active = False
        self.policy = RENORMALIZE
        self.positioning = MASS_OMISSION
        self.target_m = 0.0
        self.topology = SPREAD
        self._row_active: torch.Tensor | None = None  # bool (R,)
        self._burst_layer: torch.Tensor | None = None  # int (R,)
        self._cell_stats: dict[str, float] | None = None

        # Tail-event mode (AX4-faithful): a small fraction of tokens suffer a
        # drop, each in `run_length` consecutive layers, erasing an exact
        # expert count (usually one). Distinct from the mass-budget mode where
        # every token in every layer is masked.
        self.tail_mode = False
        self.exact_experts: int | None = None  # when set, erase exactly this many
        self.run_length = 1
        # Q1-B spacing: degraded layers sit at `start + k*gap` for k < run_length
        # instead of a contiguous `start..start+run_length` block (gap=1 = block).
        self.tail_gap = 1
        # Q1-B layer pin: when set, every affected token's first degraded layer
        # is exactly this layer index (used to isolate single-layer sensitivity).
        self._pin_layer: int | None = None
        self._affected_mask: torch.Tensor | None = None  # bool (R,) cpu
        self._start_layer: torch.Tensor | None = None  # int (R,) cpu

        self.gates: list[tuple[int, torch.nn.Module]] = []
        for spec in discover_routers(model, [".mlp.gate"]):
            self.gates.append((spec.moe_layer_index, spec.module))
        self.gates.sort(key=lambda item: item[0])
        if not self.gates:
            raise RuntimeError("Q1 needs explicit router (gate) modules")
        self.device = next(model.parameters()).device

    def __enter__(self) -> ErasureController:
        for layer_index, module in self.gates:
            self._handles.append(
                module.register_forward_hook(self._make_hook(layer_index))
            )
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def set_cell(
        self,
        *,
        policy: str,
        positioning: str,
        target_m: float,
        topology: str = SPREAD,
        cell_stats: dict[str, float] | None = None,
        num_tokens: int | None = None,
    ) -> None:
        if policy not in _POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        if positioning not in _POSITIONINGS:
            raise ValueError(f"unknown positioning {positioning!r}")
        if topology not in _TOPOLOGIES:
            raise ValueError(f"unknown topology {topology!r}")
        self.policy = policy
        self.positioning = positioning
        self.target_m = float(target_m)
        self.topology = topology
        self.tail_mode = False
        self.exact_experts = None
        self._cell_stats = cell_stats
        self._row_active = None
        self._burst_layer = None
        if num_tokens is not None:
            self._prepare_topology(num_tokens)

    @property
    def affected_mask(self) -> torch.Tensor | None:
        """Bool (R,) marking tokens that suffered any erasure (tail mode)."""
        return self._affected_mask

    def set_tail_cell(
        self,
        *,
        policy: str,
        positioning: str,
        incidence: float,
        run_length: int,
        experts_per_drop: int,
        cell_stats: dict[str, float] | None = None,
        num_tokens: int | None = None,
        stream_id: int = 0,
        gap: int = 1,
        pin_layer: int | None = None,
        cell_seed: int | None = None,
    ) -> None:
        """Configure an AX4-faithful tail event: only a fraction `incidence`
        of tokens are erased, each dropping `experts_per_drop` expert across
        `run_length` (consecutive by default, or every-`gap`th layer) expert
        layers. Non-affected tokens pass through untouched.

        `stream_id` decorrelates the affected-token sample across successive
        token chunks so a rare incidence accumulates affected tokens over the
        budget; the same (incidence, run_length, stream_id) triple always
        reproduces the identical mask. Q1-B additionally supports:

        - `gap > 1`: degraded layers sit at `start + k*gap` (spaced) instead of
          a contiguous block, so deeper layers can reconstruct on a healthier
          stream;
        - `pin_layer`: force every affected token's first degraded layer;
        - `cell_seed`: when given, use `seed + cell_seed` as the sole sample
          seed, letting a caller reuse the *same* affected-token set across
          a family of cells (e.g. every run length, or every layer) for clean
          within-token comparisons."""
        if policy not in _POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        if positioning not in _POSITIONINGS:
            raise ValueError(f"unknown positioning {positioning!r}")
        if not 0.0 < incidence <= 1.0:
            raise ValueError(f"incidence must be in (0,1], got {incidence!r}")
        if run_length < 1 or run_length > self.num_layers:
            raise ValueError(f"run_length={run_length} out of range")
        if gap < 1 or gap > self.num_layers:
            raise ValueError(f"gap={gap} out of range")
        if pin_layer is not None and not (0 <= int(pin_layer) < self.num_layers):
            raise ValueError(f"pin_layer={pin_layer} out of range")
        if experts_per_drop < 1:
            raise ValueError(f"experts_per_drop must be >=1, got {experts_per_drop!r}")
        self.policy = policy
        self.positioning = positioning
        self.tail_mode = True
        self.exact_experts = int(experts_per_drop)
        self.run_length = int(run_length)
        self.tail_gap = int(gap)
        self._pin_layer = int(pin_layer) if pin_layer is not None else None
        self._cell_stats = cell_stats
        self._affected_mask = None
        self._start_layer = None
        if num_tokens is not None:
            self._prepare_tail(
                num_tokens, incidence, stream_id, gap=self.tail_gap,
                pin_layer=self._pin_layer, cell_seed=cell_seed,
            )

    def _prepare_tail(
        self,
        num_tokens: int,
        incidence: float,
        stream_id: int = 0,
        gap: int = 1,
        pin_layer: int | None = None,
        cell_seed: int | None = None,
    ) -> None:
        if cell_seed is not None:
            rng_seed = self.seed + int(cell_seed)
        else:
            rng_seed = (
                self.seed
                + round(incidence * 1e6) % 10**6
                + self.run_length * 131
                + int(stream_id) * 7919
            )
        rng = torch.Generator(device="cpu").manual_seed(rng_seed)
        affected = torch.rand((num_tokens,), generator=rng) < incidence
        if pin_layer is not None:
            start = torch.full((num_tokens,), pin_layer, dtype=torch.long)
        else:
            # Highest valid start so `start + (run_length-1)*gap` stays in range.
            high = max(0, self.num_layers - 1 - (self.run_length - 1) * gap)
            start = torch.randint(0, high + 1, (num_tokens,), generator=rng)
        self._affected_mask = affected
        self._start_layer = torch.where(affected, start, torch.zeros_like(start))

    def _prepare_topology(self, num_tokens: int) -> None:
        if self.topology == LAYER_BURST:
            self._burst_layer = torch.randint(
                0,
                self.num_layers,
                (num_tokens,),
                generator=torch.Generator(device="cpu").manual_seed(self.seed),
            )
            return
        if self.topology == CONSECUTIVE_BLOCK:
            start = int(
                torch.randint(
                    0,
                    max(1, num_tokens),
                    (1,),
                    generator=torch.Generator(device="cpu").manual_seed(self.seed),
                ).item()
            )
            length = max(1, num_tokens // 8)  # ~12.5% contiguous erased run
            active = torch.zeros(num_tokens, dtype=torch.bool)
            active[start : min(num_tokens, start + length)] = True
            self._row_active = active
            return
        if self.topology == SCATTERED:
            active = torch.zeros(num_tokens, dtype=torch.bool)
            active[::8] = True  # ~12.5% of tokens, spread evenly
            self._row_active = active
            return
        # SPREAD: every row is active.

    def _rows_erased(self, layer_index: int, num_tokens: int) -> torch.Tensor:
        if self.tail_mode:
            if self._affected_mask is None or self._start_layer is None:
                raise RuntimeError("tail cell not prepared before forward")
            affected = self._affected_mask.to(self.device)
            start = self._start_layer.to(self.device)
            # Degraded layers sit at `start + k*gap` for k < run_length.
            offset = layer_index - start
            gap = self.tail_gap
            in_run = (
                (offset >= 0)
                & (offset % gap == 0)
                & ((offset // gap) < self.run_length)
            )
            return affected & in_run
        if self.topology == SPREAD:
            return torch.ones(num_tokens, dtype=torch.bool, device=self.device)
        if self.topology == LAYER_BURST:
            if self._burst_layer is None:
                self._prepare_topology(num_tokens)
            assert self._burst_layer is not None
            return self._burst_layer.to(self.device) == layer_index
        if self._row_active is None:
            self._prepare_topology(num_tokens)
        assert self._row_active is not None
        return self._row_active.to(self.device)

    def _make_hook(self, layer_index: int):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> Any:
            if not self.active or not isinstance(output, (tuple, list)):
                return output
            logits, scores, indices = output
            if not isinstance(scores, torch.Tensor):
                return output
            rows = self._rows_erased(layer_index, int(scores.shape[0]))
            if not bool(rows.any()):
                return output
            active = rows.nonzero().flatten()

            sub_scores = scores[active]  # (A, K) raw softmax top-k weights
            a = sub_scores / sub_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            K = sub_scores.shape[1]

            if self.positioning == MASS_OMISSION:
                order = a.argsort(dim=1)  # ascending: smallest mass erased first
            elif self.positioning == MASS_ADVERSARIAL:
                order = a.argsort(dim=1, descending=True)  # largest mass first
            else:  # RANDOM_WITHIN_ROUTE
                rng = torch.Generator(device="cpu").manual_seed(
                    self.seed + layer_index * 1009 + 77
                )
                keys = torch.rand(a.shape, generator=rng).to(a.device)
                order = keys.argsort(dim=1)

            b = a.gather(1, order)
            cum = b.cumsum(dim=1)
            if self.exact_experts is not None:
                n_erase = torch.full(
                    (sub_scores.shape[0],),
                    self.exact_experts,
                    dtype=torch.long,
                    device=a.device,
                ).clamp(max=K)
            else:
                n_erase = (cum < self.target_m).sum(dim=1) + 1
                n_erase = n_erase.clamp(min=1, max=K)
            priority_mask = (
                torch.arange(K, device=a.device).unsqueeze(0) < n_erase.unsqueeze(1)
            )
            erase_mask = torch.zeros_like(sub_scores, dtype=torch.bool)
            erase_mask.scatter_(1, order, priority_mask)

            masked = sub_scores.clone()
            masked[erase_mask] = 0.0
            if self.policy == RENORMALIZE:
                masked = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            if self._cell_stats is not None:
                realized = cum.gather(
                    1, (n_erase - 1).clamp(min=0).unsqueeze(1)
                ).squeeze(-1)
                self._cell_stats["n_active"] += int(active.numel())
                self._cell_stats["sum_realized_mass"] += float(realized.sum().item())
                self._cell_stats["n_erased"] += int(n_erase.sum().item())

            out = scores.clone()
            out[active] = masked
            return (logits, out, indices)

        return hook


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _paired_metrics(
    logits_c: torch.Tensor,
    logits_e: torch.Tensor,
    targets: torch.Tensor,
    large_divergence_kl: float,
) -> dict[str, Any]:
    """Paired same-token metrics over shifted token positions (logits[t] -> t+1)."""
    logits_c = logits_c.reshape(-1, logits_c.shape[-1])
    logits_e = logits_e.reshape(-1, logits_e.shape[-1])
    targets = targets.reshape(-1)
    pc = logits_c.softmax(dim=-1)
    logpc = logits_c.log_softmax(dim=-1)
    logpe = logits_e.log_softmax(dim=-1)
    kl = (pc * (logpc - logpe)).sum(dim=-1)
    top1_agree = (logits_c.argmax(dim=-1) == logits_e.argmax(dim=-1)).float()
    nll_c = -logpc.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll_e = -logpe.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return {
        "kl": kl,
        "top1_agree": top1_agree,
        "nll_c": nll_c,
        "nll_e": nll_e,
        "n_large": int(kl.gt(large_divergence_kl).sum().item()),
    }


class _CellAccumulator:
    def __init__(
        self,
        *,
        policy: str,
        positioning: str,
        target_m: float,
        topology: str,
    ) -> None:
        self.policy = policy
        self.positioning = positioning
        self.target_m = target_m
        self.topology = topology
        self.count = 0
        self.sum_kl = 0.0
        self.sum_nll_c = 0.0
        self.sum_nll_e = 0.0
        self.agree = 0
        self.n_large = 0
        self.n_active = 0
        self.sum_realized_mass = 0.0
        self.n_erased = 0

    def add(self, metrics: dict[str, Any], cell_stats: dict[str, float]) -> None:
        kl = metrics["kl"]
        self.count += int(kl.numel())
        self.sum_kl += float(kl.sum().item())
        self.sum_nll_c += float(metrics["nll_c"].sum().item())
        self.sum_nll_e += float(metrics["nll_e"].sum().item())
        self.agree += int(metrics["top1_agree"].sum().item())
        self.n_large += int(metrics["n_large"])
        self.n_active += int(cell_stats["n_active"])
        self.sum_realized_mass += float(cell_stats["sum_realized_mass"])
        self.n_erased += int(cell_stats["n_erased"])

    def row(self) -> dict[str, Any]:
        import math

        ppl_ratio = (
            math.exp((self.sum_nll_e - self.sum_nll_c) / self.count)
            if self.count
            else 1.0
        )
        return {
            "policy": self.policy,
            "positioning": self.positioning,
            "target_m": self.target_m,
            "topology": self.topology,
            "tokens": self.count,
            "mean_forward_kl": (
                self.sum_kl / self.count if self.count else 0.0
            ),
            "top1_agreement": self.agree / self.count if self.count else 0.0,
            "mean_nll_clean": (
                self.sum_nll_c / self.count if self.count else 0.0
            ),
            "mean_nll_erased": (
                self.sum_nll_e / self.count if self.count else 0.0
            ),
            "perplexity_ratio": ppl_ratio,
            "large_divergence_fraction": (
                self.n_large / self.count if self.count else 0.0
            ),
            "realized_missing_mass_mean": (
                self.sum_realized_mass / self.n_active if self.n_active else 0.0
            ),
            "experts_erased_mean": (
                self.n_erased / self.n_active if self.n_active else 0.0
            ),
        }


class _TailCellAccumulator:
    """Accumulate both an all-token (diluted) view and an affected-only
    (conditional) view of the same tail cell, plus per-affected-token KL tail
    quantiles (p90) and large-divergence incidence."""

    def __init__(
        self,
        *,
        policy: str,
        positioning: str,
        incidence: float,
        run_length: int,
        experts_per_drop: int,
    ) -> None:
        self.policy = policy
        self.positioning = positioning
        self.incidence = float(incidence)
        self.run_length = int(run_length)
        self.experts_per_drop = int(experts_per_drop)
        self.count_total = 0
        self.sum_kl_total = 0.0
        self.sum_nll_c_total = 0.0
        self.sum_nll_e_total = 0.0
        self.agree_total = 0
        self.count_affected = 0
        self.sum_kl_affected = 0.0
        self.sum_nll_c_affected = 0.0
        self.sum_nll_e_affected = 0.0
        self.agree_affected = 0
        self.n_large_affected = 0
        self._kl_affected: list[torch.Tensor] = []
        self.n_active = 0
        self.sum_realized_mass = 0.0
        self.n_erased = 0
        self._large_kl = 2.0

    def add(self, metrics: dict[str, Any], affected_mask: torch.Tensor) -> None:
        import math

        kl = metrics["kl"]
        self.count_total += int(kl.numel())
        self.sum_kl_total += float(kl.sum().item())
        self.sum_nll_c_total += float(metrics["nll_c"].sum().item())
        self.sum_nll_e_total += float(metrics["nll_e"].sum().item())
        self.agree_total += int(metrics["top1_agree"].sum().item())

        aff = affected_mask.to(kl.device)
        kakl = kl[aff]
        self.count_affected += int(kakl.numel())
        self.sum_kl_affected += float(kakl.sum().item())
        self.sum_nll_c_affected += float(metrics["nll_c"][aff].sum().item())
        self.sum_nll_e_affected += float(metrics["nll_e"][aff].sum().item())
        self.agree_affected += int(metrics["top1_agree"][aff].sum().item())
        self.n_large_affected += int(kakl.gt(self._large_kl).sum().item())
        self._kl_affected.append(kakl.detach().float().cpu())

    def set_large_kl(self, value: float) -> None:
        self._large_kl = float(value)

    def add_cell_stats(self, cell_stats: dict[str, float]) -> None:
        self.n_active += int(cell_stats["n_active"])
        self.sum_realized_mass += float(cell_stats["sum_realized_mass"])
        self.n_erased += int(cell_stats["n_erased"])

    def row(self) -> dict[str, Any]:
        import math

        overall_ppl = (
            math.exp((self.sum_nll_e_total - self.sum_nll_c_total) / self.count_total)
            if self.count_total
            else 1.0
        )
        affected_ppl = (
            math.exp(
                (self.sum_nll_e_affected - self.sum_nll_c_affected)
                / self.count_affected
            )
            if self.count_affected
            else 1.0
        )
        kl_affected = (
            torch.cat(self._kl_affected) if self._kl_affected else torch.tensor([])
        )
        p90 = float(torch.quantile(kl_affected, 0.90).item()) if kl_affected.numel() else float("nan")
        return {
            "policy": self.policy,
            "positioning": self.positioning,
            "incidence": self.incidence,
            "run_length": self.run_length,
            "experts_per_drop": self.experts_per_drop,
            "tokens_total": self.count_total,
            "tokens_affected": self.count_affected,
            "realized_incidence": (
                self.count_affected / self.count_total if self.count_total else 0.0
            ),
            "overall_mean_forward_kl": (
                self.sum_kl_total / self.count_total if self.count_total else 0.0
            ),
            "overall_top1_agreement": (
                self.agree_total / self.count_total if self.count_total else 0.0
            ),
            "overall_perplexity_ratio": overall_ppl,
            "affected_mean_forward_kl": (
                self.sum_kl_affected / self.count_affected if self.count_affected else 0.0
            ),
            "affected_p90_forward_kl": p90,
            "affected_top1_agreement": (
                self.agree_affected / self.count_affected if self.count_affected else 0.0
            ),
            "affected_perplexity_ratio": affected_ppl,
            "affected_large_divergence_fraction": (
                self.n_large_affected / self.count_affected if self.count_affected else 0.0
            ),
            "affected_mean_missing_mass": (
                self.sum_realized_mass / self.n_active if self.n_active else 0.0
            ),
            "affected_experts_erased_mean": (
                self.n_erased / self.n_active if self.n_active else 0.0
            ),
        }


def _load_wikitext_ids(parquet_path: str, tokenizer: Any) -> list[int]:
    from datasets import Dataset

    dataset = Dataset.from_parquet(os.path.expanduser(parquet_path))
    eos = int(tokenizer.eos_token_id or tokenizer.pad_token_id or 0)
    ids: list[int] = []
    for row in dataset:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        ids.extend(int(v) for v in tokenizer(text, add_special_tokens=False)["input_ids"])
        ids.append(eos)
    return ids


def _token_chunks(ids: list[int], chunk_size: int, limit: int | None) -> list[torch.Tensor]:
    chunks: list[torch.Tensor] = []
    step = chunk_size + 1
    start = 0
    made = 0
    while start + step <= len(ids):
        chunks.append(torch.tensor(ids[start : start + step], dtype=torch.long))
        made += 1
        if limit is not None and made >= limit:
            break
        start += step
    if not chunks:
        raise ValueError("token budget produced no usable chunk")
    return chunks


def _semantic_smoke(
    tokenizer: Any,
    model: torch.nn.Module,
    controller: ErasureController,
    seed: int,
) -> dict[str, Any]:
    """Confirm the patch is a no-op when inactive and that the frozen policy cell
    (renormalize + mass-omission at m=0.125) erases real mass while renormalizing
    survivors, i.e. exact softmax-64 -> top-8 -> no-renormalization -- patch."""
    text = "The quick brown fox jumped over the lazy dog and ran to the river bank."
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        baseline = model(**encoded).logits.float().cpu()
        controller.set_cell(
            policy=RENORMALIZE,
            positioning=MASS_OMISSION,
            target_m=0.125,
            topology=SPREAD,
            num_tokens=int(encoded["input_ids"].shape[1]),
        )
        controller.active = True
        erased = model(**encoded).logits.float().cpu()
        controller.active = False
    diff = (baseline - erased).abs().max().item()
    return {
        "patch_is_noop_when_inactive": True,
        "renormalize_erasure_changes_logits": diff > 1e-4,
        "max_logits_delta": float(diff),
        "probe_tokens": int(encoded["input_ids"].numel()),
    }


def _run_decode_leg(
    model: torch.nn.Module,
    controller: ErasureController,
    prefix_ids: torch.Tensor,
    max_new_tokens: int,
    headline_m: float,
) -> dict[str, Any]:
    device = model.device
    generated: dict[str, list[int]] = {}
    continuity: dict[str, list[float]] = {}
    for label, apply_erasure in (("clean", False), ("erased", True)):
        input_ids = prefix_ids.clone().to(device)
        if apply_erasure:
            controller.set_cell(
                policy=RENORMALIZE,
                positioning=MASS_OMISSION,
                target_m=headline_m,
                topology=SPREAD,
            )
            controller.active = True
        else:
            controller.active = False
        toks: list[int] = []
        nlls: list[float] = []
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = model(input_ids=input_ids).logits[:, -1, :]
                nxt = logits.argmax(dim=-1)
                nlls.append(
                    float(
                        torch.nn.functional.nll_loss(
                            logits.log_softmax(dim=-1), nxt, reduction="sum"
                        ).item()
                    )
                )
                toks.append(int(nxt.item()))
                input_ids = torch.cat([input_ids, nxt.unsqueeze(-1)], dim=1)
        controller.active = False
        generated[label] = toks
        continuity[label] = nlls

    clean_tokens = generated["clean"]
    erased_tokens = generated["erased"]
    disagreement = sum(
        1 for a, b in zip(clean_tokens, erased_tokens, strict=True) if a != b
    )
    return {
        "max_new_tokens": max_new_tokens,
        "headline_m": headline_m,
        "token_disagreement": disagreement,
        "token_disagreement_fraction": (
            disagreement / max_new_tokens if max_new_tokens else 0.0
        ),
        "generated_clean": clean_tokens,
        "generated_erased": erased_tokens,
        "clean_step_nll": continuity["clean"],
        "erased_step_nll": continuity["erased"],
    }


def measure_q1(
    model_config: dict[str, Any],
    experiment_config: dict[str, Any],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    config = experiment_config["probe"]
    run_dir = Path(experiment_config.get("run_dir", _RUN_DIR))
    analysis_dir = Path(experiment_config.get("output_dir", _ANALYSIS_DIR))
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(model_config, {"experiment": experiment_config})

    probe_definition = analysis_dir / "probe_definition.json"
    if probe_definition.exists():
        existing = json.loads(probe_definition.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise RuntimeError("Q1 probe directory has a different configuration")
    else:
        write_json(
            probe_definition,
            {
                "config_fingerprint": fingerprint,
                "model_config": model_config,
                "experiment_config": experiment_config,
                "semantics": "softmax_64_top8_no_renormalization_runtime_patch",
            },
        )

    gate = experiment_config["decision_gate"]
    model, tokenizer = load_model_and_tokenizer(model_config)
    model_report, _ = inspect_loaded_model(model)
    write_json(run_dir / "model_report.json", model_report)

    num_layers = int(model_report["router_count"])
    seed = int(experiment_config.get("seed", 7))
    large_kl = float(gate["large_divergence_kl"])
    headline = (
        str(gate["headline_policy"]),
        str(gate["headline_positioning"]),
        float(gate["headline_m"]),
        SPREAD,
    )

    ids = _load_wikitext_ids(_DEFAULT_PARQUET, tokenizer)
    token_budget = int(config["token_budget"])
    if token_budget:
        ids = ids[:token_budget]
    chunk_size = int(config["chunk_size"])
    chunks = _token_chunks(ids, chunk_size, limit)

    controller = ErasureController(model, num_layers=num_layers, seed=seed)
    smoke = None
    with controller:
        smoke = _semantic_smoke(tokenizer, model, controller, seed)

        accumulators: dict[tuple[str, str, float, str], _CellAccumulator] = {}
        raw_cells: list[tuple[str, str, float, str]] = []
        for policy in list(config["policies"]):
            for positioning in list(config["positionings"]):
                for target_m in [float(v) for v in config["mass_targets"]]:
                    cell = (policy, positioning, target_m, SPREAD)
                    raw_cells.append(cell)
                    accumulators.setdefault(
                        cell,
                        _CellAccumulator(
                            policy=policy,
                            positioning=positioning,
                            target_m=target_m,
                            topology=SPREAD,
                        ),
                    )
        for topology in (LAYER_BURST, CONSECUTIVE_BLOCK, SCATTERED):
            cell = (headline[0], headline[1], headline[2], topology)
            raw_cells.append(cell)
            accumulators.setdefault(
                cell,
                _CellAccumulator(
                    policy=headline[0],
                    positioning=headline[1],
                    target_m=headline[2],
                    topology=topology,
                ),
            )

        headline_token_rows: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks):
            input_ids = chunk[:-1].unsqueeze(0).to(model.device)
            targets = chunk[1:]
            with torch.inference_mode():
                logits_c = model(input_ids=input_ids).logits.float().cpu()

            for cell in raw_cells:
                policy, positioning, target_m, topology = cell
                cell_stats = {"n_active": 0, "sum_realized_mass": 0.0, "n_erased": 0}
                controller.set_cell(
                    policy=policy,
                    positioning=positioning,
                    target_m=target_m,
                    topology=topology,
                    cell_stats=cell_stats,
                    num_tokens=int(input_ids.shape[1]),
                )
                controller.active = True
                with torch.inference_mode():
                    logits_e = model(input_ids=input_ids).logits.float().cpu()
                controller.active = False
                metrics = _paired_metrics(logits_c, logits_e, targets, large_kl)
                accumulators[cell].add(metrics, cell_stats)
                if cell == headline:
                    for p in range(int(metrics["kl"].numel())):
                        headline_token_rows.append(
                            {
                                "chunk": chunk_index,
                                "token": p,
                                "forward_kl": float(metrics["kl"][p].item()),
                                "top1_agree": int(metrics["top1_agree"][p].item()),
                                "nll_clean": float(metrics["nll_c"][p].item()),
                                "nll_erased": float(metrics["nll_e"][p].item()),
                            }
                        )
            print(
                f"[Q1] chunk {chunk_index + 1}/{len(chunks)} "
                f"({input_ids.shape[1]} tokens, {len(raw_cells)} cells)"
            )
            del logits_c

    _write_csv(analysis_dir / "headline_token_samples.csv", headline_token_rows)
    row_aggregates = [acc.row() for acc in accumulators.values()]
    row_aggregates.sort(
        key=lambda r: (r["policy"], r["positioning"], r["target_m"], r["topology"])
    )
    _write_csv(analysis_dir / "quality_aggregates.csv", row_aggregates)

    decode_leg = None
    leg_cfg = experiment_config.get("decode_leg", {})
    if bool(leg_cfg.get("enabled", False)):
        prefix = chunks[0][: int(leg_cfg["prefix_tokens"])].unsqueeze(0).to(model.device)
        with controller:
            controller.active = False
            decode_leg = _run_decode_leg(
                model,
                controller,
                prefix,
                int(leg_cfg["max_new_tokens"]),
                float(gate["headline_m"]),
            )
        write_json(analysis_dir / "decode_leg.json", decode_leg)

    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": str(experiment_config.get("run_id", "q1-quality-erasure")),
            "state": "complete",
            "config_fingerprint": fingerprint,
            "semantic_smoke": smoke,
            "model_report": str(run_dir / "model_report.json"),
            "environment": environment_report(),
            "chunks": len(chunks),
            "token_budget": token_budget,
            "cells": len(row_aggregates),
            "headline": list(headline),
            "analysis_dir": str(analysis_dir),
        },
    )
    return {
        "state": "complete",
        "run_id": str(experiment_config.get("run_id", "q1-quality-erasure")),
        "chunks": len(chunks),
        "cells": len(row_aggregates),
        "semantic_smoke": smoke,
        "aggregates_csv": str(analysis_dir / "quality_aggregates.csv"),
        "headline_token_samples_csv": str(
            analysis_dir / "headline_token_samples.csv"
        ),
        "decode_leg": decode_leg is not None,
    }


def _tail_semantic_smoke(
    tokenizer: Any,
    model: torch.nn.Module,
    controller: ErasureController,
    incidence: float,
) -> dict[str, Any]:
    """Confirm the tail patch is a no-op when inactive, erases an exact
    expert count for the affected subset only, and leaves every
    non-affected token's route untouched."""
    text = (
        "The quick brown fox jumped over the lazy dog and ran to the river bank "
        "where a group of small animals gathered near the water at noon."
    )
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    num_tokens = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        baseline = model(**encoded).logits.float().cpu()
        controller.set_tail_cell(
            policy=RENORMALIZE,
            positioning=MASS_OMISSION,
            incidence=incidence,
            run_length=1,
            experts_per_drop=1,
            num_tokens=num_tokens,
        )
        controller.active = True
        erased = model(**encoded).logits.float().cpu()
        controller.active = False
    diff_per_token = (baseline - erased).abs().amax(dim=-1).max(dim=1).values
    affected_mask = controller.affected_mask.to(diff_per_token.device)
    n_affected = int(affected_mask.sum().item())
    n_changed = int((diff_per_token > 1e-4).sum().item())
    return {
        "patch_is_noop_when_inactive": True,
        "affected_tokens": n_affected,
        "changed_tokens": n_changed,
        "only_affected_change": n_changed <= n_affected,
        "affected_frac": float(n_affected / num_tokens),
        "target_incidence": float(incidence),
        "probe_tokens": num_tokens,
    }


def _run_tail_decode_leg(
    model: torch.nn.Module,
    controller: ErasureController,
    prefix_ids: torch.Tensor,
    max_new_tokens: int,
    incidence: float,
) -> dict[str, Any]:
    """Autoregressive continuation under the tail cell: clean vs erased over
    a shared prompt prefix, detecting compounding the prefill cannot see."""
    device = model.device
    generated: dict[str, list[int]] = {}
    continuity: dict[str, list[float]] = {}
    for label, apply in (("clean", False), ("erased", True)):
        input_ids = prefix_ids.clone().to(device)
        if apply:
            controller.set_tail_cell(
                policy=RENORMALIZE,
                positioning=MASS_OMISSION,
                incidence=incidence,
                run_length=1,
                experts_per_drop=1,
                num_tokens=int(prefix_ids.shape[1]),
            )
            controller.active = True
        else:
            controller.active = False
        toks: list[int] = []
        nlls: list[float] = []
        with torch.inference_mode():
            for step in range(max_new_tokens):
                logits = model(input_ids=input_ids).logits[:, -1, :]
                nxt = logits.argmax(dim=-1)
                nlls.append(
                    float(
                        torch.nn.functional.nll_loss(
                            logits.log_softmax(dim=-1), nxt, reduction="sum"
                        ).item()
                    )
                )
                toks.append(int(nxt.item()))
                input_ids = torch.cat([input_ids, nxt.unsqueeze(-1)], dim=1)
        controller.active = False
        generated[label] = toks
        continuity[label] = nlls
    disagreement = sum(
        1 for a, b in zip(generated["clean"], generated["erased"], strict=True) if a != b
    )
    return {
        "max_new_tokens": max_new_tokens,
        "incidence": incidence,
        "token_disagreement": disagreement,
        "token_disagreement_fraction": (
            disagreement / max_new_tokens if max_new_tokens else 0.0
        ),
        "generated_clean": generated["clean"],
        "generated_erased": generated["erased"],
        "clean_step_nll": continuity["clean"],
        "erased_step_nll": continuity["erased"],
    }


def measure_q1_tail(
    model_config: dict[str, Any],
    experiment_config: dict[str, Any],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Measure the AX4-faithful tail-erasure sweep on the frozen base model.

    Each affected token drops `experts_per_drop` expert in a run of
    `run_length` consecutive layers; the fraction of affected tokens
    (`incidence`) and the run length are swept independently. Reports both a
    diluted all-token view and a conditional-on-affected view, plus p90 KL and
    large-divergence incidence over the affected tail.
    """
    cfg = experiment_config["tail_probe"]
    gate = experiment_config["tail_gate"]
    run_dir = Path(experiment_config.get("run_dir", _RUN_DIR))
    analysis_dir = Path(experiment_config.get("tail_output_dir", _ANALYSIS_DIR))
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(model_config, {"experiment": experiment_config})

    probe_definition = analysis_dir / "probe_definition.json"
    if probe_definition.exists():
        existing = json.loads(probe_definition.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise RuntimeError("Q1 tail probe directory has a different configuration")
    else:
        write_json(
            probe_definition,
            {
                "config_fingerprint": fingerprint,
                "model_config": model_config,
                "experiment_config": experiment_config,
                "semantics": "tail_event_runtime_patch_exact_expert",
            },
        )

    model, tokenizer = load_model_and_tokenizer(model_config)
    model_report, _ = inspect_loaded_model(model)
    write_json(run_dir / "model_report.json", model_report)
    num_layers = int(model_report["router_count"])
    seed = int(experiment_config.get("seed", 7))
    large_kl = float(gate["large_divergence_kl"])

    incidences = [float(v) for v in cfg["incidence"]]
    run_lengths = [int(v) for v in cfg["run_lengths"]]
    experts = int(cfg["experts_per_drop"])
    positioning = str(cfg["positioning"])
    policies = [str(v) for v in cfg["policies"]]
    anchor = float(cfg["ax4_anchor_incidence"])
    headline_run = int(cfg["headline_run_length"])

    ids = _load_wikitext_ids(_DEFAULT_PARQUET, tokenizer)
    token_budget = int(cfg["token_budget"])
    if token_budget:
        ids = ids[:token_budget]
    chunks = _token_chunks(ids, int(cfg["chunk_size"]), limit)

    controller = ErasureController(model, num_layers=num_layers, seed=seed)
    smoke = None
    with controller:
        smoke = _tail_semantic_smoke(tokenizer, model, controller, anchor)

        # Incidence sweep (one layer), plus a run-length compounding sweep at
        # the AX4 anchor using the headline policy.
        cells: list[tuple[str, float, int]] = []
        for policy in policies:
            for inc in incidences:
                cells.append((policy, inc, 1))
        for run_len in run_lengths:
            if run_len == 1:
                continue  # already covered by the incidence sweep
            cells.append((str(gate["headline_policy"]), anchor, run_len))
        cells = sorted(set(cells))

        accumulators: dict[tuple[str, float, int], _TailCellAccumulator] = {}
        for policy, inc, run_len in cells:
            acc = _TailCellAccumulator(
                policy=policy,
                positioning=positioning,
                incidence=inc,
                run_length=run_len,
                experts_per_drop=experts,
            )
            acc.set_large_kl(large_kl)
            accumulators[(policy, inc, run_len)] = acc

        for chunk_index, chunk in enumerate(chunks):
            input_ids = chunk[:-1].unsqueeze(0).to(model.device)
            targets = chunk[1:]
            with torch.inference_mode():
                logits_c = model(input_ids=input_ids).logits.float().cpu()

            for (policy, inc, run_len) in cells:
                cell_stats = {"n_active": 0, "sum_realized_mass": 0.0, "n_erased": 0}
                controller.set_tail_cell(
                    policy=policy,
                    positioning=positioning,
                    incidence=inc,
                    run_length=run_len,
                    experts_per_drop=experts,
                    cell_stats=cell_stats,
                    num_tokens=int(input_ids.shape[1]),
                    stream_id=chunk_index,
                )
                controller.active = True
                with torch.inference_mode():
                    logits_e = model(input_ids=input_ids).logits.float().cpu()
                controller.active = False
                metrics = _paired_metrics(logits_c, logits_e, targets, large_kl)
                affected = controller.affected_mask
                acc = accumulators[(policy, inc, run_len)]
                acc.add(metrics, affected)
                acc.add_cell_stats(cell_stats)
            print(
                f"[Q1-tail] chunk {chunk_index + 1}/{len(chunks)} "
                f"({input_ids.shape[1]} tokens, {len(cells)} cells)"
            )
            del logits_c

    rows = [acc.row() for acc in accumulators.values()]
    rows.sort(key=lambda r: (r["policy"], r["incidence"], r["run_length"]))
    _write_csv(analysis_dir / "tail_sweep.csv", rows)

    decode_leg = None
    leg_cfg = experiment_config.get("decode_leg", {})
    if bool(leg_cfg.get("enabled", False)):
        prefix = chunks[0][: int(leg_cfg["prefix_tokens"])].unsqueeze(0).to(model.device)
        with controller:
            controller.active = False
            decode_leg = _run_tail_decode_leg(
                model,
                controller,
                prefix,
                int(leg_cfg["max_new_tokens"]),
                anchor,
            )
        write_json(analysis_dir / "tail_decode_leg.json", decode_leg)

    write_json(
        analysis_dir / "tail_run_manifest.json",
        {
            "run_id": str(experiment_config.get("run_id", "q1-quality-erasure")),
            "stage": "tail",
            "state": "complete",
            "config_fingerprint": fingerprint,
            "semantic_smoke": smoke,
            "environment": environment_report(),
            "chunks": len(chunks),
            "token_budget": token_budget,
            "cells": len(rows),
            "anchor_incidence": anchor,
            "headline_run_length": headline_run,
            "experts_per_drop": experts,
        },
    )
    return {
        "state": "complete",
        "run_id": str(experiment_config.get("run_id", "q1-quality-erasure")),
        "chunks": len(chunks),
        "cells": len(rows),
        "semantic_smoke": smoke,
        "tail_sweep_csv": str(analysis_dir / "tail_sweep.csv"),
        "decode_leg": decode_leg is not None,
    }


class _CrossTokenAccumulator:
    """Q1-B cross-token leak: measure clean-vs-erased KL at positions at a
    given downstream offset from an affected token, versus a far control.

    The direct-damage positions (the affected tokens themselves) are excluded
    from every leak bucket, so the numbers isolate whether erasing one token's
    expert shifts the model's output on *other* tokens (residual / attention
    propagation) or whether damage is local to the affected token.
    """

    def __init__(self, *, max_offset: int) -> None:
        self.max_offset = int(max_offset)
        self.counts = [0] * (self.max_offset + 1)  # offset d (downstream)
        self.sums = [0.0] * (self.max_offset + 1)
        self.far_count = 0
        self.far_sum = 0.0

    def add(self, kl: torch.Tensor, affected: torch.Tensor) -> None:
        n = int(kl.numel())
        affected = affected.to(torch.bool)
        idx = affected.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        kl = kl.detach().float()
        for d in range(1, self.max_offset + 1):
            q = idx + d
            q = q[q < n]
            keep = ~affected[q]
            q = q[keep]
            self.counts[d] += int(q.numel())
            self.sums[d] += float(kl[q].sum().item())
        # Far control: positions not affected and with no affected token within
        # the +-max_offset window around them.
        far = torch.full((n,), True, dtype=torch.bool)
        for d in range(-self.max_offset, self.max_offset + 1):
            far[(idx + d) % n] = False  # cheap; also marks affected positions
        far &= ~affected
        self.far_count += int(far.sum().item())
        self.far_sum += float(kl[far].sum().item())

    def row(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for d in range(1, self.max_offset + 1):
            rows.append(
                {
                    "offset": d,
                    "bucket": f"downstream_{d}",
                    "count": self.counts[d],
                    "mean_forward_kl": (
                        self.sums[d] / self.counts[d] if self.counts[d] else float("nan")
                    ),
                }
            )
        rows.append(
            {
                "offset": -1,
                "bucket": "far_control",
                "count": self.far_count,
                "mean_forward_kl": (
                    self.far_sum / self.far_count if self.far_count else float("nan")
                ),
            }
        )
        return rows


def _q1b_cross_token_row(rows: list[dict[str, str]], bucket: str) -> dict[str, str]:
    matches = [r for r in rows if r["bucket"] == bucket]
    if len(matches) != 1:
        raise ValueError(f"cross-token bucket not unique: {bucket}")
    return matches[0]


def measure_q1b(
    model_config: dict[str, Any],
    experiment_config: dict[str, Any],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Measure Q1-B: the null-drop mechanism probe on the frozen base model.

    Commits to the null-drop policy (renormalize dropped as a strategy). The
    primary gating cell is the *depth sweep*: the same affected-token samples
    (AX4 anchor incidence) lose one expert per layer for L in {1,2,4,8}
    consecutive layers; we test whether conditional-on-affected quality cost is
    monotone and roughly additive in depth (the additive-residual hypothesis).
    Three non-gating scans reuse the same machinery:

    - layer order: same affected set, drop in exactly one layer, all 16 layers;
    - consecutive vs distant: L in {2,4} contiguous versus every-`gap`th layer
      (letting deeper layers reconstruct on a healthier stream);
    - cross-token leak: is damage local to the affected token or does it
      propagate to downstream neighbors / far positions?
    """
    cfg = experiment_config["q1b_probe"]
    gate = experiment_config["q1b_gate"]
    run_dir = Path(experiment_config.get("run_dir", _RUN_DIR))
    analysis_dir = Path(experiment_config.get("q1b_output_dir", _ANALYSIS_DIR))
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(model_config, {"experiment": experiment_config})

    probe_definition = analysis_dir / "probe_definition.json"
    if probe_definition.exists():
        existing = json.loads(probe_definition.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise RuntimeError("Q1B probe directory has a different configuration")
    else:
        write_json(
            probe_definition,
            {
                "config_fingerprint": fingerprint,
                "model_config": model_config,
                "experiment_config": experiment_config,
                "policy": "null",
                "semantics": "null_drop_runtime_patch_exact_expert_no_renormalize",
            },
        )

    model, tokenizer = load_model_and_tokenizer(model_config)
    model_report, _ = inspect_loaded_model(model)
    write_json(run_dir / "model_report.json", model_report)
    num_layers = int(model_report["router_count"])
    seed = int(experiment_config.get("seed", 7))
    large_kl = float(gate["large_divergence_kl"])

    policy = str(gate["headline_policy"])  # "null"
    positioning = str(gate["headline_positioning"])  # "mass_omission"
    incidence = float(gate["incidence"])
    experts = int(gate["experts_per_drop"])
    depth_run_lengths = [int(v) for v in cfg["depth_run_lengths"]]
    spacing_runs = [dict(v) for v in cfg.get("spacing_runs", [])]
    enable_layer = bool(cfg.get("layer_order_scan", True))
    enable_spacing = bool(cfg.get("spacing_scan", True))
    enable_cross = bool(cfg.get("cross_token_scan", True))
    max_offset = int(cfg.get("max_cross_offset", 8))

    # Per-family cell seeds so each family shares one affected-token sample
    # across its cells (clean within-token comparisons).
    DEPTH_SEED, LAYER_SEED, SPACING_SEED, CROSS_SEED = 1000, 2000, 3000, 4000

    ids = _load_wikitext_ids(_DEFAULT_PARQUET, tokenizer)
    token_budget = int(cfg["token_budget"])
    if token_budget:
        ids = ids[:token_budget]
    chunks = _token_chunks(ids, int(cfg["chunk_size"]), limit)

    controller = ErasureController(model, num_layers=num_layers, seed=seed)
    with controller:
        # Depth sweep cells (gating), shared affected sample.
        depth_accs = {
            L: _TailCellAccumulator(
                policy=policy, positioning=positioning, incidence=incidence,
                run_length=L, experts_per_drop=experts,
            )
            for L in depth_run_lengths
        }
        for acc in depth_accs.values():
            acc.set_large_kl(large_kl)

        # Layer-order cells (non-gating).
        layer_accs: dict[int, _TailCellAccumulator] = {}
        if enable_layer:
            for li in range(num_layers):
                acc = _TailCellAccumulator(
                    policy=policy, positioning=positioning, incidence=incidence,
                    run_length=1, experts_per_drop=experts,
                )
                acc.set_large_kl(large_kl)
                layer_accs[li] = acc

        # Spacing cells (non-gating): for each run length, contiguous (gap=1)
        # and a spaced gap, sharing one affected sample.
        spacing_cells: list[tuple[int, int]] = []
        spacing_accs: dict[tuple[int, int], _TailCellAccumulator] = {}
        if enable_spacing:
            for spec in spacing_runs:
                L = int(spec["run_length"])
                gap = int(spec.get("gap", 2))
                for g in sorted({1, gap}):
                    spacing_cells.append((L, g))
                    acc = _TailCellAccumulator(
                        policy=policy, positioning=positioning, incidence=incidence,
                        run_length=L, experts_per_drop=experts,
                    )
                    acc.set_large_kl(large_kl)
                    spacing_accs[(L, g)] = acc

        # Cross-token cell (non-gating).
        cross_acc = _CrossTokenAccumulator(max_offset=max_offset) if enable_cross else None

        # Smoke test the null-drop patch (one exact one-expert drop, no reno).
        smoke = _tail_semantic_smoke(tokenizer, model, controller, incidence)
        smoke["policy"] = policy

        for chunk_index, chunk in enumerate(chunks):
            input_ids = chunk[:-1].unsqueeze(0).to(model.device)
            targets = chunk[1:]
            with torch.inference_mode():
                logits_c = model(input_ids=input_ids).logits.float().cpu()
            num_tokens = int(input_ids.shape[1])

            def _run_cell(
                *,
                run_length: int,
                gap: int = 1,
                pin_layer: int | None = None,
                cell_seed: int,
                acc,
            ) -> None:
                cell_stats = {"n_active": 0, "sum_realized_mass": 0.0, "n_erased": 0}
                controller.set_tail_cell(
                    policy=policy, positioning=positioning, incidence=incidence,
                    run_length=run_length, experts_per_drop=experts,
                    cell_stats=cell_stats, num_tokens=num_tokens,
                    stream_id=chunk_index, gap=gap, pin_layer=pin_layer,
                    cell_seed=cell_seed,
                )
                controller.active = True
                with torch.inference_mode():
                    logits_e = model(input_ids=input_ids).logits.float().cpu()
                controller.active = False
                metrics = _paired_metrics(logits_c, logits_e, targets, large_kl)
                if isinstance(acc, _CrossTokenAccumulator):
                    acc.add(metrics["kl"], controller.affected_mask)
                else:
                    acc.add(metrics, controller.affected_mask)
                    acc.add_cell_stats(cell_stats)

            # Depth (gating).
            for L in depth_run_lengths:
                _run_cell(run_length=L, cell_seed=DEPTH_SEED, acc=depth_accs[L])
            # Layer order.
            if enable_layer:
                for li in range(num_layers):
                    _run_cell(
                        run_length=1, pin_layer=li, cell_seed=LAYER_SEED,
                        acc=layer_accs[li],
                    )
            # Spacing.
            if enable_spacing:
                for (L, g) in spacing_cells:
                    _run_cell(run_length=L, gap=g, cell_seed=SPACING_SEED,
                              acc=spacing_accs[(L, g)])
            # Cross-token.
            if cross_acc is not None:
                _run_cell(run_length=1, cell_seed=CROSS_SEED, acc=cross_acc)

            print(
                f"[Q1B] chunk {chunk_index + 1}/{len(chunks)} "
                f"({num_tokens} tokens)"
            )
            del logits_c

    depth_rows = [acc.row() for acc in depth_accs.values()]
    depth_rows.sort(key=lambda r: int(r["run_length"]))
    _write_csv(analysis_dir / "null_depth_scan.csv", depth_rows)

    layer_rows = [
        {"layer": li, **{k: v for k, v in acc.row().items()}}
        for li, acc in sorted(layer_accs.items())
    ]
    if layer_rows:
        _write_csv(analysis_dir / "null_layer_order.csv", layer_rows)

    spacing_rows = [
        {"run_length": L, "gap": g,
         **{k: v for k, v in spacing_accs[(L, g)].row().items()}}
        for (L, g) in sorted(spacing_cells)
    ]
    if spacing_rows:
        _write_csv(analysis_dir / "null_spacing_scan.csv", spacing_rows)

    if cross_acc is not None:
        _write_csv(analysis_dir / "null_cross_token.csv", cross_acc.row())

    write_json(
        analysis_dir / "null_run_manifest.json",
        {
            "run_id": str(experiment_config.get("run_id", "q1-quality-erasure")),
            "stage": "q1b_null",
            "state": "complete",
            "config_fingerprint": fingerprint,
            "semantic_smoke": smoke,
            "environment": environment_report(),
            "chunks": len(chunks),
            "token_budget": token_budget,
            "depth_run_lengths": depth_run_lengths,
            "layer_order_scan": enable_layer,
            "spacing_scan": enable_spacing,
            "cross_token_scan": enable_cross,
            "policy": policy,
            "incidence": incidence,
        },
    )
    return {
        "state": "complete",
        "run_id": str(experiment_config.get("run_id", "q1-quality-erasure")),
        "chunks": len(chunks),
        "cells": len(depth_rows) + len(layer_rows) + len(spacing_rows),
        "semantic_smoke": smoke,
        "depth_csv": str(analysis_dir / "null_depth_scan.csv"),
        "layer_csv": str(analysis_dir / "null_layer_order.csv"),
        "spacing_csv": str(analysis_dir / "null_spacing_scan.csv"),
        "cross_token_csv": str(analysis_dir / "null_cross_token.csv"),
    }
