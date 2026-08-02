# Q1B protocol: null-drop mechanism probe (intrinsic cost, additive in depth?)

**Status:** draft, awaiting researcher freeze
**Track:** empirical quality measurement (closed loop back to AX4/Q2)
**Immediate experiment:** measure how the frozen base model's quality falls
under **null-drop** (drop a selected expert with no renormalization) as a
function of *where and how* the erasure is placed, holding routed-mass
incidence bounded at the AX4 tail. No renormalization: dropped from the
candidate set after Q1-tail showed it is ~40x worse than null at equal
single-expert mass and theoretically distorts the residual-stream scaling.
**New inference:** yes (forward passes on the frozen base checkpoint); no
training, no model download, no second checkpoint.

**Primary config:** `configs/experiment/q1-quality-erasure.toml`
**Defining hypothesis (mechanism):** each routed expert adds a mostly
independent, weighted update to the residual stream. Dropping an expert in
`k` separate layers therefore removes `k` roughly-additive residual updates.
If true, null-drop cost should scale *approximately additively and
monotonically* in the number of dropped expert-layers, and be ordered by the
same layers that carry the largest routed updates — not by an emergent,
depth-amplified blow-up.

## Why this follow-up exists

Q1 (mass-budget) and Q1-tail established:

- Universal every-token/every-layer erasure (mass-budget headline) is
  catastrophic (KL 5.81) — but that regime is *not* what AX4's deadline
  contract delivers.
- Treating erasure as a **rare tail event** (AX4: ~0.9% degraded waves) is
  far less destructive and, under the model's native **null-drop**, a single
  one-expert one-layer drop is nearly free: conditional-on-affected KL 0.0032,
  top-1 99.2%, PPL ~1.0. Under **renormalize** the same cell costs ~40x more
  (KL 0.128, top-1 86%), and renormalization is theoretically wrong: it
  rescales survivors, amplifying the very mass the drop removed.

Decision: drop renormalization as a viable strategy; commit to **null-drop**.
Q1B then maps the null-drop mechanism we are betting on, so AX4's
bounded-run quality contract (up to 8 consecutive degraded layers) and any
Q2 robustness-training target are defined against measured, monotonic,
depth-scaling behavior rather than one point.

## Decisive question

On the frozen base model, is **null-drop** quality loss controlled by the
*number and placement* of dropped expert-layers per token — approximately
additive and monotone in depth as the additive-residual hypothesis predicts —
or does it compound / depend on which layers are hit / leak across tokens?

Sub-questions resolved inside the same run (one decisive, rest non-gating):

1. **(Decisive) Depth additivity:** as the number of consecutive degraded
   layers on an affected token grows 1 → 8, does conditional-on-affected KL
   rise monotonically and roughly additively (per-layer marginal cost roughly
   flat), with no sudden jump?
2. **Layer-order sensitivity:** which layers are most / least sensitive to a
   single-layer, one-expert null drop?
3. **Consecutive vs distant:** for the same number of dropped layers, does
   spacing them apart (letting deeper layers reconstruct on a healthier deep
   stream) reduce cost relative to a contiguous block?
4. **Cross-token leak:** does erasing one token's expert shift the model's
   output on *other* tokens in the sequence (attention / residual
   propagation), or is quality damage local to the affected token?

## Evidence and claim boundary

Everything is **measured** on the frozen OLMoE-1B-7B-0125 base checkpoint over
WikiText-2 validation (prefill scope): forward passes + logits, paired
same-token deltas and cross-token deltas. No training, no trace-derived
predictor, no hypothetical hardware in the primary measurement. Metrics are
quality only; Q1B makes no latency, capacity, or service-rate claim.

The AX4 reduced-mass regime (0.9% degraded waves, up to 8 consecutive
degraded layers) is *referenced* to set the incidence anchor and the run-length
sweep ceiling, but the values themselves are measured from the actual frozen
model.

## Definitions (null-drop only)

For a token `t`, MoE layer `l`, selected top-8 `S`, and `M ⊆ S` the dropped
expert subset:

- **null-drop:** remove the contribution of `M` with no renormalization —
  surviving selected weights are unchanged (current OLMoE execution
  semantics). Renormalize is dropped and never gating.
- **expert drop:** always drop the **lowest-mass** selected expert
  (`mass_omission`) — matches AX4's mass-priority deadline scheduler. One
  expert per degraded layer.
- **degraded layer:** a MoE layer at which token `t` has one expert dropped.
- **run / consecutive block:** `L` degraded layers are contiguous
  (layers `s..s+L-1`).
- **spacing:** `L` degraded layers arranged with a fixed gap between them
  (e.g. start `s`, then `s+g, s+2g, ...`) instead of contiguous.
- **affected token:** a token with at least one degraded layer.
- **conditional-on-affected:** metric averaged over affected tokens only.
- `ΔQ` is a paired same-token (or cross-token) delta against the clean pass.

## Primary probe: depth additivity (gating)

- Policy: **null-drop**, positioning **mass_omission**, one expert per layer.
- Incidence: AX4 anchor **0.009** (degraded-token fraction).
- Sweep consecutive run length `L ∈ {1, 2, 4, 8}` (AX4 worst case is 8).
- Metric, conditional-on-affected: per-token forward KL, top-1 agreement,
  perplexity ratio; plus per-layer marginal `ΔKL` between consecutive `L`.
- Report the full curve and the marginal increments.

### Non-gating scans (same run, cheap)

1. **Layer-order sensitivity** (incidence 0.009, run length 1, null,
   mass_omission): for each of the 16 layers independently, drop the
   lowest-mass expert only in that layer; report conditional-on-affected KL
   by `layer_index`. Orders layers by sensitivity.
2. **Consecutive vs distant** (incidence 0.009, null, mass_omission): for a
   fixed number of dropped layers `L ∈ {2, 4}`, compare the contiguous run
   against spaced placements with the same `L` (gap such that all layers stay
   in 0..15). Does reconstruction between drops matter?
3. **Cross-token leak to neighboring tokens** (incidence 0.009, run length 1,
   null, mass_omission): for each affected token `p`, also measure the
   clean-vs-erased KL at downstream positions `p+1, p+2, p+3` (and up to +7),
   plus a far-away control, to see if quality damage propagates beyond the
   affected token or is local.

## Injection mechanism

Reuse the frozen runtime MoE-forward patch from Q1 (`ErasureController`,
`tail_mode`): each gate hook temporarily zeroes the masked expert's
contribution with no renormalization, per forward pass, then restores. Never
modifies the model file. Extend the controller with explicit layer-identity
control (drop only a specified layer for layer-order; spaced layer sets for
the spacing scan) while keeping exact softmax-64 → top-8 →
no-renormalization downstream semantics and seeded, reproducible affected-token
sampling.

## Metrics

- per-token forward KL `KL(p_clean ‖ p_erased)`;
- top-1 token agreement (fraction of tokens keeping the same argmax);
- perplexity and perplexity ratio (erased/clean);
- all **paired same-token** against the clean pass (cross-token scan uses the
  clean pass at the same positions);
- per-layer marginal KL (depth additivity);
- degraded-step / large-divergence incidence (KL ≥ 2.0);
- conditional-on-affected and diluted readouts both reported.

## Frozen stop/go gate (depth additivity, primary)

**Primary cell:** null-drop + mass_omission at incidence 0.009,
consecutive run `L = 8` (AX4 worst case), conditional-on-affected.

**GO** if all of:
- conditional-on-affected mean forward-KL increases **monotonically** in
  `L ∈ {1,2,4,8}` (no non-monotone jump);
- conditional-on-affected KL at `L=8` is compatible with additive scaling —
  the per-layer marginal `ΔKL` between consecutive sweep points does not
  increase super-linearly (roughly flat to mildly rising marginal cost;
  `ΔKL(8) ≤ 3 × ΔKL(1)`), i.e. no exponential blow-up;
- no frequent large divergence: at `L=8`, fraction of affected tokens with
  KL ≥ 2.0 is ≤ 1%.

**STOP / kill** if even a single low-depth null drop (`L=1`, sub-1% incidence)
causes frequent large divergence, or the curve is non-monotone / jumps / blows
up super-linearly by the AX4 worst case. A STOP kills the additive-residual
mechanism thesis and makes AX4's bounded-run quality contract and any Q2
null-drop training target harder to justify.

The gate is frozen before running. Layer-order, spacing, and cross-token
scans may narrow the interpretation but never rewrite the gate.

## Outputs

- `null_depth_scan.csv` — depth sweep, conditional-on-affected and diluted;
- `null_layer_order.csv` — per-layer sensitivity;
- `null_spacing_scan.csv` — consecutive vs spaced;
- `null_cross_token.csv` — leak to downstream positions;
- `null_gate.json` — frozen gate verdict;
- `NULL_REPORT.md`; `ΔKL vs L` depth curve + layer-order/spacing/cross-token
  panels, input-hash manifest.

All under `artifacts/runs/q1-quality-erasure/analysis/q1b_null/`.

## Protocol lesson expected

A clean depth-additivity result separates the viable null-drop story (AX4's
bounded-run contract is plausible, and Q2 should train mask-aware adaptation
against the measured depth distribution) from a fragile one (quality blows up
with even a few stacked drops, so the bounded-run contract needs a hard ceiling
and Q2 carries a tighter target). Layer-order and spacing additionally tell the
hardware/scheduler which layers must not be dropped and whether reconstruction
window buys headroom.
