# Q2 protocol: availability-conditioned robustness — stress and cliff mapping

**Status:** **complete** — all three arms measured and accepted by the
researcher; cross-domain GO, cliff WITH_MARGIN, decode divergence re-read as
paraphrase not degradation; **no non-free regime demonstrated, so the gated
calibration sub-step is NOT entered** (see `EXPERIMENT_LOG.md`).
**Track:** measured-quality stress (closed loop back to AX4 / Q2 training); a
minimal availability-conditioned calibration is defined but gated and is **not**
part of the probe phase
**Prerequisite:** Q1 (STOP on universal mass-budget) and Q1-B (GO on the AX4
null-drop tail) are complete and accepted.

## The problem Q1/Q1-B left open

Q1 measured quality versus missing routed mass. Q1-B then committed to
**null-drop** and showed that, on the frozen base model, quality loss under the
AX4 tail regime is **monotone and roughly additive in the number of dropped
expert-layers**: worst case L=8 is 0.013 nats conditional-on-affected KL with
zero large divergences, layer-uniform sensitivity, no reconstruction benefit
from spacing, and no cross-token leak. In other words, the model already
tolerates the AX4 erasure distribution essentially **for free** — *but only in
the regime Q1-B actually measured*:

- one frozen revision;
- one text domain (`WikiText-2` prose);
- **prefill only** (no autoregressive generation, so compounding is invisible);
- erasure bounded at AX4's nominal 0.9% incidence, ≤8 consecutive layers, one
  low-mass expert per layer.

Separately, Q1's universal mass-budget headline (KL 5.81) shows that outside
those bounds quality collapses off a **cliff**. Q2 therefore does not re-ask
"can the model tolerate erasure" — that is largely answered. Q2 asks where and
whether that measured free-tolerance holds, and only if a real failure appears,
what the *smallest* intervention must correct.

## Reframed goal

> Verify that the measured free null-drop tolerance survives the untested axes
> (cross-domain, decode compounding, contract-violation boundaries); if a real
> degradation is found, define a **minimal, mask-aware** calibration that
> softens the cliff at exactly that regime — never generic expert dropout.

This goal is deliberately minimal: if the stress probe shows tolerance still
holds everywhere it matters, then AX4's benefit is realized with **zero**
training cost and Q2 closes as a documented no-op (a GO to AX4 with no
robustness penalty). Training is justified only by a demonstrated non-free
regime.

## Decisive question

On the frozen base checkpoint, is the Q1-B null-drop tolerance
(monotone-additive, low worst-case cost, local) preserved (i) across domains,
(ii) under autoregressive decode compounding, and (iii) up to and across the
AX4 contract boundary — and if not, exactly where does it stop being free and
how steep is that cliff?

Three arms answer it; each is a forward-pass measurement, none trains.

## Experiment arms (all measured, no training)

Shared base config: `configs/experiment/q2-stress.toml`; each arm reuses the
frozen null-drop/mass-omission machinery from Q1-B (same affected-token
sampling, same conditional-on-affected and diluted metrics, same
per-layer-marginal and large-divergence computations).

### Q2-A — Cross-domain tolerance
- **Question:** does the depth-additive null-drop tolerance hold on non-prose
  domains (the tension Q1's single-domain pilot could not see)?
- **Scope:** run the Q1-B depth sweep (L∈{1,2,4,8}, AX4 anchor incidence,
  same affected sample, null + mass-omission) on at least one additional domain
  materialized from local data (code and/or math/reasoning), with the primary
  cell conditional-on-affected at L=8.
- **Gate (this arm):** cross-domain **GO** if the L=8 conditional-on-affected
  KL stays small and the depth curve stays monotone (same criteria as the Q1-B
  frozen gate applied at equal realized incidence); any domain that breaks
  monotonicity or crosses a large-divergence bar is a *candidate robustness
  target*, reported explicitly and non-gating.

### Q2-B — Decode compounding
- **Question:** does the (prefill-free) tolerance survive autoregressive
  generation, where each step's slightly-perturbed output feeds the next?
- **Scope:** a larger continuation leg (tens of generated tokens, e.g. 32–64)
  under the AX4 tail cell (incidence 0.009, L=1 and L=8), comparing clean vs
  erased token streams, step-wise and cumulative clean-vs-erased divergence,
  token-agreement over generation, and whether divergence grows, plateaus, or
  collapses with depth of generation.
- **Gate (this arm):** **GO** if the erased continuation stays coherent
  (token-agreement high and cumulative divergence bounded, no runaway
  divergence) at both L=1 and L=8; systematic compounding (divergence that
  grows monotonically beyond a small threshold, or token-stream divergence)
  is a candidate robustness target.

### Q2-C — Cliff mapping (contract-violation boundary)
- **Question:** where does null-drop stop being free, and how steep is the
  cliff? This makes the Q1 universal mass-budget cliff *quantitative and
  located* rather than a single isolated STOP point.
- **Scope:** starting from the Q1-B free cell, push erasure along the axes
  AX4's bound protects: higher incidence (e.g. 0.02→0.1→0.3→0.5), longer runs
  (L=8→12→16), and multiple experts dropped per layer (1→2→4), up to the
  mass-budget neighborhood. Report conditional-on-affected and diluted quality
  vs each axis, plus where the large-divergence/large-mass band begins.
- **Gate (this arm):** the cliff is the response-surface boundary where
  conditional-on-affected cost leaves the near-free ⟨0.02 nats⟩ regime. If that
  boundary sits **comfortably beyond** a safely-enforced AX4 bound (with
  margin), the contract stands as measured; if it appears **within** a plausible
  contract-violation margin, a robustness target is justified to soften it.

## Evidence contract

- **Measured** paired forward passes on the frozen
  `OLMoE-1B-7B-0125` base checkpoint (revision
  `9b0c1aa87e34a20052389dce1f0cf01da783f654`). No training, no model download,
  no second checkpoint.
- Quality only: no latency, capacity, or service-rate claim in any arm.
- Null-drop only; **renormalize remains dropped** as a strategy.
- Conditional-on-affected and diluted readouts both reported; per-token forward
  KL, top-1 agreement, perplexity ratio, large-divergence incidence.
- Labels: direct measured evidence; the *training* sub-step (below) is an
  explicit target, not an empirical claim.

## The gated training sub-step (not part of the probe)

Only if a probe arm demonstrates a real (non-free) degradation:

1. define the exact conditional event (which arm, which domain / generation
   depth / boundary), and the measured mask/mass distribution it produces;
2. train a **minimal, mask-aware** calibration — mask-aware renormalization
   that outperforms the naive renormalize already rejected, a tiny
   always-resident shared/fallback expert used only on a drop, or a low-rank
   (LoRA) adapter on the router/MoE — against that exact distribution;
3. hold validation loss and load balance fixed; report the Pareto tuple
   (validation loss, load balance, quality under the AX4 erasure distribution,
   modeled TPOT benefit);
4. no generic expert-dropout robustness; no expansion of scope.

This sub-step requires explicit researcher permission and is only entered if a
probe arm fails its gate.

## Outputs

- `q2_cross_domain/depth_by_domain.csv`, `q2_cross_domain/gate.json`,
  `q2_cross_domain/CROSS_REPORT.md`;
- `q2_decode/continuation.csv`, `q2_decode/gate.json`, `q2_decode/DECODE_REPORT.md`;
- `q2_cliff/cliff_surface.csv`, `q2_cliff/cliff_gate.json`, `q2_cliff/CLIFF_REPORT.md`;
- one primary figure per arm (depth-by-domain, compounding curve, cliff surface)
  with SHA-256 input manifests.

All under `artifacts/runs/q1-quality-erasure/analysis/q2_*/`.

## Protocol lesson expected

A clean outcome separates three possibilities: (1) tolerance holds broadly →
AX4 is realized with zero training and Q2 closes as a documented no-op; (2) a
narrow non-free regime appears → a tight, mask-aware calibration target is
defined and AX4's contract is scoped with an explicit margin; (3) the cliff
sits inside the contract margin → AX4's bounded-run quality contract needs a
harder ceiling before any training, and Q2 is reprioritized accordingly.
