# Q1: Expert-erasure quality probe

**Status:** protocol frozen, awaiting implementation + researcher review
**Track:** empirical quality measurement (closed loop back to AX4)
**Immediate experiment:** paired clean-vs-erased inference over WikiText-2 on the
frozen base checkpoint, measuring how quality falls with missing routed mass
**New inference:** yes (forward passes on the frozen base model); no training,
no model download, no upstream model modification

**Primary config:** `configs/experiment/q1-quality-erasure.toml`

## Why this experiment exists

AX4 established an analytical deadline-erasure regime: at K=8, 256 GB/s,
C=99%, A=1.5x, a fixed commit deadline bounds batch-1 TPOT at 11.25 ms with
~0 P99 missing routed mass, 0.93% degraded waves, and zero full fallback. But
it *cannot* establish that the model tolerates the erasure it assumes. Its
"assumed robustness" contract is exactly the gap Q1 measures.

The decisive unresolved question in the project is whether low-mass,
deadline-induced expert omissions can be made semantically cheap enough to
convert prediction into a hard latency guarantee. AX4 passes that question
*conditionally*: it requires the model to tolerate bounded missing routed mass.
Q1 measures that tolerance on the frozen checkpoint before any training is
authorized.

## Decisive question

On the frozen base model, is quality loss caused by expert erasure controlled
by missing routed mass `m_missing`, or by *which* experts are lost and *where*
they are lost?

- If `ΔQ` is smooth, modest, and monotone in `m_missing`, AX4's quality
  contract is plausible and the robustness-training target is defined.
- If even very low mass (sub-1%) causes frequent large divergence, AX4's
  deadline-erasure contract is not viable without architectural retraining.

## Evidence and claim boundary

Every result must keep the AX asset labels:

1. **Measured:** forward passes and logits on the frozen base checkpoint;
   everything in Q1 is measured on the actual model.
2. **Trace-derived:** if/where AX4 decode availability-mask statistics are used
   to shape the correlation scan, label the source.
3. **Assumed predictor / hypothetical hardware:** none used in the primary
   renormalize/null measurements. Shared-residual fallback is **explicitly
   excluded** from the measured probe (the base model has no shared expert) and
   remains paper-level architecture discussion only.

Q1 measures quality, not latency, capacity, or hardware. It cannot make any
service-rate claim.

## Definitions

For a token `t` at MoE layer `l` under an erasure policy:

- `S`: the router's selected top-8 experts;
- `M ⊆ S`: the erased (missing) expert subset;
- `a_i`: selected-expert contribution weight normalized within top-8
  (matching AX4's primary mass semantics; `sum=1`);
- missing routed mass `m_missing = Σ_{i∈M} a_i ∈ [0,1]`;
- **renormalize**: drop `M`, rescale surviving `a_i` to sum 1;
- **null-drop**: drop `M` with no renormalization (current-model semantics);
- `ΔQ` is always a **paired same-token delta** against the clean pass.

## Policies and erasure probes

Only two measured policies (matched at equal `m_missing`):

1. **Renormalize present** — primary policy. The probable future-training
   form; the number that defines the robustness-training target.
2. **Null-drop** — lower bound; mirrors current OLMoE execution semantics.

No shared-residual in the measured probe (no shared expert in the base model).

**Primary gate — mass-budget sweep** at fixed `m_missing`
`{0.01, 0.05, 0.125, 0.25, 0.50}` with positioning:

- **mass-omission** (headline): erase lowest-mass experts first — matches AX4's
  mass-priority deadline scheduler;
- **random-within-route** (reference);
- **mass-adversarial** (worst case): erase highest-mass experts first.

**Non-gating correlation scan** (does concentration/compounding change cost?):

- **layer-burst**: the `m_missing` misses for a token co-locate in one layer;
- **consecutive-token-burst**: a run of tokens each suffer erasure.

**Decode leg (non-gating):** a short greedy decode from a shared prompt prefix,
comparing clean vs erased over the generated continuation to detect
autoregressive compounding that prefill structurally cannot see.

## Injection mechanism

**Option 1 — runtime MoE-forward patch (frozen).** Temporarily wrap each
selected-moe forward with an in-memory mask that zeroes the masked experts'
contribution (renormalize or null), applied per forward pass and restored
immediately after. Never modifies the model file. This preserves exact
execution semantics: softmax over 64, top-8, no renormalization on the raw
path, and the ~0.406 raw selected-weight sum quirk.

## Metrics

- per-token forward KL divergence `KL(p_clean ‖ p_erased)`;
- top-1 token agreement (fraction of tokens keeping the same argmax);
- perplexity and perplexity-ratio (erased/clean);
- all **paired same-token** against the clean pass;
- secondary readouts: number of experts erased at each `m`, layer and domain
  sensitivity, degraded-step incidence.

Principal classifications (from
[FOUNDATIONAL_INSIGHTS](FOUNDATIONAL_INSIGHTS.md)): intrinsic-vs-recoverable
cost (renormalize-vs-null gap), and whether mass alone explains cost.

## Frozen stop/go gate

**Primary cell:** renormalize + mass-omission at headline `m = 0.125` (one
expert's nominal share), averaged over tokens with at least one erased expert.

**GO** if all:
- mean per-token forward-KL ≤ 0.05;
- top-1 agreement ≥ 99%;
- perplexity ratio (erased/clean) ≤ 1.05;
- `ΔQ` monotone in `m_missing` (no sudden jump).

**STOP / kill** (AX4 training justification dies) if sub-1% mass induces
frequent large divergence, or `ΔQ` is non-monotone or jumpy.

The gate is frozen before running. Post-hoc correlation/positioning scans may
narrow interpretation but never rewrite the gate.

## Outputs

- paired per-token quality tables by policy, positioning, `m`, layer, domain;
- `ΔQ vs m_missing` curve (primary, mass-omission) for renormalize and null;
- positioning/correlation panel (omission vs random vs adversarial;
  layer-burst vs spread);
- gate JSON verdict under `artifacts/runs/q1-quality-erasure/analysis/q1/`.

No new model, no training, no second checkpoint without explicit permission.

## Protocol lesson expected

A clean renormalize-vs-null comparison decides whether quality cost is
intrinsic (⇒ availability-conditioned training required) or largely
policy-recoverable (⇒ lighter training burden). This is the highest-information
measurement between the completed AX4 analysis and any further training work.
