# Predictive Expert Prefetching for Hierarchical MoE Inference

## Founding Research PRD

**Status:** H1–H6 and C0 empirical pilot complete; AX1–AX4 analytical architecture track complete (AX4 review pending); Q1/Q1-B/Q2 expert-erasure quality probes complete and accepted (Q2: null-drop tolerance holds across domains with cliff margin, and decode divergence is paraphrase, not degradation)
**Primary environment:** Python 3.12, `uv`, PyTorch, Hugging Face Transformers, CUDA 12.4  
**Initial hardware:** 1× NVIDIA GPU with 24 GB VRAM  
**Planned scale-up environment:** 8× AMD MI355X-class GPUs with 288 GB HBM per GPU  
**Primary audience:** AI coding assistant, ML systems researcher, GPU performance modeling engineer, HW/SW/workload co-design architect

**Curated publication insights:** [docs/FOUNDATIONAL_INSIGHTS.md](docs/FOUNDATIONAL_INSIGHTS.md)
This durable synthesis is updated only at major evidence transitions; routine
results remain in `EXPERIMENT_LOG.md` and the per-hypothesis reports.

**Active architecture protocol:**
[docs/DEADLINE_DEGRADATION_PROTOCOL.md](docs/DEADLINE_DEGRADATION_PROTOCOL.md)

---

# 1. Executive Summary

This project studies whether future Mixture-of-Experts (MoE) expert demand can be predicted sufficiently early and accurately to support a hierarchical expert-memory system.

The initial experiment will:

1. Run inference on a Hugging Face-compatible MoE model.
2. Capture per-token, per-layer expert routing traces.
3. Quantify expert popularity, skew, reuse, temporal locality, and cross-layer predictability.
4. Train lightweight skip-layer routers that predict experts needed one, two, or three MoE layers in the future.
5. Evaluate prediction using latency-relevant set-coverage metrics rather than only ordinary classification accuracy.
6. Replay routing traces through an analytical cache and prefetch simulator.
7. Build an artificial CPU-to-GPU expert-loading prototype using pinned host memory and asynchronous PCIe transfers.
8. Use measured workload statistics and calibrated transfer timing to identify the hardware regimes in which predictive expert prefetching is viable.

The project must not assume that high prediction accuracy automatically improves low-latency inference. In synchronous small-batch decode, one unpredicted cold expert can stall the execution wave. Therefore, the central question is not:

> Can future experts be predicted accurately?

It is:

> Can the full set of experts demanded by an execution wave be covered early enough, with sufficiently low speculative bandwidth and capacity overhead, to avoid extending the critical path?

The project should produce either:

- a defensible positive result identifying a viable architectural region; or
- a defensible negative result showing that prediction is primarily useful for throughput, residency planning, or replication rather than strict low-latency prefetch.

Both outcomes are valuable.

---

# 2. Vision

Future MoE inference systems should be understood as hierarchical distributed-state machines.

Three state classes dominate execution:

1. **KV cache:** request-owned, persistent, temporally reused state.
2. **Expert weights:** model-owned, globally shared, sparsely demanded state.
3. **Activations:** transient execution state that is often cheaper to move than model parameters.

Attention prefers to keep KV state stationary near compute. MoE prefers to keep expert weights stationary and move activations to the selected experts. These stationarity preferences conflict because each token alternates through attention and MoE layers.

The broader architectural vision is a runtime and hardware stack that explicitly represents:

- where each state object resides;
- when it will be needed;
- what it costs to move;
- whether it should be moved, replicated, prefetched, or remotely executed;
- how its placement interacts with routing and batching.

This project addresses one narrow but important primitive within that vision:

> **Predictive future-expert demand estimation for hierarchical expert residency and prefetch planning.**

The immediate goal is not to design the complete distributed inference architecture. The goal is to establish the workload evidence, predictive limits, analytical model, and experimentally calibrated feasibility boundaries needed to justify or reject such an architecture.

---

# 3. Core Research Thesis

MoE inference increasingly becomes a hierarchical data-movement and state-placement problem as model parameter capacity grows faster than economically local memory capacity.

The cost of an MoE layer is not adequately represented as only GEMM time. A more complete abstraction is:

\[
T_l =
\max_{e \in D_l}
\left[
T_{\text{route}}
+
T_{\text{pack},e}
+
T_{\text{move},e}
+
T_{\text{queue},e}
+
T_{\text{compute},e}
+
T_{\text{return},e}
\right]
\]

where \(D_l\) is the set of experts demanded at layer \(l\).

For hierarchical storage, a nonresident expert incurs:

\[
T_{\text{load},e}
=
\alpha_{\text{tier}}
+
\frac{S_e}{\beta_{\text{tier}}^{\text{effective}}}
+
Q_e
\]

where:

- \(S_e\) is expert weight size or transferred tile size;
- \(\alpha_{\text{tier}}\) is transfer startup latency;
- \(\beta_{\text{tier}}^{\text{effective}}\) is achieved bandwidth;
- \(Q_e\) captures queueing and contention.

A prediction issued at layer \(l\) for future layer \(l+\Delta\) is timely only if:

\[
T_{\text{available}}(l,\Delta)
=
\sum_{j=l}^{l+\Delta-1} T_j
\ge
T_{\text{remaining load},e}
\]

Prediction is therefore useful only when both conditions hold:

1. **Coverage condition:** the future demanded expert set is covered.
2. **Timing condition:** covered experts arrive before demand.

A third condition is required for architectural profitability:

3. **Resource condition:** speculative loading does not consume unacceptable bandwidth, memory capacity, or cache residency.

---

# 4. Problem Statement

## 4.1 Primary Research Question

Can lightweight skip-layer predictors infer the complete future expert demand set early enough to make hierarchical expert storage viable under realistic bandwidth, latency, capacity, and execution constraints?

## 4.2 Secondary Research Questions

1. How skewed is expert popularity globally, per layer, per workload, and within short execution windows?
2. Is global hotness actually useful for fixed residency, or does it disappear when conditioned on layer and workload?
3. How predictable is expert demand across one, two, and three future MoE layers?
4. Does hidden-state information provide predictive value beyond simple expert popularity and transition statistics?
5. How does prediction quality degrade under workload and domain shift?
6. What candidate-set size is required to reach near-complete wave-level coverage?
7. How much speculative transfer amplification is required for a given miss probability?
8. Under what hardware parameters does oracle prefetch succeed?
9. How much of the oracle benefit can a learned predictor recover?
10. When does moving experts lose to moving activations to resident experts?
11. Does prediction mainly help throughput and bandwidth, or can it materially improve strict low-latency decode?
12. What expert granularity is viable: whole expert, matrix, block, tile, quantized fragment, or factorized component?

---

# 5. Key Correction to the Naive Hypothesis

A high per-expert prediction recall does not imply low miss probability for a whole execution wave.

For:

- batch size \(B\),
- top-\(k\) routing,
- per-expert recall \(r\),

a rough independence approximation gives:

\[
P(\text{all demanded experts predicted})
\approx
r^{Bk}
\]

and:

\[
P(\text{at least one miss})
\approx
1-r^{Bk}
\]

Across \(L\) MoE layers in a decode step:

\[
P(\text{no miss in decode step})
\approx
r^{BkL}
\]

This becomes extremely demanding even when \(r\) appears high by ordinary ML standards.

Example:

\[
B=8,\quad k=2,\quad L=16,\quad r=0.99
\]

gives:

\[
P(\text{no miss}) \approx 0.99^{256} \approx 7.6\%
\]

under the simple independence model.

Therefore, the project must optimize and report:

- token-complete coverage;
- execution-wave-complete coverage;
- decode-step-complete coverage;
- residual miss-stall distributions;
- candidate-set size and speculative bytes required to achieve each coverage level.

Ordinary top-1 accuracy is not a sufficient research metric.

---

# 6. Scope

## 6.1 In Scope

- Hugging Face-compatible MoE inference.
- Router instrumentation.
- Per-token and per-layer routing trace collection.
- Expert popularity and temporal locality analysis.
- Lightweight future-expert predictors.
- Trace-driven cache and prefetch simulation.
- Analytical performance modeling.
- Artificial CPU-to-GPU expert loading over PCIe.
- Measured validation of transfer overlap and stall.
- Extrapolation to large-HBM and multi-GPU systems.
- Evaluation of whole-expert and synthetic tile-level transfer sizes.
- Explicit comparison of prediction, static hot-expert caching, and oracle prefetch.
- Tail-latency and critical-path analysis.

## 6.2 Out of Scope for the Initial Project

- Training or fine-tuning the underlying MoE model.
- Changing the model router to be locality-aware.
- Designing a production distributed inference runtime.
- Implementing expert parallelism across eight GPUs in phase one.
- Claiming production speedup from the single-GPU PCIe prototype.
- Building a cycle-accurate GPU simulator.
- Designing new silicon in RTL.
- Solving distributed KV cache and expert placement jointly in the first implementation.
- Proving that whole-expert offload is generally optimal.
- Assuming that on-chip L2 can hold complete transformer experts.

---

# 7. Research Hypotheses

## H1: Expert popularity is skewed

For at least some models, layers, and workloads, a minority of experts accounts for a disproportionate fraction of routed tokens.

### Required evidence

Report per-layer and per-workload:

- expert probability distribution;
- entropy;
- Gini coefficient;
- top-\(n\) demand coverage;
- maximum-to-median popularity ratio;
- short-window popularity stability.

### Failure interpretation

If demand is nearly uniform within operationally relevant windows, fixed hot-expert caching has weak value.

---

## H2: Temporal and conditional locality is stronger than global popularity

Future expert demand may be significantly more predictable when conditioned on:

- current hidden state;
- current expert choice;
- token position;
- request identity or domain;
- recent routing history.

### Required evidence

Compare:

\[
P(E_{l+\Delta})
\]

against:

\[
P(E_{l+\Delta}\mid E_l)
\]

and:

\[
P(E_{l+\Delta}\mid h_l)
\]

and combinations thereof.

### Failure interpretation

If a learned predictor does not materially beat popularity and transition baselines, the skip-router hypothesis is weak.

---

## H3: Lightweight predictors can recover useful future-expert candidate sets

A linear or small MLP predictor can identify a compact candidate set containing all future selected experts with high probability.

### Required evidence

For \(\Delta \in \{1,2,3\}\), report:

- recall at candidate budget;
- token-complete coverage;
- wave-complete coverage;
- decode-step-complete coverage;
- precision;
- calibration;
- speculative amplification;
- generalization under domain shift.

### Failure interpretation

If near-complete coverage requires predicting most experts, prefetch prediction is not selective enough to be architecturally useful.

---

## H4: Oracle prefetch identifies a viable hardware region

There exist combinations of:

- expert size;
- transfer bandwidth;
- transfer startup latency;
- lookahead time;
- cache capacity;
- batch size;
- demand skew;
- reuse;

for which perfect future knowledge can avoid a meaningful fraction of cold-expert stalls.

### Required evidence

A trace-driven simulator must show a nontrivial feasible region.

### Failure interpretation

If oracle prefetch fails, learned prediction cannot rescue the architecture. The bottleneck is physical timing or bandwidth, not prediction quality.

---

## H5: Learned prediction recovers a meaningful fraction of oracle benefit

Define:

\[
\eta_{\text{oracle recovery}}
=
\frac{
\text{stall reduction from learned prefetch}
}{
\text{stall reduction from oracle prefetch}
}
\]

The learned policy should recover a material fraction of oracle benefit without excessive speculative traffic.

H5 begins with a first-order requirements study rather than a higher-fidelity
performance model:

1. sweep assumed complete-route coverage, candidate amplification, capacity,
   lookahead, and normalized cold bandwidth;
2. solve the inverse problem for minimum predictor quality;
3. place the existing transition and linear streams on that surface without
   retraining.

The detailed work queue and visualization plan are in
[docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md).

### Failure interpretation

Low oracle recovery means the predictor is not sufficiently useful even if oracle feasibility exists.

### Current pilot result

H5 found a controlled first-order profitability region but no profitable
unchanged policy placement. At K=32 the existing policies cover 67–81% of
complete cold sets, yet blindly transferring every nonresident candidate costs
6.3–6.7× useful bytes. The immediate problem is selective admission, not
additional prediction capacity. A post-hoc score sweep finds strong pairwise
separation for the linear sidecar (AUROC 0.883/0.861 at Δ=3/9), but preserving
50% complete cold-set coverage still requires about 3.0–3.3× amplification.
The class-conditional score distributions have real but incomplete separation:
linear JS divergence is 0.381/0.332 bits with 38.7/42.8% overlap. These are
useful-demand versus unused scores among nonresident IDs, not globally hot
versus cold expert distributions. The 7–8% useful-candidate base rate explains
why good ranking does not automatically yield cheap admission.
See [docs/H5_RESULTS.md](docs/H5_RESULTS.md).

---

## H6: Prediction is more useful for residency planning than strict just-in-time loading

Prediction may be most valuable for:

- selecting experts to replicate;
- reserving staging capacity;
- planning short-horizon residency;
- reducing conservative prefetch bandwidth;
- informing remote execution placement;
- prioritizing transfer queues.

This may remain true even if strict low-latency just-in-time prefetch fails.

### Current pilot result

The tested form is not supported. H6 replayed static popularity,
domain-conditioned popularity, reactive LRU, transition-guided residency,
linear-guided residency, and an equal-budget next-use oracle over both phases,
all valid layer pairs, and K=8/16/32. Prediction could admit only an actually
demanded miss and could not trigger candidate-only prefetch.

At the frozen decode \(K=16,\Delta=3\) gate, transition and linear are 3.9 and
2.5 percentage points worse than the strongest matched simple baseline on
expert-stall reduction, and 0.7 and 0.6 points worse on complete resident-set
hits. Neither is positive across any domain or layer on average. The oracle
still cuts residual cold demand from LRU's 48.1% to 31.2%, showing that the
residency mechanism has headroom but the existing depth-trajectory scores do
not supply the needed temporal-reuse information.

The direct lesson is:

> Predicting \(E_{\ell+\Delta,t}\) for the current token is not the same task
> as predicting \(E_{\ell,t+\tau}\) for future tokens.

See [docs/H6_RESULTS.md](docs/H6_RESULTS.md).

---

## H7: Predictable routing can be encouraged without sacrificing model quality

A later controlled intervention will test whether adding one
trajectory-predictability objective can improve complete future-route coverage
and modeled hardware benefit while preserving validation loss and marginal
load balance.

Current H1–H6 evidence is observational and cannot support this claim. The H6
placement failure also removes the immediate justification for this training
intervention.

---

## C0: Post-training materially changes routing-trajectory predictability

This within-family confirmation compared the OLMoE Base checkpoint with its
final SFT+DPO+RLVR Instruct descendant under exactly matched input tokens.

### Current pilot result

The tested claim is not supported. At the preregistered prefill layer-0→15,
K=16 gate, transition selection gain over static popularity changes from
+11.0 points for Base to +12.6 points for Instruct. The +1.6-point difference
is below the frozen 5-point threshold.

The stronger result is stability:

- 89.7% of selected expert IDs are shared across checkpoints;
- exact top-8 route sets match for 35.7% of token-layer events;
- K=16 hot sets have 84.6% Jaccard overlap;
- transition tables transfer across endpoints with small average penalties.

For this lineage, post-training makes local expert substitutions while
preserving the pretrained trajectory scaffold and its predictability.
Do not add the intermediate SFT/DPO checkpoints after the failed endpoint
gate. See [docs/C0_RESULTS.md](docs/C0_RESULTS.md).

---

## C1: The trajectory result transfers to a more sparsely routed checkpoint

Before making a general MoE claim, repeat the decisive trace and normalized
co-design analysis on one newer top-1/top-2 checkpoint. This is a confirmation
experiment, not a reason to broaden the current OLMoE prototype.

---

## AX: Future-predictor architecture envelope

The active architecture track makes one explicit optimistic assumption:
future MoE training can produce an MTP-style routing gate whose additional
heads expose accurate multi-horizon expert demand without materially degrading
language-model loss or load balance.

AX does not treat that assumption as an empirical result. It uses:

- measured OLMoE expert size and host-to-device calibration;
- trace-derived demand sets, cold rates, reuse, and burstiness;
- independently swept complete cold-set coverage and false-positive
  amplification;
- hypothetical host/pooled-memory, HBM, and software-managed SRAM parameters.

The goal is to derive necessary and sufficient first-order conditions,
inverse-design requirements, capacity/latency Pareto frontiers, and concrete
workload/software/hardware interfaces. Wall-clock improvement on current
OLMoE is not an exit condition.

The track contains:

1. **AX1:** capacity and TPOT envelope for predictive host/pooled-memory
   offload;
2. **AX2:** bandwidth/latency/reliability/granularity regime classification;
3. **AX3:** long-horizon HBM placement plus short-horizon rolling SRAM staging.

Complete cold-set coverage and predicted/useful byte amplification are primary
predictor axes. Synthetic misses are generated at wave level so correlated
false negatives remain visible in tail latency. CPU offload is compared with
reactive CPU offload, not claimed faster than all-HBM execution.

See
[docs/ARCHITECTURE_EXPLORATION_PROTOCOL.md](docs/ARCHITECTURE_EXPLORATION_PROTOCOL.md)
for the frozen evidence contract and interpretation rules. The canonical
result is
[artifacts/runs/h1-standard-small/analysis/architecture/REPORT.md](artifacts/runs/h1-standard-small/analysis/architecture/REPORT.md).

The completed sweep finds:

- at measured PCIe and assumed C=99%, A=1.5×, wave-local P99 improves
  34–39% over reactive offload at K=8/16/32, while remaining 3.0–4.7× slower
  than the all-resident measured reference;
- the K=16, A=1× mean bandwidth bound falls from 71.3 GB/s at Δ=1 to
  22.8/11.6/8.2 GB/s at Δ=3/6/9;
- selected FCFS queue replay is materially worse than the wave-local envelope,
  so mean headroom is necessary but not tail-sufficient;
- top-8 whole-expert SRAM staging needs a 192 MiB double buffer at A=1× and
  384 MiB at A=2×, making staging capacity/pollution a first-order constraint.

### AX4 follow-on: deadline-elastic expert execution

AX4 relaxes exact expert-set execution. At a fixed layer deadline, the runtime
commits every available routed contribution and substitutes, renormalizes, or
omits unavailable contributions. Late transfers cannot stall the token.

For normalized selected weights \(a_i\), missing routed mass is

\[
m_{t,l}=1-\sum_{i\in A_{t,l}}a_i.
\]

This changes the critical resource contract:

- TPOT is bounded by reserved local compute, fallback, merging, and scheduler
  work rather than cold-transfer completion;
- prediction affects missing mass, fallback load, traffic, and quality rather
  than the committed latency;
- complete-set coverage is replaced by P99 missing mass and full-fallback
  incidence;
- importance-per-byte admission becomes more relevant than expert-ID recall.

The preferred future workload form is an always-resident shared expert plus
optional routed residual experts. Its per-layer degradation is bounded by the
missing weighted residual contribution. The immediate retained-trace replay
can quantify that normalized bound and the TPOT/capacity Pareto, but not loss
or generation quality. Availability-conditioned training remains deferred
until the analytical contract passes and requires explicit permission.

The first quantitative prediction is a current-testbed calibration, not a
quality claim: 10.23 ms local decode plus a 10% fallback/commit allowance
implies an 11.25 ms deadline cap (88.9 batch-1 tokens/s), compared with the
66.83 ms and 15.0 tokens/s K=16 reactive-offload P99 projection—a 5.94×
same-batch throughput projection. For a future 20 ms bounded step, ideal
synchronous throughput is 50/100/200 tokens/s at batch 1/2/4. AX4 determines
the missing-mass price paid for those bounds.

AX4 now passes its formal analytical gate, but only in a high-bandwidth,
mass-priority regime. The strongest capacity point keeps K=8 of 64 experts per
layer resident at 256 GB/s with C=99%, A=1.5×, Δ=1, and one layer interval of
commit slack. Its bounded TPOT is 11.25 ms versus 16.41 ms for reactive exact
offload on the same hierarchy; 0.93% of waves have any missing mass, P99 wave
missing mass is effectively zero, and full fallback is zero. K=32 at 128 GB/s
also passes with 7.2% P99 missing mass. Measured PCIe fails at every capacity
with 81–100% P99 missing mass. K=16 at 128 GB/s is a useful near-boundary point
at 20.4%.

The selected weights require a semantic caveat: current OLMoE softmaxes over
all 64 experts, selects eight, and does not renormalize them. The selected
probabilities sum to 0.406 on average. Normalized-within-top-8 missing mass is
therefore a future architecture/training contract, not current OLMoE's exact
execution scale.

See
[docs/DEADLINE_DEGRADATION_PROTOCOL.md](docs/DEADLINE_DEGRADATION_PROTOCOL.md)
for definitions, gates, low-batch/large-model projections, and the
deadline-elastic hardware proposal.

### Q1 / Q1-B: does the frozen model tolerate the erasure AX4 relies on?

Q1 measured quality versus missing routed mass directly on the frozen base
checkpoint (forward passes only). Two regimes:

- **Universal mass-budget erasure** (every token, every layer) is
  catastrophic regardless of policy: the renormalize headline cell
  (m=0.125) gives KL 5.81, top-1 9.2%, PPL 279.9. This is a **STOP** for any
  mechanism that drops mass routinely.
- **AX4-faithful tail erasure** (rare ~0.9% of tokens, bounded run of
  layers) is far less destructive — and under the model's native **null-drop**
  (no renormalization) a single one-expert one-layer drop is nearly free
  (conditional KL 0.0032, top-1 99.2%). Renormalization is ~40× worse at
  equal mass and theoretically wrong (it rescales survivors, amplifying the
  removed mass), so it is dropped as a strategy.

**Q1-B** then mapped the null-drop mechanism. On the frozen base model,
conditional-on-affected quality cost is **monotone and roughly additive in
the number of dropped expert-layers**: worst case L=8 is 0.013 nats KL with
zero large divergences; per-layer marginal is flat (last/first ratio 1.20);
sensitivity is **layer-uniform** (~1.6× spread across all 16 layers); spacing
the drops confers no reconstruction benefit; and damage does not leak to other
tokens (downstream ≈ far control). This is direct evidence for the
**additive-residual mechanism** — each routed expert adds a mostly independent
weighted update to the residual stream, so removing one per layer removes a
small independent increment.

**Interpretation:** the AX4 bounded-run quality contract is *measured*, not
assumed, and is cheap: even the worst case (8 consecutive degraded layers)
is semantically negligible. See `docs/Q1B_PROTOCOL.md` and
`artifacts/runs/q1-quality-erasure/analysis/q1b_null/NULL_REPORT.md`.

**Q2 (complete):** because the frozen model already tolerates the AX4 null-drop
tail for free in the measured regime (prefill, one domain), robust training is
no longer an unconditional requirement. Q2 verified that tolerance across the
untested axes and located the cliff with three measured, non-training stress
arms (cross-domain, decode compounding, cliff mapping) reusing the Q1-B
machinery. Outcomes: cross-domain **GO** (holds on gsm8k math), cliff
**WITH_MARGIN** (AX4 cell free, first cliff at 2 experts/layer), decode
**divergence not degradation** (clean/erased streams fly apart after a
near-tie flip, but the erased output is a fluent paraphrase — the step-KL
overstates quality loss). No real non-free regime appeared, so the gated
minimal mask-aware calibration (not generic dropout) is **not entered**.
See `docs/Q2_PROTOCOL.md` and `EXPERIMENT_LOG.md`.

---

# 8. Architectural Claims to Test, Not Assume

The project should test the following claims.

## Claim A: Hot experts justify a fast residency tier

This is supported only if hotness is:

- strong;
- stable enough;
- layer-specific;
- reusable over a meaningful residency interval.

## Claim B: Skip-layer prediction creates enough time to load cold experts

This is supported only if:

\[
T_{\text{lookahead}} \ge T_{\text{load}}
\]

for actual measured layer times and realistic transfer sizes.

## Claim C: Prediction can reduce low-latency stalls

This is supported only if wave-complete or step-complete coverage is sufficiently high. Mean accuracy is insufficient.

## Claim D: Whole-expert movement is viable

This is supported only if transfer cost is amortized over enough token demand or hidden under enough compute.

## Claim E: Hierarchical storage is profitable

This is supported only if total system cost improves after accounting for:

- transfer bandwidth;
- queueing;
- cache pollution;
- wasted prefetch;
- capacity reservation;
- synchronization;
- residual misses.

## Claim F: Future large-HBM multi-GPU systems benefit

This is supported only after comparing:

1. moving experts;
2. moving activations to resident experts;
3. selective expert replication;
4. local HBM residency;
5. peer-HBM access or migration.

---

# 9. Experimental Architecture

The implementation should be modular and divided into six subsystems:

1. **Model adapter**
2. **Trace collector**
3. **Dataset and feature pipeline**
4. **Predictor training and evaluation**
5. **Trace-driven simulator**
6. **PCIe prototype and timing calibrator**

Recommended repository structure:

```text
predictive-expert-prefetch/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── model/
│   ├── dataset/
│   ├── trace/
│   ├── predictor/
│   └── simulator/
├── src/
│   └── pep/
│       ├── cli.py
│       ├── config.py
│       ├── models/
│       │   ├── base.py
│       │   ├── hf_adapter.py
│       │   └── router_discovery.py
│       ├── tracing/
│       │   ├── hooks.py
│       │   ├── schema.py
│       │   ├── writer.py
│       │   └── validation.py
│       ├── data/
│       │   ├── datasets.py
│       │   ├── token_stream.py
│       │   ├── splits.py
│       │   └── projection.py
│       ├── analysis/
│       │   ├── popularity.py
│       │   ├── locality.py
│       │   ├── transitions.py
│       │   ├── coverage.py
│       │   └── plots.py
│       ├── predictors/
│       │   ├── base.py
│       │   ├── popularity.py
│       │   ├── transition.py
│       │   ├── linear.py
│       │   ├── mlp.py
│       │   └── metrics.py
│       ├── simulation/
│       │   ├── events.py
│       │   ├── cache.py
│       │   ├── transfer.py
│       │   ├── policies.py
│       │   ├── replay.py
│       │   └── reports.py
│       ├── prototype/
│       │   ├── pinned_memory.py
│       │   ├── async_copy.py
│       │   ├── staging_pool.py
│       │   └── timeline.py
│       └── utils/
│           ├── logging.py
│           ├── reproducibility.py
│           └── hardware.py
├── scripts/
│   ├── inspect_model.py
│   ├── collect_trace.py
│   ├── analyze_trace.py
│   ├── train_predictor.py
│   ├── simulate_prefetch.py
│   └── run_pcie_demo.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── golden/
├── notebooks/
│   ├── 01_trace_characterization.ipynb
│   ├── 02_skip_router_results.ipynb
│   ├── 03_oracle_feasibility.ipynb
│   └── 04_hardware_phase_diagrams.ipynb
└── artifacts/
    ├── traces/
    ├── checkpoints/
    ├── reports/
    └── figures/
```

The implemented prototype uses `artifacts/runs/<run-id>/analysis/` rather than
separate report and figure trees. These native paths are the canonical
publication record: compact run metadata, CSV/JSON/MD results, fitted analysis
outputs, and PDF/PNG figures are checked into ordinary Git. Large request-level
`trace/`, projected `features/`, `hidden_states/`, and `activations/`
directories are ignored and currently treated as disposable. A closeout audit
validates hashes and document references before staging all durable artifacts;
see `docs/EXPERIMENT_SOP.md`. This avoids duplicated result bundles and keeps
paper figures directly connected to their machine-readable source tables.

---

# 10. Model Selection Requirements

The first model should be selected for instrumentation simplicity, not benchmark prestige.

Required properties:

- Hugging Face Transformers compatibility.
- Explicit MoE router modules.
- Multiple MoE layers.
- At least 8 experts per MoE layer.
- Top-1 or top-2 routing.
- Fits within 24 GB VRAM in an inference-capable representation.
- Router input and router logits can be intercepted.
- Does not require a distributed runtime for basic generation.
- Supports deterministic or near-deterministic evaluation.

Preferred properties:

- Standard PyTorch modules rather than opaque custom kernels.
- Moderate hidden dimension.
- Public tokenizer and checkpoint.
- Router logits optionally exposed in model outputs.
- Configurable eager-attention or non-fused mode for debugging.
- Quantization support that does not destroy access to router modules.

The implementation must include a model inspection tool that prints:

- module tree;
- candidate router modules;
- expert modules;
- layer-to-router mapping;
- number of experts per layer;
- top-\(k\);
- hidden dimension;
- expert matrix shapes;
- estimated expert bytes by precision;
- total expert bytes;
- non-expert parameter bytes.

Do not hard-code one model architecture into the core pipeline. Use an adapter interface.

---

# 11. Dataset Design

Use a controlled workload matrix.

Minimum workload categories:

1. General prose.
2. Code.
3. Mathematical or reasoning text.
4. Conversation-style prompts.
5. Mixed shuffled workload.

Separate:

- prefill tokens;
- decode tokens;
- prompt source domain;
- generated token position.

Required splits:

## IID split

Train and test from the same workload mixture using disjoint examples.

## Domain-shift split

Train on one or several domains and test on a held-out domain.

## Temporal split

Train on earlier trace segments and evaluate on later segments to detect drift.

The initial target can be 100,000 to 1,000,000 routed token-layer events, depending on storage and runtime.

The coding assistant should make sample counts configurable and support resumable trace collection.

---

# 12. Trace Schema

The trace must preserve enough information to reconstruct routing demand and train predictors.

Recommended logical schema:

```text
TraceRecord
├── run_id: string
├── request_id: int64
├── sample_id: int64
├── phase: enum[prefill, decode]
├── token_position: int32
├── input_token_id: int32
├── layer_id: int16
├── moe_layer_index: int16
├── hidden_feature: float16[d_proj] or optional
├── selected_expert_ids: int16[top_k]
├── selected_expert_weights: float16[top_k]
├── router_logits: optional float16[n_experts]
├── timestamp_ns: optional int64
├── batch_id: int64
├── batch_size: int32
├── dataset_name: string
├── domain: string
└── metadata_version: int16
```

Recommended storage:

- Parquet or Arrow for metadata and routing decisions.
- Sharded tensor files for projected hidden states if needed.
- JSON metadata for run configuration.
- No Python pickle for durable research artifacts.

Required metadata:

- model identifier and revision;
- tokenizer identifier and revision;
- precision and quantization;
- software versions;
- CUDA and driver versions;
- random seeds;
- dataset revisions;
- generation parameters;
- router discovery mapping;
- projection matrix hash;
- trace schema version.

---

# 13. Hidden-State Storage Strategy

Saving full hidden states for every token and layer is likely too expensive.

Preferred initial strategy:

\[
x_l = R h_l
\]

where:

- \(h_l \in \mathbb{R}^{d}\) is the router input;
- \(R \in \mathbb{R}^{d_p \times d}\) is a fixed random projection;
- \(d_p \in \{64,128,256\}\).

The same projection matrix must be used across all runs for a given experiment.

Recommended options:

1. Store projected states for all events.
2. Store full states for a small sampled subset.
3. Optionally train predictors online to reduce storage.
4. Store router logits when affordable.
5. Never silently mix hidden states taken from different points in the block.

The trace metadata must state precisely whether the predictor feature is:

- residual stream entering the MoE block;
- normalized router input;
- router logits;
- selected expert embedding;
- post-attention hidden state;
- another architecture-specific location.

---

# 14. Router Instrumentation Requirements

The trace collector should support two strategies.

## Strategy A: Forward hooks

Use when router modules are explicit and hooks expose the needed tensors.

## Strategy B: Minimal source adapter

Use when the model architecture computes routing internally and hooks are insufficient.

The collector must verify:

- routing decisions match the model’s actual selected experts;
- batch and token dimensions are interpreted correctly;
- padding tokens are excluded where appropriate;
- top-\(k\) ordering is deterministic;
- tensor-parallel or fused layouts are not silently misread;
- repeated generation steps are assigned correct token positions.

The trace collector should include a validation mode that:

1. captures a tiny batch;
2. compares stored top-\(k\) experts to router logits;
3. confirms layer ordering;
4. confirms token ordering;
5. checks that each expected MoE layer emitted one record per routed token.

---

# 15. Routing Characterization

The first analysis stage must generate the following.

## 15.1 Popularity Metrics

For each layer and workload:

\[
p_{l,e}
=
\frac{\text{tokens routed to expert }e}
{\text{total expert selections at layer }l}
\]

Report:

- histogram;
- rank-frequency plot;
- entropy;
- normalized entropy;
- Gini coefficient;
- top-1, top-2, top-4, top-8, and top-quartile coverage;
- expert utilization variance.

## 15.2 Windowed Hotness

For window sizes measured in:

- tokens;
- decode steps;
- batches;
- wall-clock-equivalent trace intervals;

report:

- overlap between consecutive hot sets;
- Jaccard similarity;
- rank correlation;
- hot-set half-life;
- adaptation benefit over static global caching.

## 15.3 Reuse Distance

For each expert, measure routed-event distance between consecutive uses.

Report:

- reuse-distance CDF;
- reuse time by layer;
- reuse conditioned on workload;
- reuse conditioned on batch size;
- probability of reuse before a given cache eviction horizon.

## 15.4 Co-occurrence

Measure:

- experts selected together by one token;
- experts demanded by the same execution wave;
- layer-wise co-occurrence matrices;
- candidate replication groups;
- batch-level union size of selected experts.

## 15.5 Transition Statistics

For lookahead \(\Delta\):

\[
P(E_{l+\Delta}=e' \mid E_l=e)
\]

Report transition entropy and predictive gain over marginal popularity.

## 15.6 Cross-Workload Stability

Compare expert rankings and transition matrices across domains.

---

# 16. Predictor Tasks

For each source layer \(l\) and lookahead \(\Delta \in \{1,2,3\}\), train:

\[
f_{l,\Delta}(x_l)
\rightarrow
\hat{D}_{l+\Delta}
\]

where \(D_{l+\Delta}\) is the actual top-\(k\) future expert set.

Do not assume one predictor can be shared across layers.

Required baselines:

1. **Global popularity**
2. **Per-layer popularity**
3. **Windowed popularity**
4. **Expert transition table**
5. **Conditional transition table using recent route history**
6. **Linear classifier**
7. **Small MLP**

Optional later baselines:

- low-rank classifier;
- shared predictor with layer embeddings;
- recurrent route-history model;
- calibrated ensemble;
- conformal candidate-set predictor.

The first implementation should remain deliberately simple.

---

# 17. Predictor Evaluation Metrics

## 17.1 Per-Expert Recall

\[
r_{\text{expert}}
=
\frac{
|\hat D \cap D|
}{
|D|
}
\]

Useful but insufficient.

## 17.2 Exact Token-Complete Coverage

\[
r_{\text{token}}
=
P(D_t \subseteq \hat D_t)
\]

This asks whether all experts required for one token are covered.

## 17.3 Wave-Complete Coverage

For execution wave \(w\):

\[
D_w = \bigcup_{t \in w} D_t
\]

\[
r_{\text{wave}}
=
P(D_w \subseteq C_{\text{resident}} \cup P_{\text{prefetched}})
\]

This is the main low-latency metric.

## 17.4 Decode-Step-Complete Coverage

\[
r_{\text{step}}
=
P(
\forall l,\,
D_l \subseteq C_l \cup P_l
)
\]

This estimates the probability of no cold miss during one generated token step.

## 17.5 Candidate Budget

Evaluate candidate-set sizes such as:

\[
k_p \in \{k, 2k, 4k, 8k\}
\]

and fixed fraction-of-expert budgets.

## 17.6 Prefetch Amplification

\[
A_{\text{bytes}}
=
\frac{
\text{all predicted transfer bytes}
}{
\text{useful predicted transfer bytes}
}
\]

Also report:

\[
A_{\text{experts}}
=
\frac{
|\hat D|
}{
|D|
}
\]

## 17.7 Miss Risk

Report:

- probability of at least one miss per wave;
- probability of at least one miss per decode step;
- conditional miss probability by workload;
- miss clustering;
- maximum consecutive misses;
- expert-specific miss rates.

## 17.8 Calibration

For probabilistic predictors, measure whether predicted set confidence corresponds to actual complete-set coverage.

## 17.9 Domain-Shift Robustness

Compare IID and held-out-domain performance.

---

# 18. Tail-Safe Candidate-Set Prediction

A useful predictor should output the smallest candidate set \(S\) satisfying:

\[
P(D_{l+\Delta} \subseteq S \mid x_l)
\ge 1-\epsilon
\]

This is a set-coverage problem rather than ordinary top-\(k\) classification.

The implementation should support:

- fixed candidate budget;
- confidence-threshold candidate sets;
- per-layer adaptive candidate size;
- target miss probability;
- target bandwidth budget.

The central plot should be:

\[
\text{prefetch bytes or candidate count}
\quad\text{vs.}\quad
P(\text{wave miss})
\]

A second key plot should be:

\[
\text{prefetch bytes}
\quad\text{vs.}\quad
P(\text{decode-step miss})
\]

---

# 19. Analytical Performance Model

The simulator and analysis should share one explicit cost model.

## 19.1 Transfer Model

For tier \(i \rightarrow j\):

\[
T_{i\rightarrow j}(S)
=
\alpha_{ij}
+
\frac{S}{\beta_{ij}^{\text{effective}}}
+
Q_{ij}
\]

Support:

- fixed effective bandwidth;
- size-dependent measured bandwidth;
- bounded transfer concurrency;
- shared bandwidth;
- startup latency;
- queueing;
- transfer priorities.

## 19.2 Lookahead Slack

\[
T_{\text{slack}}(l,\Delta)
=
\sum_{j=l}^{l+\Delta-1}
T_j
\]

Use measured layer timing where available.

A prefetch is timely if:

\[
T_{\text{completion},e}
\le
T_{\text{demand},e}
\]

## 19.3 Residual Stall

\[
T_{\text{stall},e}
=
\max(
0,
T_{\text{completion},e}
-
T_{\text{demand},e}
)
\]

For a synchronous wave:

\[
T_{\text{wave stall}}
=
\max_{e \in D_w}
T_{\text{stall},e}
\]

This maximum, not the average expert stall, controls wave latency.

## 19.4 Bandwidth Constraint

For token rate \(R\):

\[
R \cdot B_{\text{prefetch/token}}
\le
u \cdot \beta_{\text{tier}}
\]

where \(u\) is an allowed utilization fraction.

## 19.5 Reuse Amortization

For expert \(e\) loaded once and reused by \(n_e\) tokens:

\[
\bar B_e
=
\frac{S_e}{n_e}
\]

The simulator must track reuse across a configurable residency lifetime.

## 19.6 Move-Token vs. Move-Expert Comparison

Expert movement:

\[
T_{\text{move expert}}
=
\alpha
+
\frac{S_e}{\beta}
+
T_{\text{local compute}}
\]

Activation movement:

\[
T_{\text{move token}}
=
\alpha
+
\frac{S_a}{\beta}
+
T_{\text{remote queue}}
+
T_{\text{remote compute}}
+
\alpha
+
\frac{S_o}{\beta}
\]

The project must not assume expert movement wins.

---

# 20. Trace-Driven Simulator

The simulator should be event-driven but not cycle-accurate.

## 20.1 State

Track:

- resident expert set by layer;
- in-flight prefetches;
- transfer queue;
- staging-buffer occupancy;
- expert last-use time;
- predicted future demand;
- actual demand;
- cache capacity;
- transfer bandwidth;
- transfer concurrency;
- layer execution timeline;
- optional batch execution waves.

## 20.2 Policies

Required policies:

1. All resident.
2. Static most-popular experts.
3. LRU on demand.
4. Reactive cold load.
5. Oracle prefetch.
6. Popularity prefetch.
7. Transition prefetch.
8. Learned skip-router prefetch.
9. Conservative union prefetch.
10. Prediction-guided admission.
11. Prediction-guided replication proxy.
12. Move-token baseline.

## 20.3 Eviction Policies

Required:

- LRU;
- LFU;
- static pinned hot set plus LRU remainder;
- Belady oracle;
- predicted next-use;
- cost-aware eviction.

## 20.4 Outputs

Report:

- resident hit rate;
- useful prefetch rate;
- late prefetch rate;
- useless prefetch rate;
- demand miss rate;
- wave miss rate;
- step miss rate;
- average stall;
- P50/P90/P95/P99 stall;
- maximum stall;
- transferred bytes;
- useful transferred bytes;
- speculative amplification;
- bandwidth utilization;
- queue depth;
- cache occupancy;
- eviction-induced misses;
- oracle gap;
- learned/oracle benefit ratio.

---

# 21. Oracle-First Gating

The implementation must evaluate oracle feasibility before optimizing learned predictors.

Required sequence:

1. Measure expert sizes and layer times.
2. Configure realistic transfer parameters.
3. Run perfect-future oracle prefetch.
4. Sweep cache capacity, bandwidth, latency, and expert granularity.
5. Identify whether any meaningful stall reduction is physically possible.
6. Only then evaluate learned prediction.

This prevents spending time improving a predictor for a physically infeasible mechanism.

---

# 22. Artificial PCIe Prototype

The prototype is a mechanism demonstration, not a production benchmark.

## 22.1 Memory Tiers

- **Fast tier:** GPU VRAM.
- **Slow tier:** pinned CPU memory.
- **Optional slower tier:** pageable host memory for contrast.

## 22.2 Required Features

- pinned host buffers;
- reusable GPU staging buffers;
- nonblocking `cudaMemcpyAsync` or equivalent PyTorch asynchronous copy;
- dedicated CUDA streams;
- CUDA events for timing;
- explicit demand synchronization;
- bounded in-flight transfers;
- transfer cancellation or obsolete-prefetch accounting where practical;
- timeline export.

## 22.3 Prototype Modes

1. Reactive load.
2. Oracle prefetch.
3. Popularity prefetch.
4. Learned prefetch.
5. Intentionally wrong prefetch.
6. Broad candidate prefetch.

## 22.4 Measurements

- transfer time by size;
- achieved PCIe bandwidth;
- startup latency;
- overlap fraction;
- residual demand stall;
- GPU staging occupancy;
- wasted bytes;
- queue depth;
- analytical-model prediction error.

## 22.5 Synthetic Payloads

Synthetic payloads may be attached to expert identities to emulate future transfer sizes.

The implementation must clearly distinguish:

- real model expert weights;
- copied proxy buffers;
- synthetic transfer-only payloads.

No synthetic result may be presented as a real end-to-end model speedup.

---

# 23. Timing Calibration

Before running the simulator, measure:

1. CPU pinned to GPU transfer time across sizes.
2. GPU to CPU transfer time across sizes.
3. Concurrent transfer scaling.
4. Transfer overlap with representative GEMM or model-layer compute.
5. CUDA event overhead.
6. Per-layer model execution time.
7. Router and MoE block time where separable.
8. Cold-start versus steady-state transfer behavior.

Recommended transfer size sweep:

```text
4 KB
16 KB
64 KB
256 KB
1 MB
4 MB
16 MB
32 MB
64 MB
128 MB
256 MB
```

Fit:

\[
T(S) = \alpha + \frac{S}{\beta_{\text{effective}}}
\]

and retain empirical interpolation for nonlinearity.

---

# 24. Small-Batch Decode Semantics

The project must distinguish semantic dependencies from implementation barriers.

Within one sequence, the current token cannot advance past an MoE block until all selected expert contributions are available.

Across independent sequences, a batch-wide barrier is often an implementation choice rather than a mathematical requirement. However, allowing independent sequence progression can fragment batching and reduce GPU utilization.

The initial simulator should support two modes:

## Synchronous wave mode

All sequences in a batch wait for the slowest expert demand.

\[
T_{\text{wave}} =
\max_t T_t
\]

This is the primary mode for strict low-latency analysis.

## Independent sequence mode

Sequences may advance independently after their own expert demands complete.

This mode should account for reduced effective batch size or an explicit efficiency penalty.

The project should not claim that asynchronous progression is free.

---

# 25. Gotchas and Failure Modes

## 25.1 Global skew may be misleading

Global popularity can appear highly skewed while per-layer or short-window demand is not.

Always report:

- per-layer;
- per-domain;
- per-phase;
- per-window statistics.

## 25.2 Layer expert namespaces may differ

Expert 7 at layer 4 is not the same object as expert 7 at layer 20.

Do not aggregate expert IDs across layers without a composite key.

## 25.3 Router predictions may leak target information

Ensure the source feature is available at the intended prefetch issue time.

Do not train on:

- future hidden states;
- future router logits;
- post-target-layer features;
- labels indirectly embedded in trace ordering.

## 25.4 Decode and prefill are different regimes

Do not combine them into one headline metric.

## 25.5 Per-expert recall hides complete-set failure

Always report token-, wave-, and step-complete coverage.

## 25.6 A cache hit is not equal to zero cost

Resident weights still require HBM or cache bandwidth and compute.

The simulator should separate:

- capacity miss;
- transfer stall;
- weight streaming;
- compute.

## 25.7 L2 is not a software object store

Do not model complete experts as ordinary named L2 objects unless explicitly studying a hypothetical architecture.

## 25.8 Whole experts may be too large

Calculate real expert sizes from parameter shapes and precision.

## 25.9 PCIe may be bandwidth-feasible but latency-infeasible

Both constraints must be checked.

## 25.10 Prediction can increase contention

False prefetches may delay useful transfers.

## 25.11 Average speedup can hide P99 regression

Always report tail latency.

## 25.12 Quantization changes both size and execution path

Record precision and kernel path for every result.

## 25.13 Hooking may perturb performance

Trace collection runs are for workload characterization. Do not use hooked timing as production timing without validation.

## 25.14 Generation settings alter routing

Record temperature, top-p, top-k sampling, repetition penalties, and seeds.

## 25.15 Dataset duplication can inflate predictability

Deduplicate or at least detect repeated passages.

## 25.16 Batch composition changes expert union size

Analyze batch size and workload mixing separately.

## 25.17 Correlated misses invalidate simple independence estimates

Use empirical wave and step miss rates, not only \(r^{BkL}\).

## 25.18 Static expert popularity may reflect model imbalance

High skew can indicate router imbalance rather than useful semantic locality.

## 25.19 Expert prediction may be trivial from current router state

That is acceptable only if the required source signal is genuinely available at the issue point and yields useful lookahead.

## 25.20 The prototype may measure copying proxy buffers rather than loading usable expert state

Keep this distinction explicit.

---

# 26. Reproducibility Requirements

Every experiment must save:

- configuration;
- random seeds;
- command line;
- Git commit;
- environment package lock;
- hardware inventory;
- driver and CUDA versions;
- model and dataset revisions;
- trace schema version;
- predictor checkpoint;
- evaluation split definition;
- raw metric tables;
- generated figures.

All scripts should be restartable and deterministic where practical.

Use structured logging.

Avoid notebooks as the only source of truth. Notebooks should call library code.

After every major experiment, prefer one plain-language headline curve and one
compact regime heatmap from immutable metric tables, then pause for human
review. Apply one simple preregistered gate unchanged; use cheap broad post-hoc
analysis of existing artifacts to find layer, domain, horizon, and capacity
regimes rather than preregistering a complex hypothesis tree. Post-hoc findings
may narrow the interpretation but must not rewrite the gate. Add request-level
uncertainty and fresh confirmation data only after a result is borderline or a
physical feasibility gate justifies the extra work. Save PDF and high-resolution
PNG plus a manifest of figure inputs.

---

# 27. CLI Requirements

Target commands:

```bash
uv run python scripts/inspect_model.py --config configs/model/example.yaml

uv run python scripts/collect_trace.py \
  --model-config configs/model/example.yaml \
  --dataset-config configs/dataset/mixed.yaml \
  --trace-config configs/trace/default.yaml

uv run python scripts/analyze_trace.py \
  --trace artifacts/traces/run_id

uv run python scripts/train_predictor.py \
  --trace artifacts/traces/run_id \
  --predictor-config configs/predictor/linear.yaml

uv run python scripts/simulate_prefetch.py \
  --trace artifacts/traces/run_id \
  --checkpoint artifacts/checkpoints/model.pt \
  --sim-config configs/simulator/pcie.yaml

uv run python scripts/run_pcie_demo.py \
  --sim-config configs/simulator/pcie.yaml \
  --mode learned
```

Each command should support `--help`, validation, and explicit output directories.

---

# 28. Testing Requirements

## Unit Tests

- router output parsing;
- top-\(k\) set construction;
- projection determinism;
- trace serialization;
- popularity metrics;
- Gini and entropy calculations;
- transition tables;
- complete-set coverage;
- cache insertion and eviction;
- transfer timing;
- event ordering;
- stall calculation.

## Integration Tests

- tiny model trace collection;
- trace round trip;
- predictor training on synthetic routing data;
- simulator replay on hand-computed cases;
- PCIe copy correctness;
- asynchronous overlap sanity check.

## Golden Tests

Create a tiny synthetic trace for which:

- exact cache hits;
- oracle prefetches;
- late prefetches;
- evictions;
- wave stalls;

can be calculated manually.

---

# 29. Milestones

## Milestone 0: Environment and model inspection

Deliverables:

- reproducible `uv` environment;
- hardware report;
- model inspection script;
- confirmed router and expert module mapping;
- expert size table.

Exit gate:

- one forward pass succeeds;
- selected experts are verified against router logits.

---

## Milestone 1: Routing trace collection

Deliverables:

- trace schema;
- sharded writer;
- one small validated trace;
- one larger trace over multiple workloads;
- trace integrity report.

Exit gate:

- all MoE layers produce expected records;
- no token or layer ordering ambiguity remains.

---

## Milestone 2: Routing characterization

Deliverables:

- per-layer popularity plots;
- top-\(n\) coverage plots;
- Gini and entropy tables;
- reuse-distance CDFs;
- transition matrices;
- cross-domain comparison.

Exit gate:

- hotness and locality are quantified without relying on a global histogram alone.

---

## Milestone 3: Skip-layer prediction

Deliverables:

- popularity baseline;
- transition baseline;
- linear predictor;
- MLP predictor;
- \(\Delta=1,2,3\) results;
- IID and domain-shift evaluation;
- candidate-budget tradeoff.

Exit gate:

- learned predictors are compared against non-ML baselines;
- token-, wave-, and step-complete coverage are reported.

---

## Milestone 4: Oracle feasibility simulator

Deliverables:

- calibrated PCIe transfer model;
- cache simulator;
- oracle prefetch policy;
- static caching baseline;
- reactive load baseline;
- phase diagrams.

Exit gate:

- a viable or nonviable physical region is identified before learned prefetch is claimed useful.

---

## Milestone 5: Learned prefetch simulation

Deliverables:

- controlled prediction-quality × hardware-assumption sweep;
- inverse minimum-predictor-quality curves;
- learned-policy replay;
- prefetch amplification;
- residual stall distribution;
- oracle-recovery ratio;
- bandwidth and capacity sensitivity.

Exit gate:

- an analytical profitability window is identified;
- required predictor quality is explicit;
- architectural conclusions are tied to the existing predictor behavior.

---

## Milestone 5A: Assumption-driven architecture envelope

Deliverables:

- wave-level synthetic predictor streams spanning complete coverage and
  transfer amplification;
- capacity-versus-P99 Pareto frontier for predictive host/pooled-memory
  offload;
- inverse bandwidth, coverage, amplification, object-size, and residency
  requirements;
- latency/bandwidth/reliability/granularity phase classification;
- three-tier pooled-memory/HBM/rolling-SRAM projection;
- explicit measured, trace-derived, assumed, and hypothetical evidence labels.

Exit condition:

- necessary and sufficient first-order regions are quantified;
- capacity viability, improvement over reactive hierarchy, and SLO safety are
  reported separately;
- future-router assumptions are not described as current-model measurements;
- at least one concrete architecture configuration or an empty feasible region
  is identified for each hierarchy.

This milestone is complete. It required no model
training, new inference, or live asynchronous implementation.

---

## Milestone 5B: Deadline-bounded graceful expert degradation

Deliverables:

- weighted deadline replay with zero post-commit transfer wait;
- P50/P95/P99 missing routed mass and full-fallback incidence;
- bounded TPOT and low-batch throughput projections;
- normalized null, renormalization, and shared-residual perturbation bounds;
- capacity–throughput–degradation Pareto for current and labeled large sparse
  geometries;
- deadline-elastic fallback/refinement HW architecture and telemetry contract.

Exit condition:

- at least one configuration with at least half the experts offloaded improves
  P99 TPOT by 25% over reactive exact offload;
- bounded TPOT is at most 1.5× the all-local anchor including fallback;
- P99 missing mass is at most 20% and full-fallback waves at most 1%;
- the regime repeats across at least two domains and two layer bands;
- every quality statement remains an explicit training target, not an
  empirical claim.

This milestone is complete and used existing traces only. The exit gate passes
under the explicit assumed-predictor, assumed-robustness, and hypothetical
high-bandwidth hardware contract. Human review remains pending, and no
language-quality claim follows from the pass.

---

## Milestone 6: PCIe prototype

Deliverables:

- pinned-memory staging;
- asynchronous transfers;
- overlap timeline;
- reactive/oracle/learned comparison;
- model-versus-measurement error.

Exit gate:

- measured transfer and stall behavior agrees sufficiently with the analytical model to support extrapolation.

This milestone is deferred until H5 identifies a policy region worth
validating. It is not a prerequisite for the first-order viability and
profitability-window study.

---

## Milestone 7: Multi-GPU extrapolation package

Deliverables:

- parameterized peer-HBM and scale-up-fabric model;
- move-token vs move-expert comparison;
- 8× MI355X scenario tables;
- proposed work-system measurements;
- prioritized hardware implications.

Exit gate:

- extrapolation assumptions are explicit and separated from measured home-system evidence.

---

# 30. Required Figures

The first complete research package should generate at least:

1. Per-layer expert rank-frequency plots.
2. Top-\(n\) expert demand coverage.
3. Gini coefficient by layer and workload.
4. Expert reuse-distance CDF.
5. Transition entropy by lookahead.
6. Recall vs candidate budget.
7. Token-complete coverage vs candidate budget.
8. Wave-complete coverage vs batch size.
9. Decode-step miss probability vs candidate budget.
10. Prefetch amplification vs wave miss probability.
11. Oracle stall reduction vs bandwidth and cache capacity.
12. Learned/oracle recovery ratio.
13. Residual stall CDF.
14. Transfer timeline for reactive, oracle, and learned prefetch.
15. Move-token vs move-expert phase diagram.
16. Viable region by expert size and lookahead time.
17. Prediction-quality × cold-service-headroom profitability map.
18. Minimum required complete-route coverage versus lookahead.
19. Existing-policy location relative to the analytical requirement boundary.

---

# 31. Required Tables

1. Model architecture and expert-size summary.
2. Dataset and token-count summary.
3. Expert popularity statistics.
4. Predictor results by layer and lookahead.
5. Complete-set coverage metrics.
6. Simulator configuration.
7. Oracle feasibility results.
8. Learned-policy results.
9. PCIe calibration measurements.
10. Analytical model error.
11. Multi-GPU extrapolation assumptions.
12. Supported and unsupported architectural claims.

---

# 32. Decision Gates

Use one gate at a time. These are kill switches, not a requirement to encode
every interaction as an early threshold. After each gate, scan cheap structural
axes already present in the data and record any regimes as exploratory.

## Gate A: Is there useful expert skew?

Proceed with static hot-residency analysis only if skew is strong and stable at relevant scopes.

## Gate B: Is future demand predictable beyond trivial baselines?

Record where hidden-state predictors beat popularity and transition baselines,
but do not optimize them until Gate C identifies issue points where transfers
can physically complete.

## Gate C: Is oracle prefetch physically viable?

Stop claiming latency hiding if oracle cannot hide useful transfers.

If Gate C passes, evaluate predictor and transition policies only inside the
viable issue-point/bandwidth/capacity region rather than against a global
accuracy average.

Current pilot result: the frozen measured-bandwidth \(K=16,\Delta=1\)–3 gate
failed, while a descriptive \(K=32,\Delta=3\) region passed the same physical
thresholds. Treat H4 as a capacity–lead-time–bandwidth boundary, not a universal
whole-expert feasibility claim. For the prototype, advance to the normalized
H5 analytical sweep without requiring a higher-fidelity overlap experiment;
retain the absence of live overlap validation as a claim limitation.

## Gate D: Is complete-set coverage affordable?

If wave-safe coverage requires prefetching nearly all experts, prediction has little selectivity value.

Current pilot result: complete residual-cold-set coverage is substantially
better than complete top-8 route coverage at K=32, but the raw candidate lists
are unaffordable as transfer lists. None of eight H5-C placements meets the 2×
candidate/useful-byte screen.

## Gate E: Is learned policy close enough to oracle?

If learned/oracle recovery is low, use prediction for planning rather than just-in-time loading.

Current pilot result: the K=32, Δ=9 policies recover 67–77% of the first-order
oracle before the traffic gate is applied. This is enough information for a
selective admission/residency experiment, but not evidence for blind
just-in-time prefetch. H6 subsequently showed that the unchanged depth
predictors do not convert this information into better on-demand residency:
both fail against the strongest static/domain/LRU comparator. Do not optimize
the predictor or proceed to routing intervention as a rescue.

## Gate F: Does moving expert state beat moving activations?

Do not recommend expert movement where token dispatch is cheaper.

---

# 33. Interpretation Framework

The result space has four major regimes.

## Regime 1: High skew, high predictability, feasible transfer

Implication:

- strongest case for hierarchical residency and predictive prefetch;
- evaluate selective replication and tile-level movement;
- consider hardware support for prefetch queues and object-aware movement.

## Regime 2: High skew, low predictability

Implication:

- static or slowly adaptive hot-expert residency is useful;
- skip-layer prediction adds limited value;
- replication may outperform just-in-time prefetch.

## Regime 3: Low skew, high predictability

Implication:

- fixed hot caches are weak;
- per-request or short-horizon prefetch may still work;
- candidate-set prediction may be valuable.

## Regime 4: Low skew, low predictability

Implication:

- hierarchical expert offload is unattractive for strict latency;
- favor all-HBM residency, expert parallelism, activation movement, or model redesign.

A fifth practical regime may appear:

## Regime 5: Predictable but physically unprefetchable

Implication:

- prediction is real but insufficient;
- use it for queue reservation, placement, replication, or throughput scheduling;
- do not claim strict low-latency benefit.

---

# 34. Hardware/Software/Workload Co-Design Implications

The project should evaluate, not merely speculate about, the following implications.

## Hardware

Potentially useful features:

- larger HBM capacity;
- lower-latency peer-HBM access;
- high message-rate fabric;
- more concurrent DMA channels;
- transfer priority classes;
- tensor multicast;
- asynchronous gather/scatter;
- object-aware movement descriptors;
- tile-level quantize/dequantize during transfer;
- dedicated staging SRAM;
- programmable prefetch queues.

## Software

Potentially useful mechanisms:

- expert placement runtime;
- prediction-guided admission;
- residency epochs;
- expert replication;
- batch formation by predicted expert demand;
- transfer deadline scheduling;
- asynchronous sequence progression;
- trace-driven policy selection;
- topology-aware expert mapping.

## Workload and Model Architecture

Potential changes:

- route slack among semantically acceptable experts;
- locality-aware router training;
- shared expert bases plus small expert-specific deltas;
- factorized experts;
- expert clustering;
- smaller transferable expert tiles;
- predictable routing regularization;
- expert substitutability;
- architecture-level separation of hot shared and cold specialized capacity.

These are future directions. The initial experiment should only claim support where trace and model evidence exists.

---

# 35. Strongly Supported Conclusions vs Speculative Conclusions

## Strongly supportable if measured

- expert demand is skewed at specified scopes;
- future routing is predictable at specified lookahead;
- prediction reduces candidate bandwidth at a fixed complete-set coverage;
- oracle prefetch is feasible in a defined hardware region;
- learned prefetch recovers a measured fraction of oracle benefit;
- cold misses dominate tail latency under synchronous waves;
- whole-expert movement is or is not viable for measured sizes;
- activation movement is cheaper in specified regimes.

## Speculative until later work

- dedicated expert SRAM is broadly optimal;
- future production systems should offload experts to host memory;
- learned prefetch will improve end-to-end TPOT;
- three-tier expert caching is a complete architecture;
- skip-layer prediction generalizes across models;
- routing can be modified for locality without quality loss;
- peer-HBM migration will beat token dispatch at scale;
- a single-GPU PCIe result directly predicts rack-scale behavior.

---

# 36. Deliverable Format

For the prototype, maintain only:

1. `STATUS.md` and one chronological experiment log;
2. machine-readable metrics and immutable configuration/manifest files;
3. one or two scripted PDF/PNG figures;
4. one concise Markdown result report containing decisions, limitations, and
   important assumptions.

Add a separate assumption ledger, failure database, tracking service, or final
report structure only when architecture-scale evidence makes it necessary.

The results summary must include a human visual-review checkpoint. The next
major hypothesis starts only after the researcher reviews the generated
figures and records the next action.

---

# 37. Initial Implementation Order

Use this lean order:

1. Inspect the model, compute exact expert sizes, and validate hook semantics.
2. Collect one restartable trace with integrity checks.
3. Run H1/H2 routing baselines and one fixed linear H3 sidecar.
4. Use cheap post-hoc scans of the existing trace to expose layer/horizon
   regimes; do not add predictor ablations.
5. Measure a small unhooked layer-timing sample and host-to-device transfer
   curve.
6. Build the minimum oracle H4 feasibility calculation.
7. Sweep assumed predictor quality and normalized hardware parameters to map
   H5 viability/profitability windows and inverse predictor requirements.
8. Place the existing transition and linear streams on that surface without
   retraining.
9. Compare residency roles only with the existing policies first. H6 did so
   and failed; stop this mechanism rather than escalating predictor complexity.
10. Sweep an explicitly assumed MTP-router envelope over complete cold-set
    coverage and amplification; retain correlated wave misses.
11. Derive host/pooled-memory capacity–P99 frontiers and inverse interconnect
    requirements before adding timing fidelity.
12. Add transfer granularity and rolling HBM-to-SRAM staging using the same
    event model.
13. Defer broad predictor training and asynchronous-copy prototypes until an
    analytical configuration is worth calibrating.

---

# 38. Minimal Success Criteria

The project is minimally successful if it produces:

- a validated expert-routing trace;
- per-layer hotness and locality statistics;
- one skip-layer predictor;
- correct complete-set coverage metrics;
- a calibrated transfer model;
- an oracle prefetch simulation;
- one learned prefetch result;
- one trace-calibrated future-predictor architecture envelope;
- one fast-tier capacity versus tail-latency Pareto result;
- one defensible positive or negative architecture conclusion.

The project does not require end-to-end model speedup or a live asynchronous
PCIe implementation to be successful. A later overlap demonstration is a
calibration step for a selected design point, not a prerequisite for deriving
the architecture.

---

# 39. Stretch Goals

After the core experiment:

- batch-level demand prediction;
- prediction of expert union rather than token-level labels;
- conformal candidate sets with guaranteed empirical coverage;
- dynamic cache partitioning across layers;
- tile-level transfer simulation;
- factorized expert-size sensitivity;
- real expert tensor offload;
- asynchronous sequence scheduling;
- multi-request workload mixing;
- multi-tenant expert hotness;
- integration with vLLM or another serving runtime;
- eight-GPU expert-parallel trace collection;
- peer-HBM transfer calibration;
- selective expert replication prototype;
- joint KV-location and expert-location scheduler.

---

# 40. Final Research Position

The project should begin from a skeptical position.

The interesting result is not that an ML predictor can guess future experts. That is likely achievable.

The difficult question is whether prediction changes the architecture.

A convincing architectural result must jointly establish:

1. **Demand structure:** expert demand is skewed, local, or predictable.
2. **Coverage:** future demanded expert sets can be covered at sufficiently low miss risk.
3. **Timing:** predicted transfers complete before demand.
4. **Bandwidth:** speculative movement fits within the hierarchy.
5. **Capacity:** prefetched experts or tiles fit without destructive eviction.
6. **Critical path:** residual misses do not dominate TPOT.
7. **Competition:** moving experts is better than moving activations or simply adding residency.
8. **Scalability:** the conclusion remains plausible on a large-HBM multi-GPU system.

The most likely high-value conclusion may be narrower than the original idea:

> Skip-layer expert prediction is unlikely to guarantee strict low-latency execution by itself, but it may substantially reduce the cost of conservative residency, replication, broad candidate prefetch, and transfer scheduling.

That conclusion would still justify concrete GPU architecture and runtime work.

The project should be implemented so that it can prove this conclusion wrong.
