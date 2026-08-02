# Next experiments: first-order co-design and predictable routing

**Updated:** 2026-08-02  
**Current next action:** run Q1, the expert-erasure quality probe (frozen
protocol in [Q1_PROTOCOL.md](Q1_PROTOCOL.md)); measure whether quality loss is
controlled by missing routed mass before any AX4 robustness-training is
authorized
**Operating rule:** use the cheapest existing artifacts first; model broad
viability and expected benefit before improving timing fidelity or predictors.
After AX4, the decisive open question is whether low-mass deadline-induced
omissions are semantically cheap — Q1 measures that on the frozen model before
any training.

This plan now separates four questions:

1. Under what hardware and prediction assumptions is hierarchical expert
   management analytically worthwhile? (answered by the AX track)
2. Where do the existing transition and linear candidate streams land within
   that design space? (H5/H6)
3. Does the frozen model tolerate the bounded expert erasure AX4's deadline
   contract relies on? (Q1 — the new immediate priority)
4. Can routing later be trained to hold quality under that erasure without
   harming loss or load balance? (deferred robustness-training, gated on Q1)

This plan separates three questions that should not be conflated:

1. Under what hardware and prediction assumptions is hierarchical expert
   management analytically worthwhile?
2. Where do the existing transition and linear candidate streams land within
   that design space?
3. Can routing later be trained to move the prediction–quality Pareto frontier
   without harming language-model quality or load balance?

The active AX track answers the first question under an explicit optimistic
assumption: a future MTP-style router can expose multi-horizon expert-demand
predictions without degrading model quality. It does not claim that current
OLMoE or the fixed H3 predictor achieves those points. The frozen design is in
[ARCHITECTURE_EXPLORATION_PROTOCOL.md](ARCHITECTURE_EXPLORATION_PROTOCOL.md).

## Experiment list

| ID | Question | New inference/training? | Status |
|---|---|---:|---|
| AX1 | What model-capacity and TPOT envelope can future predictive host/pooled-memory prefetch provide? | No | Complete; projected region exists, review pending |
| AX2 | How do bandwidth, latency, coverage, amplification, and transfer granularity divide the design space? | No | Complete; inverse bounds and phase map generated |
| AX3 | What local-HBM and rolling-SRAM capacities suit a multi-horizon three-tier hierarchy? | No | Complete; physical staging envelope generated |
| AX4 | Can hard-deadline expert erasure produce a tight low-batch TPOT bound with a plausible quality-robustness contract? | No for completed replay; later training requires permission | Complete; analytical gate passes only in a high-bandwidth mass-priority regime, review pending |
| Q1 | Does the frozen model tolerate the expert erasure AX4's deadline contract relies on? Is quality loss controlled by missing routed mass? | Yes; forward passes on frozen base model, no training | **Frozen** protocol; awaiting implementation |
| Q2 | Can availability-conditioned robustness training make the model tolerate the AX4 erasure distribution without harming loss/load balance? | Yes; small intervention | Deferred; gated on Q1 GO |
| H5-A | What prediction × hardware combinations create a first-order profitability window? | No | Complete; region exists |
| H5-B | What minimum predictor quality is required at each capacity, bandwidth, and lookahead? | No; derived from H5-A | Complete |
| H5-C | How much analytical oracle benefit do the existing transition and linear streams recover? | No retraining; reconstruct existing predictions | Complete; raw streams fail traffic gate |
| H5-D | Do existing scores separate useful from useless cold candidates? | No | Complete; signal present, shared threshold insufficient |
| H6 | Does prediction-guided on-demand residency beat static/domain/LRU at equal capacity and movement budget? | No | Complete; frozen gate failed |
| H7 | Can a routing-predictability objective improve modeled benefit without harming loss or load balance? | Yes; small controlled intervention | Deferred after H6 failure |
| C0 | Does Base→Instruct post-training materially change matched-token trajectory predictability? | Yes; two endpoint traces | Complete; frozen stage-effect gate failed |
| C1 | Does the trajectory/co-design result transfer to one newer top-1/top-2 checkpoint? | Yes; one trace collection | Deferred; explicit permission required |

Detailed timing validation, concurrent-copy microbenchmarks, MLPs, predictor
training, new inference, and model downloads remain deferred. AX is an
analytical architectural exploration, not an attempt to rescue the current H3
or H6 policies.

AX4 is the immediate exception to the exact-execution contract, not to the
no-training rule. It asks whether late experts can be converted from a latency
failure into bounded routed-mass erasure. Current traces establish the
resource/erasure contract; they cannot establish model quality under erasure.

C0 adds a within-family control, not a new placement mechanism. Base and
Instruct preserve 89.7% of expert selections and differ by only +1.6 points on
the frozen long-range conditional-predictability metric. Do not spend two more
checkpoint downloads on SFT/DPO stage localization. The next generalization
experiment, if explicitly approved later, should change routing architecture
rather than add another OLMoE post-training stage.

## AX architecture-exploration sequence

### Assumption boundary

Use measured and trace-derived OLMoE demand as the workload anchor, but sweep
hypothetical future-router quality independently:

- complete cold-set coverage
  \(C\in\{0.50,0.75,0.90,0.95,0.99,0.999\}\);
- predicted/useful byte amplification
  \(A\in\{1,1.25,1.5,2,4,8\}\);
- correlated wave misses rather than independent expert-label corruption.

Every output must label measured inputs, trace-derived inputs, assumed
predictor behavior, and hypothetical hardware parameters separately.

### AX1 — Capacity-first predictive offload

Extend the existing H4/H5 replay over K=8/16/32, selected lookaheads through
Δ=15, and cold-tier bandwidths from 16–512 GB/s. Report HBM bytes retained,
maximum offloaded expert capacity, useful/false/late/missed bytes, and
mean/P95/P99 slowdown relative to the **reactive hierarchy**.

CPU-memory prefetch is a capacity-enabling mechanism. Do not compare it as a
speedup over an otherwise identical all-HBM model.

### AX2 — Reliability and interconnect regimes

Add startup latency, transfer concurrency, and 12/4/1/0.25 MiB transfer
objects. Derive \(\beta_{\min}\), \(C_{\min}\), \(A_{\max}\), and
\(S_{\max}\). Classify bandwidth-, latency-, reliability-, capacity-, and
SLO-limited regions. Use normalized unique demand U=1/2/4/8 only as a clearly
labeled sensitivity for future top-1/top-2 routing.

### AX3 — Predictive three-tier hierarchy

Model pooled or host memory → local HBM → rolling software-managed SRAM.
Long-horizon heads plan HBM placement; short-horizon heads plan SRAM staging;
the ordinary router confirms demand. Sweep 32–512 MiB global SRAM capacity
with double buffering, not a persistent per-layer expert cache.

### Required synthesis

Produce at most:

1. complete coverage versus cold-service-headroom profitability map;
2. fast-tier capacity versus P99 TPOT Pareto frontier;
3. minimum bandwidth versus lookahead inverse-design curve.

The result is quantitative bounds and architecture regimes, not a measured
wall-clock benefit. Live validation is optional and follows only after a
representative point is selected.

### Completed AX result and review gate

The canonical result and three figures are under
`artifacts/runs/h1-standard-small/analysis/architecture/`. The immediate
decision is not another sweep:

1. review the profitability phase map, memory–P99 Pareto, and inverse
   bandwidth/lookahead curve;
2. accept or reject the wave-local model as a useful architecture envelope in
   light of the more pessimistic selected FCFS queue points;
3. select at most one calibration point only if measuring live asynchronous
   behavior would change the architectural conclusion;
4. otherwise preserve C1 top-1/top-2 confirmation and H7 predictable-routing
   training as future work requiring explicit permission.

## AX4 — Deadline-bounded graceful degradation

The frozen protocol is
[DEADLINE_DEGRADATION_PROTOCOL.md](DEADLINE_DEGRADATION_PROTOCOL.md), with
configuration in
`configs/experiment/ax4-deadline-degradation.toml`.

### Immediate question

At batch-1 decode, can a hard layer commit deadline remove cold-transfer
queueing from the critical path while keeping P99 missing normalized routed
mass small enough to define a credible future training target?

### Completed result

OLMoE uses probabilities from a 64-way softmax after top-8 selection without
renormalizing them. AX4 therefore reports normalized-within-top-8 mass as the
architecture contract and preserves absolute missing router probability as a
secondary result.

The trace-ordered FCFS boundary gives:

- measured 24.14 GB/s PCIe fails at K=8/16/32 with 100%/100%/81% P99 missing
  normalized mass;
- K=32 at 128 GB/s passes with 7.2% P99 missing mass;
- K=8 and K=16 at 256 GB/s pass with effectively zero P99 wave mass;
- K=16 at 128 GB/s is a sharp near miss at 20.4%;
- K=32 at 256 GB/s delivers the mass but fails the 25% benefit threshold
  because reactive offload is already too fast.

The gate requires at least 50% expert offload, ≥25% P99 TPOT improvement over
reactive exact offload, ≤1.5× all-local TPOT including fallback allowance,
P99 missing mass ≤20%, and ≤1% full-fallback waves across multiple domains
and layer bands. It passes at K=8, 256 GB/s, C=99%, A=1.5×, Δ=1, one-layer
slack, and mass-priority ordering: 1.5 GiB is resident, 10.5 GiB is offloaded,
bounded TPOT is 11.25 ms, and the same-hierarchy reactive comparison is
16.41 ms. Only 0.93% of waves are degraded and none takes full fallback.
Passing identifies a robustness target worth training for; it does not
validate quality.

Concrete frozen prediction: the 10.23 ms measured local anchor plus 10%
fallback/commit allowance gives an 11.25 ms deadline cap and 88.9 batch-1
tokens/s, versus the existing 66.83 ms and 15.0 tokens/s K=16 reactive P99
projection. This is a 5.94× same-batch throughput projection. Whether that
large resource gain is useful is decided by the resulting P99 missing routed
mass, not assumed in advance.

The aligned hardware proposal has an always-resident shared/identity/null
plane, optional routed residual experts, fixed commit bitmaps, deadline-aware
mass-per-byte scheduling, bounded speculative credits, and missing-mass
telemetry. Transfers that miss commit cannot delay dispatch.

The immediate next step is human review of the three figures and evidence
boundary. Do not start erasure training, inference collection, or a new model
without explicit permission.

## Q1 — Expert-erasure quality probe (active)

**Updated:** 2026-08-02. Frozen protocol:
[Q1_PROTOCOL.md](Q1_PROTOCOL.md).

### Decisive question

After AX4, the project's decisive open question is whether low-mass,
deadline-induced expert omissions can be made semantically cheap enough to
convert prediction into a hard latency guarantee. AX4 passes only
conditionally on the model tolerating bounded missing routed mass. Q1 measures
that tolerance directly on the frozen base checkpoint.

### Immediate design (frozen)

- **Checkpoint/dataset:** base `OLMoE-1B-7B-0125` over WikiText-2; forward
  passes only, no training, no second checkpoint.
- **Primary scope (gating):** prefill quality probe with paired same-token
  deltas — per-token forward KL, top-1 agreement, perplexity ratio — as a
  function of missing routed mass `m_missing`.
- **Policies:** renormalize (primary) and null-drop (lower bound), matched at
  equal `m_missing`. Shared-residual is excluded from the measured probe (the
  base model has no shared expert) and stays paper-level discussion.
- **Erasure probes (gate):** mass-budget sweep at
  `m ∈ {0.01, 0.05, 0.125, 0.25, 0.50}`, positioning swept across
  mass-omission (headline), random-within-route, mass-adversarial.
- **Correlation scan (non-gating):** layer-burst and consecutive-token-burst,
  plus a short greedy decode leg to detect autoregressive compounding that
  prefill structurally cannot see.
- **Injection:** runtime MoE-forward patch (temporary in-memory mask, restored
  after each pass; upstream model file never modified).
- **Frozen gate (renormalize + mass-omission at headline `m=0.125`):**
  forward-KL ≤ 0.05, top-1 agreement ≥ 99%, perplexity ratio ≤ 1.05, and
  `ΔQ` monotone in `m_missing`. STOP/kill if sub-1% mass causes frequent large
  divergence or non-monotone/jumpy `ΔQ`.

### Steps

1. Run a 1–2 request smoke test of the probe; verify the erasure patch
   reproduces exact softmax-64→top-8→no-renormalization semantics before
   scaling (SOP step 2).
2. Materialize the paired table and apply the frozen gate.
3. Generate one `ΔQ vs m_missing` curve and one positioning/correlation panel
   with hashed inputs.
4. Human visual review, then decide GO (proceed to Q2 robustness-training) or
   STOP (AX4's training justification dies).

## Q2 — Availability-conditioned robustness training (deferred)

**Gated on Q1 GO.** Only if the frozen model tolerates the AX4 erasure
distribution should a minimal intervention train a small fallback/calibration
mechanism (mask-aware renormalization, small shared fallback expert, or a
low-rank adapter) against the actual availability-mask distribution produced by
the target hierarchy and deadline policy. Measure the Pareto tuple
(validation loss, load balance, quality under the AX4 erasure distribution,
modeled TPOT benefit). Generic expert-dropout robustness is explicitly not the
target; only the trace/mass-derived distribution counts.

## H5-A — Controlled prediction × hardware sweep

### Question

For the observed OLMoE cold-demand and reuse structure, which combinations of
prediction quality, speculative traffic, fast-tier capacity, lookahead, and
cold-path bandwidth produce useful expected benefit?

### First-order model

Use the dimensionless cold-service pressure:

\[
\rho =
\frac{
\bar N_{\text{candidate}} S_{\text{expert}}
}{
B_{\text{cold}}\Delta T_{\text{layer}}
}.
\]

Reuse existing trace-derived demand, LRU hit/cold rates, exact 12 MiB expert
size, and the effective layer interval. Do not add a more detailed performance
model for this experiment.

Sweep:

- complete-route coverage from 0% to 100%;
- candidate amplification \(A\in\{1,2,4\}\);
- per-layer capacity \(K\in\{8,16,32\}\);
- lookahead \(\Delta=1\ldots15\);
- normalized cold bandwidth \(0.25\times,0.5\times,1\times,2\times,4\times\).

Treat expert size and bandwidth as the same first-order \(S/B\) control. Add a
separate expert-granularity axis only if a result cannot be explained by that
normalized ratio.

### Proposed screening gate

Call a point analytically profitable only if it achieves all three:

- at least 25% modeled reactive-stall reduction;
- at least 50% oracle recovery;
- no more than 2× predicted/useful transfer bytes.

These are prototype screening thresholds, not end-to-end speedup claims.
Freeze them in the H5 protocol before running the sweep.

### Outputs

- `h5_design_points.csv`: every assumption cell and category;
- `h5_windows.csv`: qualifying horizon range/count by capacity, bandwidth, and
  predictor assumption;
- expected stall reduction, oracle recovery, useful/false bytes, and
  amplification;
- one categorical phase diagram over physical headroom and complete coverage,
  with amplification as compact panels.

## H5-B — Inverse predictor requirements

### Question

What minimum complete-route coverage and maximum candidate amplification are
required to cross the H5-A screening gate?

Compute:

\[
C_{\min}(K,\Delta,B,A)
\]

and, where useful:

\[
A_{\max}(K,\Delta,B,C).
\]

This converts the hardware sweep into an ML and training target. The headline
output is one curve of minimum required complete coverage versus lookahead for
K=8/16/32, with bandwidth shown as a small number of line styles or panels.

Report empty windows explicitly. Do not hide a requirement above 100% or a
candidate amplification below the demanded top-k.

## H5-C — Existing-policy placement and replay

### Question

At representative points on the H5 surface, how much oracle benefit do the
existing untuned transition and linear candidate streams recover after false
candidate bytes are charged?

Reconstruct candidates from existing H2/H3 artifacts. Do not retrain, tune, or
add an MLP.

Primary representative cells:

- K=32, Δ=1: prediction-good/physics-limited control;
- K=32, Δ=3: current boundary candidate;
- K=32, Δ=9: long-range linear advantage;
- K=16, Δ=9: oracle-feasible/prediction-limited control.

Report cold-set rather than only total-route coverage:

- useful, false, and late candidate bytes;
- cold-expert and complete-cold-set coverage;
- expected stall reduction;
- oracle recovery;
- candidate amplification and churn.

The visualization should place actual policies on the H5-A phase diagram and
show a small actual-versus-required table. Do not create a separate dashboard.

## H6 — Mechanism competition

H6 is complete. It compared static popularity, domain popularity, reactive
LRU, transition-guided residency, linear-guided residency, and an equal-budget
next-use oracle at K=8/16/32. Prediction could admit only an actually demanded
miss; no broad candidate prefetch was allowed.

At the frozen decode K=16, Δ=3 gate, transition and linear lose 3.9 and 2.5 pp
of expert-stall reduction and 0.7 and 0.6 pp of complete-set hits relative to
the strongest matched simple baseline. The oracle remains strong, but existing
depth-trajectory scores do not predict temporal reuse well enough to select
residency.

Decision: stop this placement mechanism after human figure review. Do not fit
the previously proposed cost-sensitive head, tune an MLP, collect fresh
confirmation, begin H7, or download a second model. Any later work must first
pose a genuinely different mechanism or a direct temporal-reuse hypothesis and
receive explicit permission.

## H7 — Controlled routing-predictability intervention

H1–H6 observe a normally trained model; they do not show that predictability
can be increased without quality loss. H6 also removes the immediate placement
justification for this intervention.

If H5-B produces a plausible predictor target, run one small matched pilot:

1. standard load-balancing continuation objective;
2. the same objective plus one trajectory-predictability term.

Start with one seed and a short token budget. Measure the Pareto tuple:

\[
(\text{validation loss},\ \text{load balance},\
\text{complete trajectory coverage},\ \text{modeled HW benefit}).
\]

Proceed only if the intervention moves the modeled-benefit frontier without a
material validation-loss or load-balance regression. Replication, regularizer
ablations, and broader training wait for that result.

This intervention is a distinct project phase because it modifies training.
It must not be retroactively inferred from the current hook-only evidence.

## C1 — Sparse-model transfer check

Before generalizing beyond OLMoE, repeat only the decisive trace and analytical
steps on one checkpoint with top-1/top-2 routing and more sparsity:

- routing integrity and exact expert bytes;
- H1/H2 trajectory structure;
- the H5-A normalized co-design map;
- existing simple transition baseline;
- a linear sidecar only if transitions leave a meaningful information gap.

The purpose is to test whether OLMoE top-8 demand makes complete-set coverage
and cold-service pressure unusually harsh.

## Insight mining after each major experiment

After applying the frozen gate, use cheap post-hoc analysis to extract:

- boundary locations and empty opportunity windows;
- capacity–bandwidth–lookahead substitution rates;
- whether mean headroom disagrees with trace-driven tail behavior;
- the source layers and domains that create or destroy a viable region;
- whether complete coverage, speculative traffic, or physics is limiting;
- dimensionless quantities that transfer across hardware assumptions;
- negative results that redirect prediction toward residency, replication, or
  activation movement.

Label each conclusion as directly supported, analytical inference, or
speculation. Update `docs/FOUNDATIONAL_INSIGHTS.md` only when a result changes
the durable thesis.

## Visualization standard for the next phase

Use at most two primary figures per major experiment:

1. a categorical co-design phase diagram;
2. an inverse-design curve showing the predictor quality required for a chosen
   benefit threshold.

Use a compact actual-versus-required table for H5-C. Every plot must name:

- prediction metric and set semantics;
- physical normalization;
- capacity and candidate-budget semantics;
- profitability assumptions;
- whether the result is measured, trace-driven, or analytical.

Complete the human review checkpoint before beginning H7 or C1.
