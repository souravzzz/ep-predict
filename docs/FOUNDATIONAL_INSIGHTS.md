# Foundational insights and publication thesis

## Purpose and curation policy

This document preserves the durable ideas, perspective shifts, negative
results, and publication-level interpretations produced by the project. It is
not an experiment log and should not be updated after every run.

Update it only when at least one of the following occurs:

- a major hypothesis gate materially changes the architectural thesis;
- a broad post-hoc analysis reveals a new organizing principle;
- H4–H6 connects prediction to physical timing or policy benefit;
- a second model confirms or contradicts a claimed general principle;
- a result changes the strongest defensible paper claim.

Routine metrics, commands, implementation changes, and transient next steps
belong in `EXPERIMENT_LOG.md`, `STATUS.md`, and the per-hypothesis reports.

**Evidence snapshot:** 2026-08-02, after H1–H6, the all-layer H2/H3 scan, the
C0 Base–Instruct matched-token comparison, and the Q1/Q1-B expert-erasure
quality probe.

---

## Headline thesis

> **MoE routing is a structured trajectory through expert space.**

Global expert popularity is an incomplete description of sparse-model
execution. Even when a workload does not have a sufficiently stable universal
hot set, the token's path through layer-local expert spaces can remain strongly
predictable.

The emerging architectural view is:

> Early hidden state supports long-range resource planning; later routing
> decisions support short-range correction. A hierarchical runtime can
> progressively refine placement, replication, or scheduling as execution
> reveals more information.

This is presently an OLMoE pilot result, not a universal MoE law. H4 found
that physical actionability is conditional: the preregistered compact tier and
short horizon failed, while more residency, lead time, or bandwidth exposed an
oracle-feasible region.

---

## Empirical foundation

The following measurements anchor the interpretation.

### Static hotness is insufficient globally

- OLMoE routes top-8 of 64 experts across 16 MoE layers.
- A mixed-workload static top-8 tier passed the complete H1 gate in only 2/16
  layers.
- Domain-conditioned routing distributions were much more distinct and useful
  than one workload-agnostic placement.

### Current routing predicts nearby future demand

At decode \(K=16\), the H2 transition table achieved:

| Lookahead | Selection coverage | Complete top-8 coverage |
|---:|---:|---:|
| \(n+1\) | 79.0% | 24.1% |
| \(n+2\) | 77.9% | 23.5% |
| \(n+3\) | 76.8% | 22.2% |

All 168 original layer-domain comparisons improved over static popularity.

### A linear sidecar is not a universal replacement

At the formal H3 primary gate—decode \(K=16,\Delta=1\)—the fixed linear
hidden-state sidecar achieved 79.4% selection and 28.7% complete coverage,
versus 79.0% and 24.1% for transitions. It failed the global replacement gate
because selection gain and cross-domain consistency were insufficient, while
candidate churn rose from 42.8% to 52.8%.

### Early hidden states dominate at long range

The all-layer scan changed the interpretation without changing the formal H3
decision:

| Decode, K=16 | Transition | Linear hidden state |
|---|---:|---:|
| Layer 0→15 selection | 53.8% | 69.2% |
| Layer 0→15 complete top-8 | 4.6% | 19.7% |

Across all 120 domain-balanced source-target pairs:

- linear beats transition in 100/120 pairs for selection coverage;
- linear beats transition in 112/120 pairs for complete-token coverage;
- average selection gain is +14.7 points from source layer 0 and +11.0 points
  from source layer 1;
- selection gain becomes negative from approximately source layer 10 onward.

### Distance alone does not explain predictability

Holding source layer 0 fixed, linear selection coverage is:

- 69.9% for target layer 1;
- 80.0% for target layer 9;
- 69.2% for target layer 15.

Prediction does not monotonically degrade with layer distance. Source-target
identity and target-layer routing structure matter at least as much as
\(\Delta\).

### H4 separates information from physical actionability

- Hook-free cached-token forward: 10.229 ms median, or 0.639 ms per effective
  MoE-layer interval.
- Exact 12 MiB pinned-host expert copy: 0.524 ms median at 24.14 GB/s.
- At measured bandwidth, \(K=16,\Delta=3\): 32.8% of cold bytes timely and
  38.9% oracle stall reduction.
- At measured bandwidth, \(K=32,\Delta=3\): 55.5% timely and 61.8% stall
  reduction.

Perfect prediction cannot compensate for an overloaded transfer queue. It
becomes actionable only after residency has reduced the cold set, the issue
point supplies enough lead time, or hardware supplies more bandwidth.

### H5 separates prediction from transfer admission

On held-out decode requests at K=32, the existing policies cover 67–81% of
complete residual cold sets. Yet transferring every nonresident candidate
costs 6.3–6.7× useful cold bytes, so none passes the frozen 2× traffic screen.

This is a major semantic correction: a candidate set is an information
envelope, not an action list. Broad trajectory prediction can be valuable even
when literal candidate prefetch is not, provided a later admission policy
converts uncertainty into selective movement or residency.

The score distributions confirm that this is not simply a signal-absence
failure. For the linear sidecar, useful-versus-unused nonresident scores have
JS divergence of 0.381 bits at \(\Delta=3\) and 0.332 bits at \(\Delta=9\),
with AUROC 0.883 and 0.861. But only 7–8% of scored nonresident IDs are useful.
At 50% complete cold-set coverage, even the linear ranking still needs
3.0–3.3× transferred/useful bytes. The rare-event base rate and set-completion
requirement—not only distribution overlap—control admission profitability.

### H6 separates depth trajectory from temporal reuse

At held-out decode \(K=16,\Delta=3\), reactive LRU leaves 48.1% residual cold
expert demand. Transition- and linear-guided residency leave 50.2% and 48.8%,
while an equal-movement-budget next-use oracle reaches 31.2%.

The oracle establishes a real residency opportunity, but the existing
predictors do not recover it. They predict a target layer for the same token;
residency needs to predict reuse by later tokens. Linear and transition
movements earn later hits only 66.3% and 60.0% of the time, versus 69.7% for
LRU and 94.7% for the oracle.

This negative result changes the architectural interpretation: trajectory
information is not a generic cache-control signal. The conditioning axis must
match the mechanism's reuse axis.

### C0 shows that post-training preserves the trajectory scaffold

Under identical prefill tokens, OLMoE Base and its final SFT+DPO+RLVR
descendant share 89.7% of selected expert IDs across depth. Their K=16 hot
sets overlap by 84.6%, while expert-popularity JS divergence is only 0.0029
nats. Exact top-8 sets still match for just 35.7% of token-layer events:
post-training frequently substitutes one or a few experts without replacing
the overall path.

Conditional predictability is equally stable. At layer 0→15, transition gain
over static popularity changes from +11.0 points in Base to +12.6 points in
Instruct, below the preregistered 5-point stage-effect gate. Transition tables
also transfer between endpoints with small average penalties.

This first checkpoint-lineage result suggests that structured routing is
established primarily during pretraining and survives later behavioral
alignment. Post-training edits a pretrained computational scaffold rather
than creating trajectory predictability from scratch. This remains
`single-family`: it is not evidence across MoE architectures.

### Erasure cost is mass-controlled and, under null-drop, additive in depth

Q1 measured quality versus missing routed mass directly on the frozen base
checkpoint (paired prefill forward passes):

- **Universal mass-budget erasure is catastrophic** regardless of policy. The
  renormalize headline cell (m=0.125) gives forward KL 5.81, top-1 9.2%, and
  PPL 279.9. Removing routed mass *routinely* always breaks the model.
- **Rare tail erasure is nearly free under the model's native null-drop.** A
  single one-expert one-layer drop on ~0.9% of tokens gives conditional KL
  ~0.003, top-1 99.2%. Renormalization is ~40× worse at equal mass and
  theoretically wrong (it rescales survivors, amplifying the very mass it
  removed), so it is dropped as a strategy.

Q1-B mapped the null-drop mechanism across depth, layer order, spacing, and
cross-token leak (same affected-token samples, conditional-on-affected):

| L | affected KL | top-1 | large-div frac |
|---:|---:|---:|---:|
| 1 | 0.0041 | 97.9% | 0 |
| 2 | 0.0053 | 96.2% | 0 |
| 4 | 0.0073 | 98.4% | 0 |
| 8 | 0.0133 | 93.6% | 0 |

- **Monotone and roughly additive in depth:** worst case L=8 is 0.013 nats
  conditional KL; per-layer marginal is flat (last/first ratio 1.20, gate ≤3);
  zero KL≥2 events at any depth.
- **Layer-uniform:** single-layer sensitivity spans only ~1.6× across all 16
  layers (0.0017–0.0027 nats) — no lynchpin layer.
- **Spacing-insensitive:** no reconstruction benefit from spreading drops; at
  L=4 spaced is even slightly worse. Independent contributions, not emergent
  depth compounding.
- **Local:** cross-token leak at downstream offsets ≈ far control (~0.0017
  nats); no propagating, distance-growing damage.

This is direct evidence for the **additive-residual mechanism**: each routed
expert adds a mostly independent weighted update to the residual stream, so
removing one per layer removes a small independent increment and cost scales
~linearly in the number of dropped expert-layers.

---

## Foundational principles

### 1. Routing is a trajectory, not a histogram

Popularity describes occupancy. Prediction describes motion.

A nearly flat or unstable global histogram does not imply unpredictable
execution. Tokens can follow regular conditional paths through expert space
without producing one globally useful hot set. Hardware that observes only
frequency can miss structure visible in transitions and hidden state.

**Paper-level implication:** characterize MoE workloads using expert-flow
trajectories and source-target predictability, not only expert-popularity
histograms.

### 2. Predictability and cacheability are different properties

The linear sidecar retains substantial long-range coverage while changing more
than half of its candidate set per token. Demand can therefore be predictable
but mobile.

- Cacheability requires stability and reuse.
- Schedulability requires advance information.

Prediction may be more valuable for prioritizing transfers, reserving
bandwidth, planning replication, or ordering work than for blindly loading
every candidate.

### 3. Router decisions are useful but lossy telemetry

An early top-8 route is a severe quantization of a 2,048-dimensional hidden
state. It reports the local execution choice but discards semantic information
that remains relevant to later routing.

The evidence is consistent with an information-bottleneck interpretation:

- hidden state is a richer long-range planning signal;
- the current route becomes a better locally sufficient signal near the
  target;
- neither signal is globally dominant.

This is an interpretation, not yet a causal or information-theoretic proof.

### 4. Network depth is a control horizon

The early-linear/late-transition regime resembles a two-stage controller:

1. **Feed-forward planning:** an early hidden state forms a coarse long-range
   resource plan.
2. **Feedback correction:** later routing observations refine or correct that
   plan as the deadline approaches.

The natural policy is hybrid rather than winner-take-all. Predictor choice
should depend on source layer, target layer, lead time, confidence, and
resource cost.

### 5. Layer-pair identity is more fundamental than lookahead distance

A scalar \(\Delta\) hides the model's internal regimes. The correct object is a
directed source-target predictability graph whose nodes are MoE layers and
whose edges carry coverage, confidence, churn, and available lead time.

Some distant targets are easier to predict than nearby targets. This may
reflect differences in routing entropy, specialization, or the extent to which
a target layer expresses semantics already present in the early residual
stream.

### 6. Complete-set coverage is a distinct structured objective

Nearly identical marginal selection coverage can produce meaningfully
different complete top-8 coverage. The linear sidecar's primary H3 result
demonstrated this directly.

Hardware waits for demanded sets, not independent labels. Predictor evaluation
must therefore preserve correlated misses and report token-, wave-, and
eventually decode-step-complete coverage.

Ordinary recall is insufficient for architectural conclusions.

### 7. Candidate capacity can create a reliability threshold

For H2 transition prediction at \(n+1\):

- \(K=8\): 58.0% selection, 1.2% complete coverage;
- \(K=16\): 79.0% selection, 24.1% complete coverage;
- \(K=32\): 93.2% selection, 64.1% complete coverage.

Capacity does not translate smoothly into safe execution. A fast tier can be
large enough for respectable average recall yet too small to prevent nearly
universal partial misses. Hardware capacity should be studied as a reliability
regime transition, not only a hit-rate curve.

### 8. Routing behavior is conditional, not a scalar model property

The evidence depends jointly on:

\[
\text{checkpoint}
\times \text{source-target layer pair}
\times \text{domain}
\times \text{phase}
\times \text{candidate budget}.
\]

Statements such as “this model has skew” or “MoE routing is predictable” are
usually too coarse. Conditional structure weakens universal static policies
but strengthens the case for adaptive resource control.

### 9. The memory hierarchy may encode confidence as well as latency

A predictive hierarchy can be organized by how knowledge evolves:

- persistent tier for stable/domain-hot experts;
- planned tier for early, long-lead predictions;
- immediate tier for later, higher-confidence corrections;
- reactive path for residual misses.

This reframes fast, medium, and slow storage as physical representations of
different certainty and deadline regimes—not merely different access times.

### 10. The model may expose an accidental resource-control surface

A fixed 128-dimensional random projection of a 2,048-dimensional router input
retains strong long-range routing information. One possible explanation is
that future-routing intent occupies a low-dimensional or redundant subspace.

If confirmed, a compact external control plane may be extractable without
modifying or retraining the base model. The sidecar would interpret latent
computational intent for resource planning.

This is provocative but preliminary: only one projection size, seed, model,
and workload have been tested.

### 11. Prediction and residency are complements, not substitutes

At \(K=16,\Delta=3\), the average eligible wave still needs 3.46 cold expert
copies. Three layer intervals provide 1.92 ms of nominal lead time, while those
copies require about 1.81 ms only when the queue is empty. Sustained arrivals
create backlog and make 67.2% of cold bytes late.

At \(K=32\), residency removes about four-fifths of demand before prediction
acts. The oracle then crosses the physical gate at the same lookahead and
bandwidth. A hierarchy is therefore not justified by prediction replacing
capacity; prediction makes finite capacity more effective after capacity has
already reduced transfer pressure.

### 12. A useful hardware unit is experts transferable per layer interval

For the measured system:

\[
\frac{T_{\text{layer interval}}}{T_{\text{12 MiB copy}}}
=
\frac{0.639}{0.524}
\approx 1.22
\]

This dimensionless exchange rate connects model execution, expert granularity,
and interconnect bandwidth directly. Compare it with cold experts per wave,
not raw GB/s alone. It is a compact first-order screen for whether a proposed
issue point can drain demand faster than demand is created.

### 13. Prediction should target the residual cold set, then separate belief from action

Complete route coverage is the wrong final metric once a fast tier already
holds part of the route. The operational target is:

\[
D_{\mathrm{cold}} = D_{\mathrm{route}} - D_{\mathrm{resident}}.
\]

At K=32, this conditioning raises complete-set coverage into the 67–81% range.
But the same experiment shows why prediction and action must remain separate:
false nonresident candidates dominate bytes unless an admission mechanism
filters them.

The resulting control stack has three distinct objects:

1. a broad belief over future experts;
2. a residency state that removes already-satisfied demand;
3. a resource-aware admission decision that commits bytes.

Conflating these objects makes a good predictor look like a bad architecture,
or a bad transfer policy look like a prediction failure.

### 14. Useful ranking is not sufficient admission

The H5 score analysis finds useful-versus-useless AUROC of 0.883 at Δ=3 and
0.861 at Δ=9 for the linear sidecar. The latent signal is therefore not weak.
Yet a shared threshold needs about 3.0–3.3× transfer amplification to preserve
50% complete cold-set coverage.

Set completion amplifies modest score overlap: several useful experts must all
survive admission, while many more useless expert IDs each have a chance to
cross the threshold. Admission must therefore be trained and evaluated as a
resource-constrained set decision, not inferred from pairwise ranking quality
alone.

### 15. Architectural value is a three-gate intersection

The experiments separate three necessary properties:

1. **Information:** future demand is predictable.
2. **Service:** the hierarchy can act before the deadline.
3. **Selectivity:** acting on predictions does not waste excessive bytes or
   capacity.

H2/H3 support the information gate. H4 exposes conditional service regions.
H5 finds an analytical intersection but shows that the unchanged candidate
streams miss its selectivity boundary.

No single accuracy, bandwidth, or cache-hit metric establishes architectural
value. A system is profitable only in the intersection of all three regions.
This prevents two symmetric errors: dismissing useful trajectory information
because one transfer policy fails, and claiming a viable hierarchy because a
predictor or oracle looks strong in isolation.

### 16. Prediction must match the mechanism's time axis

The project now exposes two different prediction problems:

\[
\text{depth: }P(E_{\ell+\Delta,t}\mid \text{state}_{\ell,t}),
\qquad
\text{time: }P(E_{\ell,t+\tau}\mid \text{history}).
\]

H2/H3 establish strong depth prediction. H6 shows that feeding those scores
into a temporal residency policy does not beat simple caching, despite a
substantial oracle gap.

This is more than a failed heuristic. It is a workload–mechanism alignment
principle: advance information has architectural value only when it predicts
the future event that consumes the managed resource. A depth trajectory can
schedule within-token transfers; cache retention needs cross-token reuse;
replication needs cross-request or cross-device demand. These tasks may share
features, but one cannot be substituted for another without evidence.

### 17. Erasure cost is a measurable, additive distortion budget — not a binary yes/no

Q1/Q1-B replace the vague "does the model tolerate missing experts?" with a
measured, mechanism-ordered curve. On the frozen base model the answer is
**regime-dependent**: universal mass erasure is catastrophic (KL 5.81), but
rarer null-drop erasure is monotone-additive, layer-uniform, and local, with
worst-case L=8 cost ~0.013 nats — semantically negligible. This matters in
three ways:

1. **It is the AX4 contract made measured.** The deadline-erasure hardware
   proposal (already in this document) turns a synchronization failure into a
   distortion budget \(m=\sum_{i\in M}a_i\); Q1-B now prices that budget
   empirically for the tail regime rather than assuming robustness.
2. **Training is not automatically the answer.** If the model already tolerates
   the target erasure distribution, robustness training adds cost and risk with
   no measured benefit. The honest default is a **non-training stress probe**
   first (the untested axes: cross-domain, decode compounding, contract
   boundaries), and a minimal mask-aware calibration only where a real non-free
   regime appears.
3. **The additive-residual mechanism gives a scaling law.** Cost ≈ (number of
   dropped expert-layers) × a small, layer-uniform constant, with no compounding
   blow-up — a concrete, testable property a training target or scheduler can
   be designed against.

---

## Perspective shifts produced by the experiments

| Initial framing | Evidence-driven framing |
|---|---|
| Find globally hot experts | Model conditional expert-flow trajectories |
| Predictability implies caching value | Predictability may enable scheduling even when caching is weak |
| One predictor should beat another globally | Use different signals at different control horizons |
| Accuracy should decay with layer distance | Source-target identity dominates a simple distance law |
| Marginal expert recall measures success | Complete demanded-set coverage controls stalls |
| More capacity gives proportionally better behavior | Complete coverage can exhibit threshold-like regimes |
| Fast memory tiers differ only by speed | Tiers can represent confidence, lead time, and commitment |
| Prediction can compensate for a small cache | Residency must first reduce cold demand below transfer service rate |
| A predictor candidate list is a prefetch list | Prediction is a belief envelope; admission commits scarce bytes |
| High AUROC or visible score divergence implies efficient admission | Rare-event base rate and complete-set survival determine byte efficiency |
| Complete route coverage is the placement target | Complete residual-cold-set coverage is the operational target |
| Quote bandwidth in GB/s | Compare experts transferable per layer interval with cold experts per wave |
| One good metric establishes architectural value | Information, physical service, and selectivity must pass together |
| Future-expert prediction is a generic cache signal | The predictor's axis must match the mechanism: depth for within-token action, time for reuse |
| A failed primary gate ends the idea | A failed global policy can expose a valuable conditional regime |
| Preregister every important interaction | Freeze one decision, then use cheap post-hoc scans for discovery |

---

## Hard-earned research lessons

### Negative gates should narrow claims, not trigger complexity

H1 rejected universal static placement but revealed domain structure. H3
rejected universal linear replacement but revealed early-layer long-range
value. Neither failure justified an MLP or a tuning sweep.

### Preregistration protects decisions, not discovery

The formal H3 decision remains valid. The post-hoc source-target scan did not
rewrite it; it found a different, narrower architectural claim.

Simple gates plus broad inexpensive analysis were more productive than a large
early hypothesis tree would have been.

### Aggregation can hide the architectural regime

The \(\Delta=1\) global mean averaged fifteen source layers and obscured the
strong layer-0/layer-1 behavior. Any aggregate should be paired with a compact
heterogeneity view when layer or domain interactions are plausible.

### Accuracy before physics is only workload evidence

H4 confirms why predictor accuracy alone was insufficient. One exact expert
can arrive within one effective layer interval, but several continuously
arriving cold experts overload the serialized copy path. Deadline-feasible
bytes and residual wave stall—not isolated transfer latency—decide
actionability.

### High churn changes the likely mechanism

The observed candidate replacement rates argue against literal per-token
loading. Candidate unions, reuse, wave aggregation, priorities, and residency
must be modeled before interpreting prediction as data movement.

### A repeatedly inspected test split becomes discovery data

The current 32 held-out requests supported valid frozen gates, but subsequent
post-hoc policy discovery means they cannot independently confirm the new
hybrid policy. Fresh requests are required only if H4 makes confirmation worth
the cost.

### A policy bottleneck does not automatically justify model training

H5 identified excessive speculative movement, not a failure of the language
model or proof that routing should be reshaped. Turning score separation into
a base-model training objective would introduce a different hypothesis about
loss, load balance, and routing controllability.

The next aligned question is whether existing trajectory information improves
residency or replication over simple policies. Co-training predictable routing
remains later work and should not be used to rescue an unproven placement
mechanism.

### A strong oracle gap can coexist with the wrong predictor

H6's next-use oracle materially beats LRU, so residency is not intrinsically
empty. Yet the depth predictors fail. Oracle headroom identifies an opportunity
for information; it does not prove that the information already collected is
the right information.

Before tuning a model, write the exact conditional event the controller needs
to predict. This check would distinguish same-token layer lookahead from
cross-token reuse and prevents optimizing an impressive but causally misaligned
metric.

---

## Provocative hypotheses for later work

These are not established results.

### Sparse top-1/top-2 routers may strengthen the sidecar case

OLMoE's top-8 route gives transition tables eight source observations while
making complete-set coverage unusually demanding. A top-1/top-2 model may
provide weaker route telemetry but require much smaller complete sets. Hidden
state could become more valuable relative to transitions.

### Routing may reveal a latent computational program

Predictable expert trajectories may be an observable projection of a token's
internal computational plan. Early residual state may encode not only meaning,
but an approximate future sequence of specialized transformations.

### The best output may be a resource action, not an expert label

A future controller might predict:

- reserve bandwidth;
- retain or evict an expert;
- replicate to another tier;
- move an activation instead of weights;
- prefetch a tile or quantized fragment;
- defer commitment until later feedback.

Direct resource-action prediction could eventually dominate independent
expert-label prediction, but only after H4/H5 define the relevant cost.

### Whole-expert movement may fail while the principle survives

Even if 12 MiB experts are physically too large for just-in-time movement,
trajectory prediction may still benefit:

- slow-timescale replication;
- fast-tier residency planning;
- expert-tile or matrix-fragment movement;
- token dispatch to resident experts;
- bandwidth and queue reservation;
- workload admission and batching.

The predictive-control thesis is broader than whole-expert PCIe prefetch.

### Lookahead is bandwidth currency; amplification is its exchange rate

The AX inverse sweep makes the co-design relationship unusually simple:

\[
\beta_{\min}\propto \frac{A}{\Delta}.
\]

At trace-derived K=16 cold demand with 12 MiB experts and A=1×, the mean
bandwidth bound falls from 71.3 GB/s at Δ=1 to 22.8, 11.6, and 8.2 GB/s at
Δ=3, 6, and 9. Longer prediction is valuable even without greater accuracy
because it converts semantic foreknowledge into physical service time.
False-positive amplification spends that time almost linearly.

This suggests a clean router/interconnect contract: a predictor should expose
not merely accuracy, but a deadline-indexed amplification frontier.

### Mean feasibility and tail schedulability are different architectural gates

The AX phase map contains regions with adequate mean cold-service headroom,
yet selected FCFS replay is much worse than the wave-local bound. For K=16,
Δ=9, C=99%, and A=1.5× at measured PCIe, wave-local P99 stall is 33.0 ms while
the within-token FCFS replay reaches 65.2 ms.

The missing concept is schedulability: bursts, false candidates, and correlated
misses share a finite queue. A link can satisfy \(\rho<1\) on average and
still violate the token tail. Predictor quality therefore resembles fault
coverage more than ordinary classifier accuracy. Complete-wave misses are
rare synchronous recovery events, and their acceptable rate is set by the SLO
percentile.

### Routing sparsity is a memory-hierarchy multiplier

For whole-expert rolling staging, fast-tier occupancy scales as
\(2AkS_{\mathrm{expert}}\) under double buffering. OLMoE's top-8, 12 MiB route
therefore requires 192 MiB even at A=1× and 384 MiB at A=2×. No 512 MiB
whole-expert cell can tolerate A=4×.

Top-k is consequently not just a model-compute parameter. It jointly controls:

- complete-set reliability difficulty;
- speculative occupancy;
- link traffic per wave;
- minimum useful SRAM staging capacity.

This sharpens why a top-1/top-2 confirmation matters: lower routing width may
alter the architecture envelope more profoundly than a modest bandwidth
increase, even if its raw trajectory predictor is less accurate.

---

## Unanswered foundational questions

1. Does the descriptive \(K=32,\Delta=3\) oracle region survive measured
   concurrent copy/compute contention?
2. Can prediction be aggregated across requests or token waves so that reuse
   offsets candidate churn?
3. Can a direct temporal-reuse predictor, request-level aggregation, or
   replication objective exploit the H6 oracle gap without becoming a new
   high-complexity project?
4. Is future-routing information genuinely low-dimensional, or did one random
   projection happen to work well?
5. Does the early-linear/late-transition regime reproduce on fresh requests?
6. Does it generalize to a newer top-1/top-2 MoE checkpoint?
7. Are middle layers easier to predict because of lower router entropy,
   semantic specialization, residual-stream geometry, or another mechanism?
8. Is whole-expert movement ever preferable to moving activations toward
   resident experts?
9. Should the hierarchy optimize strict latency, throughput, bandwidth,
   replication quality, or different objectives in different regimes?
10. Can routing or control behavior eventually be co-trained without harming
    model quality?

---

## Deadline erasure turns a synchronization problem into a distortion budget

Exact sparse execution has an AND-style completion rule: a layer cannot finish
until every demanded cold expert is available. One late expert therefore makes
the token inherit the maximum transfer/queue delay. AX4 shows that an atomic
commit plus optional residual experts changes the mathematical object being
managed:

\[
\max_i T_{\mathrm{expert},i}
\quad\longrightarrow\quad
m=\sum_{i\in M} a_i.
\]

This is a foundational co-design shift. The interconnect no longer determines
the committed latency after the deadline; it determines how much optional
expert mass arrives before commit. Hardware schedules distortion reduction per
byte, software enforces the deadline, and training determines whether a given
tail of missing mass is acceptable.

The shift is useful only in a real service regime. Trace-ordered FCFS replay
finds no credible whole-expert PCIe point: P99 normalized missing mass remains
81–100% at 24.14 GB/s. A boundary appears around 128–256 GB/s under C=99%,
A=1.5× mass-priority admission. K=16 at 128 GB/s misses the 20% contract by
only 0.4 point; K=32 passes at 7.2%; K=8/16 at 256 GB/s are effectively exact
at the wave P99.

Three deeper lessons follow:

1. **Approximation can bound latency without making transfer cheap.** It
   removes late transfer from the critical path, but only adequate service
   capacity keeps the quality/distortion tail small.
2. **More residency is not monotonically more profitable.** K=32 at 256 GB/s
   delivers essentially all mass, yet fails the 25% improvement gate because
   reactive offload is already fast. A profitable Pareto point needs both a
   low erasure tail and enough avoided reactive cost.
3. **A zero P99 does not mean no degradation.** At the K=8 headline point,
   0.93% of waves are degraded and the worst wave loses 27.4% normalized mass,
   even though wave P99 is numerically zero. Publication claims should pair
   P99 with degraded-wave incidence, token-level maxima, and worst observed
   mass.

There is also a model-semantics warning. OLMoE does not renormalize after
selecting top-8; its selected weights sum to 0.406 on average. Normalizing
within the selected set is appropriate for a future routed-contribution
contract, but it is not a literal description of current OLMoE execution.

---

## Strongest defensible claims today

### Directly supported for the pinned OLMoE workload

- Global static hotness is insufficient, while conditional routing structure
  is strong.
- Current routes predict nearby future expert demand far better than marginal
  popularity.
- A fixed projected-hidden-state linear readout retains substantial
  long-range routing information.
- Linear prediction is strongest from early source layers; transitions are
  preferable in several late-layer regimes.
- Complete-set coverage and candidate churn materially change the
  interpretation of ordinary selection coverage.
- On the measured platform, the compact \(K=16,\Delta=1\)–3 whole-expert
  oracle target is physically insufficient.
- A larger \(K=32,\Delta=3\) oracle region exists in the analytical replay.
- A controlled first-order profitability region exists, but the unchanged
  transition and linear candidate streams do not enter it.
- Under an explicitly assumed C=99%, A=1.5× future router, the AX wave-local
  projection improves P99 by 34–39% over reactive PCIe offload at equal HBM
  capacity, while remaining much slower than all-resident execution.
- The AX inverse bound quantifies the approximately A/Δ bandwidth trade:
  K=16 whole-expert A=1× demand needs 71.3/22.8/11.6/8.2 GB/s at
  Δ=1/3/6/9.
- Selected FCFS replay is materially worse than the mean/wave-local envelope,
  establishing that service headroom is necessary but not tail-sufficient.
- The current linear ranking contains real useful-versus-unused separation,
  but needs roughly 3.0–3.3× transferred/useful bytes to preserve 50%
  complete cold-set coverage.
- Existing transition/linear depth scores do not beat static/domain/LRU
  on-demand residency at equal capacity and movement budget.
- A strong equal-budget next-use oracle gap remains: at decode K=16, Δ=3,
  residual cold demand is 31.2% for oracle versus 48.1% for LRU.
- Under identical inputs, OLMoE Base and Instruct retain 89.7% of expert
  selections, and their layer-0→15 conditional prediction gain differs by only
  +1.6 points.
- Under assumed C=99%, A=1.5× mass-priority prediction and a nonblocking
  commit, trace-ordered FCFS replay identifies a deadline-erasure regime:
  K=8/16 pass at 256 GB/s and K=32 passes at 128 GB/s.
- The same AX4 replay rejects measured PCIe for this mechanism: P99 normalized
  missing mass is 100%/100%/81% at K=8/16/32.
- Under rare tail erasure on the frozen base checkpoint, **null-drop** quality
  cost is monotone and roughly additive in the number of dropped expert-layers:
  worst case L=8 is 0.013 nats conditional-on-affected KL with zero large
  divergence, layer-uniform sensitivity, no spacing benefit, and no cross-token
  leak (prefill, single domain). Universal mass-budget erasure is catastrophic
  (KL 5.81). Renormalization is ~40× worse than null at equal mass and is
  dropped as a strategy.

### Plausible architectural inference

- A hybrid early-planning/late-correction controller is more appropriate than
  one universal prediction policy.
- Predictive information may help within-token scheduling even when literal
  prefetch is too expensive; residency requires a separate temporal-reuse
  signal.
- A memory hierarchy could be organized around certainty and deadline as well
  as speed and capacity.
- Experts transferable per layer interval is a useful co-design quantity, and
  prediction becomes actionable only after residency reduces cold demand.
- An always-resident competence path plus optional routed residual experts
  could turn expert availability into an explicit, telemetry-visible
  distortion budget rather than a token-level synchronization failure.

### Not yet supported

- The analytical \(K=32,\Delta=3\) region survives real concurrent
  copy/compute contention.
- Prediction reduces end-to-end latency or TPOT.
- The current prediction-guided transfer policy is profitable.
- The current depth predictors improve on-demand expert residency.
- The result generalizes beyond one top-8 OLMoE checkpoint.
- The within-family Base–Instruct stability result generalizes across model
  families or routing sparsities.
- The base model learned to manage hardware resources.
- Making routing more predictable would preserve model loss and load balance.
- Whole-expert movement beats activation movement or additional local memory.
- The null-drop tail tolerance holds beyond the measured regime: across other
  domains, under autoregressive decode compounding, or past AX4's
  incidence/run-length/multi-expert boundary (the Q2 stress arms test this).
- Renormalization or shared-residual substitution preserves quality as well as
  null-drop does.
- A future availability-trained model can meet the AX4 ≤20% P99 missing-mass
  contract without harming exact-mode quality or load balance.

---

## Candidate publication framing

### Possible title

**MoE Routing Is a Structured Trajectory Through Expert Space**

### Possible subtitle

**Early Hidden States Enable Long-Range Expert-Demand Planning While Later
Routes Provide Short-Range Correction**

### Candidate contributions under the current evidence

1. Show that static expert popularity misses predictable conditional
   trajectories.
2. Identify a source-layer-dependent crossover between hidden-state and
   transition prediction.
3. Demonstrate why complete-set coverage and candidate churn—not ordinary
   recall—govern architectural usefulness.
4. Map the capacity–lead-time–bandwidth region in which predictive information
   is physically actionable.
5. Separate predictive belief, residency state, and byte-committing admission,
   showing why information, service, and selectivity are independent gates.
6. Show that depth-trajectory predictability and temporal cache reuse are
   distinct, and that policy value requires matching the prediction axis to
   the resource-consumption axis.
7. Derive the deadline-indexed \(A/\Delta\) interconnect law and the
   \(2AkS\) rolling-staging capacity law, separating mean feasibility from
   tail schedulability.
8. Recast late experts as deadline erasures under an always-resident competence
   path, and quantify the bandwidth–residency–distortion region in which a hard
   TPOT bound becomes a plausible training target.

The defensible contribution is a workload and co-design boundary, not a
profitable end-to-end prefetch implementation. Strong long-range routing
information can coexist with insufficient physical time or excessive
speculative traffic; this redirects the mechanism toward residency,
replication, scheduling, or finer-grained movement without weakening the
central trajectory result.

---

## Revision history

- **2026-08-01:** Initial synthesis after H1, H2, H3, and the complete
  source-target horizon analysis. H4 physical feasibility remains open.
- **2026-08-01:** H4 added the capacity–lead-time–bandwidth boundary,
  established the experts-per-layer-interval screening quantity, and showed
  that prediction and residency are complements.
- **2026-08-01:** H5 separated predictive belief from transfer admission,
  established information–service–selectivity as three independent gates, and
  showed how rare-event base rates make strong ranking insufficient for
  profitable movement.
- **2026-08-01:** H6 separated within-token depth prediction from cross-token
  reuse prediction. Existing depth scores failed equal-budget residency despite
  a strong next-use oracle ceiling, establishing the prediction-axis and
  mechanism-axis alignment principle.
- **2026-08-01:** C0 compared matched Base and Instruct endpoints. Post-training
  preserved about 90% of expert selections and did not materially change the
  frozen long-range predictability metric, suggesting a pretrained trajectory
  scaffold with local post-training edits.
- **2026-08-01:** AX1–AX3 converted the future-router assumption into
  bandwidth/lookahead, capacity/P99, and rolling-SRAM bounds; it established
  A/Δ as the mean service trade, 2AkS as the staging-capacity bound, and
  separated mean feasibility from tail schedulability.
- **2026-08-01:** AX4 converted cold-transfer synchronization into an explicit
  missing-mass contract. It rejected measured PCIe, identified a
  mass-priority 128–256 GB/s FCFS regime, and established that a hard latency
  bound, service capacity, and erasure robustness are three separate
  requirements.
- **2026-08-02:** Q1/Q1-B priced the AX4 erasure budget empirically on the
  frozen base model. Universal mass-budget erasure is catastrophic (KL 5.81);
  rare tail erasure under null-drop is monotone-additive and semantically
  near-free (worst L=8 0.013 nats), layer-uniform and local. This supports the
  additive-residual mechanism, moves AX4's bounded-run contract from assumed
  to *measured*, and reframes Q2 as a non-training stress probe (cross-domain,
  decode compounding, cliff mapping) with a gated minimal mask-aware
  calibration rather than an unconditional robustness-training requirement.
