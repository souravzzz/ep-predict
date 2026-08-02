# Project status

**Current focus:** Q2 — availability-conditioned robustness, stress and cliff
mapping (**measured, closed by the researcher**): the three arms are run,
gated, and accepted. Cross-domain and cliff confirm the free null-drop
tolerance; the decode leg produces *trajectory divergence under a near-tie
flip that manual inspection of the generated text shows is paraphrase, not
degradation*. No non-free regime is demonstrated, so **no mask-aware
calibration is justified** and Q2 closes.
**Current stage:** Q1/Q1-B accepted by the researcher. The universal
mass-budget probe is a decisive STOP (KL 5.81); the AX4-faithful **null-drop**
tail is a decisive **GO** — quality cost is monotone and roughly additive in
depth L=1→8 (worst case 0.013 nats conditional KL), with zero large divergence,
layer-uniform sensitivity, no spacing benefit, and no cross-token leak. The
model already tolerates the AX4 tail essentially for free in **prefill**.
Q2 adds the untested axes: the tolerance holds **across domains** (math +
WikiText-2) and **with margin at the cliff** (first boundary at 2
experts/layer, KL 0.028). The **decode** leg shows the two streams diverge
after a rare near-tie flip (`1.6→1.5` in the replayed case), but the erased
stream keeps generating fluent, on-topic, equally valid text — a **paraphrase,
not a quality collapse**. The large step-KL on the diverged trajectory is a
trajectory artifact (different-but-valid contexts), not evidence of
degradation. No calibration or guard is warranted.
**Last updated:** 2026-08-02 (Q2 stress arms measured, gated, accepted; decode = paraphrase, no calibration)

| Gate | Question | State | Exit evidence |
|---|---|---|---|
| M0 | Can we inspect and validate the model without source changes? | Passed | Model report, complete hook traces, and zero integrity failures |
| H1 | Is hotness strong and stable enough for a fast tier? | Mixed; global gate failed | 2/16 mixed decode layers passed; code and math are locally strong |
| H2 | Does conditional locality beat marginal popularity? | Pilot supported | All Δ=1/2/3 transition baselines passed the held-out decode gate |
| H3 | Can a small predictor beat transition tables at equal candidate budget? | Formal pilot not supported; reviewed | Global replacement failed; post-hoc scan found strong early-layer/long-range value |
| H4 | Is oracle prefetch physically viable? | Formal K=16 short-horizon gate not supported; broader region mixed | At K=16, Δ=3: 32.8% timely cold bytes and 38.9% stall reduction; K=32, Δ=3 reaches 55.5%/61.8% |
| H5-A | What prediction × hardware assumptions create a profitable analytical window? | Supported analytically | 22,618/68,175 controlled cells pass the frozen screen |
| H5-B | What predictor quality is required to enter that window? | Complete | 25–50% minimum complete cold-set coverage in nonempty windows |
| H5-C | Where do existing transition/linear policies land? | Raw streams not supported | 3.4–6.7× transfer amplification; no representative row passes |
| H5-D | Do existing scores separate useful from useless cold candidates? | Mixed; strong ranking, insufficient threshold | Linear AUROC 0.883/0.861 at Δ=3/9; C≥50% needs A≈3.0–3.3× |
| H6 | Does prediction-guided residency beat static/domain/LRU placement? | Pilot not supported; reviewed | At decode K=16, Δ=3, transition/linear lose 3.9/2.5 pp expert-stall reduction and 0.7/0.6 pp complete hits versus the strongest matched baseline |
| H7 | Can routing be made more predictable without harming loss or balance? | Deferred after H6 failure | Requires a new mechanism and explicit permission |
| C0 | Does post-training materially change matched-token trajectory predictability? | Pilot not supported; review pending | Base/Instruct retain 89.7% of selections; layer-0→15 conditional-gain change is +1.6 pp versus a 5 pp gate |
| C1 | Does the result transfer to a top-1/top-2 checkpoint? | Deferred; explicit permission required | No model download or testbed change authorized |
| Q1 | Does the frozen model tolerate the expert erasure AX4's deadline contract relies on? Is quality loss controlled by missing routed mass? | **STOP (universal) / near-free null-drop tail; accepted by researcher** | Universal mass-budget headline (renormalize, m=0.125): KL 5.81, top-1 9.2%, PPL 279.9, non-monotone. Tail-event sub-track (rare one-expert one-layer drops): null-drop near-free (conditional KL 0.0032, top-1 99.2%); renormalize ~40× worse and dropped as a strategy |
| Q1B | Under null-drop, is quality loss additive and monotone in the number/placement of dropped expert-layers (depth, layer order, spacing, cross-token leak)? | **GO; accepted by researcher** | Depth sweep L=1→8 conditional-on-affected: monotone (KL 0.0041→0.0133), flat per-layer marginal (ratio 1.20 ≤3), 0% large divergence at L=8; layer-uniform, spacing-insensitive, no cross-token leak. Additive-residual mechanism supported |
| Q2 | Does availability-conditioned robustness make the model tolerate the AX4 erasure distribution (and if not, where)? | **Measured; accepted by the researcher — no non-free regime, no calibration** | Q2-A **GO**: depth-additive tolerance holds on math (L=8 KL 0.0090) and WikiText-2 (0.0145), both monotone, no super-linear, zero large divergence. Q2-C **WITH_MARGIN**: AX4 nominal cell free (L=8 KL 0.0102); cliff only at 2 experts/layer (KL 0.028); no incidence/run-length crossing through 0.5/16. Q2-B **decode**: streams diverge after a rare near-tie flip, but the replayed generated text shows the erased stream is a fluent, on-topic **paraphrase, not a quality collapse** — divergence (trajectory artifact), not degradation. No calibration or guard justified |
| AX1 | Under assumed future MTP-style routing quality, what capacity/TPOT envelope does predictive offload enable? | Projected region exists; review pending | At measured PCIe and assumed C=99%, A=1.5×, wave-local P99 improves 34–39% versus reactive offload; FCFS queue tails are materially worse |
| AX2 | What bandwidth, latency, reliability, amplification, and granularity bounds define viable regions? | Complete; review pending | K=16, A=1× needs 71.3/22.8/11.6/8.2 GB/s at Δ=1/3/6/9; reliability remains orthogonal |
| AX3 | What HBM and rolling-SRAM organization suits a three-tier predictive hierarchy? | Physical staging envelope complete; review pending | Top-8 whole-expert double buffering needs 192 MiB at A=1× and 384 MiB at A=2×; no SRAM execution speedup is claimed |
| AX4 | Can deadline-controlled expert erasure bound low-batch TPOT while retaining a plausible routed-mass/quality contract? | Supported analytically under explicit assumptions; review pending | K=8, 256 GB/s, C=99%, A=1.5× passes with 1/8 experts resident, 11.25 ms bounded TPOT, zero full fallback, and <1% degraded waves; measured PCIe fails |

## Immediate run checklist

- [x] Implement the Q1 probe (runtime MoE-forward patch) and its measure/analyze/plot
      CLI subcommands; semantic smoke confirms the patch is a no-op when inactive
      and erases real mass under the headline cell (softmax-64→top-8→no-renormalization).
- [x] Materialize WikiText-2 paired clean-vs-erased tables, apply the frozen
      Q1 gate, and generate the `ΔQ vs m_missing` curve + positioning/correlation
      panel with hashed inputs.
- [x] Human reviewed the Q1 figures (fig1/fig2 mass-budget + fig_tail
      sub-track) and accepted the result: universal erasure STOPS; the
      AX4-faithful null-drop tail is near-free.
- [x] Add the AX4-faithful tail-event sub-track: rare one-expert one-layer
      drops swept by incidence and consecutive-layer run length, conditional
      on the affected tokens (CLI `measure/analyze/plot-q1-tail`).
- [x] Human-decide whether Q1's decisive cell is the universal renormalize
      mass sweep (STOP) or the AX4-anchored tail (null-drop near-free).
- [x] Freeze the Q1-B protocol (commit to null-drop; renormalize dropped).
- [x] Implement Q1-B: extend the erasure controller (pin layer, spacing gap,
      shared cell seed), measure/analyze/plot subcommands, config, and tests.
- [x] Execute Q1-B on the frozen base checkpoint and apply the frozen gate
      (depth additivity L=1→8 at AX4 anchor, conditional-on-affected): **GO**.
- [x] Human reviewed the Q1-B figures (fig1 depth additivity + fig2 mechanism)
      and accepted the result; the Q1-B additive-residual mechanism is closed.
- [x] Freeze the Q2 protocol and config (docs/Q2_PROTOCOL.md,
      configs/experiment/q2-stress.toml), reframed to verify the measured free
      null-drop tolerance across the untested axes.
- [x] Implement and run the Q2 stress arms (cross-domain, decode compounding,
      cliff mapping) on the frozen base checkpoint once authorized.
- [x] Apply the frozen per-arm gates: Q2-A GO (cross-domain), Q2-B decode
      divergence observed, Q2-C WITH_MARGIN (cliff outside the contract bound).
      Generate the three hashed figures; human visual review skipped (no vision
      on this session's model) — verified determinism and hashes
      programmatically instead.
- [x] Re-run the Q2-B decode legs capturing the generated token IDs and decode
      both clean/erased streams to text for manual inspection; the erased
      stream is a fluent, on-topic paraphrase, so the decode divergence is a
      trajectory artifact, not a quality collapse.
- [x] Researcher accepted Q2 as measured: no non-free regime demonstrated, so
      the gated mask-aware calibration sub-step is **not entered** and Q2
      closes. (SOP updated to always retain original/perturbed decoded tokens
      and to skip visual analysis when the session model has no vision.)
- [ ] Confirm the actual machine exposes the intended 24 GB NVIDIA GPU.
- [x] Install the `data` and `inference` dependency groups.
- [x] Materialize and review the revision-pinned standard-small workload.
- [x] Run model inspection and retain `model_report.json`.
- [x] Run two standard examples with `--limit 2`; require zero integrity errors.
- [x] Run the 128-request standard pilot.
- [x] Review H1 gate output before expanding the workload.
- [x] Generate the three scripted H1 decision figures.
- [x] Human reviewed H1 and approved advancing with the mixed interpretation.
- [ ] If borderline, run the confirmation workload with more requests and
      request-level bootstrap intervals.
- [x] Decide H1 and record the outcome in `EXPERIMENT_LOG.md`.
- [x] Preregister H2 with a request-level held-out split and fixed gate.
- [x] Reuse the H1 trace to evaluate static, domain, lagged, and transition
      baselines at K=8/16/32 and Δ=1/2/3.
- [x] Generate H2 diagnostics, then replace the primary view with two simple
      lookahead plots.
- [x] Human reviewed the simplified H2 figures and approved advancing to H3.
- [x] Preregister the minimal H3 proof/disproof gate before collecting hidden
      features or training a predictor.
- [x] Validate deterministic projected router-input capture on two requests.
- [x] Collect all 128 H3 requests with aligned routing and feature shards.
- [x] Train the single fixed linear recipe on the preserved 96/32 split.
- [x] Apply the primary H3 gate and reproduce all H2 transition metrics.
- [x] Generate two simple H3 decision figures and hash their inputs.
- [x] Extend H2/H3 descriptively through Δ=15 without new inference.
- [x] Generate the global horizon curve and source-target gain heatmap.
- [x] Human reviewed H3 and the extended figures; formal H3 failure stands,
      with early-layer linear prediction retained as an exploratory regime.
- [x] Freeze one simple H4 question and oracle stop/go rule.
- [x] Measure unhooked inter-layer time and a small host-to-device transfer
      curve; hooked timings remain excluded.
- [x] Replay exact 12 MiB experts over a small bandwidth/capacity/issue-point
      grid and plot the oracle feasible region.
- [x] Apply the formal gate before interpreting the broader scan; the
      preregistered K=16, Δ=1–3 target failed.
- [x] Do not overlay or tune transition/linear policies after the formal H4
      failure.
- [x] Researcher accepted first-order analytical modeling as sufficient for
      the next viability/profitability-window study; timing fidelity is not the
      current focus.
- [x] Add the post-hoc cold-service-headroom versus complete-prediction regime
      map; keep “candidate region” distinct from demonstrated profitability.
- [x] Preregister H5-A’s simple analytical-profit screen: ≥25% modeled stall
      reduction, ≥50% oracle recovery, and ≤2× predicted/useful bytes.
- [x] Sweep complete coverage, candidate amplification, K=8/16/32, Δ=1–15,
      and 0.25×–4× normalized cold bandwidth using existing artifacts.
- [x] Derive H5-B minimum complete coverage and maximum amplification rather
      than tuning a predictor blindly.
- [x] Generate one profitability phase diagram and one inverse-requirement
      curve.
- [x] Researcher explicitly advanced from H5 to the minimal H6 residency
      study; the H5 formal result remains unchanged.
- [x] Reconstruct and place existing transition/linear streams at four
      representative H5-C points without retraining.
- [x] Sweep a shared standardized score threshold and plot useful-versus-false
      expert separation at K=32, Δ=3/9.
- [x] Defer the cost-sensitive admission head; test placement value first.
- [x] Preregister H6 with held-out decode K=16, Δ=3, one demanded-miss
      insertion per wave, and a breadth gate across layers and domains.
- [x] Replay static, domain, LRU, transition, linear, and equal-budget oracle
      residency at K=8/16/32 over both phases and all valid lookaheads.
- [x] Report residual cold demand, complete-set hits, useful/wasted movement
      bytes, evictions/churn, first-order stall reduction, and oracle recovery.
- [x] Generate the compact layer/lookahead/capacity gain heatmap and hash its
      inputs.
- [x] Apply the H6 gate unchanged: neither guided policy passes.
- [x] Complete human review of the H6 heatmap and record the final
      interpretation.
- [x] Defer overlap microbenchmarks, MLPs, fresh routing collection, H7, C1,
      and timing
      fidelity until a selective policy identifies a worthwhile mechanism.
- [x] Preregister C0 before Base collection with exact matched-token
      serialization and a fixed layer-0→15 conditional-gain gate.
- [x] Download and qualify OLMoE Base only after explicit researcher approval;
      verify identical 16-layer, 64-expert, top-8 geometry.
- [x] Collect Base and Instruct raw-prefill traces on the same 128 prompts with
      one forward per request and zero router mismatches.
- [x] Verify all 13,918 input tokens match exactly across checkpoints.
- [x] Fit H2 tables independently on the same 96 requests and evaluate the same
      32 held-out requests through Δ=15.
- [x] Apply the C0 gate unchanged: +1.6 pp is below the 5 pp stage-effect
      threshold.
- [x] Generate and programmatically inspect the predictability and matched-route
      figures with hashed inputs.
- [ ] Complete researcher visual review; do not add SFT/DPO unless that review
      overturns the frozen endpoint stop decision.
- [x] Freeze the AX evidence contract separating measured, trace-derived,
      assumed-predictor, and hypothetical-hardware inputs.
- [x] Freeze AX1 predictor-quality, capacity, lookahead, bandwidth, latency,
      concurrency, granularity, and SLO sweep axes.
- [x] Define capacity viability, reactive-hierarchy profitability, and SLO
      safety without claiming speedup over all-HBM execution.
- [x] Define AX2 inverse requirements and AX3 rolling three-tier SRAM
      semantics.
- [x] Implement AX1 by extending the existing H4/H5 replay.
- [x] Reproduce the measured H4/H5 anchors before interpreting synthetic
      future-router points.
- [x] Generate the capacity–P99 Pareto frontier and execute the factorized AX2
      inverse-bound and AX3 rolling-SRAM sweeps.
- [x] Generate the three principal PDF/PNG figures and inspect them
      programmatically.
- [ ] Researcher reviews the AX figures and selects one representative
      architecture point, or closes the track without live calibration.
- [x] Freeze AX4's hard commit deadline, normalized routed-mass definitions,
      null/renormalized/shared-residual policies, and perturbation bounds.
- [x] Freeze the analysis-only weighted-route replay and mass-priority oracle
      without authorizing model training or inference collection.
- [x] Define low-batch bounded TPOT/tokens-s projections and clearly label
      top-1/top-2/large-model geometry as sensitivity rather than evidence.
- [x] Freeze the deadline-elastic HW proposal: always-resident fallback,
      optional refinements, commit bitmap, deadline-aware DMA, traffic
      isolation, and degradation telemetry.
- [x] Implement AX4 and first verify the selected-weight execution semantics.
- [x] Replay the retained trace, generate the three principal figures, and
      apply the plausible-degradation-contract gate before any training work.
- [ ] Researcher reviews the AX4 figures and accepts, narrows, or rejects the
      erasure-robustness target before any training or new-model work.

Full plan: [docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md).
Active protocol: [docs/Q2_PROTOCOL.md](docs/Q2_PROTOCOL.md).
Prior probe: [docs/Q1_PROTOCOL.md](docs/Q1_PROTOCOL.md).
Q1-B mechanism probe: [docs/Q1B_PROTOCOL.md](docs/Q1B_PROTOCOL.md).
Prior result: [docs/H4_RESULTS.md](docs/H4_RESULTS.md).
Architecture protocol:
[docs/ARCHITECTURE_EXPLORATION_PROTOCOL.md](docs/ARCHITECTURE_EXPLORATION_PROTOCOL.md).
Latest empirical result: [docs/C0_RESULTS.md](docs/C0_RESULTS.md).

## Evidence policy

For empirical hypotheses, `Ready` means code and protocol exist. AX results
are explicitly projected: they combine measured calibration, trace-derived
demand, assumed predictor quality, and hypothetical hardware. The canonical
tables, report, and figures are under
`artifacts/runs/h1-standard-small/analysis/architecture/`; none is a measured
end-to-end speedup. AX4's canonical result is under
`artifacts/runs/h1-standard-small/analysis/ax4_deadline_degradation/`. Its gate
pass identifies a future training contract; it does not mean current OLMoE
tolerates missing experts.
