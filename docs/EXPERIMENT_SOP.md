# Lean experiment standard operating procedure

Use this loop for every major hypothesis experiment. It keeps a human in the
decision path without adding a tracking system or notebook ceremony.

## 1. Freeze

- State one simple decisive question, one primary scope, and one stop/go rule.
- Pin model, dataset, generation, and analysis configurations.
- Treat every gate and conclusion as scoped to the pinned model revision. When
  introducing another checkpoint, follow
  `docs/ADDING_MODEL_TESTBED.md` so results and insights remain separated until
  an explicit cross-model synthesis.
- Keep cheap descriptive and post-hoc analysis non-gating. Do not preregister
  every possible layer, domain, or hardware interaction.
- For assumption-driven architecture studies, freeze an evidence contract that
  labels every input as measured, trace-derived, assumed predictor behavior,
  or hypothetical hardware. Do not let an optimistic predictor sweep alter an
  empirical model result.

## 2. Validate

- Run the smallest instrumentation smoke test.
- Require trace/schema/router integrity before scaling.
- Confirm that resumed artifacts match the frozen configuration fingerprint.
- For paired-checkpoint experiments, force one shared serialization and require
  exact input-token-ID equality before interpreting route differences.

## 3. Collect

- Write restartable request-level artifacts.
- For any perturbation, erasure, or generation probe, retain the **original and
  perturbed token sequences** (decoded text and/or token IDs) so that post-hoc
  human inspection of the actual model output is always possible without
  re-running measurement. Q2 exposed this the hard way: only step-KL/agreement
  aggregates were saved, and interpreting the "runaway" decode divergence as
  degradation — versus paraphrase — required re-running the legs to recover the
  streams. Store the streams once as a disposable trace; do not reduce them to
  a scalar at collection time.
- Never repair raw traces in place after analysis. Routing traces, projected
  hidden features, full hidden states, and activations are local replay inputs:
  they are ignored by Git and may be discarded once the derived evidence is
  safely committed. Original/perturbed token sequences follow the same
  disposable, Git-ignored rule as other raw traces — persist them, then commit
  only the derived compact tables.
- Record environment, hardware, revisions, and request completion.
- Keep the compact run definition, run manifest, model report, dataset
  manifest, and exact prepared prompts under `artifacts/`; these are durable
  Git-tracked provenance, not disposable trace data.

## 4. Analyze

- Generate machine-readable tables before figures.
- Keep layer, phase, domain, and top-k semantics explicit.
- In checkpoint comparisons, report conditional gain over the checkpoint's own
  marginal baseline; raw accuracy alone can confound predictability with skew.
- Apply the frozen decision gate before exploratory interpretation.
- Then run cheap broad scans over structure already present in the artifacts,
  such as all source-target layer pairs. These scans may narrow the conclusion
  but never rewrite the original gate.
- Mine the result for boundary locations, empty windows, crossover points,
  dimensionless ratios, and which constraint is active. Keep this cheap and
  post-hoc; do not turn every observation into a new preregistered branch.
- When studying co-design, solve the inverse requirement as well as the forward
  result: report what predictor quality or hardware headroom would be needed
  to change the decision.
- For capacity-enabling offload, compare predicted policies with the same
  reactive hierarchy. Do not call CPU/pooled-memory offload faster than an
  otherwise identical all-resident execution.
- Evaluate future predictors by complete cold-set coverage and
  predicted/useful byte amplification. Generate correlated misses at the wave
  level; independent label flips may be used only as a labeled sensitivity.
- Report mean service pressure separately from trace-replayed P95/P99 queue
  behavior. Mean headroom is necessary but not sufficient.
- For deadline-degraded execution, separate the latency guarantee from the
  assumed quality contract. Enforce zero waiting after commit, report missing
  normalized routed mass and full-fallback incidence, and state the compute,
  fallback, and traffic-isolation assumptions required for a hard bound.
- Add request-level uncertainty only for a confirmation run or a genuinely
  borderline decision; do not bootstrap every pilot table.

## 5. Visualize

- Prefer one simple headline curve and, when heterogeneity matters, one
  layer/domain/regime heatmap. Add a third figure only if it changes a decision.
- For assumption-driven co-design work, prefer one categorical phase diagram
  and one inverse-design curve. Clearly distinguish candidate, analytically
  profitable, and experimentally demonstrated regions.
- Use checked-in scripts, not manual plotting or a notebook as the source of
  truth.
- Save vector PDF and high-resolution PNG plus a manifest that hashes the
  figure inputs.
- Write figures beside their canonical analysis tables under
  `artifacts/runs/<run-id>/analysis/`. Do not copy them into a second results
  tree or manually edit a duplicate for publication.
- Label selection-, token-, wave-, and step-level quantities precisely.
- Avoid uncertainty marks until the corresponding resampling unit is defined.

## 6. Human visual-review checkpoint

Pause before starting the next hypothesis. The researcher reviews the figures
and records. **When the working session's model cannot see images** (e.g. a
non-vision model), do not fake or guess a visual read: skip the pixel-level
figure review and instead run the programmatic equivalent — (a) confirm the
figure-input hashes and manifest cover the exact tables that produced them,
(b) recompute the machine-readable headline values from the analysis tables
and assert they match the gate, and (c) review the underlying numbers and
generated token streams directly. Mark the figure review note as
"visual review skipped (no vision); verified programmatically" so the
substitution is explicit. The checkpoint records:

- whether axes, units, aggregation, baselines, and thresholds are correct;
- whether machine-readable headline values agree with the plots;
- visible regimes, outliers, layer clusters, saturation points, and confounds;
- whether the visual evidence supports, weakens, or changes the automated
  decision;
- the single next action.

The review is not permission to change a preregistered gate after seeing the
result. Post-hoc findings must remain explicitly exploratory.

## 7. Decide and hand off

- Update `EXPERIMENT_LOG.md` with the formal result and human interpretation.
- Update `STATUS.md` with supported, mixed, rejected, or inconclusive.
- Link the metric report, figures, and review note.
- Run the artifact-retention closeout below and commit its staged outputs with
  the code, configs, and documentation that produced or interpreted them.
- Advance only after the human has reviewed the result.
- Once a held-out set drives post-hoc policy discovery, treat it as development
  data. Use fresh requests only if the physical gate justifies confirmation.

For a fast prototype, a Markdown checklist in the generated figure directory
is sufficient evidence of review. Do not introduce an experiment-tracking
service unless artifact discovery becomes a real bottleneck.

## 8. Artifact-retention closeout

The native `artifacts/` paths are the single source of truth for publication
evidence. Ordinary Git tracks all compact and derived products, including:

- CSV analysis tables and measured samples;
- JSON definitions, manifests, summaries, gates, and integrity results;
- Markdown reports and figure-review notes;
- PDF and PNG figures;
- compact fitted analysis outputs such as linear-predictor NPZ files;
- prepared dataset manifests and exact prompts.

Only large replay inputs are ignored:

- `artifacts/runs/*/trace/`;
- `artifacts/runs/*/features/`;
- `artifacts/runs/*/hidden_states/`;
- `artifacts/runs/*/activations/`.

Treat these ignored files as disposable until external archival storage is
deliberately introduced. Do not let a result document depend on an ignored raw
file when a compact supporting table can be generated instead.

After every major analysis and visualization, run this fixed closeout:

```bash
uv run ep-predict audit-artifacts
git add artifacts
uv run ep-predict audit-artifacts --require-tracked
```

The first command checks that figure inputs and outputs exist and still match
their recorded SHA-256 hashes, that result/status documents do not contain
stale artifact paths, and that only approved raw-data directories are ignored.
`git add artifacts` stages every durable product automatically while respecting
the raw-data exclusions. The final command fails if any durable product was
missed.

Do not hand-select extensions, copy a “best results” subset, or create a second
publication directory. If a generated artifact is no longer valuable, remove
it and its references in the same change. The audit warns about any individual
durable file above 50 MiB so repo growth receives an explicit human decision
rather than silently expanding.
