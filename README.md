# ep-predict

Hook-only experiments for testing whether MoE expert demand is skewed,
predictable, and useful for hierarchical memory placement. The model and
Transformers source remain unmodified.

Follow a "tracer bullet" development approach that develops a narrow vertical slice to a result instead of breadth and ceremony. This is a scientist's tool for rapid prototyping and experimentation, not a production software. Keep the human in the loop. Explain your findings, analysis, and interpretation from each experiment before moving to the next one.

The project advances one research gate at a time. H1 was mixed: one
workload-agnostic static tier failed, while domain-conditioned demand was
strong. H2 was pilot-supported: held-out layer-transition tables strongly beat
per-layer marginal popularity. H3 did not support globally replacing that
simple policy with a fixed linear hidden-state sidecar at the primary \(n+1\)
gate. The post-hoc all-layer scan found a narrower use: early hidden states
strongly improve long-range prediction, while late source layers favor
transitions. H4 then rejected the preregistered \(K=16,\Delta=1\)–3
whole-expert PCIe target: even a perfect oracle made only 32.8% of cold bytes
timely and removed 38.9% of stall at the best primary point. A broader
descriptive region exists with \(K=32,\Delta=3\), longer lead time, or twice
the measured bandwidth.
See [STATUS.md](STATUS.md), [docs/H4_PROTOCOL.md](docs/H4_PROTOCOL.md), and
[docs/H4_RESULTS.md](docs/H4_RESULTS.md).

The current evidence supports source-target-aware planning for the pinned
OLMoE checkpoint: linear hidden-state candidates for early planning and
transition candidates for late refinement. H4 now adds a physical boundary:
the measured platform can hide whole experts only after enough residency,
lead time, or bandwidth reduces transfer pressure. It still does not establish
live copy/compute overlap, end-to-end latency improvement, or universal MoE
behavior. H5 found a first-order profitable region, but the unchanged
transition/linear streams transfer 3.4–6.7× useful cold bytes and do not pass
the frozen policy screen. H6 then tested prediction-guided on-demand residency
at equal capacity and movement budget. It failed: neither existing predictor
beat the strongest static/domain/LRU baseline across layers and domains,
despite a strong oracle ceiling. The key distinction is that predicting a
token's trajectory down network depth does not automatically predict expert
reuse across later tokens. See [docs/H6_RESULTS.md](docs/H6_RESULTS.md).

C0 then compared the pretrained Base checkpoint with its final
SFT+DPO+RLVR Instruct descendant under exactly matched input tokens. The two
checkpoints retain 89.7% of selected expert IDs across depth, and their
layer-0→15 conditional prediction gain differs by only +1.6 points—below the
frozen 5-point gate. The structured trajectory is already present in Base and
largely preserved by post-training for this lineage. See
[docs/C0_RESULTS.md](docs/C0_RESULTS.md).

The assumption-driven AX1–AX3 architecture exploration is now complete and
awaits human figure review. It treats a future MTP-style routing gate with swept complete-set
coverage and false-positive amplification as a workload/software contract,
then derives the host/pooled-memory, HBM, and software-managed SRAM capacities,
bandwidths, lookaheads, and transfer granularities required for useful
hierarchical execution. The goal is quantitative bounds and co-design regimes,
not wall-clock speedup on current OLMoE or the current GPU. At the measured
PCIe anchor, assumed C=99%/A=1.5× prediction improves the wave-local P99
projection by 34–39% versus reactive offload, but FCFS queue replay remains
much worse and all offload points remain slower than all-resident execution.
See the [protocol](docs/ARCHITECTURE_EXPLORATION_PROTOCOL.md) and canonical
[AX report](artifacts/runs/h1-standard-small/analysis/architecture/REPORT.md).

AX4, deadline-bounded graceful expert degradation, is now complete pending
human figure review.
Instead of waiting for every cold expert, execution commits at a fixed layer
deadline and uses an always-resident shared/identity/null path for missing
contributions. This can make TPOT independent of transfer completion while
moving prediction failures into an explicit missing-routed-mass and quality
contract. The formal analytical gate passes only in a high-bandwidth,
mass-priority regime: K=8 at 256 GB/s keeps 1/8 of experts resident, bounds
TPOT at 11.25 ms, and degrades fewer than 1% of waves. Measured PCIe fails at
every capacity with 81–100% P99 missing mass. This is an architecture/training
target, not evidence that current OLMoE preserves quality under erasure. See
the [protocol](docs/DEADLINE_DEGRADATION_PROTOCOL.md) and
[AX4 report](artifacts/runs/h1-standard-small/analysis/ax4_deadline_degradation/REPORT.md).

Q1 then measured that erasure cost directly on the frozen base checkpoint.
Universal every-layer erasure is catastrophic (KL 5.81, STOP), but the
AX4-faithful tail regime under the model's native **null-drop** is nearly free.
Q1-B mapped the mechanism: quality cost is monotone and roughly additive in the
number of dropped expert-layers (worst case L=8 is 0.013 nats conditional KL,
zero large divergence), layer-uniform, spacing-insensitive, and local — direct
support for the additive-residual hypothesis, and the first point where AX4's
bounded-run quality contract is *measured* rather than assumed. Q2 (ready)
verifies that tolerance across the untested axes (cross-domain, decode
compounding, cliff mapping) and only then considers a minimal mask-aware
calibration. See [docs/Q2_PROTOCOL.md](docs/Q2_PROTOCOL.md) and the
[Q1-B report](artifacts/runs/q1-quality-erasure/analysis/q1b_null/NULL_REPORT.md).

The active experiment queue, insight-mining questions, and visualization
deliverables are in
[docs/NEXT_EXPERIMENTS.md](docs/NEXT_EXPERIMENTS.md).

The durable publication thesis, foundational principles, perspective shifts,
and hard-earned lessons are curated separately in
[docs/FOUNDATIONAL_INSIGHTS.md](docs/FOUNDATIONAL_INSIGHTS.md). Unlike the
experiment log, that document changes only at major evidence transitions.

## Testbed

The primary model is `allenai/OLMoE-1B-7B-0125-Instruct`:

- 7B total and about 1.3B active parameters;
- 16 MoE layers;
- 64 routed experts per layer;
- top-8 routing;
- an explicit router module returning logits, routing weights, and selected
  expert IDs.

The BF16 checkpoint is about 13.8 GB and fits the target 24 GB GPU. The
implementation discovers routers by module behavior and attributes rather than
hard-coding OLMoE into the trace format.

Future checkpoints may use one-time model-specific loading and hook wiring.
Before adding one, follow
[docs/ADDING_MODEL_TESTBED.md](docs/ADDING_MODEL_TESTBED.md) to keep run
artifacts, status, conclusions, and publication insights scoped by model.

## Quick start

Install the inference dependencies:

```bash
uv sync --extra inference
```

Install plotting dependencies:

```bash
uv sync --all-extras
```

Inspect the model before collecting data:

```bash
uv run ep-predict inspect \
  --config configs/model/olmoe-1b-7b-instruct.toml
```

Materialize the small standard workload:

```bash
uv sync --extra data --extra inference

uv run ep-predict prepare-dataset \
  --config configs/dataset/h1-standard-small.toml
```

Run H1:

```bash
uv run ep-predict collect \
  --model-config configs/model/olmoe-1b-7b-instruct.toml \
  --experiment-config configs/experiment/h1-standard-small.toml

uv run ep-predict analyze-h1 \
  --run artifacts/runs/h1-standard-small \
  --config configs/experiment/h1-standard-small.toml

uv run ep-predict plot-h1 \
  --run artifacts/runs/h1-standard-small \
  --config configs/experiment/h1-standard-small.toml
```

Reuse the trace for H2; no new inference is required:

```bash
uv run ep-predict analyze-h2 \
  --run artifacts/runs/h1-standard-small \
  --config configs/experiment/h2-standard-small.toml

uv run ep-predict plot-h2 \
  --run artifacts/runs/h1-standard-small \
  --config configs/experiment/h2-standard-small.toml
```

Run the hook-only H3 feature collection, fixed linear analysis, and figures:

```bash
uv run ep-predict collect \
  --model-config configs/model/olmoe-1b-7b-instruct.toml \
  --experiment-config configs/experiment/h3-standard-small.toml

uv run ep-predict analyze-h3 \
  --run artifacts/runs/h3-standard-small \
  --config configs/experiment/h3-standard-small.toml

uv run ep-predict plot-h3 \
  --run artifacts/runs/h3-standard-small \
  --config configs/experiment/h3-standard-small.toml
```

Reuse the same H3 artifacts for the post-hoc all-layer horizon analysis:

```bash
uv run ep-predict analyze-h3 \
  --run artifacts/runs/h3-standard-small \
  --config configs/experiment/h23-extended-horizon.toml

uv run ep-predict plot-extended-horizon \
  --run artifacts/runs/h3-standard-small \
  --config configs/experiment/h23-extended-horizon.toml
```

This trains the same fixed linear recipe for all 120 valid source-target pairs
per phase; it performs no additional inference. See
[docs/H23_EXTENDED_HORIZON_RESULTS.md](docs/H23_EXTENDED_HORIZON_RESULTS.md).

Run the hook-free H4 calibration, oracle replay, and two figures:

```bash
uv run ep-predict measure-h4 \
  --model-config configs/model/olmoe-1b-7b-instruct.toml \
  --experiment-config configs/experiment/h4-oracle.toml

uv run ep-predict analyze-h4 \
  --run artifacts/runs/h1-standard-small \
  --config configs/experiment/h4-oracle.toml

uv run ep-predict plot-h4 \
  --run artifacts/runs/h1-standard-small \
  --config configs/experiment/h4-oracle.toml

uv run ep-predict analyze-codesign-map \
  --config configs/experiment/h4-codesign-map.toml

uv run ep-predict plot-codesign-map \
  --config configs/experiment/h4-codesign-map.toml
```

H4 installs no hooks and collects no new routing data. The timing run measures
ordinary cached-token forwards and pinned host-to-device copies; the simulator
then replays the existing decode trace.
The final two commands create a post-hoc regime map combining cold-transfer
headroom with complete-route prediction coverage; it does not alter H3 or H4.

Run the analysis-only H5 requirements sweep, inverse design, existing-policy
placement, and figures:

```bash
uv run ep-predict analyze-h5 \
  --config configs/experiment/h5-first-order.toml

uv run ep-predict plot-h5 \
  --config configs/experiment/h5-first-order.toml
```

H5 performs no inference or training. It reconstructs held-out H2/H3
candidates and charges nonresident false candidate bytes.

Run the post-hoc H5 score-separation and admission-threshold analysis:

```bash
uv run ep-predict analyze-h5-admission \
  --config configs/experiment/h5-admission.toml

uv run ep-predict plot-h5-admission \
  --config configs/experiment/h5-admission.toml
```

This scores all 64 expert IDs after K=32 resident filtering and tests whether
the unchanged predictor confidence can reduce speculative traffic without
retraining.

Run the analysis-only H6 residency replay and its compact heatmap:

```bash
uv run ep-predict analyze-h6 \
  --config configs/experiment/h6-residency.toml

uv run ep-predict plot-h6 \
  --config configs/experiment/h6-residency.toml
```

H6 reuses the same 96/32 split, routes, projected features, transition tables,
and fixed linear heads. Prediction can admit only an actually demanded miss;
it never triggers candidate-only prefetch. No inference, training, model
download, or library modification occurs.

Run the assumption-driven AX1–AX3 architecture analysis and three principal
figures:

```bash
uv run ep-predict analyze-architecture \
  --config configs/experiment/ax-future-predictor-architecture.toml

uv run ep-predict plot-architecture \
  --config configs/experiment/ax-future-predictor-architecture.toml
```

This reuses the retained H1 decode trace and H4 measurement. It performs no
inference or training. Results are trace-calibrated projections with explicit
measured, trace-derived, assumed-predictor, and hypothetical-hardware labels.

Run the matched Base–Instruct C0 endpoint comparison:

```bash
uv run ep-predict collect \
  --model-config configs/model/olmoe-1b-7b-base.toml \
  --experiment-config configs/experiment/c0-olmoe-base-collect.toml

uv run ep-predict collect \
  --model-config configs/model/olmoe-1b-7b-instruct.toml \
  --experiment-config configs/experiment/c0-olmoe-instruct-collect.toml

uv run ep-predict analyze-h2 \
  --run artifacts/runs/olmoe-base-c0-paired \
  --config configs/experiment/c0-olmoe-base-h2.toml

uv run ep-predict analyze-h2 \
  --run artifacts/runs/olmoe-instruct-c0-paired \
  --config configs/experiment/c0-olmoe-instruct-h2.toml

uv run ep-predict analyze-checkpoint-trajectories \
  --config configs/experiment/c0-posttraining-trajectory.toml

uv run ep-predict plot-checkpoint-trajectories \
  --config configs/experiment/c0-posttraining-trajectory.toml
```

C0 uses raw prompt serialization, one generation token, identical input-ID
validation, and only matched prefill evidence. It does not compare divergent
free-running outputs or train a predictor.

The collector writes one compressed JSONL routing trace per request. H3 also
writes one aligned numeric NPZ shard containing compact projected router
inputs. Both are crash-safe and resumable at request granularity; no full
hidden states or Python pickle artifacts are stored.

`data/prompts/h1-pilot.jsonl` remains only an instrumentation smoke fixture.
Research evidence uses the revision-pinned standard mixture described in
[docs/DATASET_PROTOCOL.md](docs/DATASET_PROTOCOL.md).

Every major experiment ends with scripted visualization and a human review
before the next hypothesis begins. See
[docs/EXPERIMENT_SOP.md](docs/EXPERIMENT_SOP.md).

The operating rule is deliberately lean: freeze one simple decision, apply it
unchanged, then use cheap post-hoc scans of existing artifacts to discover
regimes. Complex hypothesis trees, broad ablations, uncertainty on every cell,
and confirmation workloads wait until a physical feasibility result justifies
them.

## Artifact retention

The existing `artifacts/` paths are canonical and publication-ready; there is
no duplicate results directory. Git tracks prepared workloads, run metadata,
CSV/JSON/MD analysis products, compact fitted predictors, and both PDF and PNG
figures. Only large request-level routing traces and captured
feature/hidden-state tensors remain local and disposable.

Close every major experiment with:

```bash
uv run ep-predict audit-artifacts
git add artifacts
uv run ep-predict audit-artifacts --require-tracked
```

This validates figure hashes and result-document references, then ensures every
durable artifact is staged while ignored raw data remains excluded. The full
policy is in [artifacts/README.md](artifacts/README.md) and
[docs/EXPERIMENT_SOP.md](docs/EXPERIMENT_SOP.md).

## Invariants

- Inference only: no model training or router modification.
- Hooks capture the model's actual router output.
- Every hook run validates selected IDs against top-k router logits.
- Prefill and decode are never combined in headline metrics.
- Expert IDs are always keyed by layer.
- Derived evidence under `artifacts/` is tracked in Git at its canonical path;
  only raw traces and captured hidden/feature tensors are disposable.
- No latency claim is made from a hooked run.

The long-form research agenda remains in [RESEARCH.md](RESEARCH.md).
