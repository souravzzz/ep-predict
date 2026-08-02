# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`ep-predict` is a researcher's instrumentation toolkit that uses PyTorch forward
hooks to capture MoE expert-routing traces from `allenai/OLMoE-1B-7B-0125-Instruct`
and runs offline simulations to test whether expert demand is skewed,
predictable, and useful for hierarchical memory placement. It follows a
["tracer bullet"](docs/EXPERIMENT_STRATEGY.md) approach: advance one narrow
vertical slice to a result, keep a human in the loop, and explain findings
before moving on. The upstream model and Transformers source are never modified.

The project advances one research gate at a time. Completed gates include the
hypotheses H1–H6, the paired-checkpoint control C0, and the assumption-driven
architecture analyses AX1–AX4. **Before adding a new experiment, read
`docs/EXPERIMENT_SOP.md` (the required operating loop) and
`docs/NEXT_EXPERIMENTS.md` (the active queue) and follow an existing protocol
(e.g. `docs/H5_PROTOCOL.md`) as a template.** Current status lives in
`STATUS.md`; `EXPERIMENT_LOG.md` records formal results and human
interpretation; `docs/FOUNDATIONAL_INSIGHTS.md` holds durable lessons.
`RESEARCH.md` is the full research narrative.

## Commands

Environment is managed with `uv` (Python 3.12). The package is an editable
src-layout install with no runtime dependencies; optional extras enable the
toolchain tiers:

```bash
uv sync --extra inference          # model loading + hook tracing (torch, transformers, accelerate)
uv sync --all-extras               # add data (datasets) and viz (matplotlib, numpy)
```

All work is driven through the single `ep-predict` CLI (`ep_predict.cli:main`):

```bash
uv run ep-predict <subcommand> --config/-model-config/-experiment-config ...
```

Subcommands follow the pipeline `collect → analyze → plot` (only some gates use
all three; analysis commands are offline and reuse existing traces). The core
flow for a new gate (see README for the full worked examples):

```bash
uv run ep-predict collect --model-config configs/model/olmoe-1b-7b-instruct.toml \
  --experiment-config configs/experiment/<gate>.toml
uv run ep-predict analyze-<gate> --run artifacts/runs/<run-id> --config configs/experiment/<gate>.toml
uv run ep-predict plot-<gate> --run artifacts/runs/<run-id> --config configs/experiment/<gate>.toml
uv run ep-predict inspect --config configs/model/olmoe-1b-7b-instruct.toml   # sanity-check router discovery
uv run ep-predict prepare-dataset --config configs/dataset/h1-standard-small.toml
```

`collect` accepts `--limit N` for a smoke test. Pass `--limit 2` and verify
trace/schema/router integrity before scaling (SOP step 2).

Run tests with pytest. The suite is plain `pytest` (no config section); a full
test run needs the inference extra because `test_hooks.py` and `test_storage.py`
import `torch`:

```bash
uv sync --all-extras
uv run pytest -q
uv run pytest tests/test_h5_analysis.py -q          # a single file
uv run pytest tests/test_h5_analysis.py -k gate -q  # a single test / keyword
```

After every major analysis + visualization, run the [artifact-retention
closeout](docs/EXPERIMENT_SOP.md#8-artifact-retention-closeout):

```bash
uv run ep-predict audit-artifacts
git add artifacts
uv run ep-predict audit-artifacts --require-tracked   # fails if any durable product was missed
```

## Architecture

Everything hangs off the thin `cli.py`, which parses subcommands and dispatches
to functions in three mirrored per-gate module groups. Adding a gate means
adding one module to each of `analysis/` and `visualize/` (and a config) plus a
subcommand in `cli.py` — there is no shared "engine"; each gate is self-contained.

- **`ep_predict/tracing/`** — the instrumentation core. `hooks.py` discovers
  routers **generically by name and module attributes** (`.mlp.gate`, `num_experts`,
  `top_k`), not by hardcoding OLMoE, via `RouterTracer`/`RouterSpec`. It also
  provides `RouterInputProjector`, a fixed random-sign projection of router
  inputs used as compact H3 features. `storage.py` writes one atomic gzip-JSONL
  member per request (`RequestTraceStore`) plus aligned NPZ feature shards
  (`RequestFeatureStore`) — both crash-safe and resumable at request
  granularity. `schema.py` holds the trace record types and
  `TRACE_SCHEMA_VERSION`.
- **`ep_predict/analysis/`** — offline replayers/simulators (one per gate:
  `h1`, `h2`, `h3`, `h4`, `h5`, `h6`, plus `checkpoint`, `admission`, `codesign`,
  `architecture`, `degradation`). Each produces machine-readable CSV tables and
  a JSON `gate` verdict, then the SOP applies the frozen decision gate. The
  functions take a run directory + TOML config.
- **`ep_predict/visualize/`** — mirror modules that read the analysis tables and
  emit publication-style PDF + PNG figures plus a manifest that SHA-256-hashes
  their inputs. `hardware/h4.py` is the hook-free decode/timing measurer used by
  `measure-h4`. `data/standard.py` materializes the revision-pinned workload.

### Configuration and provenance

Configs are TOML, separated by role under `configs/{model,dataset,experiment}/`:
`model/*` pins `model_id`/`revision`/`tokenizer_revision`/`dtype` (`machine` →
use a pinned revision; results are scoped to it); `dataset/*` defines the source
mixture; `experiment/*` sets run id, output paths, seeds, `[analysis]` params,
and the `[decision_gate]` stop/go rule. `config.py` hashes configs into a
`config_fingerprint` so resumed artifacts can be checked against the frozen
definition (see `run_definition.json` under each run).

### Artifacts layout and provenance rules

The native `artifacts/` tree is the single source of truth for publication
evidence and is Git-tracked (see `.gitattributes` for the CSV CRLF/whitespace
handling):

```
artifacts/runs/<run-id>/
  run_definition.json, run_manifest.json, model_report.json   # durable provenance
  analysis/<gate>/                                             # tables, figures, REPORT.md, manifests
  trace/ features/ hidden_states/ activations/                # DISPOSABLE, gitignored
```

Raw replay inputs under `trace/`, `features/`, `hidden_states/`, `activations/`
are disposable and **ignored by Git** (per `.gitignore`). Compact derived
tables, manifests, gate JSON, reports, fitted NPZ, and figures are durable and
must be committed. Never repair a raw trace in place after analysis, never copy
figures into a second results tree, and don't let a result document depend on an
ignored raw file when a compact table suffices — `audit-artifacts` enforces
these invariants (hash checks, stale-path detection, approved ignore dirs,
large-file warnings).

### Research-workflow conventions (from docs/EXPERIMENT_SOP.md)

- Freeze one decisive question, primary scope, and stop/go rule per gate; keep
  cheap post-hoc scans non-gating.
- Report conditional gain over the model's own marginal baseline
  (raw accuracy confounds predictability with skew).
- For capacity/offload studies, compare against the same reactive
  hierarchy and never call offload faster than identical all-resident execution.
- Pair checkpoint comparisons require exact input-token-ID equality.
- Store tables before figures; human reviews figures before advancing;
  handle off via `EXPERIMENT_LOG.md` / `STATUS.md`.

## Testbed caveat

OLMoE uses top-8 routing (each token requests 8 of 64 experts per layer, 16
layers), which is intentionally demanding. Once a decisive gate is understood,
replicate only the decisive traces on a top-2/top-4 model before generalizing.
See `docs/TESTBED.md` for the model choice rationale. `docs/ADDING_MODEL_TESTBED.md`
governs adding any second checkpoint — this is a deliberate qualification task,
not something to attempt casually, and results must stay model-scoped.
