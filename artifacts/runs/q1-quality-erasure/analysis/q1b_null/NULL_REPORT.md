# Q1-B null-drop mechanism probe result

**Decision:** `GO` (kill signal: `False`)

## Frozen null-drop gate (primary: depth additivity)

- Cell: null + mass_omission, one expert per degraded layer, same affected-token sample at incidence 0.009 (AX4 anchor), conditional on affected tokens.
- Run lengths swept: [1, 2, 4, 8], worst case L = 8.
- Monotone in L: **True**
- Super-linear marginal blow-up: **False** (last/first per-layer marginal ratio 1.20, gate ≤ 3.0)
- Large divergence at L=8: **False** (fraction 0.00000, gate ≤ 0.01)

### Depth sweep, conditional-on-affected

| L | affected mean KL | affected top-1 | affected PPL | large-div frac | n affected |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0041 | 97.85% | 1.0026 | 0.00000 | 186 |
| 2 | 0.0053 | 96.24% | 0.9971 | 0.00000 | 186 |
| 4 | 0.0073 | 98.39% | 1.0066 | 0.00000 | 186 |
| 8 | 0.0133 | 93.55% | 0.9901 | 0.00000 | 186 |

### Per-layer marginal cost (per extra degraded layer)

| range | delta L | per-layer marginal KL |
|---:|---:|---:|
| 1→2 | 1 | 0.00126 |
| 2→4 | 2 | 0.00097 |
| 4→8 | 4 | 0.00151 |

## Non-gating scans

### Layer-order sensitivity (one expert, one layer)

| layer | affected mean KL | affected top-1 |
|---:|---:|---:|
| 0 | 0.0027 | 98.06% |
| 1 | 0.0027 | 97.42% |
| 4 | 0.0026 | 96.77% |
| 12 | 0.0026 | 99.35% |
| 10 | 0.0026 | 96.77% |
| 7 | 0.0025 | 98.71% |
| 3 | 0.0025 | 98.06% |
| 15 | 0.0024 | 98.71% |
| 5 | 0.0024 | 96.13% |
| 13 | 0.0023 | 98.06% |
| 9 | 0.0022 | 97.42% |
| 8 | 0.0022 | 98.71% |
| 6 | 0.0021 | 99.35% |
| 11 | 0.0021 | 95.48% |
| 2 | 0.0020 | 95.48% |
| 14 | 0.0017 | 96.13% |

### Consecutive vs distant (same affected sample)

| L | gap | affected mean KL | affected top-1 |
|---:|---:|---:|---:|
| 2 | 1 | 0.0043 | 96.77% |
| 2 | 7 | 0.0039 | 95.97% |
| 4 | 1 | 0.0059 | 95.97% |
| 4 | 3 | 0.0078 | 92.74% |

### Cross-token leak (downstream offsets from affected tokens)

| bucket | mean forward KL | count |
|---:|---:|---:|
| downstream_1 | 0.00219 | 124 |
| downstream_2 | 0.00127 | 124 |
| downstream_3 | 0.00197 | 124 |
| downstream_4 | 0.00186 | 124 |
| downstream_5 | 0.00213 | 124 |
| downstream_6 | 0.00144 | 124 |
| downstream_7 | 0.00229 | 124 |
| downstream_8 | 0.00274 | 93 |
| far_control | 0.00171 | 13764 |

## Evidence boundary

All metrics are measured paired forward passes on the frozen OLMoE-1B-7B-0125 base checkpoint over WikiText-2 validation (prefill scope), null-drop only (renormalize dropped as a strategy). Each affected token drops exactly one expert per degraded layer with no renormalization; the same affected-token sample is reused within each scan family for clean within-token comparisons. Conditional metrics average only tokens that suffered erasure. This measures quality, not latency or capacity, and is scoped to this single revision.
