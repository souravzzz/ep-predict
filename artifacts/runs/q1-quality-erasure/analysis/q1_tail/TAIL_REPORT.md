# Q1 tail-event (AX4-faithful) erasure probe result

**Decision:** `STOP` (stop signal: `False`)

## Frozen tail headline gate

- Primary cell: renormalize + mass_omission, one expert in 1 consecutive layer(s), at incidence 0.009 (AX4 anchor), conditional on the 132/15872 affected tokens (realized incidence 0.0083).
- Conditional-on-affected mean forward-KL: **0.1277** (gate ≤ 0.05)
- Conditional-on-affected top-1 agreement: **86.36%** (gate ≥ 99%)
- Conditional-on-affected perplexity ratio: **1.1296** (gate ≤ 1.05)
- Overall diluted mean KL grows monotonically in incidence: **True**

### Incidence sweep (one layer), conditional-on-affected vs diluted

| incidence | affected mean KL | affected top-1 | diluted mean KL |
|---:|---:|---:|---:|
| 0.0020 | 0.0811 | 89.66% | 0.0009 |
| 0.0050 | 0.1464 | 80.77% | 0.0027 |
| 0.0090 | 0.1277 | 86.36% | 0.0032 |
| 0.0200 | 0.1278 | 84.94% | 0.0088 |
| 0.0500 | 0.1302 | 86.89% | 0.0145 |
| 0.1000 | 0.1168 | 84.38% | 0.0255 |
| 0.2000 | 0.1416 | 84.13% | 0.0489 |
| 0.3000 | 0.1501 | 83.53% | 0.0689 |

### Run-length compounding (AX4 anchor incidence), affected KL

| run length | affected mean KL | affected top-1 |
|---:|---:|---:|
| 1 | 0.1277 | 86.36% |
| 2 | 0.1540 | 79.73% |
| 4 | 0.3171 | 73.08% |
| 8 | 0.8073 | 64.49% |

## Evidence boundary

Measured paired forward passes on the frozen OLMoE-1B-7B-0125 base checkpoint over WikiText-2 validation (prefill scope). Tail mode erases an exact expert count for a bounded fraction of tokens (swept by incidence) in a bounded run of consecutive layers, matching AX4's ~0.9-1% degraded-wave regime. Conditional metrics are over only the tokens that suffered erasure; diluted metrics include the untouched majority.
