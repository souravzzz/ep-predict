# Q2-C cliff mapping result

**Reading:** `WITH_MARGIN`

## Where null-drop stops being free

- Free band: conditional-on-affected mean KL ≤ 0.02 nats.
- AX4 nominal cell (incidence 0.009, L=8, 1 expert/layer): affected KL = 0.010229129659672898, free = **True**.

| axis | first crossing | margin over AX4 nominal |
|---|---:|---:|
| incidence | none within swept range | ∞ |
| run length | none within swept range | ∞ |
| experts/layer | 2.0 (KL 0.028) | 2.0 |

### Cliff surface cells (conditional-on-affected)

| axis | incidence | run length | experts/layer | affected KL | top-1 |
|---:|---:|---:|---:|---:|---:|
| experts_per_layer | 0.0090 | 8 | 2 | 0.0276 | 92.74% |
| experts_per_layer | 0.0090 | 8 | 4 | 0.1437 | 77.42% |
| incidence | 0.0090 | 8 | 1 | 0.0102 | 95.97% |
| incidence | 0.0200 | 8 | 1 | 0.0132 | 95.16% |
| incidence | 0.0500 | 8 | 1 | 0.0131 | 95.56% |
| incidence | 0.1000 | 8 | 1 | 0.0147 | 94.30% |
| incidence | 0.3000 | 8 | 1 | 0.0157 | 94.49% |
| incidence | 0.5000 | 8 | 1 | 0.0175 | 94.06% |
| run_length | 0.0090 | 12 | 1 | 0.0148 | 94.35% |
| run_length | 0.0090 | 16 | 1 | 0.0185 | 94.35% |

## Interpretation

the free band holds at the AX4 nominal cells; cliffs sit beyond them, so the measured contract stands with margin.

## Evidence boundary

Measured forward passes on the frozen base checkpoint, null-drop only, WikiText-2. Pushes erasure past AX4's nominal bound (incidence, run length, experts per layer) to locate the cliff. Quality only; AX4's deadline regime remains a separate contract.
