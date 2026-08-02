# Q1 expert-erasure quality probe result

**Decision:** `STOP` (stop signal: `True`)

## Frozen headline gate

- Primary cell: renormalize + mass_omission at m = 0.125 (realized 0.1794) over 3584 tokens.
- Mean forward-KL: **5.8141** (gate ≤ 0.05)
- Top-1 agreement: **9.18%** (gate ≥ 99%)
- Perplexity ratio: **279.8925** (gate ≤ 1.05)
- Monotone in m: **False**

| realized m | mean KL | top-1 | PPL ratio |
|---:|---:|---:|---:|
| 0.0818 | 4.9348 | 13.36% | 103.6493 |
| 0.0847 | 4.8804 | 13.50% | 99.5401 |
| 0.1794 | 5.8141 | 9.18% | 279.8925 |
| 0.3043 | 6.7328 | 6.50% | 625.3855 |
| 0.5676 | 8.8478 | 1.76% | 5154.1401 |

## Non-gating scans

### Erase positioning (renormalize), mean forward-KL by mass

| m (target) | mass-omission | random-in-route | mass-adversarial |
|---:|---:|---:|---:|
| 0.010 | 4.9348 | 5.3487 | 6.1434 |
| 0.050 | 4.8804 | 5.3998 | 6.1434 |
| 0.125 | 5.8141 | 5.9848 | 6.1434 |
| 0.250 | 6.7328 | 6.7855 | 6.8086 |
| 0.500 | 8.8478 | 8.3138 | 7.6052 |

### Correlation topology (headline m), mean forward-KL

| topology | mean KL | top-1 | PPL ratio |
|---:|---:|---:|---:|
| spread | 5.8141 | 9.18% | 279.8925 |
| layer_burst | 0.2664 | 77.59% | 1.2037 |
| consecutive_block | 0.7955 | 80.97% | 2.0638 |
| scattered | 0.4687 | 77.20% | 1.4395 |

## Evidence boundary

Everything is a measured paired forward pass on the frozen OLMoE-1B-7B-0125 base checkpoint over WikiText-2 validation (prefill scope). It measures quality, not latency or capacity. Missing mass is normalized within the routed top-8 matching AX4's primary semantics. Layer/domain sensitivity is aggregated across the 16 layers and a single domain in this pilot.
