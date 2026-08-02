# Q2-A cross-domain tolerance result

**Decision:** `GO`

## Frozen per-domain additive-depth gate

- Cell: null + mass_omission, one expert per degraded layer at incidence 0.009 (AX4 anchor), domain text from local parquet only (no download).
- Headline run length L = 8, conditional on affected tokens.

| domain | monotone | super-linear | large-div frac (L8) | L8 KL | n affected |
|---:|:---:|:---:|---:|---:|---:|
| math | True | False (ratio 0.90) | 0.00000 | 0.0090 | 186 |
| ref_wikitext2 | True | False (ratio 0.64) | 0.00000 | 0.0145 | 279 |

### Depth sweep, conditional-on-affected KL by domain

| domain | L | affected mean KL | affected top-1 |
|---:|---:|---:|---:|
| math | 1 | 0.0013 | 97.85% |
| math | 2 | 0.0023 | 97.85% |
| math | 4 | 0.0054 | 97.85% |
| math | 8 | 0.0090 | 97.85% |
| ref_wikitext2 | 1 | 0.0035 | 96.77% |
| ref_wikitext2 | 2 | 0.0059 | 97.85% |
| ref_wikitext2 | 4 | 0.0085 | 97.49% |
| ref_wikitext2 | 8 | 0.0145 | 94.98% |

## Evidence boundary

Measured forward passes on the frozen OLMoE-1B-7B-0125 base checkpoint, null-drop only. The reference domain is WikiText-2 (Q1-B in-family); math is gsm8k word problems from local parquet. Code was not locally materialized, so it is not in this arm. Quality only; no latency or capacity claim.
