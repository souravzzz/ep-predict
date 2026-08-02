# Q2-B decode compounding result

**Decision:** `CANDIDATE_ROBUSTNESS_TARGET`

## Frozen decode-coherence gate

- Clean vs erased autoregressive continuation from a shared [1, 8] prefix over 64 steps, AX4 tail incidence 0.009, null-drop.
- GO requires high token agreement, bounded mean step KL, and no runaway late-window divergence at both L=1 and L=8.

| L | token agreement | mean step KL | final cum KL | late/early | runaway | pass |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | 62.5% | 3.70622 | 3.70622 | 18620.77 | True | False |
| 8 | 62.5% | 4.54665 | 4.54665 | 14019.51 | True | False |

## Evidence boundary

Measured two-stream generation on the frozen base checkpoint over WikiText-2, null-drop only, AX4 anchor incidence. Step KL is the clean-vs-erased next-token KL at each generation step; the streams advance by their own argmax so a changed token propagates. This detects compounding the prefill cannot see; it is not a throughput or latency claim.
