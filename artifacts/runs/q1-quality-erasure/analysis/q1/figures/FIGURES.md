# Q1 figure review

## Automated headline

Formal decision: `STOP` (stop signal `True`). Headline renormalize + mass-omission at m=0.125 gives mean forward-KL 5.8141, top-1 agreement 9.18%, and perplexity ratio 279.8925 over 3584 tokens.

## Human review checklist

- [ ] The renormalize-vs-null gap (intrinsic vs recoverable cost) is visible on fig 1.
- [ ] The headline m=0.125 marker and the gate thresholds are correct.
- [ ] Fig 2(a) ordering (omission low, adversarial high), 2(b) concentration, and `gate.json` agree.
- [ ] Single-domain, layer-aggregated pilot limitation is accepted.
- [ ] One next action is recorded before Q2 / AX4 hand-off.
