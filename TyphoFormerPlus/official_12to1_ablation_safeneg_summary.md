# Official 12->1 Safe-Negative Ablation

Protocol: official HURDAT2 strict-6h lead-specific clean 12->1, seed 42.

| Variant | 12h DeltaR km | 24h DeltaR km |
|---|---:|---:|
| B3 leak-free TyphoFormer | 103.547 | 266.448 |
| B3 + dual CV-residual head | 86.488 | 239.978 |
| B3 + positive analog only | 92.977 | 249.365 |
| B3 + alignment/rank only | 104.104 | 283.239 |
| B3 + analog + alignment/rank | 109.549 | 265.998 |
| B4 full TyphoFormer++ | 91.640 | 230.797 |
