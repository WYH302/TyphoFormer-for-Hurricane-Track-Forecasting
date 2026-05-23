# Official 12->1 Simple Residual Baselines

Values are test DeltaR km under the strict-HURDAT2 safe-negative protocol. Ridge rows select alpha on validation; neural rows select epoch on validation.

| Model | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| Direct ridge | 37.983 | 94.706 | 166.554 | 246.246 |
| Ridge residual-on-CV | 35.568 | 90.183 | 159.166 | 236.474 |
| MLP residual-on-CV | 36.351 +/- 0.161 | 92.662 +/- 2.016 | 160.603 +/- 0.310 | 236.702 +/- 1.054 |
| Tiny dual MLP, no Transformer | 37.242 +/- 0.323 | 93.738 +/- 1.148 | 164.018 +/- 1.654 | 241.679 +/- 3.499 |
