# Official 12->1 Storm-Level Bootstrap vs Best Non-Proposed Baseline

B4 - best baseline paired DeltaR in km. Negative means B4 is better. B4 and neural baselines average seeds first, then bootstrap test storms.

| Lead | Storms | Best baseline | Baseline storm mean | B4 storm mean | B4-baseline delta | 95% CI | B4 better storms |
|---:|---:|---|---:|---:|---:|---:|---:|
| 6h | 41 | constant_velocity | 39.647 | 35.698 | -3.949 | [-7.031, -1.434] | 75.6% |
| 12h | 36 | cliper_ridge | 103.399 | 97.596 | -5.802 | [-9.285, -2.216] | 72.2% |
| 18h | 35 | informer | 171.967 | 170.528 | -1.439 | [-13.107, 13.549] | 57.1% |
| 24h | 30 | informer | 255.258 | 249.723 | -5.535 | [-24.496, 16.881] | 63.3% |
