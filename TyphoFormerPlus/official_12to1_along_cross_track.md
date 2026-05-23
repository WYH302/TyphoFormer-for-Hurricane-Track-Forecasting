# Official 12->1 Along/Cross-Track Diagnostics

Values are km on the test split. Along/cross components are computed in local displacement space against the observed final displacement vector.

| Lead | Model | FDE | Abs along | Abs cross | Signed along | Signed cross |
|---:|---|---:|---:|---:|---:|---:|
| 6h | B4 TyphoFormer++ | 34.153 | 21.193 | 22.731 | -7.802 | -3.687 |
| 6h | Best baseline (constant_velocity) | 36.869 | 22.045 | 25.105 | -8.526 | -0.903 |
| 12h | B4 TyphoFormer++ | 86.712 | 57.243 | 54.486 | -30.118 | -9.852 |
| 12h | Best baseline (cliper_ridge) | 93.245 | 59.644 | 59.727 | -26.205 | -3.849 |
| 18h | B4 TyphoFormer++ | 156.200 | 104.236 | 96.836 | -56.635 | -9.517 |
| 18h | Best baseline (informer) | 162.763 | 110.776 | 97.829 | -57.114 | -1.197 |
| 24h | B4 TyphoFormer++ | 230.528 | 160.162 | 136.730 | -108.146 | -12.122 |
| 24h | Best baseline (informer) | 243.785 | 166.593 | 147.521 | -87.032 | 1.483 |
