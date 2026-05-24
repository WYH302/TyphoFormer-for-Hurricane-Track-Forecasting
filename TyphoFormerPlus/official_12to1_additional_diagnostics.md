# Additional Official 12->1 Diagnostics

## Common 24h-Clean Test Subset

| Lead | W/S | Best non-B4 | Best mean | B3 | B4 | B4 gain vs best |
|---:|---:|---|---:|---:|---:|---:|
| 6h | 630/30 | constant_velocity | 35.943 | 37.680 | 33.441 | 7.0% |
| 12h | 630/30 | gru | 90.427 | 98.343 | 83.788 | 7.3% |
| 18h | 630/30 | informer | 160.797 | 168.106 | 153.320 | 4.6% |
| 24h | 630/30 | informer | 243.785 | 260.721 | 230.528 | 5.4% |

## 24h Turning Subgroups

| Turn group | W/S | Mean turn deg | CV | B3 | B4 | B4-CV | B4-B3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 210/26 | 2.1 | 214.543 | 238.620 | 199.891 | -14.652 | -38.729 |
| medium | 210/27 | 9.5 | 251.671 | 255.506 | 231.792 | -19.879 | -23.714 |
| high | 210/24 | 32.1 | 302.946 | 288.036 | 259.901 | -43.045 | -28.135 |

## Runtime

| Model | Runs | Median epochs | Median train min/run |
|---|---:|---:|---:|
| GRU | 12/12 | 32 | 1.3 |
| LSTM | 12/12 | 30 | 1.2 |
| Informer-style | 12/12 | 24 | 0.9 |
| Autoformer-style | 12/12 | 37 | 1.4 |
| TSMixer | 12/12 | 29 | 1.2 |
| B2 Numeric Transformer | 12/12 | 26 | 5.1 |
| B3 TyphoFormer-style | 12/12 | 26 | 5.8 |
| B4 TyphoFormer++ | 12/12 | 23 | 8.7 |
| Numeric Transformer + CV-residual | 12/12 | 24 | 5.3 |

## B4 Inference Timing

| Lead | Windows | Seconds | Windows/s |
|---:|---:|---:|---:|
| 6h | 759 | 0.597 | 1271.1 |
| 12h | 710 | 0.475 | 1496.1 |
| 18h | 668 | 0.499 | 1339.0 |
| 24h | 630 | 0.472 | 1334.2 |
