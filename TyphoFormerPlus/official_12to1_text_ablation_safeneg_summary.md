# Official 12->1 Safe-Negative Text-Template Ablation

Values are DeltaR km. Neural rows are mean +/- std over seeds 42/123/2024.

| Row | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| B2/B3 without text-template | 39.185 +/- 0.707 | 101.201 +/- 4.731 | 173.365 +/- 5.238 | 265.277 +/- 8.332 |
| B3 with hash text-template | 39.462 +/- 0.665 | 101.775 +/- 1.686 | 170.988 +/- 4.322 | 260.721 +/- 6.654 |
| B4 without text-template | 33.099 +/- 0.669 | 84.987 +/- 0.605 | 155.363 +/- 1.802 | 240.685 +/- 6.107 |
| B4 full with hash text-template | 34.153 +/- 0.507 | 86.712 +/- 1.081 | 156.200 +/- 4.628 | 230.528 +/- 1.302 |
| B4 full + val CV calibration | 34.295 +/- 0.458 | 86.160 +/- 0.825 | 154.985 +/- 4.902 | 228.959 +/- 2.043 |
