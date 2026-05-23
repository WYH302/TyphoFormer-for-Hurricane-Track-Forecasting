# TyphoFormer++ Strict-6h Experiments

Environment: `D:\App1\environment\envs\ai_base\python.exe`, PyTorch 2.1.2+cu118, RTX 3090.

Bundled-sample formal data: `data_pp_6h`, leak-free 12 -> 12 with strict 6-hour HURDAT2 synoptic records.

Official-protocol data: `data_official_pp_12to4`, official NHC Atlantic HURDAT2 1851-2024 download, filtered to 2004-2024 with train-period storms from 2004-2021 and test storms from 2022-2024.

Important correction: the older `data_pp` runs used raw adjacent records and therefore mixed in HURDAT2 special intermediate times. Those results are useful only as directional debugging, not as paper-grade 24h/48h/72h comparisons.

## Data Audit

Generation command:

```powershell
D:\App1\environment\envs\ai_base\python.exe prepare_typhoformerpp_data.py --output-dir data_pp_6h --standard-times-only --strict-6h --interval-hours 6 --force
```

Audit:

| Item | Value |
|---|---:|
| Raw rows | 3004 |
| Rows after standard-time filter | 2906 |
| Dropped non-standard rows | 98 |
| Train samples | 759 |
| Val samples | 86 |
| Test samples | 177 |
| Bad dt windows found in saved samples | 0 |
| Bad 24h/48h/72h lead checks | 0 |
| Train same-storm analog violations | 0 |
| Val/test non-train analog references | 0 |

All saved windows use only `0000/0600/1200/1800`, every adjacent step is exactly 6h, and the 4th/8th/12th future points are exactly 24h/48h/72h after the last observation.

## Main Strict-6h Results

Seeds: 42, 123, 2024. Values are test km, mean +/- std.

| Model | ADE | FDE | 24h | 48h | 72h |
|---|---:|---:|---:|---:|---:|
| Persistence | 597.30 | 1044.41 | 394.02 | 738.50 | 1044.41 |
| Constant Velocity | 451.01 | 933.25 | 224.42 | 565.91 | 933.25 |
| B2 Numeric Transformer | 441.99 +/- 22.85 | 828.85 +/- 70.84 | 266.68 +/- 5.65 | 544.50 +/- 28.63 | 828.85 +/- 70.84 |
| B3 TyphoFormer leak-free | 474.05 +/- 8.63 | 863.10 +/- 7.06 | 292.42 +/- 7.46 | 584.22 +/- 10.79 | 863.10 +/- 7.06 |
| B4 TyphoFormer++ dual-head raw | 394.84 +/- 26.28 | 751.61 +/- 34.80 | 228.60 +/- 17.66 | 509.29 +/- 24.18 | 751.61 +/- 34.80 |
| B4 TyphoFormer++ dual-head + val CV calibration | 376.25 +/- 11.32 | 723.89 +/- 23.16 | 211.43 +/- 1.49 | 477.98 +/- 7.27 | 723.89 +/- 23.16 |

The calibrated row uses validation-only, lead-time-wise global blending:

```text
Y_calibrated(k) = alpha_k * Y_model(k) + (1 - alpha_k) * Y_CV(k)
```

The `alpha_k` values are selected on the validation split only, then frozen for test evaluation. This should be reported as a calibrated deterministic variant, not as the raw model.

## Per-Seed B4 Calibrated

| Seed | ADE | FDE | 24h | 48h | 72h |
|---:|---:|---:|---:|---:|---:|
| 42 | 370.02 | 718.22 | 210.33 | 471.93 | 718.22 |
| 123 | 389.32 | 749.36 | 213.13 | 486.05 | 749.36 |
| 2024 | 369.43 | 704.09 | 210.83 | 475.95 | 704.09 |

## Original-Horizon Alignment: Strict-6h 12 -> 4

Purpose: align the forecast horizon with the original TyphoFormer-style 6h/12h/18h/24h reporting, while keeping the same strict 6h, leak-free split protocol.

Data: `data_pp_6h_12to4`, input 12 steps = past 72h, output 4 steps = future 24h.

Audit:

| Item | Value |
|---|---:|
| Raw rows | 3004 |
| Rows after standard-time filter | 2906 |
| Dropped non-standard rows | 98 |
| Train samples | 1130 |
| Val samples | 149 |
| Test samples | 258 |
| Bad dt windows found in saved samples | 0 |
| Bad 6h/12h/18h/24h lead checks | 0 |
| Split storm overlaps | 0 |
| Train same-storm analog violations | 0 |
| Val/test non-train analog references | 0 |

Seeds: 42, 123, 2024. Values are test km, mean +/- std.

| Model | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| Persistence | 111.79 | 223.02 | 332.34 | 439.74 |
| Constant Velocity | 35.29 | 93.96 | 169.40 | 256.87 |
| B2 Numeric Transformer | 45.61 +/- 2.31 | 103.58 +/- 3.70 | 170.55 +/- 9.05 | 246.56 +/- 13.50 |
| B3 TyphoFormer leak-free | 50.42 +/- 4.03 | 109.32 +/- 7.07 | 180.48 +/- 13.58 | 260.97 +/- 13.18 |
| B4 TyphoFormer++ dual-head raw | 34.26 +/- 1.65 | 83.80 +/- 3.65 | 145.86 +/- 6.54 | 218.65 +/- 15.53 |
| B4 TyphoFormer++ dual-head + val CV calibration | 33.76 +/- 1.45 | 83.49 +/- 3.53 | 144.74 +/- 7.03 | 216.28 +/- 17.48 |

Conclusion: under the original-horizon-aligned strict-6h 12 -> 4 protocol, TyphoFormer++ raw dual-head beats the leak-free TyphoFormer baseline B3 at every reported lead time. The validation-only CV calibration further improves all four lead times and, on the three-seed mean, also beats pure Constant Velocity at 6h/12h/18h/24h.

## Official NHC HURDAT2 Protocol: Strict-6h 12 -> 4

Purpose: rerun the main paper-facing comparison on the official NHC Atlantic HURDAT2 source instead of the bundled 2020-2024 CSV sample.

Data source:

- `data_official/hurdat2_atl_1851_2024.txt`, downloaded from the NHC HURDAT2 archive.
- `data_official/hurdat2_format_atl.pdf`, official format reference.

Generation command:

```powershell
D:\App1\environment\envs\ai_base\python.exe prepare_official_hurdat2_pp_data.py --hurdat2 data_official\hurdat2_atl_1851_2024.txt --output-dir data_official_pp_12to4 --input-len 12 --pred-len 4 --train-years 2004-2021 --test-years 2022-2024 --val-ratio 0.10 --split-seed 42 --force
```

Protocol:

| Item | Value |
|---|---:|
| Source rows | 55230 |
| Source storms | 1991 |
| Protocol years | 2004-2024 |
| Rows in protocol years | 11510 |
| Non-standard rows in protocol years | 350 |
| Train storms / samples | 188 / 3349 |
| Val storms / samples | 20 / 285 |
| Test storms / samples | 30 / 630 |
| Input | 12 strict 6h points = past 72h |
| Output | 4 strict 6h points = future 24h |
| Lead times | 6h / 12h / 18h / 24h |
| Stats fit split | train only |
| Analog candidate split | train only |

Audit:

| Check | Value |
|---|---:|
| Bad saved time count | 0 |
| Bad adjacent dt count | 0 |
| Bad lead count | 0 |
| Train/val/test storm overlap | 0 |
| Test storms only from 2022-2024 | true |
| Train same-storm analog violations | 0 |
| Val/test non-train analog references | 0 |
| Calibration split | val only |

Text embeddings in this official run are reproducible 384-dimensional hashing-vectorizer embeddings of rule-based storm descriptions. They are not Sentence-BERT or GPT text embeddings.

### Official Full Test: 2022-2024

Seeds for neural models: 42, 123, 2024. Values are test mean +/- std for neural models. `CLIPER-style ridge` is a train-only ridge-regression CLIPER-style baseline, not the official NHC CLIPER implementation.

MAE:

| Model | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| Persistence | 0.780 | 1.553 | 2.312 | 3.050 |
| Constant Velocity | 0.233 | 0.613 | 1.113 | 1.713 |
| CLIPER-style ridge | 0.237 | 0.598 | 1.073 | 1.632 |
| B2 Numeric Transformer | 0.303 +/- 0.013 | 0.674 +/- 0.015 | 1.129 +/- 0.027 | 1.661 +/- 0.014 |
| B3 TyphoFormer leak-free | 0.307 +/- 0.016 | 0.671 +/- 0.030 | 1.122 +/- 0.044 | 1.659 +/- 0.062 |
| B4 TyphoFormer++ raw | 0.196 +/- 0.003 | 0.470 +/- 0.009 | 0.832 +/- 0.026 | 1.273 +/- 0.048 |
| B4 TyphoFormer++ + val CV calibration | 0.196 +/- 0.005 | 0.466 +/- 0.011 | 0.826 +/- 0.027 | 1.259 +/- 0.044 |

DeltaR km:

| Model | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| Persistence | 119.86 | 238.65 | 355.38 | 468.84 |
| Constant Velocity | 35.94 | 92.65 | 167.63 | 256.39 |
| CLIPER-style ridge | 36.67 | 91.59 | 163.32 | 247.66 |
| B2 Numeric Transformer | 46.880 +/- 2.133 | 104.154 +/- 2.622 | 172.895 +/- 4.502 | 253.047 +/- 2.872 |
| B3 TyphoFormer leak-free | 47.645 +/- 2.119 | 103.992 +/- 3.565 | 172.517 +/- 4.821 | 253.431 +/- 6.937 |
| B4 TyphoFormer++ raw | 30.364 +/- 0.352 | 72.234 +/- 1.778 | 126.960 +/- 4.426 | 193.733 +/- 8.097 |
| B4 TyphoFormer++ + val CV calibration | 30.258 +/- 0.603 | 71.302 +/- 1.639 | 125.232 +/- 3.779 | 190.351 +/- 6.485 |

### Official 2024 Subset

DeltaR km:

| Model | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| Persistence | 146.52 | 299.35 | 458.30 | 623.18 |
| Constant Velocity | 32.63 | 83.24 | 153.97 | 243.45 |
| CLIPER-style ridge | 34.34 | 83.37 | 148.25 | 229.03 |
| B2 Numeric Transformer | 47.406 +/- 2.829 | 110.421 +/- 1.880 | 178.283 +/- 5.534 | 261.474 +/- 8.751 |
| B3 TyphoFormer leak-free | 44.283 +/- 4.624 | 97.875 +/- 11.785 | 160.202 +/- 14.079 | 240.958 +/- 16.632 |
| B4 TyphoFormer++ raw | 29.428 +/- 0.911 | 72.467 +/- 3.191 | 126.752 +/- 5.927 | 194.023 +/- 6.899 |
| B4 TyphoFormer++ + val CV calibration | 28.939 +/- 0.978 | 70.227 +/- 2.702 | 123.716 +/- 5.837 | 189.774 +/- 7.311 |

### Historical Joint 12 -> 4 Official Table 1 Boundary

The official TyphoFormer paper Table 1 reports TyphoFormer DeltaR (All) of 31.539 / 38.084 / 42.435 / 49.562 km at 6h/12h/18h/24h, and TyphoFormer DeltaR (2024) of 31.274 / 37.934 / 42.973 / 50.881 km.

On the historical joint 12 -> 4 official-data run above, TyphoFormer++ raw is 30.364 / 72.234 / 126.960 / 193.733 km on 2022-2024 and 29.428 / 72.467 / 126.752 / 194.023 km on the 2024 subset. This run is now supplementary because the paper uses the safe-negative lead-specific 12 -> 1 protocol as the main evidence. Therefore:

- Correct claim: TyphoFormer++ consistently outperforms our leak-free TyphoFormer reproduction under a strict official-HURDAT2 6-hour, 12-to-4 protocol aligned with the original 6h-24h horizon.
- Correct claim: TyphoFormer++ also beats Constant Velocity and the CLIPER-style ridge baseline on this official-data protocol.
- Do not claim yet: TyphoFormer++ outperforms the official TyphoFormer Table 1 results. It only slightly beats the official Table 1 value at 6h; it is still much worse at 12h/18h/24h.

The remaining discrepancy is large: even our official-data CV and CLIPER-style ridge baselines are around 247-256 km at 24h, while the paper's CLIPER row reports 58.268 km at 24h. This strongly suggests that the paper's exact preprocessing/evaluation protocol is still not fully reconstructed from the public repository and paper table. Treat the official Table 1 comparison as unresolved until that protocol gap is explained.

## Official NHC HURDAT2 Lead-Specific Clean 12 -> 1

Purpose: test the closest local reconstruction of the public repo's default next-step style without decoder leakage. Four independent single-horizon datasets/models are trained:

| Task | Input | Target |
|---|---|---|
| 6h | past 12 strict 6h points | future step 1 |
| 12h | past 12 strict 6h points | future step 2 |
| 18h | past 12 strict 6h points | future step 3 |
| 24h | past 12 strict 6h points | future step 4 |

The decoder/base seed is always the last observed point. No target future point is used as decoder seed.

Implementation notes:

- Data directories: `data_official_pp_12to1_lead1` through `data_official_pp_12to1_lead4`.
- Checkpoints: `checkpoints_official_12to1_leadspecific`.
- Summary: `official_leadspecific_12to1_summary.json`.
- Script: `run_official_leadspecific_12to1_experiments.ps1`.
- CV and CV-residual use the true lead multiplier: 1/2/3/4 for 6h/12h/18h/24h.
- Metrics are labeled by actual lead hour, so a 24h single-output model reports `ERR24`, not `ERR6`.
- Each lead uses all windows where that target lead is available, so sample counts differ by lead.

Data audit:

| Lead | Train samples | Val samples | Test samples | Bad time | Bad dt | Bad lead | Analog leak |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6h | 4050 | 354 | 759 | 0 | 0 | 0 | 0 |
| 12h | 3797 | 329 | 710 | 0 | 0 | 0 | 0 |
| 18h | 3565 | 306 | 668 | 0 | 0 | 0 | 0 |
| 24h | 3349 | 285 | 630 | 0 | 0 | 0 | 0 |

DeltaR km, 2022-2024 test:

| Model | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| Persistence | 126.68 | 248.17 | 362.02 | 468.84 |
| Constant Velocity | 36.87 | 95.22 | 170.27 | 256.39 |
| CLIPER-style ridge | 37.28 | 93.24 | 165.42 | 247.66 |
| B2 Numeric Transformer | 39.185 +/- 0.707 | 101.201 +/- 4.731 | 173.365 +/- 5.238 | 265.277 +/- 8.332 |
| B3 TyphoFormer leak-free | 39.462 +/- 0.665 | 101.775 +/- 1.686 | 170.988 +/- 4.322 | 260.721 +/- 6.654 |
| B4 TyphoFormer++ raw | 30.968 +/- 0.832 | 71.968 +/- 0.567 | 123.232 +/- 1.673 | 185.461 +/- 8.025 |
| B4 TyphoFormer++ + val CV calibration | 30.036 +/- 0.492 | 71.968 +/- 0.567 | 123.232 +/- 1.673 | 185.461 +/- 8.025 |

DeltaR km, 2024 subset:

| Model | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| Persistence | 158.38 | 315.43 | 467.04 | 623.18 |
| Constant Velocity | 35.20 | 90.91 | 162.28 | 243.45 |
| CLIPER-style ridge | 34.93 | 84.94 | 148.64 | 229.03 |
| B2 Numeric Transformer | 39.690 +/- 0.936 | 99.091 +/- 2.155 | 169.686 +/- 14.170 | 266.164 +/- 26.016 |
| B3 TyphoFormer leak-free | 38.849 +/- 2.989 | 102.153 +/- 4.142 | 171.264 +/- 8.768 | 247.609 +/- 18.687 |
| B4 TyphoFormer++ raw | 30.189 +/- 1.521 | 73.787 +/- 6.327 | 125.898 +/- 3.248 | 180.160 +/- 4.133 |
| B4 TyphoFormer++ + val CV calibration | 29.225 +/- 0.802 | 73.787 +/- 6.327 | 125.898 +/- 3.248 | 180.160 +/- 4.133 |

Lead-specific conclusion:

- TyphoFormer++ beats our leak-free TyphoFormer reproduction B3 at all four horizons.

## Safe-Negative Revision for Paper Draft

After auditing the analog pipeline, the paper-facing TyphoFormer++ results were regenerated/evaluated with target-free validation/test negative analogs and with negative analog fusion disabled by default. Training still uses target-aware hard negatives only as supervised ranking examples.

Summary files:

- `official_leadspecific_12to1_safeneg_summary.json`
- `official_leadspecific_12to1_safeneg_summary.md`
- `official_12to1_sequence_baselines_safeneg_summary.json`
- `official_12to1_sequence_baselines_safeneg_summary.md`
- `official_12to1_ablation_safeneg_summary.json`
- `official_12to1_ablation_safeneg_summary.md`
- `official_12to1_ablation_full_safeneg_summary.json`
- `official_12to1_ablation_full_safeneg_summary.md`
- `official_12to1_text_ablation_safeneg_summary.json`
- `official_12to1_text_ablation_safeneg_summary.md`
- `official_12to1_storm_bootstrap_safeneg_summary.json`
- `official_12to1_storm_bootstrap_safeneg_summary.md`

Safe-negative lead-specific DeltaR km, 2022-2024 test:

| Model | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| Persistence | 126.68 | 248.17 | 362.02 | 468.84 |
| Constant Velocity | 36.87 | 95.22 | 170.27 | 256.39 |
| CLIPER-style ridge | 37.28 | 93.24 | 165.42 | 247.66 |
| GRU | 37.681 +/- 0.170 | 93.649 +/- 1.433 | 165.224 +/- 2.867 | 244.747 +/- 4.995 |
| LSTM | 38.305 +/- 0.293 | 97.459 +/- 0.282 | 169.154 +/- 3.960 | 251.756 +/- 6.716 |
| Informer-style | 37.183 +/- 0.448 | 94.618 +/- 0.826 | 162.763 +/- 4.692 | 243.785 +/- 3.859 |
| Autoformer-style | 66.212 +/- 8.190 | 152.669 +/- 15.581 | 248.854 +/- 21.787 | 357.299 +/- 33.088 |
| TSMixer | 37.314 +/- 0.973 | 94.168 +/- 2.226 | 169.437 +/- 5.755 | 247.496 +/- 1.437 |
| B3 TyphoFormer leak-free | 39.462 +/- 0.665 | 101.775 +/- 1.686 | 170.988 +/- 4.322 | 260.721 +/- 6.654 |
| B4 TyphoFormer++ safe raw | 34.153 +/- 0.507 | 86.712 +/- 1.081 | 156.200 +/- 4.628 | 230.528 +/- 1.302 |
| B4 TyphoFormer++ safe + val CV calibration | 34.295 +/- 0.458 | 86.160 +/- 0.825 | 154.985 +/- 4.902 | 228.959 +/- 2.043 |

Full three-seed attribution ablation:

| Variant | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| B3 leak-free TyphoFormer | 39.462 +/- 0.665 | 101.775 +/- 1.686 | 170.988 +/- 4.322 | 260.721 +/- 6.654 |
| B3 + dual CV-residual head | 33.833 +/- 0.468 | 86.462 +/- 0.326 | 153.768 +/- 2.346 | 236.483 +/- 3.212 |
| B3 + positive analog only | 39.200 +/- 0.340 | 96.898 +/- 3.395 | 175.832 +/- 6.034 | 260.644 +/- 10.107 |
| B3 + alignment/rank only | 43.047 +/- 1.205 | 106.165 +/- 2.973 | 176.821 +/- 2.943 | 281.220 +/- 8.686 |
| B4 safe TyphoFormer++ | 34.153 +/- 0.507 | 86.712 +/- 1.081 | 156.200 +/- 4.628 | 230.528 +/- 1.302 |

Storm-level paired bootstrap for B4 safe raw versus B3:

| Lead | Storms | B3 storm mean | B4 storm mean | B4-B3 delta | 95% CI | B4 better storms |
|---:|---:|---:|---:|---:|---:|---:|
| 6h | 41 | 41.023 | 35.698 | -5.326 | [-7.088, -3.571] | 85.4% |
| 12h | 36 | 108.862 | 97.596 | -11.266 | [-19.156, -3.011] | 80.6% |
| 18h | 35 | 185.140 | 170.528 | -14.612 | [-27.522, 1.069] | 74.3% |
| 24h | 30 | 278.032 | 249.723 | -28.309 | [-51.322, -0.417] | 83.3% |

Text-template ablation:

| Row | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| B2/B3 without text-template | 39.185 +/- 0.707 | 101.201 +/- 4.731 | 173.365 +/- 5.238 | 265.277 +/- 8.332 |
| B3 with hash text-template | 39.462 +/- 0.665 | 101.775 +/- 1.686 | 170.988 +/- 4.322 | 260.721 +/- 6.654 |
| B4 without text-template | 33.099 +/- 0.669 | 84.987 +/- 0.605 | 155.363 +/- 1.802 | 240.685 +/- 6.107 |
| B4 full with hash text-template | 34.153 +/- 0.507 | 86.712 +/- 1.081 | 156.200 +/- 4.628 | 230.528 +/- 1.302 |
| B4 full + val CV calibration | 34.295 +/- 0.458 | 86.160 +/- 0.825 | 154.985 +/- 4.902 | 228.959 +/- 2.043 |

Updated interpretation: safe TyphoFormer++ still beats the local leak-free TyphoFormer reproduction and the added GRU/LSTM/Informer-style/Autoformer-style/TSMixer baselines under the same strict official protocol, but it does not beat the published TyphoFormer Table 1 at any lead. Informer-style is the strongest local generic sequence baseline at most leads, while Autoformer-style is weak in this small strict-split setting. The complete ablation shows that the dual CV-residual head is the dominant source of gain; analog, ranking, and text-template features should be framed as supporting signals, not as equally proven standalone contributions. The text-template ablation shows text is not a short-lead driver and helps mainly at 24h in the full model. The current paper draft uses this narrower, defensible claim.
- TyphoFormer++ beats CV and CLIPER-style ridge at all four horizons.
- Lead-specific 12 -> 1 improves over joint 12 -> 4, especially at 6h and 24h, but the improvement is not enough to reproduce the official Table 1 numbers at 12h/18h/24h.
- Against official TyphoFormer Table 1, safe TyphoFormer++ does not beat the published row at any lead. Do not claim official Table 1 superiority.

## Gate Behavior

For strict-6h B4 raw seed42, mean test-set dual gate values are CV-residual weights:

```text
step 01-12: 0.874 0.838 0.808 0.740 0.645 0.534 0.471 0.400 0.308 0.252 0.208 0.163
24h gate: 0.740
48h gate: 0.400
72h gate: 0.163
```

The learned schedule matches the design: short lead times lean on CV-residual, medium lead times mix, and 72h mostly uses the direct head.

## Diagnostics

Extra strict-6h seed42/123 diagnostics:

| Variant | Observation |
|---|---|
| B4 select by val ADE | Worse than val-FDE selection on test. |
| B4 heavier 24/48/72 loss weights `0.5/1.0/1.5` | Worse FDE than recommended `0.3/0.5/1.0`. |
| B4 w/o positive analog | Worse than full on strict-6h seed42. |
| Stronger gate prior + short residual anchor | Did not solve 24h cleanly and hurt FDE. |
| Validation CV calibration | Improved ADE/FDE/24h/48h/72h for all three seeds. |

## Current Conclusion

The closest local reconstruction of the public repo's next-step setup is now the official NHC HURDAT2 lead-specific clean 12 -> 1 experiment: strict 6h, train-period storms from 2004-2021, test storms from 2022-2024, and four independently trained horizons at 6h/12h/18h/24h. In that setting, safe TyphoFormer++ raw clearly exceeds our leak-free TyphoFormer reproduction B3, Numeric Transformer B2, GRU, LSTM, Informer-style, Autoformer-style, TSMixer, Constant Velocity, and the CLIPER-style ridge baseline on the three-seed mean.

This is not a claim of beating the official TyphoFormer Table 1 numbers. The correct claim is narrower and cleaner: TyphoFormer++ outperforms our leak-free TyphoFormer reproduction under official-source, strict-6h, original-horizon-aligned local reconstruction protocols. After the safe-negative correction, it no longer beats the official Table 1 TyphoFormer row at 6h, and it remains much worse at 12h/18h/24h. Treat the published table mismatch as unresolved.

The official 12 -> 4 result should be reported as the joint multi-horizon variant. The 12 -> 12 strict-6h results should be reported as a long-horizon extension or appendix experiment. Under that corrected strict 6h, leak-free 12 -> 12 setting, TyphoFormer++ raw dual-head exceeds the leak-free TyphoFormer baseline B3 on all reported metrics across three seeds. It also beats Numeric Transformer B2 on three-seed average ADE/FDE/24h/48h/72h, which supports that the gain is not merely from using a Transformer.

The raw dual-head still has a small 24h issue against pure Constant Velocity on the three-seed mean: 228.60 km versus CV's 224.42 km. The validation-only CV-calibrated deterministic variant fixes that and improves every metric, giving 211.43 km at 24h and 723.89 km FDE.

For paper tables, report both:

- `TyphoFormer++ dual-head raw` as the main learned model.
- `TyphoFormer++ dual-head + val CV calibration` as the robust operational deterministic variant.

Do not claim the calibrated row is uncalibrated raw model performance.
