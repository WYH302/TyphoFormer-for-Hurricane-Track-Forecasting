# TyphoFormer++ Results Index

This folder keeps both exploratory and formal experiment outputs. Nothing below is deleted during the project split.

## Exploratory Mixed-Interval Results

- `checkpoints_pp/`
- `data_pp/`

These runs used adjacent HURDAT2 rows before enforcing standard synoptic times. They are useful for debugging model direction, but they are not paper-grade because special intermediate HURDAT2 records can change the real lead time.

## Formal Strict-6h 12 -> 12 Results

- `data_pp_6h/`
- `checkpoints_pp_6h/`
- `EXPERIMENT_SUMMARY_TyphoFormerPP.md`

These are the formal long-horizon results: input 12 strict 6h steps, output 12 strict 6h steps, evaluated at 24h/48h/72h.

## Formal Strict-6h 12 -> 4 Horizon-Aligned Results

- `data_pp_6h_12to4/`
- `checkpoints_pp_6h_12to4/`
- `strict6h_12to4_summary.json`

These are the minimal fair experiment aligned to the original TyphoFormer horizon: input 12 strict 6h steps, output 4 strict 6h steps, evaluated at 6h/12h/18h/24h.

## Official NHC HURDAT2 12 -> 4 Results

- `data_official/`
- `data_official_pp_12to4/`
- `checkpoints_official_12to4/`
- `experiment_logs/official_12to4_stdout.log`
- `experiment_logs/official_12to4_stderr.log`
- `official_12to4_summary.json`

These are the paper-facing official-source experiments: NHC Atlantic HURDAT2 official download, train-period storms from 2004-2021, validation storms held out from the training period, test storms from 2022-2024, strict 6h windows, input 12 steps, output 4 steps, evaluated at 6h/12h/18h/24h.

Important interpretation: TyphoFormer++ beats our leak-free TyphoFormer reproduction B3 on this official protocol, but it does not yet beat the official TyphoFormer Table 1 values beyond 6h. Keep that distinction explicit in paper text.

## Official NHC HURDAT2 Lead-Specific 12 -> 1 Results

- `data_official_pp_12to1_lead1/`
- `data_official_pp_12to1_lead2/`
- `data_official_pp_12to1_lead3/`
- `data_official_pp_12to1_lead4/`
- `checkpoints_official_12to1_leadspecific/`
- `run_official_leadspecific_12to1_experiments.ps1`
- `summarize_official_leadspecific_12to1.py`
- `official_leadspecific_12to1_summary.json`

These are the clean single-horizon reconstructions closest to the public repo's 12 -> 1 next-step setting: separate models for 6h, 12h, 18h, and 24h targets, always seeded by the last observed point rather than a future ground-truth point.

Important interpretation: this preliminary non-safe-negative table is superseded by the safe-negative revision below. It helped identify the lead-specific setting, but the paper should use the safe-negative results as main evidence and should not claim superiority over the published Table 1 values.

## Official Lead-Specific Safe-Negative Revision

- `data_official_pp_12to1_lead1_safeneg/`
- `data_official_pp_12to1_lead2_safeneg/`
- `data_official_pp_12to1_lead3_safeneg/`
- `data_official_pp_12to1_lead4_safeneg/`
- `checkpoints_official_12to1_leadspecific_safeneg/`
- `official_leadspecific_12to1_safeneg_summary.json`
- `official_leadspecific_12to1_safeneg_summary.md`

This revision fixes the operational ambiguity in hard-negative analogs. Training samples may use target-aware hard negatives for ranking supervision, but validation/test samples use target-free query-neighbor negatives. The default model also does not fuse negative analogs as inference-time context.

Safe-negative TyphoFormer++ still beats the local leak-free TyphoFormer reproduction at all four lead times:

| Model | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| B3 TyphoFormer leak-free | 39.462 +/- 0.665 | 101.775 +/- 1.686 | 170.988 +/- 4.322 | 260.721 +/- 6.654 |
| B4 TyphoFormer++ safe raw | 34.153 +/- 0.507 | 86.712 +/- 1.081 | 156.200 +/- 4.628 | 230.528 +/- 1.302 |
| B4 TyphoFormer++ safe + val CV calibration | 34.295 +/- 0.458 | 86.160 +/- 0.825 | 154.985 +/- 4.902 | 228.959 +/- 2.043 |

Important interpretation update: after the safe-negative correction, TyphoFormer++ no longer beats the published TyphoFormer Table 1 at 6h. The correct paper claim is only that TyphoFormer++ improves over the local leak-free TyphoFormer reproduction under the same strict official HURDAT2 protocol.

## Official 12 -> 1 Safe-Negative Attribution Ablation

- `checkpoints_official_12to1_ablation_safeneg/`
- `run_official_12to1_ablation_safeneg.ps1`
- `summarize_official_12to1_ablation_safeneg.py`
- `official_12to1_ablation_safeneg_summary.json`
- `official_12to1_ablation_safeneg_summary.md`

This is a seed-42 minimal attribution study on the 12h and 24h lead-specific tasks. It shows that the dual CV-residual head is the largest single contributor, positive analog memory helps by itself, and alignment/ranking without the physics head is not sufficient.

## Official 12 -> 1 Strong Sequence Baselines

- `checkpoints_official_12to1_sequence_baselines_safeneg/`
- `run_official_12to1_sequence_baselines_safeneg.ps1`
- `train_sequence_baseline.py`
- `summarize_official_12to1_sequence_baselines_safeneg.py`
- `official_12to1_sequence_baselines_safeneg_summary.json`
- `official_12to1_sequence_baselines_safeneg_summary.md`

These runs add local GRU, LSTM, Informer-style, Autoformer-style, and TSMixer controls under the same official strict-6h, safe-negative, lead-specific 12 -> 1 protocol. Values are DeltaR km, mean +/- std over seeds 42/123/2024:

| Model | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| GRU | 37.681 +/- 0.170 | 93.649 +/- 1.433 | 165.224 +/- 2.867 | 244.747 +/- 4.995 |
| LSTM | 38.305 +/- 0.293 | 97.459 +/- 0.282 | 169.154 +/- 3.960 | 251.756 +/- 6.716 |
| Informer-style | 37.183 +/- 0.448 | 94.618 +/- 0.826 | 162.763 +/- 4.692 | 243.785 +/- 3.859 |
| Autoformer-style | 66.212 +/- 8.190 | 152.669 +/- 15.581 | 248.854 +/- 21.787 | 357.299 +/- 33.088 |
| TSMixer | 37.314 +/- 0.973 | 94.168 +/- 2.226 | 169.437 +/- 5.755 | 247.496 +/- 1.437 |

## Official 12 -> 1 Full Safe-Negative Ablation

- `checkpoints_official_12to1_ablation_full_safeneg/`
- `run_official_12to1_ablation_full_safeneg.ps1`
- `summarize_official_12to1_ablation_full_safeneg.py`
- `official_12to1_ablation_full_safeneg_summary.json`
- `official_12to1_ablation_full_safeneg_summary.md`

This is the paper-facing attribution table: four leads and three seeds. It shows that the dual CV-residual head is the dominant contribution; positive analog memory is weaker and uneven; alignment/rank without the physics head is unstable; full safe TyphoFormer++ is strongest at 24h.

| Variant | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| B3 leak-free TyphoFormer | 39.462 +/- 0.665 | 101.775 +/- 1.686 | 170.988 +/- 4.322 | 260.721 +/- 6.654 |
| B3 + dual CV-residual head | 33.833 +/- 0.468 | 86.462 +/- 0.326 | 153.768 +/- 2.346 | 236.483 +/- 3.212 |
| B3 + positive analog only | 39.200 +/- 0.340 | 96.898 +/- 3.395 | 175.832 +/- 6.034 | 260.644 +/- 10.107 |
| B3 + alignment/rank only | 43.047 +/- 1.205 | 106.165 +/- 2.973 | 176.821 +/- 2.943 | 281.220 +/- 8.686 |
| B4 safe TyphoFormer++ | 34.153 +/- 0.507 | 86.712 +/- 1.081 | 156.200 +/- 4.628 | 230.528 +/- 1.302 |

## Official 12 -> 1 Storm-Level Bootstrap

- `storm_bootstrap_official_12to1_safeneg.py`
- `official_12to1_storm_bootstrap_safeneg_summary.json`
- `official_12to1_storm_bootstrap_safeneg_summary.md`

This check aggregates errors by test storm before comparing B4 safe raw against B3. Negative B4-B3 values mean TyphoFormer++ is better.

| Lead | Storms | B3 storm mean | B4 storm mean | B4-B3 delta | 95% CI | B4 better storms |
|---:|---:|---:|---:|---:|---:|---:|
| 6h | 41 | 41.023 | 35.698 | -5.326 | [-7.088, -3.571] | 85.4% |
| 12h | 36 | 108.862 | 97.596 | -11.266 | [-19.156, -3.011] | 80.6% |
| 18h | 35 | 185.140 | 170.528 | -14.612 | [-27.522, 1.069] | 74.3% |
| 24h | 30 | 278.032 | 249.723 | -28.309 | [-51.322, -0.417] | 83.3% |

## Official 12 -> 1 Text-Template Ablation

- `checkpoints_official_12to1_text_ablation_safeneg/`
- `run_official_12to1_text_ablation_safeneg.ps1`
- `summarize_official_12to1_text_ablation_safeneg.py`
- `official_12to1_text_ablation_safeneg_summary.json`
- `official_12to1_text_ablation_safeneg_summary.md`

This ablation checks whether deterministic hash text-template features are a primary performance driver. They are not: B4 without text is better at 6h/12h/18h, while full B4 is better at 24h. The paper therefore frames text-template features as a weak long-lead auxiliary signal, not as a language-semantics contribution.

| Row | 6h | 12h | 18h | 24h |
|---|---:|---:|---:|---:|
| B2/B3 without text-template | 39.185 +/- 0.707 | 101.201 +/- 4.731 | 173.365 +/- 5.238 | 265.277 +/- 8.332 |
| B3 with hash text-template | 39.462 +/- 0.665 | 101.775 +/- 1.686 | 170.988 +/- 4.322 | 260.721 +/- 6.654 |
| B4 without text-template | 33.099 +/- 0.669 | 84.987 +/- 0.605 | 155.363 +/- 1.802 | 240.685 +/- 6.107 |
| B4 full with hash text-template | 34.153 +/- 0.507 | 86.712 +/- 1.081 | 156.200 +/- 4.628 | 230.528 +/- 1.302 |
| B4 full + val CV calibration | 34.295 +/- 0.458 | 86.160 +/- 0.825 | 154.985 +/- 4.902 | 228.959 +/- 2.043 |
