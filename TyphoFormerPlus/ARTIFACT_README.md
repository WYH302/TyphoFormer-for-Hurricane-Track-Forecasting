# TyphoFormer++ Strict-HURDAT2 Artifact README

This artifact supports the paper claim under one audited setting: official NHC Atlantic HURDAT2, strict 6-hour synoptic windows, 12 observed records, lead-specific 12 -> 1 targets at 6h/12h/18h/24h, storm-level splits, train-only normalization, and training-storm-only analog retrieval.

It does not support a cross-protocol state-of-the-art claim against the published TyphoFormer table or an operational forecast claim.

## Fixed Data Source

- HURDAT2 file: `data_official/hurdat2_atl_1851_2024.txt`
- Local SHA256: `D595ADA8AAA64E402C26EEB80EFB344B52EF420B1627D8C0EE6EE65F0729923B`
- Protocol years: training/validation storms from 2004-2021; test storms from 2022-2024
- Validation split: 10% random holdout from training-period storm IDs, split seed 42
- 2025 storms are not part of this artifact and should be treated as a future locked temporal-extension check.

## Primary Dataset Folders

Each folder contains `metadata.json` plus `train/`, `val/`, and `test/` `.npy` records. The metadata includes split storm IDs, feature names, normalization statistics, source audit counts, data audit counts, and analog-negative policy.

| Lead | Folder | Train W/S | Val W/S | Test W/S |
|---:|---|---:|---:|---:|
| 6h | `data_official_pp_12to1_lead1_safeneg/` | 4050/219 | 354/22 | 759/41 |
| 12h | `data_official_pp_12to1_lead2_safeneg/` | 3797/207 | 329/21 | 710/36 |
| 18h | `data_official_pp_12to1_lead3_safeneg/` | 3565/196 | 306/20 | 668/35 |
| 24h | `data_official_pp_12to1_lead4_safeneg/` | 3349/188 | 285/20 | 630/30 |

## Leakage Audit

Run the full file-level audit:

```powershell
D:\App1\environment\envs\ai_base\python.exe audit_official_protocol.py --output-json official_12to1_protocol_audit.json
```

Expected result: `total_failures` is `0`.

The audit checks:

- Split storm IDs are disjoint.
- Saved history/future times are only 0000/0600/1200/1800 UTC.
- History timestamps are strict 6-hour sequences.
- Target timestamp equals the requested lead.
- Test windows come from 2022-2024 and train/validation windows from 2004-2021.
- Validation/test analog storms are training storms only.
- Training analog candidates exclude the same storm.
- Validation/test negative analogs use `target_free_query_neighbor`.
- Training negative analogs use `train_target_hard_negative`.

For a quick smoke test, add `--max-files-per-split 20`; do not use the smoke test as paper evidence.

## Result Summaries

- Main B4/B2/B3 lead-specific results: `official_leadspecific_12to1_safeneg_summary.json` and `.md`
- Strong sequence baselines: `official_12to1_sequence_baselines_safeneg_summary.json` and `.md`
- Full ablation table: `official_12to1_ablation_full_safeneg_summary.json` and `.md`
- Text-template ablation: `official_12to1_text_ablation_safeneg_summary.json` and `.md`
- Numeric CV-residual diagnostic: `official_12to1_numeric_dual_safeneg_summary.json` and `.md`
- Storm-level bootstrap against B3: `official_12to1_storm_bootstrap_safeneg_summary.json` and `.md`
- Storm-level bootstrap against strongest non-B4 baselines: `official_12to1_storm_bootstrap_best_baseline_summary.json` and `.md`
- Common-subset, turning subgroup, runtime, and inference timing diagnostics: `official_12to1_additional_diagnostics.json` and `.md`
- Simple residual controls: `official_12to1_simple_residual_baselines.json` and `.md`
- Year/regime/gate diagnostics: `official_12to1_year_regime_gate_diagnostics.json` and `.md`
- Along/cross-track diagnostics: `official_12to1_along_cross_track.json` and `.md`

## Checkpoints and Logs

- Full B4 safe-negative checkpoints: `checkpoints_official_12to1_leadspecific_safeneg/`
- Full ablation checkpoints: `checkpoints_official_12to1_ablation_full_safeneg/`
- Text ablation checkpoints: `checkpoints_official_12to1_text_ablation_safeneg/`
- Numeric CV-residual checkpoints: `checkpoints_official_12to1_numeric_dual_safeneg/`
- Sequence baseline checkpoints: `checkpoints_official_12to1_sequence_baselines_safeneg/`

Each run folder contains `config.json`, `train_log.jsonl`, `best_model.pt`, and `last_model.pt` when the checkpoint is included. Model selection uses validation FDE before test evaluation.

## Interpretation Guardrails

- B4 is the representative full-context row, not the validation-best row over every ablation.
- The CV-residual family, especially the direct/CV-residual head, is the primary model contribution.
- Storm-level bootstrap evidence against the strongest non-B4 baseline is strongest at 6h and 12h; 18h and 24h are positive but less conclusive under that comparison.
- The CLIPER-style ridge row is a local retrospective ridge control, not operational CLIPER or modern NHC guidance.
- Text-template features are deterministic structured-template hash embeddings, not evidence for free-form language semantics.
- The benchmark is retrospective and best-track-only; it must not be used as an operational hurricane forecast system.
- The local material in this artifact ends in 2024; a true 2025 locked temporal-extension test is not included.
