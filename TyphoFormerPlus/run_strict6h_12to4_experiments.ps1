$ErrorActionPreference = "Stop"

$Python = "D:\App1\environment\envs\ai_base\python.exe"
$DataDir = "data_pp_6h_12to4"
$SaveDir = "checkpoints_pp_6h_12to4"
$Seeds = @(42, 123, 2024)

& $Python prepare_typhoformerpp_data.py `
  --output-dir $DataDir `
  --input-len 12 `
  --pred-len 4 `
  --standard-times-only `
  --strict-6h `
  --interval-hours 6 `
  --force

foreach ($Seed in $Seeds) {
  & $Python train_typhoformerpp.py `
    --data-dir $DataDir `
    --save-dir $SaveDir `
    --run-name "b2_numeric_direct_6h_12to4_256_e120_s$Seed" `
    --variant numeric `
    --decoder deterministic `
    --residual none `
    --loss weighted_haversine `
    --horizon-weight-24 0.3 `
    --horizon-weight-48 0.5 `
    --horizon-weight-72 1.0 `
    --select-key ade `
    --epochs 120 `
    --early-stop-patience 20 `
    --batch-size 32 `
    --lr 0.0002 `
    --weight-decay 0.0001 `
    --d-model 256 `
    --num-heads 4 `
    --num-layers 3 `
    --dropout 0.15 `
    --modality-dropout 0.2 `
    --lambda-smooth 0.02 `
    --seed $Seed `
    --disable-progress

  & $Python eval_typhoformerpp.py `
    --checkpoint "$SaveDir\b2_numeric_direct_6h_12to4_256_e120_s$Seed\best_model.pt" `
    --data-dir $DataDir `
    --split test `
    --batch-size 64 `
    --output-json "$SaveDir\b2_numeric_direct_6h_12to4_256_e120_s$Seed\eval_test.json" `
    --disable-progress
}

foreach ($Seed in $Seeds) {
  & $Python train_typhoformerpp.py `
    --data-dir $DataDir `
    --save-dir $SaveDir `
    --run-name "b3_typhoformer_leakfree_direct_6h_12to4_256_e120_s$Seed" `
    --variant typhoformer `
    --decoder deterministic `
    --residual none `
    --loss weighted_haversine `
    --horizon-weight-24 0.3 `
    --horizon-weight-48 0.5 `
    --horizon-weight-72 1.0 `
    --select-key ade `
    --epochs 120 `
    --early-stop-patience 20 `
    --batch-size 32 `
    --lr 0.0002 `
    --weight-decay 0.0001 `
    --d-model 256 `
    --num-heads 4 `
    --num-layers 3 `
    --dropout 0.15 `
    --modality-dropout 0.2 `
    --lambda-smooth 0.02 `
    --seed $Seed `
    --disable-progress

  & $Python eval_typhoformerpp.py `
    --checkpoint "$SaveDir\b3_typhoformer_leakfree_direct_6h_12to4_256_e120_s$Seed\best_model.pt" `
    --data-dir $DataDir `
    --split test `
    --batch-size 64 `
    --output-json "$SaveDir\b3_typhoformer_leakfree_direct_6h_12to4_256_e120_s$Seed\eval_test.json" `
    --disable-progress
}

foreach ($Seed in $Seeds) {
  & $Python train_typhoformerpp.py `
    --data-dir $DataDir `
    --save-dir $SaveDir `
    --run-name "b4_plus_dual_aux_fde_6h_12to4_256_e160_s$Seed" `
    --variant plus `
    --decoder deterministic `
    --residual dual `
    --loss weighted_haversine `
    --horizon-weight-24 0.3 `
    --horizon-weight-48 0.5 `
    --horizon-weight-72 1.0 `
    --select-key fde `
    --epochs 160 `
    --early-stop-patience 25 `
    --batch-size 32 `
    --lr 0.0002 `
    --weight-decay 0.0001 `
    --d-model 256 `
    --num-heads 4 `
    --num-layers 3 `
    --dropout 0.15 `
    --modality-dropout 0.2 `
    --lambda-align 0.05 `
    --lambda-rank 0.10 `
    --lambda-smooth 0.02 `
    --lambda-direct-aux 0.25 `
    --lambda-cv-aux 0.25 `
    --lambda-gate-prior 0.03 `
    --rank-margin 0.3 `
    --seed $Seed `
    --disable-progress

  & $Python eval_typhoformerpp.py `
    --checkpoint "$SaveDir\b4_plus_dual_aux_fde_6h_12to4_256_e160_s$Seed\best_model.pt" `
    --data-dir $DataDir `
    --split test `
    --batch-size 64 `
    --output-json "$SaveDir\b4_plus_dual_aux_fde_6h_12to4_256_e160_s$Seed\eval_test_calibrated.json" `
    --calibrate-cv-blend `
    --calibration-split val `
    --calibration-step 0.05 `
    --disable-progress
}

& $Python summarize_strict6h_12to4.py `
  --checkpoints-dir $SaveDir `
  --output-json strict6h_12to4_summary.json
