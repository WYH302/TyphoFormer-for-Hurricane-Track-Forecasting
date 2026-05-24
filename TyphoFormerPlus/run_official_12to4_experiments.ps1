$ErrorActionPreference = "Stop"

$Python = "D:\App1\environment\envs\ai_base\python.exe"
$DataDir = "data_official_pp_12to4"
$SaveDir = "checkpoints_official_12to4"
$Seeds = @(42, 123, 2024)

& $Python eval_official_cliper_baselines.py --data-dir $DataDir --split test --output-json "$DataDir\official_baselines_test.json"
& $Python eval_official_cliper_baselines.py --data-dir $DataDir --split test --year-filter 2024 --output-json "$DataDir\official_baselines_test_2024.json"

foreach ($Seed in $Seeds) {
  & $Python train_typhoformerpp.py `
    --data-dir $DataDir --save-dir $SaveDir `
    --run-name "b2_numeric_official_12to4_s$Seed" `
    --variant numeric --decoder deterministic --residual none `
    --loss weighted_haversine --horizon-weight-24 0.3 --horizon-weight-48 0.5 --horizon-weight-72 1.0 `
    --select-key fde --epochs 100 --early-stop-patience 20 `
    --batch-size 8 --lr 0.0001 --weight-decay 0.00001 `
    --d-model 256 --num-heads 4 --num-layers 3 --dropout 0.1 --modality-dropout 0.2 `
    --lambda-smooth 0.02 --seed $Seed --disable-progress

  & $Python eval_typhoformerpp.py `
    --checkpoint "$SaveDir\b2_numeric_official_12to4_s$Seed\best_model.pt" `
    --data-dir $DataDir --split test --batch-size 64 `
    --output-json "$SaveDir\b2_numeric_official_12to4_s$Seed\eval_test.json" `
    --disable-progress

  & $Python eval_typhoformerpp.py `
    --checkpoint "$SaveDir\b2_numeric_official_12to4_s$Seed\best_model.pt" `
    --data-dir $DataDir --split test --batch-size 64 `
    --year-filter 2024 `
    --output-json "$SaveDir\b2_numeric_official_12to4_s$Seed\eval_test_2024.json" `
    --disable-progress
}

foreach ($Seed in $Seeds) {
  & $Python train_typhoformerpp.py `
    --data-dir $DataDir --save-dir $SaveDir `
    --run-name "b3_typhoformer_official_12to4_s$Seed" `
    --variant typhoformer --decoder deterministic --residual none `
    --loss weighted_haversine --horizon-weight-24 0.3 --horizon-weight-48 0.5 --horizon-weight-72 1.0 `
    --select-key fde --epochs 100 --early-stop-patience 20 `
    --batch-size 8 --lr 0.0001 --weight-decay 0.00001 `
    --d-model 256 --num-heads 4 --num-layers 3 --dropout 0.1 --modality-dropout 0.2 `
    --lambda-smooth 0.02 --seed $Seed --disable-progress

  & $Python eval_typhoformerpp.py `
    --checkpoint "$SaveDir\b3_typhoformer_official_12to4_s$Seed\best_model.pt" `
    --data-dir $DataDir --split test --batch-size 64 `
    --output-json "$SaveDir\b3_typhoformer_official_12to4_s$Seed\eval_test.json" `
    --disable-progress

  & $Python eval_typhoformerpp.py `
    --checkpoint "$SaveDir\b3_typhoformer_official_12to4_s$Seed\best_model.pt" `
    --data-dir $DataDir --split test --batch-size 64 `
    --year-filter 2024 `
    --output-json "$SaveDir\b3_typhoformer_official_12to4_s$Seed\eval_test_2024.json" `
    --disable-progress
}

foreach ($Seed in $Seeds) {
  & $Python train_typhoformerpp.py `
    --data-dir $DataDir --save-dir $SaveDir `
    --run-name "b4_plus_dual_official_12to4_s$Seed" `
    --variant plus --decoder deterministic --residual dual `
    --loss weighted_haversine --horizon-weight-24 0.3 --horizon-weight-48 0.5 --horizon-weight-72 1.0 `
    --select-key fde --epochs 100 --early-stop-patience 20 `
    --batch-size 8 --lr 0.0001 --weight-decay 0.00001 `
    --d-model 256 --num-heads 4 --num-layers 3 --dropout 0.1 --modality-dropout 0.2 `
    --lambda-align 0.05 --lambda-rank 0.10 --lambda-smooth 0.02 `
    --lambda-direct-aux 0.25 --lambda-cv-aux 0.25 --lambda-gate-prior 0.03 `
    --rank-margin 0.3 --seed $Seed --disable-progress

  & $Python eval_typhoformerpp.py `
    --checkpoint "$SaveDir\b4_plus_dual_official_12to4_s$Seed\best_model.pt" `
    --data-dir $DataDir --split test --batch-size 64 `
    --output-json "$SaveDir\b4_plus_dual_official_12to4_s$Seed\eval_test_calibrated.json" `
    --calibrate-cv-blend --calibration-split val --calibration-step 0.05 `
    --disable-progress

  & $Python eval_typhoformerpp.py `
    --checkpoint "$SaveDir\b4_plus_dual_official_12to4_s$Seed\best_model.pt" `
    --data-dir $DataDir --split test --batch-size 64 `
    --year-filter 2024 `
    --output-json "$SaveDir\b4_plus_dual_official_12to4_s$Seed\eval_test_2024_calibrated.json" `
    --calibrate-cv-blend --calibration-split val --calibration-step 0.05 `
    --disable-progress
}

& $Python summarize_official_12to4.py `
  --checkpoints-dir $SaveDir `
  --baselines-json "$DataDir\official_baselines_test.json" `
  --baselines-2024-json "$DataDir\official_baselines_test_2024.json" `
  --output-json official_12to4_summary.json
