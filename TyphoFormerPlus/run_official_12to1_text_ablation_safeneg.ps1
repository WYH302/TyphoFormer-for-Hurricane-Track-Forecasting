$ErrorActionPreference = "Stop"

$Python = "D:\App1\environment\envs\ai_base\python.exe"
$SaveDir = "checkpoints_official_12to1_text_ablation_safeneg"
$Seeds = @(42, 123, 2024)
$Leads = @(1, 2, 3, 4)

function Invoke-IfMissing {
  param(
    [string]$OutputPath,
    [scriptblock]$Command
  )
  if (Test-Path $OutputPath) {
    Write-Host "SKIP existing $OutputPath"
  } else {
    & $Command
  }
}

New-Item -ItemType Directory -Force -Path $SaveDir | Out-Null

foreach ($Lead in $Leads) {
  $DataDir = "data_official_pp_12to1_lead${Lead}_safeneg"
  foreach ($Seed in $Seeds) {
    $RunName = "b4_plus_dual_no_text_lead${Lead}_s$Seed"
    Invoke-IfMissing "$SaveDir\$RunName\best_model.pt" {
      & $Python train_typhoformerpp.py `
        --data-dir $DataDir --save-dir $SaveDir `
        --run-name $RunName `
        --variant plus --decoder deterministic --residual dual --no-text `
        --loss weighted_haversine --select-key fde `
        --epochs 100 --early-stop-patience 20 `
        --batch-size 8 --lr 0.0001 --weight-decay 0.00001 `
        --d-model 256 --num-heads 4 --num-layers 3 --dropout 0.1 --modality-dropout 0.2 `
        --lambda-align 0.05 --lambda-rank 0.10 --lambda-smooth 0.02 `
        --horizon-weight-24 0.5 --horizon-weight-48 1.0 --horizon-weight-72 1.5 `
        --rank-margin 0.5 --seed $Seed --disable-progress
    }
    Invoke-IfMissing "$SaveDir\$RunName\eval_test.json" {
      & $Python eval_typhoformerpp.py `
        --checkpoint "$SaveDir\$RunName\best_model.pt" `
        --data-dir $DataDir --split test --batch-size 64 `
        --output-json "$SaveDir\$RunName\eval_test.json" `
        --disable-progress
    }
  }
}

& $Python summarize_official_12to1_text_ablation_safeneg.py `
  --checkpoints-dir $SaveDir `
  --output-json official_12to1_text_ablation_safeneg_summary.json `
  --output-md official_12to1_text_ablation_safeneg_summary.md
