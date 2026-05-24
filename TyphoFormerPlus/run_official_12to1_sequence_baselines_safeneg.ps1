$ErrorActionPreference = "Stop"

$Python = "D:\App1\environment\envs\ai_base\python.exe"
$SaveDir = "checkpoints_official_12to1_sequence_baselines_safeneg"
$Seeds = @(42, 123, 2024)
$Leads = @(1, 2, 3, 4)
$Models = @("gru", "lstm", "informer", "autoformer", "tsmixer")

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
  foreach ($Model in $Models) {
    foreach ($Seed in $Seeds) {
      $RunName = "${Model}_lead${Lead}_s$Seed"
      Invoke-IfMissing "$SaveDir\$RunName\eval_test.json" {
        & $Python train_sequence_baseline.py `
          --data-dir $DataDir `
          --save-dir $SaveDir `
          --run-name $RunName `
          --model $Model `
          --epochs 100 `
          --early-stop-patience 20 `
          --batch-size 32 `
          --lr 0.0001 `
          --weight-decay 0.00001 `
          --hidden-dim 192 `
          --num-layers 2 `
          --dropout 0.1 `
          --seed $Seed `
          --disable-progress
      }
    }
  }
}

& $Python summarize_official_12to1_sequence_baselines_safeneg.py `
  --checkpoints-dir $SaveDir `
  --output-json official_12to1_sequence_baselines_safeneg_summary.json `
  --output-md official_12to1_sequence_baselines_safeneg_summary.md
