$ErrorActionPreference = "Stop"

$Python = "D:\App1\environment\envs\ai_base\python.exe"
$SaveDir = "checkpoints_official_12to1_ablation_full_safeneg"
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

function Train-Variant {
  param(
    [int]$Lead,
    [int]$Seed,
    [string]$RunName,
    [string[]]$ExtraArgs
  )
  $DataDir = "data_official_pp_12to1_lead${Lead}_safeneg"
  Invoke-IfMissing "$SaveDir\$RunName\best_model.pt" {
    & $Python train_typhoformerpp.py `
      --data-dir $DataDir --save-dir $SaveDir `
      --run-name $RunName `
      --loss weighted_haversine --select-key fde `
      --epochs 100 --early-stop-patience 20 `
      --batch-size 8 --lr 0.0001 --weight-decay 0.00001 `
      --d-model 256 --num-heads 4 --num-layers 3 --dropout 0.1 --modality-dropout 0.2 `
      --lambda-smooth 0.02 --seed $Seed --disable-progress `
      @ExtraArgs
  }
  Invoke-IfMissing "$SaveDir\$RunName\eval_test.json" {
    & $Python eval_typhoformerpp.py `
      --checkpoint "$SaveDir\$RunName\best_model.pt" `
      --data-dir $DataDir --split test --batch-size 64 `
      --output-json "$SaveDir\$RunName\eval_test.json" `
      --disable-progress
  }
}

New-Item -ItemType Directory -Force -Path $SaveDir | Out-Null

foreach ($Lead in $Leads) {
  $DataDir = "data_official_pp_12to1_lead${Lead}_safeneg"
  foreach ($Seed in $Seeds) {
    $B3Run = "b3_leakfree_eval_lead${Lead}_s$Seed"
    New-Item -ItemType Directory -Force -Path "$SaveDir\$B3Run" | Out-Null
    Invoke-IfMissing "$SaveDir\$B3Run\eval_test.json" {
      & $Python eval_typhoformerpp.py `
        --checkpoint "checkpoints_official_12to1_leadspecific\b3_typhoformer_official_12to1_lead${Lead}_s$Seed\best_model.pt" `
        --data-dir $DataDir --split test --batch-size 64 `
        --output-json "$SaveDir\$B3Run\eval_test.json" `
        --disable-progress
    }

    Train-Variant $Lead $Seed "b3_dual_head_lead${Lead}_s$Seed" @(
      "--variant", "typhoformer",
      "--decoder", "deterministic",
      "--residual", "dual",
      "--lambda-direct-aux", "0.25",
      "--lambda-cv-aux", "0.25",
      "--lambda-gate-prior", "0.03",
      "--rank-margin", "0.3"
    )

    Train-Variant $Lead $Seed "b3_positive_analog_lead${Lead}_s$Seed" @(
      "--variant", "plus",
      "--decoder", "deterministic",
      "--residual", "none",
      "--no-negative-analog",
      "--lambda-align", "0.0",
      "--lambda-rank", "0.0",
      "--rank-margin", "0.3"
    )

    Train-Variant $Lead $Seed "b3_alignment_rank_lead${Lead}_s$Seed" @(
      "--variant", "plus",
      "--decoder", "deterministic",
      "--residual", "none",
      "--no-positive-analog",
      "--lambda-align", "0.05",
      "--lambda-rank", "0.10",
      "--rank-margin", "0.3"
    )

    $B4Run = "b4_safe_existing_lead${Lead}_s$Seed"
    New-Item -ItemType Directory -Force -Path "$SaveDir\$B4Run" | Out-Null
    Invoke-IfMissing "$SaveDir\$B4Run\eval_test.json" {
      & $Python eval_typhoformerpp.py `
        --checkpoint "checkpoints_official_12to1_leadspecific_safeneg\b4_plus_dual_safeneg_lead${Lead}_s$Seed\best_model.pt" `
        --data-dir $DataDir --split test --batch-size 64 `
        --output-json "$SaveDir\$B4Run\eval_test.json" `
        --disable-progress
    }
  }
}

& $Python summarize_official_12to1_ablation_full_safeneg.py `
  --checkpoints-dir $SaveDir `
  --output-json official_12to1_ablation_full_safeneg_summary.json `
  --output-md official_12to1_ablation_full_safeneg_summary.md
