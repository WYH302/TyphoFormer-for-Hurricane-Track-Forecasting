# Lagrangian Kinematic Residual Learning (`kinematic_residual/`)

This directory contains the core implementation of **Lagrangian Kinematic Residual Learning** and the four-tier output parameterization hierarchy evaluated across 6-, 12-, 18-, and 24-hour tropical cyclone track forecasting horizons on the North Atlantic HURDAT2 archive.

## Overview of Output Parameterizations (`controlled_heads.py`)

Given a 12-step (66-hour) synoptic observation sequence projected onto a local equirectangular tangent plane centered at forecast initialization position $p_T$, let $D_s^{\mathrm{cv}} = s\,\mathcal{M}(p_T; p_{T-1})$ denote analytical constant-velocity (CV) vortex persistence over lead time $s \in \{1, 2, 3, 4\}$ six-hour steps, and let $c \in \mathbb{R}^{256}$ denote the latent sequence representation:

- **$A$ (`direct` — Direct Eulerian Coordinate Regression):** $\widehat{D}^A = f(c)$
- **$B$ (`single_residual` — Single Lagrangian Kinematic Residual Head):** $\widehat{D}^B = D_s^{\mathrm{cv}} + r(c)$
- **$C$ (`fixed_dual` — Fixed Dual-Branch Blend):** $\widehat{D}^C = \tfrac{1}{2}(D_s^{\mathrm{cv}} + r(c)) + \tfrac{1}{2}f(c)$
- **$D$ (`adaptive_dual` — Adaptive Gated Dual Head):** $\widehat{D}^D = g(c)(D_s^{\mathrm{cv}} + r(c)) + (1 - g(c))f(c)$

## Key Scripts

| File | Description |
| :--- | :--- |
| `prepare_frozen.py` | Local equirectangular tangent-plane coordinate projection $\mathcal{M}(p; p_0)$, inverse spherical mapping, exact Haversine evaluation, and 6-hourly HURDAT2 sequence builder. |
| `controlled_heads.py` | Modular PyTorch implementation of output heads $A$, $B$, $C$, and $D$. |
| `run_controlled.py` | Controlled training and evaluation pipeline for Transformer sequence models across heads $A$–$D$ and optional intermediate-step supervision. |
| `run_matrix.py` | Multi-seed, multi-horizon (6, 12, 18, 24 h) experiment matrix runner with validation early stopping and sealed 2025 chronological extension evaluation. |
| `ridge_baseline.py` | Regularized linear Ridge kinematic residual predictor on flattened 12-step tangent-plane trajectories. |
| `mlp_residual_baseline.py` | Two-layer GELU multilayer perceptron (MLP) kinematic residual predictor. |
| `analyze_revision.py` | Whole-storm cluster bootstrap ($B=10{,}000$ replicates) confidence interval estimation and paired hypothesis testing across window-weighted ($\overline{E}_{\mathrm{win}}$) and equal-storm ($\overline{E}_{\mathrm{storm}}$) metrics. |
| `benchmark_inference.py` | Single-CPU and GPU latency, throughput, and parameter count benchmarking. |
| `vendor/model.py` | Spatio-temporal Transformer sequence encoder backbone (`STTransformer`). |
| `tests/` | Unit tests for output heads, MLP baselines, year-split validation, and scheduler reproducibility. |
