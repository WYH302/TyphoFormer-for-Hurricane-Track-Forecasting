# Beyond Static Coordinates: Physics-Guided Kinematic Residual Learning for Landfall-Complete Tropical Cyclone Track Forecasting
### 超越静态坐标：面向完整近海登陆评估的物理引导运动学残差热带气旋路径预测

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Datasets: NOAA HURDAT2 AL & EP/CP](https://img.shields.io/badge/Archives-NOAA%20HURDAT2%20(AL%20%2B%20EP%2FCP)-0f766e)](https://www.nhc.noaa.gov/data/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release Artifacts](https://img.shields.io/badge/Release-Manuscript%20%26%20Data-blue)](https://github.com/WYH302/TyphoFormer-for-Hurricane-Track-Forecasting/releases/tag/kinematic-residuals-2026-09-24)

> **English** | **[简体中文说明](#-项目简介与中文说明)**

An open-source empirical study and reproducibility toolkit exploring **co-moving Lagrangian kinematic residual learning** and **synoptic-first best-track temporal reconstruction** for short-range (6–24 h) tropical cyclone track forecasting across **two official NOAA NHC HURDAT2 ocean basins** (**North Atlantic `AL`** and **Eastern/Central North Pacific `EP`/`CP`**). We share these implementations, ten-family baseline comparisons, leakage-free five-fold storm-grouped cross-validation, FLOPs/FPS profiling scripts, and automated data-verification contracts (`C1`–`C8`) in the hope that they may serve as a modest, practical reference for researchers working on physics-guided sequence modeling and coastal hazard verification. Feedback, issues, and discussions are warmly welcome.

本项目整理并开源了我们在**北大西洋（`AL`）与东/中北太平洋（`EP`/`CP`）双海域**短临（6–24 小时）热带气旋路径预测中的方法探索、十类模型家族对照、无泄漏五折风暴分组交叉验证、计算复杂度剖析与数据重构复现工具。我们将相关代码、自动化契约测试（`C1`–`C8`）与可编辑矢量图表一并公开，希望能为从事气象序列建模与近海台风/飓风灾害评估的同行提供一点参考与便利，诚挚欢迎批评指正与交流。

<p align="center">
  <img src="assets/Figure_1_Framework_Overview.png" alt="Framework Overview: Physics-Guided Lagrangian Kinematic Residual Learning and Synoptic-First Reconstruction" width="98%">
</p>

---

## 📖 Overview (English)

1. **Co-Moving Kinematic Residual Parameterization (`kinematic_residual/`):**  
   Direct coordinate regression in a static Eulerian frame ($A$) forces sequence encoders to approximate dominant first-order vortex translation alongside subtle second-order turning accelerations, inducing systematic along-track lag (`−66.17 km` at 24 h). Anchoring predictions analytically to local tangent-plane constant-velocity persistence ($B: \widehat{D}_s = D_s^{\mathrm{cv}} + r(c)$) eliminates **87.8%** of along-track lag with **0 extra parameters and 0 extra FLOPs**. Across **ten model families** (Ridge, MLP, LSTM, GRU, TSMixer, PatchTST, iTransformer, **ModernTCN [ICLR 2024]**, **TimeMixer [ICLR 2024]**, and our ST-Transformer), Lagrangian anchoring wins **72/72 (100%)** North Atlantic multi-horizon comparisons, **8/8 (100%)** primary 44-storm Pacific comparisons, and **6/6 (100%)** neural families under leakage-free **five-fold storm-grouped cross-validation (287 storms)** while cutting cross-fold SD across every family. Our right-sized compact encoder (`B64`: **163.1K params**, **3.73 MFLOPs**, **0.62 MB** FP32 weights) runs at **690.6 FPS (1.45 ms)** on a single CPU thread.

2. **Synoptic-First Temporal Reconstruction (`synoptic_reconstruction/`):**  
   Best-track archives interleave regular 6-hourly synoptic fixes (`00, 06, 12, 18 UTC`) with unscheduled minute-precise landfall (`L`) entries. Conventional contiguous-row slicing misclassifies intervening `L` records as cadence breaks and discards 100% of off-hour future-landfall windows. Our timestamp-indexed protocol (verified by 8 unit-tested contracts `C1`–`C8`) restores **613 future-landfall windows** across both ocean basins (**506** Atlantic across 1980–2025 and **107** Pacific across 2004–2025, including **Category-5 Hurricane Otis** and **2025 Hurricane Melissa**), reducing Atlantic landfall error by **51.89 km** (**102.22 km** on Melissa) and cutting Hurricane Otis endpoint error by up to **117.9 km**.

<p align="center">
  <img src="assets/fig_sota_universality.png" alt="Eight-Family North Atlantic SOTA Universality" width="96%">
</p>

<p align="center">
  <img src="assets/fig_cross_basin_and_sota_cases.png" alt="Cross-Basin Pacific Generalization, 5-Fold Storm-Grouped CV, and Visual Cat-5 Hurricane Otis SOTA Comparison" width="96%">
</p>

---

## 📖 项目简介与中文说明

1. **随动拉格朗日运动学残差参数化（`kinematic_residual/`）：**  
   常规序列模型在静止欧拉坐标系下直接回归未来位移（Head $A$），容易在有限历史样本下产生沿路径滞后偏差（北大西洋 24 h 达 `−66.17 km`）。我们将预测端解析锚定于局部切平面等速惯性外推基准（$B: \widehat{D}_s = D_s^{\mathrm{cv}} + r(c)$），在**零额外参数、零额外 FLOPs** 开销下消除 **87.8%** 滞后偏差。在涵盖 **10 类模型家族**（Ridge、MLP、LSTM、GRU、TSMixer、PatchTST、iTransformer、**ModernTCN [ICLR 2024]**、**TimeMixer [ICLR 2024]** 与 ST-Transformer）的跨海域评测中，拉格朗日残差在北大西洋 **72/72（100%）** 组对比、东太平洋 44 风暴主测试集 **8/8（100%）** 组对比，以及无泄漏 **5 折风暴分组交叉验证（287 个气旋）6/6（100%）** 神经架构中均取得误差下降与跨折方差收敛。轻量化紧凑编码器 `B64` 仅 **163.1K 参数、3.73 MFLOPs、0.62 MB 权重**，单核 CPU 实测 **690.6 FPS（1.45 ms）**。
2. **天气尺度优先重构协议（`synoptic_reconstruction/`）：**  
   针对 HURDAT2 档案在整点观测间穿插非整点登陆（`L`）记录导致传统连续行切片 100% 漏删非整点登陆样本的问题，我们提出基于时间戳索引的重构协议与 8 项契约单元测试（`C1`–`C8`），在双海域累计恢复 **613 个未来登陆窗口**（北大西洋 506 个、东太平洋 107 个，含 2023 五级飓风 Otis 与 2025 飓风 Melissa）；在近海登陆子集上使北大西洋等风暴误差降低 **51.89 km**（Melissa 降低 **102.22 km**），在五级飓风 Otis 登陆窗口最高降低 **117.9 km**。

---

## 📊 Efficiency, Ten-Family Multi-Basin & 5-Fold CV Highlights / 核心实测汇总

| Model Family / 模型家族 | Params | MFLOPs | CPU FPS (ms) | Atlantic 24 h (22–24 / 25) | Pacific 24 h (22–24 / 25) | 5-Fold Storm CV (Direct → Res.) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constant-Velocity (`CV`)** | 0 | 0.000 | — | 256.4 / 247.5 | 125.2 / 119.4 | 244.97 ± 18.50 |
| **Ridge Residual (`Ridge-KR`)** | 962 | 0.002 | >10k (0.12) | 236.5 / 209.5 | 114.4 / 114.8 | <u>225.54 ± 12.45</u> |
| **LSTM + KR** | 227.8K | 5.227 | 1,248 (0.80) | 233.4 / 202.3 | **114.0** / 115.6 | 262.95 → **228.90 ± 8.48** (5/5) |
| **GRU + KR** | 173.0K | 3.924 | 857 (1.17) | 233.9 / 201.9 | 114.8 / 115.1 | 250.84 → **230.45 ± 12.56** (5/5) |
| **TSMixer + KR** (KDD 2023) | 102.4K | 0.911 | 766 (1.31) | 238.1 / 208.1 | 117.7 / 117.0 | 283.35 → **233.74 ± 12.84** (5/5) |
| **PatchTST + KR** (ICLR 2023) | 421.2K | 3.302 | 1,005 (1.00) | <u>232.7</u> / 204.7 | 115.9 / 115.8 | 255.41 → **230.56 ± 10.79** (5/5) |
| **iTransformer + KR** (ICLR 2024) | 428.5K | 33.268 | 699 (1.43) | 250.1 / 238.9 | 121.9 / 117.7 | — |
| **ModernTCN + KR** (ICLR 2024) | 280.9K | 6.474 | 634 (1.58) | **226.0** / 203.8 | <u>114.3</u> / 117.9 | 227.12 → **225.47 ± 7.36** |
| **TimeMixer + KR** (ICLR 2024) | 461.4K | 5.835 | 390 (2.56) | 233.5 / **198.8** | 116.3 / 117.3 | — |
| **Compact ST-Trans (`B64`, Ours)** | 163.1K | 3.728 | **691 (1.45)** | 236.0 / 206.3 | 116.8 / **114.2** | 259.81 → **230.46 ± 16.12** (5/5) |
| **Compact ST-Trans (`D64`, Ours)** | 175.9K | 3.753 | 669 (1.50) | 238.5 / <u>200.2</u> | — | — |

---

## ⚙️ Compute Footprint & Quickstart / 运行环境与快速上手

- **Measured GPU Budget:** Full 123-run RTX 3090 (24 GB VRAM) ST-Transformer suite finishes in **5.44 GPU hours** (<1.2 GB peak VRAM; 108 primary multi-horizon runs = **5.04 GPU hours**).
- **Dependencies:** Standard Python 3.10–3.12 scientific stack pinned cleanly in [`requirements.txt`](requirements.txt).

```bash
# 1. Install dependencies / 安装依赖
pip install -r requirements.txt

# 2. Run C1-C8 data-contract & unit tests / 运行自动化契约测试
pytest synoptic_reconstruction/test_window_contract.py
pytest kinematic_residual/tests/

# 3. Run multi-architecture SOTA, Pacific cross-basin, 5-fold storm-CV & FLOPs profiling
python kinematic_residual/sota_sequence_baselines.py
python kinematic_residual/eastern_pacific_cross_basin.py
python kinematic_residual/stage2_sota_cv_profile.py
python kinematic_residual/stage2_cv_and_cases.py
python kinematic_residual/system_module_roi_ablation.py
```

---

## 📦 Editable Figures & Release Bundles / 可编辑矢量图与复现附件

- Native editable framework schematic (`Comic Sans MS`): `figures/Figure_1_Editable_One_Page.pptx`, `.svg`, and `.pdf`.
- Complete manuscript PDF, 40-page supplementary statistics (`ESM_2.pdf`), single-file LaTeX sources (`editable_manuscript.zip`), and verification archives are hosted under **[GitHub Releases](https://github.com/WYH302/TyphoFormer-for-Hurricane-Track-Forecasting/releases/tag/kinematic-residuals-2026-09-24)**.

---

## 📝 Citation / 引用参考

```bibtex
@article{wu2026kinematic,
  title   = {Beyond Static Coordinates: Physics-Guided Kinematic Residual Learning for Landfall-Complete Tropical Cyclone Track Forecasting},
  author  = {Wu, Yonghao and Wu, Jiayang and He, Sirui and Liu, Chang and Filaretov, Vladimir and Yukhimets, Dmitry},
  year    = {2026},
  url     = {https://github.com/WYH302/TyphoFormer-for-Hurricane-Track-Forecasting}
}
```
