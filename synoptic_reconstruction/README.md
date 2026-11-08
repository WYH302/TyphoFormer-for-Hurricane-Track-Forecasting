# Synoptic-First Temporal Reconstruction (`synoptic_reconstruction/`)

This directory implements the **Synoptic-First Temporal Reconstruction** protocol, which resolves **Event-Induced Hazard Censorship (EIHC)** in mixed-cadence best-track archives (such as NOAA HURDAT2), along with the eight formal software verification invariants (`C1`–`C8`) and end-to-end synoptic retraining scripts.

## Why Conventional Contiguous-Row Slicing Censors Coastal Landfalls

Operational best-track archives interleave routine 6-hourly synoptic observations (`0000`, `0600`, `1200`, `1800 UTC`) with asynchronous special records documenting coastal landfall (record identifier `L`) or intensity peaks. Conventional sequence preprocessing pipelines that slice fixed-length windows by requiring contiguous raw archive rows separated by 6 hours treat every asynchronous `L` record as a cadence break—inadvertently purging **100% of future-landfall windows** (101 windows across 20 storms in the 2022–2024 North Atlantic test cohort).

Our four-stage **Synoptic-First Temporal Reconstruction** (`prepare_revision.py`):
1. **Synoptic Timestamp Filtering:** Extracts canonical synoptic timestamps (`{00, 06, 12, 18 UTC}`) while setting aside asynchronous records as hazard annotations.
2. **Temporal Window Verification:** Verifies 12 consecutive synoptic steps ($\Delta t = 6\,\mathrm{h}$) over $[T-66\,\mathrm{h}, T]$ and valid synoptic targets at $T+s$.
3. **Causal Motion Derivation:** Computes zonal/meridional velocities, speed, heading, and accelerations strictly across verified 6-hour synoptic steps.
4. **Coastal Hazard Annotation:** Cross-references asynchronous `L` records within $(T, T+24\,\mathrm{h}]$ to annotate landfall-spanning sequences without corrupting sequence cadence.

## Formal Software Verification Invariants (`test_window_contract.py`)

Run the automated contract verification suite:
```bash
pytest synoptic_reconstruction/test_window_contract.py
```

The test suite verifies all eight data-integrity invariants (`C1`–`C8`):
- **`C1` (Asynchronous Insertion Invariance):** Inserting off-hour landfall (`L`) rows does not alter extracted synoptic windows or features.
- **`C2` (Missing-Synoptic Rejection):** Any genuine missing 6-hour synoptic step is strictly rejected.
- **`C3` (Future-Information Isolation):** Perturbing archive rows after $T$ leaves all input features on $[T-66\,\mathrm{h}, T]$ bitwise identical.
- **`C4` (Training-Only Normalization):** Z-score statistics are fit exclusively on 2004–2021 training windows.
- **`C5` (Duplicate-Timestamp Signaling):** Malformed duplicate timestamps raise an explicit `ValueError`.
- **`C6` (Cryptographic Archive Traceability):** Verifies the SHA-256 digest of the official HURDAT2 snapshot.
- **`C7` (Elapsed-Hour Velocity Scaling):** Verifies exact $\Delta t_k = 6\,\mathrm{h}$ velocity and acceleration scaling.
- **`C8` (Landfall Interval Boundary Precision):** Verifies half-open interval $(T, T+24\,\mathrm{h}]$ coastal hazard attribution at 1-minute precision.
