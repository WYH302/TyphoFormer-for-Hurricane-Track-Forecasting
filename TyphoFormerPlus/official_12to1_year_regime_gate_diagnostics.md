# Official 12->1 Year, Regime, and Gate Diagnostics

Local HURDAT material max year: 2024. 2025 locked evaluation available: False.

## Yearly B4 vs Best Baseline

| Lead | Year | Windows | Storms | Best baseline | Best mean | B4 mean | B4-best |
|---:|---:|---:|---:|---|---:|---:|---:|
| 6h | 2022 | 175 | 11 | constant_velocity | 48.898 | 44.097 | -4.801 |
| 6h | 2023 | 454 | 17 | constant_velocity | 32.709 | 30.703 | -2.006 |
| 6h | 2024 | 130 | 13 | constant_velocity | 35.201 | 32.816 | -2.385 |
| 12h | 2022 | 163 | 9 | cliper_ridge | 119.430 | 112.527 | -6.904 |
| 12h | 2023 | 433 | 17 | cliper_ridge | 85.574 | 77.373 | -8.201 |
| 12h | 2024 | 114 | 10 | cliper_ridge | 84.941 | 85.273 | 0.332 |
| 18h | 2022 | 153 | 9 | informer | 198.129 | 204.303 | 6.174 |
| 18h | 2023 | 412 | 17 | informer | 151.086 | 140.262 | -10.824 |
| 18h | 2024 | 103 | 9 | informer | 156.935 | 148.496 | -8.439 |
| 24h | 2022 | 143 | 9 | informer | 294.711 | 289.599 | -5.112 |
| 24h | 2023 | 393 | 16 | informer | 228.560 | 208.309 | -20.251 |
| 24h | 2024 | 94 | 5 | informer | 229.966 | 233.560 | 3.594 |

## 24h Regime Snapshot

| Grouping | Group | Windows | Storms | Best | B4 | No-text | Numeric CV-residual |
|---|---|---:|---:|---:|---:|---:|---:|
| intensity_group | hurricane_ge64kt | 211 | 16 | 203.059 | 195.640 | 206.307 | 197.997 |
| intensity_group | tropical_storm_34_63kt | 291 | 27 | 271.572 | 266.667 | 276.591 | 263.733 |
| intensity_group | weak_lt34kt | 128 | 12 | 247.749 | 205.880 | 215.726 | 205.266 |
| latitude_group | high_lat_ge30 | 310 | 18 | 306.862 | 310.437 | 327.894 | 308.106 |
| latitude_group | low_lat_lt20 | 150 | 13 | 188.871 | 142.499 | 143.034 | 141.119 |
| latitude_group | mid_lat_20_30 | 170 | 16 | 177.217 | 162.484 | 167.821 | 165.394 |
| speed_group | fast | 210 | 26 | 337.549 | 333.159 | 347.695 | 333.988 |
| speed_group | medium | 210 | 23 | 196.696 | 184.025 | 194.033 | 183.651 |
| speed_group | slow | 210 | 17 | 197.110 | 174.400 | 180.328 | 171.874 |
| turn_group | high | 210 | 24 | 265.628 | 254.700 | 270.570 | 257.526 |
| turn_group | low | 210 | 26 | 220.803 | 200.261 | 207.887 | 199.562 |
| turn_group | medium | 210 | 27 | 244.924 | 236.623 | 243.599 | 232.425 |
| high_lat_recurvature | no | 506 | 30 | 227.653 | 211.208 | 217.680 | 209.509 |
| high_lat_recurvature | yes | 124 | 14 | 309.613 | 309.367 | 334.561 | 312.793 |

## B4 Dual-Gate Means by Turn Group

| Lead | Turn group | Count | Mean | P10 | P90 |
|---:|---|---:|---:|---:|---:|
| 6h | high | 759 | 0.949 | 0.912 | 0.981 |
| 6h | low | 765 | 0.955 | 0.922 | 0.982 |
| 6h | medium | 753 | 0.958 | 0.929 | 0.982 |
| 12h | high | 711 | 0.888 | 0.781 | 0.955 |
| 12h | low | 711 | 0.913 | 0.858 | 0.957 |
| 12h | medium | 708 | 0.911 | 0.851 | 0.958 |
| 18h | high | 669 | 0.837 | 0.662 | 0.941 |
| 18h | low | 669 | 0.876 | 0.770 | 0.947 |
| 18h | medium | 666 | 0.876 | 0.778 | 0.950 |
| 24h | high | 630 | 0.748 | 0.545 | 0.894 |
| 24h | low | 630 | 0.802 | 0.676 | 0.902 |
| 24h | medium | 630 | 0.788 | 0.637 | 0.910 |
