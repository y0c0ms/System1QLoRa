# Decision-model evaluation report

Global temperature per model (fit on pooled val, applied to all sets): `q06_final_nf4` T=2.15, `p3_m1` T=2.00, `q06_final_bf16inf` T=2.20, `base_q06_zeroshot` T=1.00

## Accuracy / ECE (global T)

| set | q06_final_nf4 | p3_m1 | q06_final_bf16inf | base_q06_zeroshot |
|---|---|---|---|---|
| bfcl.test.orig | 0.887 / 0.125 | 0.907 / 0.146 | 0.887 / 0.138 | - |
| bfcl.test.permavg | 0.912 / 0.160 (agree 0.88) | 0.935 / 0.178 (agree 0.92) | 0.930 / 0.186 (agree 0.89) | - |
| bfcl.test.rev | 0.912 / 0.145 | 0.902 / 0.135 | 0.912 / 0.153 | - |
| bfcl.val.orig | 0.975 / 0.069 | 0.983 / 0.081 | 0.983 / 0.079 | - |
| bfcl.val.permavg | 0.975 / 0.095 (agree 0.97) | 0.983 / 0.093 (agree 0.98) | 0.992 / 0.095 (agree 0.97) | - |
| bfcl.val.rev | 0.958 / 0.075 | 0.967 / 0.089 | 0.967 / 0.103 | - |
| bfcl_irr.test.orig | 0.960 / 0.035 | 0.836 / 0.049 | 0.960 / 0.056 | - |
| bfcl_irr.test.permavg | 0.965 / 0.055 (agree 0.98) | 0.846 / 0.053 (agree 0.95) | 0.965 / 0.075 (agree 0.97) | - |
| bfcl_irr.test.rev | 0.967 / 0.066 | 0.838 / 0.045 | 0.965 / 0.087 | - |
| bfcl_irr.val.orig | 0.963 / 0.059 | 0.812 / 0.083 | 0.938 / 0.072 | - |
| bfcl_irr.val.permavg | 0.975 / 0.065 (agree 0.97) | 0.825 / 0.088 (agree 0.95) | 0.963 / 0.079 (agree 0.97) | - |
| bfcl_irr.val.rev | 0.975 / 0.084 | 0.800 / 0.064 | 0.963 / 0.106 | - |
| jevbench.test.orig | 0.619 / 0.110 | 0.632 / 0.041 | 0.602 / 0.059 | 0.476 / 0.317 |
| jevbench.test.permavg | 0.632 / 0.105 (agree 0.84) | 0.667 / 0.075 (agree 0.87) | 0.619 / 0.109 (agree 0.84) | 0.541 / 0.129 (agree 0.41) |
| jevbench.test.rev | 0.619 / 0.084 | 0.675 / 0.076 | 0.615 / 0.095 | 0.502 / 0.256 |
| kevsuite.test.orig | 0.665 / 0.035 | 0.776 / 0.049 | 0.671 / 0.032 | - |
| kevsuite.test.permavg | 0.691 / 0.059 (agree 0.80) | 0.790 / 0.053 (agree 0.93) | 0.674 / 0.036 (agree 0.85) | - |
| kevsuite.test.rev | 0.671 / 0.039 | 0.792 / 0.060 | 0.671 / 0.033 | - |
| kevsuite.val.orig | 0.674 / 0.049 | 0.815 / 0.048 | 0.684 / 0.040 | - |
| kevsuite.val.permavg | 0.707 / 0.065 (agree 0.82) | 0.817 / 0.059 (agree 0.93) | 0.691 / 0.048 (agree 0.85) | - |
| kevsuite.val.rev | 0.698 / 0.047 | 0.818 / 0.059 | 0.696 / 0.049 | - |
| oadk.test.orig | 0.750 / 0.324 | 0.812 / 0.334 | 0.625 / 0.215 | - |
| oadk.test.permavg | 0.688 / 0.303 (agree 0.88) | 0.688 / 0.181 (agree 0.69) | 0.688 / 0.255 (agree 0.88) | - |
| oadk.test.rev | 0.688 / 0.305 | 0.625 / 0.384 | 0.750 / 0.308 | - |
| oadk.val.orig | 0.625 / 0.205 | 0.625 / 0.318 | 0.562 / 0.253 | - |
| oadk.val.permavg | 0.688 / 0.290 (agree 0.62) | 0.688 / 0.200 (agree 0.62) | 0.625 / 0.201 (agree 0.88) | - |
| oadk.val.rev | 0.750 / 0.303 | 0.688 / 0.278 | 0.562 / 0.155 | - |
| reflex.test.orig | 0.552 / 0.057 | 0.562 / 0.052 | 0.570 / 0.060 | - |
| reflex.test.permavg | 0.565 / 0.074 (agree 0.82) | 0.565 / 0.063 (agree 0.67) | 0.568 / 0.065 (agree 0.76) | - |
| reflex.test.rev | 0.563 / 0.085 | 0.545 / 0.076 | 0.550 / 0.053 | - |
| reflex.val.orig | 0.580 / 0.085 | 0.568 / 0.096 | 0.540 / 0.046 | - |
| reflex.val.permavg | 0.570 / 0.090 (agree 0.83) | 0.548 / 0.070 (agree 0.69) | 0.537 / 0.045 (agree 0.80) | - |
| reflex.val.rev | 0.548 / 0.069 | 0.532 / 0.071 | 0.543 / 0.068 | - |
| sni.test.orig | 0.616 / 0.062 | 0.628 / 0.073 | 0.627 / 0.058 | - |
| sni.test.permavg | 0.623 / 0.053 (agree 0.87) | 0.631 / 0.070 (agree 0.89) | 0.625 / 0.060 (agree 0.91) | - |
| sni.test.rev | 0.621 / 0.054 | 0.635 / 0.080 | 0.619 / 0.053 | - |
| sni.val.orig | 0.632 / 0.055 | 0.642 / 0.063 | 0.639 / 0.044 | - |
| sni.val.permavg | 0.638 / 0.045 (agree 0.93) | 0.651 / 0.057 (agree 0.93) | 0.640 / 0.040 (agree 0.93) | - |
| sni.val.rev | 0.637 / 0.047 | 0.654 / 0.057 | 0.640 / 0.041 | - |

## Paired bootstrap vs `q06_final_nf4` (accuracy difference, 95% CI, 2000 resamples)

| set | p3_m1 | q06_final_bf16inf | base_q06_zeroshot |
|---|---|---|---|
| bfcl.test.orig | +0.020 [-0.008, +0.045] | +0.000 [-0.020, +0.020] | - |
| bfcl.test.permavg | **+0.023 [+0.003, +0.043]** | **+0.018 [+0.003, +0.033]** | - |
| bfcl.test.rev | -0.010 [-0.035, +0.013] | +0.000 [-0.015, +0.018] | - |
| bfcl.val.orig | +0.008 [-0.017, +0.042] | +0.008 [-0.017, +0.042] | - |
| bfcl.val.permavg | +0.008 [-0.017, +0.042] | +0.017 [+0.000, +0.042] | - |
| bfcl.val.rev | +0.008 [-0.025, +0.050] | +0.008 [+0.000, +0.025] | - |
| bfcl_irr.test.orig | **-0.124 [-0.162, -0.088]** | +0.000 [-0.013, +0.013] | - |
| bfcl_irr.test.permavg | **-0.119 [-0.157, -0.086]** | +0.000 [-0.008, +0.008] | - |
| bfcl_irr.test.rev | **-0.129 [-0.164, -0.093]** | -0.003 [-0.010, +0.005] | - |
| bfcl_irr.val.orig | **-0.150 [-0.237, -0.062]** | -0.025 [-0.062, +0.000] | - |
| bfcl_irr.val.permavg | **-0.150 [-0.237, -0.075]** | -0.013 [-0.037, +0.000] | - |
| bfcl_irr.val.rev | **-0.175 [-0.263, -0.100]** | -0.013 [-0.037, +0.000] | - |
| jevbench.test.orig | +0.013 [-0.035, +0.061] | -0.017 [-0.065, +0.030] | **-0.143 [-0.216, -0.065]** |
| jevbench.test.permavg | +0.035 [-0.022, +0.082] | -0.013 [-0.052, +0.026] | **-0.091 [-0.165, -0.017]** |
| jevbench.test.rev | **+0.056 [+0.009, +0.108]** | -0.004 [-0.039, +0.030] | **-0.117 [-0.199, -0.035]** |
| kevsuite.test.orig | **+0.111 [+0.084, +0.137]** | +0.006 [-0.010, +0.023] | - |
| kevsuite.test.permavg | **+0.099 [+0.073, +0.125]** | -0.017 [-0.035, +0.002] | - |
| kevsuite.test.rev | **+0.121 [+0.094, +0.146]** | +0.001 [-0.019, +0.020] | - |
| kevsuite.val.orig | **+0.141 [+0.116, +0.169]** | +0.011 [-0.004, +0.026] | - |
| kevsuite.val.permavg | **+0.110 [+0.085, +0.135]** | -0.016 [-0.034, +0.002] | - |
| kevsuite.val.rev | **+0.120 [+0.094, +0.146]** | -0.002 [-0.020, +0.019] | - |
| oadk.test.orig | +0.062 [+0.000, +0.188] | -0.125 [-0.375, +0.125] | - |
| oadk.test.permavg | +0.000 [-0.250, +0.250] | +0.000 [-0.188, +0.188] | - |
| oadk.test.rev | -0.062 [-0.312, +0.188] | +0.062 [+0.000, +0.188] | - |
| oadk.val.orig | +0.000 [-0.312, +0.312] | -0.062 [-0.250, +0.125] | - |
| oadk.val.permavg | +0.000 [-0.250, +0.250] | -0.062 [-0.312, +0.125] | - |
| oadk.val.rev | -0.062 [-0.312, +0.188] | -0.188 [-0.438, +0.062] | - |
| reflex.test.orig | +0.010 [-0.027, +0.045] | +0.018 [-0.012, +0.047] | - |
| reflex.test.permavg | +0.000 [-0.045, +0.043] | +0.003 [-0.022, +0.028] | - |
| reflex.test.rev | -0.018 [-0.065, +0.027] | -0.013 [-0.048, +0.020] | - |
| reflex.val.orig | -0.012 [-0.047, +0.022] | **-0.040 [-0.067, -0.013]** | - |
| reflex.val.permavg | -0.022 [-0.067, +0.023] | **-0.033 [-0.060, -0.007]** | - |
| reflex.val.rev | -0.017 [-0.065, +0.032] | -0.005 [-0.035, +0.027] | - |
| sni.test.orig | +0.012 [-0.007, +0.031] | +0.011 [-0.005, +0.025] | - |
| sni.test.permavg | +0.009 [-0.011, +0.029] | +0.003 [-0.012, +0.018] | - |
| sni.test.rev | +0.015 [-0.006, +0.035] | -0.001 [-0.016, +0.015] | - |
| sni.val.orig | +0.010 [-0.002, +0.021] | +0.006 [-0.003, +0.016] | - |
| sni.val.permavg | **+0.013 [+0.001, +0.024]** | +0.001 [-0.008, +0.010] | - |
| sni.val.rev | **+0.017 [+0.005, +0.029]** | +0.003 [-0.006, +0.012] | - |

### JevBench public tiers - `q06_final_nf4` `jevbench.test.orig` (intelligence proxy 45.4)

| tier | n | acc | chance | chance-corrected |
|---|---|---|---|---|
| easy | 48 | 0.979 | 0.284 | 0.971 |
| hard | 111 | 0.414 | 0.336 | 0.118 |
| standard | 72 | 0.694 | 0.311 | 0.556 |

### JevBench public tiers - `q06_final_nf4` `jevbench.test.rev` (intelligence proxy 45.6)

| tier | n | acc | chance | chance-corrected |
|---|---|---|---|---|
| easy | 48 | 1.000 | 0.284 | 1.000 |
| hard | 111 | 0.396 | 0.336 | 0.091 |
| standard | 72 | 0.708 | 0.311 | 0.577 |

### JevBench public tiers - `p3_m1` `jevbench.test.orig` (intelligence proxy 48.0)

| tier | n | acc | chance | chance-corrected |
|---|---|---|---|---|
| easy | 48 | 1.000 | 0.284 | 1.000 |
| hard | 111 | 0.396 | 0.336 | 0.091 |
| standard | 72 | 0.750 | 0.311 | 0.637 |

### JevBench public tiers - `p3_m1` `jevbench.test.rev` (intelligence proxy 54.7)

| tier | n | acc | chance | chance-corrected |
|---|---|---|---|---|
| easy | 48 | 1.000 | 0.284 | 1.000 |
| hard | 111 | 0.441 | 0.336 | 0.159 |
| standard | 72 | 0.819 | 0.311 | 0.738 |

### JevBench public tiers - `q06_final_bf16inf` `jevbench.test.orig` (intelligence proxy 43.2)

| tier | n | acc | chance | chance-corrected |
|---|---|---|---|---|
| easy | 48 | 1.000 | 0.284 | 1.000 |
| hard | 111 | 0.369 | 0.336 | 0.050 |
| standard | 72 | 0.694 | 0.311 | 0.556 |

### JevBench public tiers - `q06_final_bf16inf` `jevbench.test.rev` (intelligence proxy 44.6)

| tier | n | acc | chance | chance-corrected |
|---|---|---|---|---|
| easy | 48 | 1.000 | 0.284 | 1.000 |
| hard | 111 | 0.405 | 0.336 | 0.104 |
| standard | 72 | 0.681 | 0.311 | 0.536 |

### JevBench public tiers - `base_q06_zeroshot` `jevbench.test.orig` (intelligence proxy 22.6)

| tier | n | acc | chance | chance-corrected |
|---|---|---|---|---|
| easy | 48 | 0.854 | 0.284 | 0.796 |
| hard | 111 | 0.342 | 0.336 | 0.009 |
| standard | 72 | 0.431 | 0.311 | 0.173 |

### JevBench public tiers - `base_q06_zeroshot` `jevbench.test.rev` (intelligence proxy 26.2)

| tier | n | acc | chance | chance-corrected |
|---|---|---|---|---|
| easy | 48 | 0.896 | 0.284 | 0.854 |
| hard | 111 | 0.369 | 0.336 | 0.050 |
| standard | 72 | 0.444 | 0.311 | 0.194 |
