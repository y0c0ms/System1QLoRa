# 9-hour loop summary

## Timeline

```
[01:35:00] START p0_smoke
[01:37:29] FAIL p0_smoke rc=3 (see results/loop9h/p0_smoke.log)
[01:37:29] ABORT: smoke gate failed - nothing else runs
[01:41:35] START p0_smoke
[01:45:40] DONE p0_smoke
[01:45:40] START p1_screen
[10:13:04] START p0_smoke
[10:18:00] DONE p0_smoke
[10:18:00] START p1_screen
[10:38:17] DONE p1_screen
[10:38:17] START p2_gate
[10:38:17] DONE p2_gate
[10:38:17] START p3_main
[11:54:00] DONE p3_main
[11:54:00] START p4_eval
[12:20:12] DONE p4_eval
[12:20:12] START p5_deploy
[18:54:34] SKIP p0_smoke (done)
[18:54:34] SKIP p1_screen (done)
[18:54:34] SKIP p2_gate (done)
[18:54:34] SKIP p3_main (done)
[18:54:34] SKIP p4_eval (done)
[18:54:34] START p5_deploy
[19:11:49] DONE p5_deploy
[19:11:49] START p6_summary
[19:11:49] DONE p6_summary
[19:11:49] LOOP COMPLETE
```

## P1 - regime screen (H1a quantization, H1b adapter capacity)

Identical corpus/steps/seed; only precision and LoRA rank differ. Dev sets only.

Rule: winner = argmax mean dev accuracy over sni.val.orig, reflex.val.orig, kevsuite.val.orig

| arm | dev macro | sni.val.orig | reflex.val.orig | kevsuite.val.orig |
|---|---|---|---|---|
| p1_nf4_r16 | 0.5275 | 0.5517 | 0.4267 | 0.6042 |
| p1_bf16_r16 | 0.6005 | 0.6304 | 0.4967 | 0.6745 |
| p1_bf16_r64 | 0.4311 | 0.4813 | 0.4450 | 0.3671 |

**Winner: `p1_bf16_r16`**

<details><summary>P1 full report (paired bootstrap CIs)</summary>

# Decision-model evaluation report

Global temperature per model (fit on pooled val, applied to all sets): `p1_nf4_r16` T=1.05, `p1_bf16_r16` T=1.15, `p1_bf16_r64` T=1.00, `p1_distill_ckpt500` T=1.50

## Accuracy / ECE (global T)

| set | p1_nf4_r16 | p1_bf16_r16 | p1_bf16_r64 | p1_distill_ckpt500 |
|---|---|---|---|---|
| bfcl.val.orig | 0.975 / 0.130 | 0.975 / 0.086 | 0.508 / 0.120 | 0.942 / 0.064 |
| bfcl_irr.val.orig | 0.950 / 0.160 | 0.850 / 0.157 | 0.925 / 0.438 | 0.738 / 0.103 |
| kevsuite.val.orig | 0.604 / 0.043 | 0.674 / 0.025 | 0.367 / 0.048 | 0.728 / 0.033 |
| reflex.val.orig | 0.427 / 0.039 | 0.497 / 0.045 | 0.445 / 0.074 | 0.515 / 0.032 |
| sni.val.orig | 0.552 / 0.022 | 0.630 / 0.022 | 0.481 / 0.013 | 0.616 / 0.024 |

## Paired bootstrap vs `p1_nf4_r16` (accuracy difference, 95% CI, 2000 resamples)

| set | p1_bf16_r16 | p1_bf16_r64 | p1_distill_ckpt500 |
|---|---|---|---|
| bfcl.val.orig | +0.000 [-0.025, +0.025] | **-0.467 [-0.558, -0.375]** | -0.033 [-0.083, +0.008] |
| bfcl_irr.val.orig | **-0.100 [-0.175, -0.025]** | -0.025 [-0.100, +0.050] | **-0.212 [-0.300, -0.125]** |
| kevsuite.val.orig | **+0.070 [+0.044, +0.099]** | **-0.237 [-0.270, -0.205]** | **+0.124 [+0.094, +0.153]** |
| reflex.val.orig | **+0.070 [+0.030, +0.112]** | +0.018 [-0.025, +0.063] | **+0.088 [+0.042, +0.130]** |
| sni.val.orig | **+0.079 [+0.062, +0.096]** | **-0.070 [-0.090, -0.051]** | **+0.064 [+0.046, +0.081]** |

</details>

## P2 gate (H2 base) and P3 plan

- Qwen3.5-0.8B gate: **DEFERRED**
- P3 plan: `{"winner": "p1_bf16_r16", "precision": "bf16", "r": 16, "steps": 2000, "gate": "DEFERRED", "sps_m1_1024": 2.5131028334299725, "m2_cost_ratio": 2.0, "deadline": 1790185384}`

### Training runs as executed

| run | regime | steps done/planned | steady s/step | wall min | stopped at deadline |
|---|---|---|---|---|---|
| P1 p1_nf4_r16 | nf4 r16 | 500/500 | - | 16.5 | no |
| P1 p1_bf16_r16 | bf16 r16 | 500/500 | 1.77 | 14.8 | no |
| P1 p1_bf16_r64 | bf16 r64 | 500/500 | 2.25 | 15.0 | no |
| P3 twin Qwen3-0.6B | bf16 r16 | 20/20 | 2.51 | 0.8 | no |
| P3 M1 Qwen3-0.6B | bf16 r16 | 2000/2000 | 2.24 | 74.6 | no |

## P4 - locked final evaluation

Final model by the pre-registered dev rule: **`p3_m1`**

### JevBench public items (231) - ours vs the published board, same items

| system | all | easy | standard | hard | source |
|---|---|---|---|---|---|
| **q06_final_nf4** | 0.619 (perm-avg 0.632) | 0.979 | 0.694 | 0.414 | measured here |
| **p3_m1** | 0.632 (perm-avg 0.667) | 1.000 | 0.750 | 0.396 | measured here |
| **q06_final_bf16inf** | 0.602 (perm-avg 0.619) | 1.000 | 0.694 | 0.369 | measured here |
| **base_q06_zeroshot** | 0.476 (perm-avg 0.541) | 0.854 | 0.431 | 0.342 | measured here |
| Jev 1.13.0 (TypeSafe AI) | 0.866 | 1.000 | 0.986 | 0.730 | JevBench board |
| SemIf, formerly OpenJev (Qwen3.5-4B, TheoLee | 0.810 | 1.000 | 0.986 | 0.613 | JevBench board |
| decider-2b (Mapika) | 0.710 | 1.000 | 0.847 | 0.495 | JevBench board |
| kev 0.6B (research preview) | 0.667 | 1.000 | 0.806 | 0.432 | JevBench board |
| Open-Jev 2B (Zefan Cai) | 0.645 | 1.000 | 0.764 | 0.414 | JevBench board |
| Laya (Convai Innovations, ModernBERT-large 4 | 0.584 | 0.958 | 0.694 | 0.351 | JevBench board |
| kev 0.5B | 0.494 | 0.958 | 0.486 | 0.297 | JevBench board |
| Dohnuts-0.1.0-0.8B | 0.658 | - | - | - | self-reported (model card) |

Board rows: outcomes published in fstandhartinger/jevbench `jevbench-v1.2-per-task.json`
(public items only; the board's composite score also uses 303 held-out items we cannot run).

### Full final report (global temperature, order sensitivity, paired bootstrap vs shipped model)

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


## P5 - small-hardware deployment (CPU only, no GPU, offline)

llama.cpp with `-ngl 0`, one request at a time, the first 5 decisions discarded, thread count and loadavg recorded per report. `GPU (same rows)` is the GPU evaluation of the same model recomputed on exactly the rows the CPU scored.

| build | size | set | n | GPU (same rows) | CPU | delta | agree | p50 | p90 |
|---|---|---|---|---|---|---|---|---|---|
| q4_k_m | 378 MB | jevbench.test | 231 | 0.632 | 0.632 | +0.000 | 0.876 | 189 ms | 5885 ms |
| q4_k_m | 378 MB | sni.val | 300 | 0.517 | 0.520 | +0.003 | 0.843 | 316 ms | 377 ms |
| q4_k_m | 378 MB | reflex.val | 150 | 0.587 | 0.513 | -0.073 | 0.853 | 194 ms | 744 ms |
| q8_0 | 610 MB | jevbench.test | 231 | 0.632 | 0.628 | -0.004 | 0.973 | 231 ms | 6453 ms |
| q8_0 | 610 MB | sni.val | 300 | 0.517 | 0.517 | +0.000 | 0.980 | 360 ms | 433 ms |
| q8_0 | 610 MB | reflex.val | 150 | 0.587 | 0.540 | -0.047 | 0.913 | 228 ms | 779 ms |

Conditions: AMD Ryzen 5 7600X 6-Core Processor, 6 threads, loadavg 11.46 11.53 6.91 at start and 12.81 13.10 9.60 at end (the run saturates the CPU itself, so these are loaded-machine figures). `agree` is the fraction of rows where CPU and GPU choose the same option; a row whose option letters are not all in the returned top-N is counted as `missing` and never renormalised over the letters that came back.

## Pre-registered caveats

- P1 screens at 500 steps (~0.3 epoch); a regime that learns faster early can look better than it ends.
- One seed per arm; CIs cover row sampling, not seed variance.
- M1/M2 train on kev public sources, so kevsuite is in-distribution for them and not for q06_final.
- M1/M2 differ from q06_final in several ways at once (precision/rank, corpus, max_len 1024, permutation);
  only P1 (regime) and M1-vs-M2 (base) are single-variable comparisons.
- If M2 stopped at the training deadline (table above), M1 vs M2 is not an equal-steps comparison.
- The JevBench intelligence proxy renormalizes over the public tiers (no judge items are public).
