# P3 - confounds: surrogates, summaries, embeddings (anchor EEG)


## (a) Surrogate test of the ictal change (G10) - 39 surrogates per type, pre (240, 340) s vs ictal (365, 425) s, 261 s

Delta = mean total persistence (ictal) - (pre-ictal). p_delta: does the real Delta differ from the Delta of surrogates that keep the spectra (and, for IAAFT, the amplitude distributions)? z_ict: is the ictal level itself beyond its own surrogates (nonlinearity within the seizure)?

| surrogate | normalized | dim | real Delta | surrogate Delta (mean +/- sd) | z | p_delta (2-sided) | z_ict | p_ict (2-sided) |
|---|---|---|---|---|---|---|---|---|
| PT | False | H0 | 251 | 316 +/- 1.1 | -56.9 | 0.050 | -64.0 | 0.050 |
| PT | False | H1 | 14.2 | 17.6 +/- 0.56 | -6.0 | 0.050 | -7.8 | 0.050 |
| PT | False | H2 | 0.749 | 1.8 +/- 0.14 | -7.4 | 0.050 | -8.0 | 0.050 |
| PT | True | H0 | 37.8 | 75.9 +/- 0.77 | -49.3 | 0.050 | -59.8 | 0.050 |
| PT | True | H1 | 1.53 | 2.78 +/- 0.35 | -3.5 | 0.050 | -6.7 | 0.050 |
| PT | True | H2 | 0.153 | 0.742 +/- 0.087 | -6.8 | 0.050 | -7.8 | 0.050 |
| IAAFT | False | H0 | 251 | 313 +/- 1.3 | -46.8 | 0.050 | -51.0 | 0.050 |
| IAAFT | False | H1 | 14.2 | 16.4 +/- 0.43 | -5.2 | 0.050 | -5.8 | 0.050 |
| IAAFT | False | H2 | 0.749 | 1.57 +/- 0.11 | -7.5 | 0.050 | -7.3 | 0.050 |
| IAAFT | True | H0 | 37.8 | 74.3 +/- 1 | -36.7 | 0.050 | -54.4 | 0.050 |
| IAAFT | True | H1 | 1.53 | 2.7 +/- 0.32 | -3.7 | 0.050 | -4.2 | 0.050 |
| IAAFT | True | H2 | 0.153 | 0.654 +/- 0.068 | -7.4 | 0.050 | -7.2 | 0.050 |

## (b) Persistence summaries and (c) embeddings - p-values (49 s)

sn = self-normalized CUSUM, lrv = long-run-variance CUSUM, hfd_block = HFD with global mean + block permutation (L=20). cp = CUSUM change-point estimate (onset in review: 363 s). 'normalized' = per-window amplitude normalization before persistence (corr-graph is scale-free).

| part | variant | normalized | dim | sn | lrv | hfd_block | cp (s) |
|---|---|---|---|---|---|---|---|
| b summary | tp1 | False | H0 | 0.0052 | 2.7e-06 | 0.0010 | 356 |
| b summary | tp1 | False | H1 | 0.0852 | 0.00012 | 0.0020 | 353 |
| b summary | tp1 | False | H2 | 0.0152 | 2e-05 | 0.0050 | 362 |
| b summary | tp2 | False | H0 | 0.0057 | 3.3e-06 | 0.0010 | 358 |
| b summary | tp2 | False | H1 | 0.1117 | 0.0002 | 0.0020 | 352 |
| b summary | tp2 | False | H2 | 0.0162 | 1.6e-05 | 0.0010 | 362 |
| b summary | maxpers | False | H0 | 0.0117 | 2.3e-05 | 0.0010 | 358 |
| b summary | maxpers | False | H1 | 0.1142 | 0.0012 | 0.0050 | 350 |
| b summary | maxpers | False | H2 | 0.0147 | 1.6e-05 | 0.0040 | 362 |
| b summary | entropy | False | H0 | 0.6263 | 0.17 | 0.1030 | 394 |
| b summary | entropy | False | H1 | 0.2847 | 0.027 | 0.0550 | 372 |
| b summary | entropy | False | H2 | 0.2704 | 0.16 | 0.0490 | 371 |
| b summary | landscape1 | False | H0 | 0.0332 | 0.00035 | 0.0010 | 358 |
| b summary | landscape1 | False | H1 | 0.0845 | 0.00057 | 0.0010 | 352 |
| b summary | landscape1 | False | H2 | 0.0217 | 2.9e-05 | 0.0010 | 362 |
| b summary | tp1 | True | H0 | 0.1362 | 0.0014 | 0.0320 | 358 |
| b summary | tp1 | True | H1 | 0.1402 | 0.0034 | 0.0850 | 415 |
| b summary | tp1 | True | H2 | 0.1935 | 0.09 | 0.0340 | 362 |
| b summary | tp2 | True | H0 | 0.1330 | 0.0013 | 0.0600 | 358 |
| b summary | tp2 | True | H1 | 0.0002 | 0.00024 | 0.1630 | 428 |
| b summary | tp2 | True | H2 | 0.2782 | 0.17 | 0.0240 | 280 |
| b summary | maxpers | True | H0 | 0.0770 | 0.004 | 0.2300 | 358 |
| b summary | maxpers | True | H1 | 0.0002 | 0.00088 | 0.2360 | 372 |
| b summary | maxpers | True | H2 | 0.4761 | 0.37 | 0.0510 | 280 |
| b summary | entropy | True | H0 | 0.6263 | 0.17 | 0.0930 | 394 |
| b summary | entropy | True | H1 | 0.2847 | 0.027 | 0.0620 | 372 |
| b summary | entropy | True | H2 | 0.2704 | 0.16 | 0.0530 | 371 |
| b summary | landscape1 | True | H0 | 0.0740 | 0.0046 | 0.1880 | 358 |
| b summary | landscape1 | True | H1 | 0.0002 | 0.00095 | 0.1960 | 372 |
| b summary | landscape1 | True | H2 | 0.2989 | 0.16 | 0.0280 | 280 |
| c embedding | corr-graph (channels as points) | True | H0 | 0.2389 | 0.071 | 0.0530 | 359 |
| c embedding | corr-graph (channels as points) | True | H1 | 0.2507 | 0.0065 | 0.0430 | 352 |
| c embedding | delay embedding of PC1 (d=3, tau=50 ms) | False | H0 | 0.0550 | 5.6e-05 | 0.0010 | 353 |
| c embedding | delay embedding of PC1 (d=3, tau=50 ms) | False | H1 | 0.1635 | 0.00052 | 0.0010 | 353 |
| c embedding | delay embedding of PC1 (d=3, tau=50 ms) | True | H0 | 0.2154 | 0.0026 | 0.0410 | 416 |
| c embedding | delay embedding of PC1 (d=3, tau=50 ms) | True | H1 | 0.2724 | 0.00095 | 0.0960 | 416 |
