# P0 - anchor replication and re-test (per-window amplitude normalization)

Windows: 416 x 1.2 s; W=50; k_max=10; B=9999; seed=2026

## Replication

| dim | ours | anchor |
|---|---|---|
| H0 | 1.230832 | 1.076437 |
| H1 | 1.323584 | 1.125810 |
| H2 | 1.339348 | 1.212369 |

**Not comparable (per-window normalization)**

## Lower-tail p-values (Holm-adjusted across H0-H2 in brackets)

| detrend | permutation | H0 | H1 | H2 |
|---|---|---|---|---|
| trailing | window | 0.0001 [0.0003] (HFD 1.231) | 0.0001 [0.0003] (HFD 1.324) | 0.0001 [0.0003] (HFD 1.339) |
| trailing | block (L=7) | 0.0058 [0.0174] (HFD 1.231) | 0.0256 [0.0334] (HFD 1.324) | 0.0167 [0.0334] (HFD 1.339) |
| trailing | block (L=20) | 0.1072 [0.2144] (HFD 1.231) | 0.0662 [0.1986] (HFD 1.324) | 0.1366 [0.2144] (HFD 1.339) |
| global | window | 0.0001 [0.0003] (HFD 1.200) | 0.0001 [0.0003] (HFD 1.269) | 0.0001 [0.0003] (HFD 1.344) |
| global | block (L=7) | 0.0001 [0.0003] (HFD 1.200) | 0.0016 [0.0032] (HFD 1.269) | 0.0082 [0.0082] (HFD 1.344) |
| global | block (L=20) | 0.0315 [0.0924] (HFD 1.200) | 0.0892 [0.0924] (HFD 1.269) | 0.0308 [0.0924] (HFD 1.344) |

## CUSUM change-point estimate

H0: window 298 = 358 s, H1: window 346 = 415 s, H2: window 302 = 362 s
