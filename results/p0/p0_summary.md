# P0 - anchor replication and re-test

Windows: 416 x 1.2 s; W=50; k_max=10; B=9999; seed=2026

## Replication

| dim | ours | anchor |
|---|---|---|
| H0 | 1.076437 | 1.076437 |
| H1 | 1.125810 | 1.125810 |
| H2 | 1.212369 | 1.212369 |

**Exact match**

## Lower-tail p-values (Holm-adjusted across H0-H2 in brackets)

| detrend | permutation | H0 | H1 | H2 |
|---|---|---|---|---|
| trailing | window | 0.0001 [0.0003] (HFD 1.076) | 0.0001 [0.0003] (HFD 1.126) | 0.0001 [0.0003] (HFD 1.212) |
| trailing | block (L=7) | 0.0049 [0.0049] (HFD 1.076) | 0.0014 [0.0028] (HFD 1.126) | 0.0001 [0.0003] (HFD 1.212) |
| trailing | block (L=20) | 0.1733 [0.2430] (HFD 1.076) | 0.0698 [0.2094] (HFD 1.126) | 0.1215 [0.2430] (HFD 1.212) |
| global | window | 0.0001 [0.0003] (HFD 1.001) | 0.0001 [0.0003] (HFD 1.040) | 0.0001 [0.0003] (HFD 1.115) |
| global | block (L=7) | 0.0001 [0.0003] (HFD 1.001) | 0.0001 [0.0003] (HFD 1.040) | 0.0001 [0.0003] (HFD 1.115) |
| global | block (L=20) | 0.0001 [0.0003] (HFD 1.001) | 0.0007 [0.0014] (HFD 1.040) | 0.0018 [0.0018] (HFD 1.115) |

## CUSUM change-point estimate

H0: window 297 = 356 s, H1: window 294 = 353 s, H2: window 302 = 362 s
