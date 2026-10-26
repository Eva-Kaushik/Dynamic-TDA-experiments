# P5 - localization on the anchor EEG

Onset reference: 363 s. CI = 95% moving-block bootstrap (L=20, B=999). Binary segmentation: self-normalized CUSUM, alpha=0.05, min segment 30 s.

| normalized | dim | change (s) | 95% CI (s) | error vs onset (s) | onset in CI | binary segmentation (s) |
|---|---|---|---|---|---|---|
| False | H0 | 356 | [293, 360] | -7 | False | 356;438 |
| False | H1 | 353 | [240, 384] | -10 | True | - |
| False | H2 | 362 | [277, 367] | -1 | True | 362;438 |
| True | H0 | 358 | [247, 371] | -5 | True | - |
| True | H1 | 415 | [263, 418] | +52 | True | - |
| True | H2 | 362 | [166, 397] | -1 | True | - |
