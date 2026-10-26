
## B. Rejection rate with a change at window 249 (reps=100, 637 s)

lock s = fraction of channels that phase-lock (topology change; s=0 is no change). amplitude a = all channels scaled by a (nuisance change). Higher is better for lock s>0; for amplitude a topology test should stay near 5% after normalization.


### HFD, block perm L=20

| series | lock 0.0 | lock 0.25 | lock 0.5 | lock 1.0 | amplitude 1.25 | amplitude 1.5 | amplitude 2.0 |
|---|---|---|---|---|---|---|---|
| TP H0 raw | 5% | 7% | 6% | 58% | 100% | 100% | 100% |
| TP H0 norm | 4% | 3% | 8% | 84% | 9% | 2% | 6% |
| TP H1 raw | 3% | 3% | 7% | 23% | 100% | 100% | 100% |
| TP H1 norm | 4% | 5% | 8% | 26% | 2% | 6% | 5% |
| amp-baseline | 5% | 5% | 6% | 3% | 100% | 100% | 100% |
| corr-baseline | 5% | 8% | 19% | 26% | 8% | 13% | 10% |
| pca2-baseline | 1% | 8% | 15% | 100% | 4% | 7% | 7% |

### self-normalized CUSUM

| series | lock 0.0 | lock 0.25 | lock 0.5 | lock 1.0 | amplitude 1.25 | amplitude 1.5 | amplitude 2.0 |
|---|---|---|---|---|---|---|---|
| TP H0 raw | 4% | 4% | 21% | 99% | 100% | 100% | 100% |
| TP H0 norm | 4% | 2% | 34% | 100% | 4% | 5% | 7% |
| TP H1 raw | 8% | 3% | 19% | 91% | 100% | 100% | 100% |
| TP H1 norm | 7% | 4% | 22% | 94% | 8% | 2% | 5% |
| amp-baseline | 5% | 4% | 5% | 6% | 100% | 100% | 100% |
| corr-baseline | 4% | 11% | 17% | 60% | 3% | 4% | 6% |
| pca2-baseline | 3% | 6% | 72% | 100% | 1% | 4% | 7% |

### LRV CUSUM (Bartlett)

| series | lock 0.0 | lock 0.25 | lock 0.5 | lock 1.0 | amplitude 1.25 | amplitude 1.5 | amplitude 2.0 |
|---|---|---|---|---|---|---|---|
| TP H0 raw | 3% | 4% | 25% | 100% | 100% | 100% | 100% |
| TP H0 norm | 2% | 2% | 46% | 100% | 2% | 4% | 3% |
| TP H1 raw | 4% | 5% | 21% | 98% | 100% | 100% | 100% |
| TP H1 norm | 5% | 5% | 22% | 99% | 5% | 0% | 5% |
| amp-baseline | 7% | 4% | 4% | 3% | 100% | 100% | 100% |
| corr-baseline | 2% | 17% | 23% | 76% | 2% | 2% | 6% |
| pca2-baseline | 2% | 3% | 80% | 100% | 2% | 1% | 3% |

### anchor (HFD, window perm)

| series | lock 0.0 | lock 0.25 | lock 0.5 | lock 1.0 | amplitude 1.25 | amplitude 1.5 | amplitude 2.0 |
|---|---|---|---|---|---|---|---|
| TP H0 raw | 5% | 7% | 5% | 21% | 100% | 100% | 100% |
| TP H0 norm | 4% | 3% | 10% | 23% | 7% | 7% | 8% |
| TP H1 raw | 4% | 2% | 9% | 6% | 79% | 99% | 100% |
| TP H1 norm | 5% | 4% | 9% | 6% | 6% | 6% | 3% |
| amp-baseline | 6% | 7% | 5% | 9% | 100% | 100% | 100% |
| corr-baseline | 73% | 72% | 62% | 55% | 66% | 72% | 68% |
| pca2-baseline | 2% | 4% | 7% | 61% | 6% | 5% | 3% |

Median localization error (windows of 1.2 s) of the CUSUM argmax when the self-normalized CUSUM rejects, lock s=1:

- TP H0 raw: 4.0
- TP H0 norm: 2.0
- TP H1 raw: 6.0
- TP H1 norm: 6.0
- amp-baseline: 61.0
- corr-baseline: 24.5
- pca2-baseline: 1.0
