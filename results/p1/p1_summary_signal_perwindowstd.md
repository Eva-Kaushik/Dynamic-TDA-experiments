# P1 size check - level: signal_perwindowstd

reps=100, B=199, alpha=0.05, n_windows=416, seed=2026, runtime=2.2 min. Cells: rejection rate under NO change (target 5%). ** = outside the 3.5-6.5% gate.


## H0

| null | trailing / window | trailing / block L=7 | trailing / block L=20 | global / window | global / block L=7 | global / block L=20 |
|---|---|---|---|---|---|---|
| iid | 4.0% | 4.0% | 7.0% ** | 7.0% ** | 4.0% | 9.0% ** |
| VAR1 | 8.0% ** | 3.0% ** | 8.0% ** | 5.0% | 2.0% ** | 7.0% ** |
| stoch.vol (VAR1 x slow amplitude) | 16.0% ** | 8.0% ** | 4.0% | 18.0% ** | 7.0% ** | 4.0% |

## H1

| null | trailing / window | trailing / block L=7 | trailing / block L=20 | global / window | global / block L=7 | global / block L=20 |
|---|---|---|---|---|---|---|
| iid | 8.0% ** | 5.0% | 6.0% | 8.0% ** | 6.0% | 6.0% |
| VAR1 | 8.0% ** | 5.0% | 5.0% | 7.0% ** | 4.0% | 5.0% |
| stoch.vol (VAR1 x slow amplitude) | 9.0% ** | 7.0% ** | 5.0% | 10.0% ** | 7.0% ** | 2.0% ** |

## any (Holm)

| null | trailing / window | trailing / block L=7 | trailing / block L=20 | global / window | global / block L=7 | global / block L=20 |
|---|---|---|---|---|---|---|
| iid | 7.0% ** | 4.0% | 6.0% | 4.0% | 5.0% | 7.0% ** |
| VAR1 | 7.0% ** | 3.0% ** | 6.0% | 6.0% | 2.0% ** | 5.0% |
| stoch.vol (VAR1 x slow amplitude) | 16.0% ** | 7.0% ** | 3.0% ** | 14.0% ** | 7.0% ** | 3.0% ** |
