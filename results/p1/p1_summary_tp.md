# P1 size check - level: tp

reps=500, B=199, alpha=0.05, n_windows=416, seed=2026, runtime=2.9 min. Cells: rejection rate under NO change (target 5%). ** = outside the 3.5-6.5% gate.


## TP

| null | trailing / window | trailing / block L=7 | trailing / block L=20 | global / window | global / block L=7 | global / block L=20 |
|---|---|---|---|---|---|---|
| iid | 5.4% | 6.8% ** | 6.2% | 6.0% | 5.6% | 7.6% ** |
| AR1(0.5) | 100.0% ** | 38.6% ** | 18.6% ** | 100.0% ** | 37.8% ** | 18.8% ** |
| AR1(0.9) | 100.0% ** | 100.0% ** | 86.4% ** | 100.0% ** | 100.0% ** | 89.2% ** |
| slowAR1(0.98) | 100.0% ** | 100.0% ** | 97.2% ** | 100.0% ** | 100.0% ** | 98.2% ** |
| longmem(d=0.3) | 99.4% ** | 80.4% ** | 37.0% ** | 100.0% ** | 89.8% ** | 48.2% ** |
