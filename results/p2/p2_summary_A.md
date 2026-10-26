# P2 - size and power of change statistics


## A. Size under no change (target 5%; reps=500, 91 s)

| null | anchor (HFD, window perm) | HFD, block perm L=20 | CUSUM, block perm L=20 | self-normalized CUSUM | LRV CUSUM (Bartlett) |
|---|---|---|---|---|---|
| iid | 6.2% | 7.6% | 6.4% | 6.0% | 5.8% |
| AR1(0.17) EEG-like | 43.6% | 6.4% | 4.2% | 6.2% | 3.6% |
| AR1(0.5) | 99.8% | 18.0% | 7.8% | 6.6% | 6.4% |
| AR1(0.9) | 100.0% | 87.0% | 16.4% | 6.8% | 35.6% |
| longmem(d=0.3) | 99.6% | 47.6% | 32.2% | 16.2% | 46.8% |
