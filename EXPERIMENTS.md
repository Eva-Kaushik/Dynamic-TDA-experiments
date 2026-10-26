# Experiments (terminal)

Re-implementation and stress test of El-Yaagoubi, Chung & Ombao (2024),
*Front. Neuroinform.* 18:1387400. The original notebooks are untouched; the
experiments live in `experiments/`, tables in `results/`, figures in `figures/`.

| phase | script | question | gaps |
|---|---|---|---|
| P0 | `p0_replicate.py` | Do we reproduce the anchor? Does the seizure result survive block permutation, global-mean centring, Holm, amplitude normalization? | G14, G2, G3, G5 |
| P1 | `p1_size.py` | Does the test keep its 5% false-alarm rate when nothing changes? | G1, G2, G3 |
| P2 | `p2_power.py` | Size and power of better statistics (self-normalized CUSUM, LRV CUSUM, block-perm CUSUM) vs non-topological baselines | G4, G6, G7 |
| P3 | `p3_confounds.py` | Is the ictal change beyond spectrum/amplitude (multivariate PT and IAAFT surrogates)? Other summaries and embeddings | G9, G10, G11 |
| P4 | `p4_sensitivity.py` | Dependence on window length, k_max, W, block length | G12 |
| P5 | `p5_localization.py` | Change time with bootstrap CI; binary segmentation | G8 |
| P6 | `p6_chbmit.py` | External validation on a CHB-MIT subset (downloads ~400 MB) | G13 |

Shared code: `dtda.py` (pipeline and all tests), `figs.py` (plots; `python experiments/figs.py` rebuilds P1 figures).

## Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run everything

```bash
bash run_all.sh            # P0-P6, ~30-40 min on an M-series Mac
bash run_all.sh --no-p6    # skip the CHB-MIT download
```

Or one phase at a time (each script takes `--help`; `--quick` exists for P1/P2):

```bash
python experiments/p2_power.py
python experiments/p3_confounds.py
python experiments/p4_sensitivity.py
python experiments/p5_localization.py
python experiments/p6_chbmit.py
```

Seeds are fixed (`--seed 2026`), so reruns give the same numbers.

## Where things go

- `results/pX/*.md` readable summaries, `results/pX/*.csv` full tables
- `figures/pX/*.png` all figures
- `results/_cache/` and `data/` (CHB-MIT downloads) are git-ignored and rebuilt on demand

## Notes on the anchor code

- Channels: ch01-ch21 minus SP1 (11), SP2 (12), Cz (20) = 18; ch22-ch31 unused.
- Non-overlapping 120-sample windows (1.2 s at 100 Hz) -> 416 windows; last partial window dropped.
- Each window's point cloud is its 120 time points in R^18 (no delay embedding).
- The notebook builds the permutation null but never computes a p-value; we use the lower tail.
- The notebook sets no random seed.
