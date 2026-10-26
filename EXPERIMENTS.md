# Experiments (terminal)

Re-implementation and stress test of El-Yaagoubi, Chung & Ombao (2024),
*Front. Neuroinform.* 18:1387400. The original notebooks are untouched; the
experiments live in `experiments/` and write to `results/`.

| file | what it does |
|---|---|
| `experiments/dtda.py` | core pipeline as functions: windows -> Rips PH -> total persistence -> detrend -> CUSUM -> HFD -> permutation test. Anchor choices are the defaults; alternatives are switches (`detrend="global"`, `perm="block"`, per-window amplitude normalization). |
| `experiments/p0_replicate.py` | **P0**: reproduces the anchor's HFDs on the seizure EEG, then re-tests with block permutation, global-mean centring and Holm across H0-H2; estimates the change time. |
| `experiments/p1_size.py` | **P1 pilot**: false-alarm rate (size) on data with *no* change: i.i.d., AR(1), slow AR(1), long memory (tp level) and i.i.d./VAR(1)/stochastic-volatility 18-channel signals (full pipeline). |

## Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# P0: replication + re-test (~3 min; total persistence is cached after the first run)
python experiments/p0_replicate.py
python experiments/p0_replicate.py --per_window_std      # amplitude-confound check

# P1: size under no change
python experiments/p1_size.py --level both --quick       # smoke test first (~2 min)
python experiments/p1_size.py --level tp                 # ~5-10 min
python experiments/p1_size.py --level signal             # ~10-20 min (H0,H1)
python experiments/p1_size.py --level signal --per_window_std --nulls stoch.vol   # amplitude fix
```

Every script takes `--help`. Seeds are fixed (`--seed 2026`), so runs are reproducible.

## Outputs

- `results/p0/p0_summary.md`, `p0_results.csv`, `p0_eeg.png`
- `results/p0_perwindowstd/...` (same, amplitude-normalized)
- `results/p1/p1_summary_tp.md`, `p1_summary_signal.md`, and CSVs

## Notes on the anchor code

- Channels used: ch01-ch21 minus SP1 (11), SP2 (12) and Cz (20) = 18 channels; ch22-ch31 are not used.
- Windows are non-overlapping, 120 samples (1.2 s at 100 Hz) -> 416 windows; the last partial window is dropped.
- Each window's point cloud is the 120 time points in R^18 (no delay embedding).
- The notebook builds the permutation null but never computes a p-value; we use the lower tail,
  p = (1 + #{null HFD <= observed}) / (B + 1).
- The notebook does not set a random seed.
