"""Core pipeline for the Dynamic-TDA HFD-of-CUSUM change test.

Reimplements El-Yaagoubi, Chung & Ombao (2024), Front. Neuroinform. 18:1387400
(code: github.com/A-EL-YAAGOUBI/Dynamic-TDA, Dynamic_TDA_HFD_Application.ipynb)
as plain functions, with the anchor's choices as defaults and our alternatives
as switches:

    detrend : "trailing" (anchor, W=50) | "global" | "none"
    perm    : "window"   (anchor, i.i.d. shuffle) | "block" (Kirch 2007)

Pipeline: EEG -> non-overlapping windows -> Vietoris-Rips persistence of the
window's point cloud (time points as points in R^p, no delay embedding) ->
total persistence per homology dimension -> deviations D_t -> CUSUM S_t ->
Higuchi fractal dimension of S_t -> permutation null for the HFD.
A change makes S_t smoother, so the test rejects in the LOWER tail.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import hfda
import numpy as np
from ripser import ripser

FS = 100  # Hz, Malow seizure EEG
# Anchor: ch01..ch21 minus SP1 (11), SP2 (12) (corrupted) and Cz (20, reference) -> 18 channels
ANCHOR_CHANNELS = [i for i in range(1, 22) if i not in (11, 12, 20)]
CHANNEL_NAMES = {1: "Fp1", 2: "Fp2", 3: "F3", 4: "F4", 5: "C3", 6: "C4", 7: "P3", 8: "P4",
                 9: "O1", 10: "O2", 11: "SP1", 12: "SP2", 13: "F7", 14: "F8", 15: "T3",
                 16: "T4", 17: "T5", 18: "T6", 19: "Fz", 20: "Cz", 21: "Pz"}


# --------------------------------------------------------------------------- data
def load_eeg(data_dir: str, channels=ANCHOR_CHANNELS, standardize: bool = True) -> np.ndarray:
    """Return a (T, p) array. Matches the anchor notebook exactly when defaults are used."""
    ts = np.array([np.loadtxt(os.path.join(data_dir, f"ch{c:02d}.inp")) for c in channels]).T
    if standardize:
        ts = ts - ts.mean(axis=0)
        ts = ts / ts.std(axis=0)
    return ts


# ------------------------------------------------------------------ topology step
def total_persistence(dgm: np.ndarray) -> float:
    """Sum of (death - birth) over finite bars (anchor's getTotalPersistence)."""
    if len(dgm) == 0:
        return 0.0
    finite = np.isfinite(dgm[:, 1])
    return float(np.sum(dgm[finite, 1] - dgm[finite, 0]))


def window_bounds(T: int, sw: int):
    """Anchor's windowing: zip(range(0, T-SW, SW), range(SW, T, SW)) (drops the last window)."""
    return list(zip(range(0, T - sw, sw), range(sw, T, sw)))


def window_total_persistence(ts: np.ndarray, sw: int = 120, maxdim: int = 2,
                             per_window_standardize: bool = False) -> np.ndarray:
    """(n_windows, maxdim+1) array of total persistence for H0..H_maxdim."""
    out = []
    for s, e in window_bounds(ts.shape[0], sw):
        X = ts[s:e]
        if per_window_standardize:  # G3/G9 option: remove amplitude scale per window
            X = (X - X.mean(axis=0)) / (X.std() + 1e-12)
        dgms = ripser(X, maxdim=maxdim)["dgms"]
        out.append([total_persistence(d) for d in dgms])
    return np.asarray(out)


# ----------------------------------------------------------------- inference step
def detrend(tp: np.ndarray, method: str = "trailing", W: int = 50) -> np.ndarray:
    """Deviations D_t = TP_t - mu_hat_t.

    trailing: anchor. First W points centred on mean(TP[:W]); afterwards on the
              mean of the previous W points.
    global  : centred on the full-series mean (classical CUSUM, Aue & Horvath 2013).
    none    : no centring.
    """
    tp = np.asarray(tp, dtype=float)
    if method == "none":
        return tp.copy()
    if method == "global":
        return tp - tp.mean()
    if method == "trailing":
        d = np.empty_like(tp)
        d[:W] = tp[:W] - tp[:W].mean()
        c = np.concatenate([[0.0], np.cumsum(tp)])
        idx = np.arange(W, len(tp))
        d[W:] = tp[W:] - (c[idx] - c[idx - W]) / W
        return d
    raise ValueError(method)


def hfd_of_cusum(d: np.ndarray, k_max: int = 10) -> float:
    return float(hfda.measure(np.cumsum(d), k_max))


def permute(d: np.ndarray, rng: np.random.Generator, perm: str = "window", block_len: int = 8) -> np.ndarray:
    if perm == "window":
        return d[rng.permutation(len(d))]
    if perm == "block":
        # Kirch (2007): cut into consecutive blocks, shuffle the blocks, keep order within.
        n = len(d)
        starts = np.arange(0, n, block_len)
        order = rng.permutation(len(starts))
        return np.concatenate([d[starts[i]:starts[i] + block_len] for i in order])
    raise ValueError(perm)


@dataclass
class TestResult:
    hfd: float
    p_value: float          # lower tail: P(null HFD <= observed)
    null_mean: float
    null_sd: float


def hfd_test(d: np.ndarray, k_max: int = 10, B: int = 999, perm: str = "window",
             block_len: int = 8, rng: np.random.Generator | None = None,
             return_null: bool = False):
    rng = np.random.default_rng() if rng is None else rng
    obs = hfd_of_cusum(d, k_max)
    null = np.fromiter((hfd_of_cusum(permute(d, rng, perm, block_len), k_max) for _ in range(B)),
                       dtype=float, count=B)
    p = (1 + np.sum(null <= obs)) / (B + 1)
    res = TestResult(obs, float(p), float(null.mean()), float(null.std()))
    return (res, null) if return_null else res


def cusum_changepoint(tp: np.ndarray) -> int:
    """Classical CUSUM estimate of a single mean change: argmax_t |S_t - t/n S_n|."""
    tp = np.asarray(tp, float)
    s = np.cumsum(tp - tp.mean())
    return int(np.argmax(np.abs(s))) + 1


def default_block_len(n: int) -> int:
    return max(2, int(round(n ** (1 / 3))))
