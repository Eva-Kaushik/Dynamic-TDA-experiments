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


# =============================================================================
# Extensions used by P2-P6
# =============================================================================
import json as _json
from scipy.stats import kstwobign as _kstwobign

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
FIG_DIR = os.path.join(REPO_ROOT, "figures")
RES_DIR = os.path.join(REPO_ROOT, "results")


def fig_path(phase: str, name: str) -> str:
    d = os.path.join(FIG_DIR, phase)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m, running = len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ------------------------------------------------------- per-window diagrams
def window_diagrams(ts: np.ndarray, sw: int = 120, maxdim: int = 2, per_window_standardize: bool = False):
    """List (one per window) of persistence diagrams [H0, ..., H_maxdim]."""
    out = []
    for s, e in window_bounds(ts.shape[0], sw):
        X = ts[s:e]
        if per_window_standardize:
            X = (X - X.mean(axis=0)) / (X.std() + 1e-12)
        out.append(ripser(X, maxdim=maxdim)["dgms"])
    return out


def _finite(dgm):
    if len(dgm) == 0:
        return np.zeros((0, 2))
    return dgm[np.isfinite(dgm[:, 1])]


def summary(dgm: np.ndarray, kind: str = "tp1") -> float:
    """Scalar summaries of one diagram (G9)."""
    d = _finite(dgm)
    pers = d[:, 1] - d[:, 0] if len(d) else np.zeros(0)
    if kind == "tp1":            # total persistence, p=1 (anchor)
        return float(pers.sum())
    if kind == "tp2":            # p=2 (Divol & Polonik weighting)
        return float(np.sqrt((pers ** 2).sum()))
    if kind == "maxpers":
        return float(pers.max()) if len(pers) else 0.0
    if kind == "entropy":        # persistent entropy
        if pers.sum() <= 0:
            return 0.0
        q = pers / pers.sum()
        q = q[q > 0]
        return float(-(q * np.log(q)).sum())
    if kind == "landscape1":     # L2 norm of the first persistence landscape
        if not len(d):
            return 0.0
        grid = np.linspace(d[:, 0].min(), d[:, 1].max(), 200)
        tents = np.clip(np.minimum(grid[None, :] - d[:, :1], d[:, 1:] - grid[None, :]), 0, None)
        lam1 = tents.max(axis=0)
        return float(np.sqrt(np.trapezoid(lam1 ** 2, grid)) if hasattr(np, "trapezoid")
                     else np.sqrt(np.trapz(lam1 ** 2, grid)))
    raise ValueError(kind)


def summaries_from_diagrams(dgms_list, kind="tp1", maxdim=None):
    maxdim = len(dgms_list[0]) - 1 if maxdim is None else maxdim
    return np.array([[summary(dg[k], kind) for k in range(maxdim + 1)] for dg in dgms_list])


# ---------------------------------------------------------- change statistics
def cusum_stat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    n = len(x)
    s = np.cumsum(x - x.mean())
    return float(np.abs(s).max() / np.sqrt(n))


def _sn_parts(x):
    """Vectorized Shao & Zhang (2010) self-normalized CUSUM pieces for k = 1..n-1."""
    x = np.asarray(x, float)
    n = len(x)
    S = np.cumsum(x)
    k = np.arange(1, n)                       # split after k
    T = (S[k - 1] - k / n * S[-1]) / np.sqrt(n)

    def left_sums(S):
        t = np.arange(1, len(S) + 1)
        cS2, ctS, ct2 = np.cumsum(S ** 2), np.cumsum(t * S), np.cumsum(t ** 2.0)
        return cS2, ctS, ct2

    # left: sum_{t<=k} (S_t - t/k S_k)^2
    cS2, ctS, ct2 = left_sums(S)
    Sk = S[k - 1]
    Lk = cS2[k - 1] - 2 * Sk / k * ctS[k - 1] + (Sk / k) ** 2 * ct2[k - 1]
    # right: same on the reversed series with k' = n - k
    R = np.cumsum(x[::-1])
    rS2, rtS, rt2 = left_sums(R)
    kp = n - k
    Rk = R[kp - 1]
    Rr = rS2[kp - 1] - 2 * Rk / kp * rtS[kp - 1] + (Rk / kp) ** 2 * rt2[kp - 1]
    V = (Lk + Rr) / n ** 2
    return T, V


def sn_stat(x: np.ndarray, trim: float = 0.05) -> float:
    T, V = _sn_parts(x)
    n = len(x)
    lo, hi = int(np.ceil(trim * n)), int(np.floor((1 - trim) * n))
    sl = slice(max(lo - 1, 0), max(hi - 1, 1))
    return float(np.max(T[sl] ** 2 / np.maximum(V[sl], 1e-300)))


_SN_NULL_CACHE = {}


def sn_null(n: int, reps: int = 4000, seed: int = 12345, disk: bool = True) -> np.ndarray:
    """Pivotal null of the SN statistic, simulated with i.i.d. N(0,1) (cached on disk)."""
    if n in _SN_NULL_CACHE:
        return _SN_NULL_CACHE[n]
    path = os.path.join(RES_DIR, "_cache", f"sn_null_n{n}_r{reps}.npy")
    if not disk:
        rng = np.random.default_rng(seed)
        arr = np.sort([sn_stat(rng.standard_normal(n)) for _ in range(reps)])
    elif os.path.exists(path):
        arr = np.load(path)
    else:
        rng = np.random.default_rng(seed)
        arr = np.sort([sn_stat(rng.standard_normal(n)) for _ in range(reps)])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, arr)
    _SN_NULL_CACHE[n] = arr
    return arr


def sn_test(x: np.ndarray) -> float:
    """p-value of the self-normalized CUSUM (upper tail)."""
    null = sn_null(len(x))
    return float((1 + np.sum(null >= sn_stat(x))) / (len(null) + 1))


def sn_test_any_n(x: np.ndarray) -> float:
    """Same, for arbitrary segment lengths (null simulated in memory, 2000 reps)."""
    null = sn_null(len(x), reps=2000, disk=False)
    return float((1 + np.sum(null >= sn_stat(x))) / (len(null) + 1))


def lrv_bartlett(x: np.ndarray, bw: int | None = None) -> float:
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    bw = int(np.floor(n ** (1 / 3))) if bw is None else bw
    g0 = x @ x / n
    s = g0
    for h in range(1, bw + 1):
        s += 2 * (1 - h / (bw + 1)) * (x[h:] @ x[:-h]) / n
    return max(s, 1e-12)


def lrv_cusum_test(x: np.ndarray) -> float:
    """CUSUM / sqrt(long-run variance); Kolmogorov (Brownian-bridge sup) p-value."""
    stat = cusum_stat(x) / np.sqrt(lrv_bartlett(x))
    return float(_kstwobign.sf(stat))


def block_perm_cusum_test(x: np.ndarray, B: int = 199, block_len: int = 20, rng=None) -> float:
    rng = np.random.default_rng() if rng is None else rng
    obs = cusum_stat(x)
    null = np.array([cusum_stat(permute(np.asarray(x, float), rng, "block", block_len)) for _ in range(B)])
    return float((1 + np.sum(null >= obs)) / (B + 1))


def hfd_block_test(x: np.ndarray, B: int = 199, block_len: int = 20, k_max: int = 10, rng=None) -> float:
    """Best anchor variant from P1: global-mean centring + block permutation."""
    return hfd_test(detrend(x, "global"), k_max, B, "block", block_len, rng).p_value


def anchor_test(x: np.ndarray, B: int = 199, k_max: int = 10, W: int = 50, rng=None) -> float:
    """The original anchor test: trailing mean + i.i.d. window permutation."""
    return hfd_test(detrend(x, "trailing", W), k_max, B, "window", 8, rng).p_value


TESTS = {
    "anchor (HFD, window perm)": lambda x, rng, B: anchor_test(x, B, rng=rng),
    "HFD, block perm L=20":      lambda x, rng, B: hfd_block_test(x, B, rng=rng),
    "CUSUM, block perm L=20":    lambda x, rng, B: block_perm_cusum_test(x, B, rng=rng),
    "self-normalized CUSUM":     lambda x, rng, B: sn_test(x),
    "LRV CUSUM (Bartlett)":      lambda x, rng, B: lrv_cusum_test(x),
}


def argmax_changepoint(x: np.ndarray) -> int:
    """Window index (0-based start of the new regime) of the single CUSUM change."""
    return cusum_changepoint(x)


# --------------------------------------------------------------- surrogates
def phase_randomize_mv(X: np.ndarray, rng) -> np.ndarray:
    """Prichard & Theiler (1994): add the SAME random phase to every channel at each
    frequency -> keeps every auto- and cross-spectrum, destroys nonlinear structure."""
    T = X.shape[0]
    F = np.fft.rfft(X, axis=0)
    ph = rng.uniform(0, 2 * np.pi, F.shape[0])
    ph[0] = 0.0
    if T % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(F * np.exp(1j * ph)[:, None], n=T, axis=0)


def iaaft_mv(X: np.ndarray, rng, iters: int = 20) -> np.ndarray:
    """Multivariate IAAFT (Schreiber & Schmitz 2000, sec. on multivariate data):
    alternately impose each channel's amplitude spectrum (keeping the cross-channel
    phase differences of the original) and each channel's value distribution."""
    T, p = X.shape
    F0 = np.fft.rfft(X, axis=0)
    amp0, ph0 = np.abs(F0), np.angle(F0)
    sorted_vals = np.sort(X, axis=0)
    Y = phase_randomize_mv(X, rng)
    Y = sorted_vals[np.argsort(np.argsort(Y, axis=0), axis=0), np.arange(p)]
    for _ in range(iters):
        F = np.fft.rfft(Y, axis=0)
        # common rotation alpha(f) that best aligns original phases to current ones
        alpha = np.angle(np.sum(F * np.conj(np.exp(1j * ph0)) * amp0, axis=1))
        Z = np.fft.irfft(amp0 * np.exp(1j * (ph0 + alpha[:, None])), n=T, axis=0)
        Y = sorted_vals[np.argsort(np.argsort(Z, axis=0), axis=0), np.arange(p)]
    return Y


# ---------------------------------------------------------- other embeddings
def corr_graph_tp(ts: np.ndarray, sw: int = 120, maxdim: int = 1) -> np.ndarray:
    """Channels as points, distance sqrt(2(1-corr)) within each window (TADA-like)."""
    out = []
    for s, e in window_bounds(ts.shape[0], sw):
        C = np.corrcoef(ts[s:e].T)
        D = np.sqrt(np.clip(2 * (1 - C), 0, None))
        np.fill_diagonal(D, 0)
        dg = ripser(D, maxdim=maxdim, distance_matrix=True)["dgms"]
        out.append([total_persistence(d) for d in dg])
    return np.asarray(out)


def delay_embed_tp(ts: np.ndarray, sw: int = 120, dim: int = 3, tau: int = 5, maxdim: int = 1,
                   per_window_standardize: bool = True) -> np.ndarray:
    """Delay embedding (Takens) of the window's first principal component."""
    out = []
    for s, e in window_bounds(ts.shape[0], sw):
        X = ts[s:e] - ts[s:e].mean(0)
        u, sv, vt = np.linalg.svd(X, full_matrices=False)
        z = X @ vt[0]
        if per_window_standardize:
            z = z / (z.std() + 1e-12)
        m = len(z) - (dim - 1) * tau
        P = np.stack([z[i * tau:i * tau + m] for i in range(dim)], axis=1)
        dg = ripser(P, maxdim=maxdim)["dgms"]
        out.append([total_persistence(d) for d in dg])
    return np.asarray(out)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        _json.dump(obj, f, indent=2, default=float)
