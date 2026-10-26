"""P1 pilot: does the HFD-of-CUSUM test keep its false-alarm rate (size) when
there is NO change?  Every null below has no change point, so the rejection
rate at alpha=0.05 should be about 5%.  Gate from the review: 5% +/- 1.5 points.

Two levels:
  tp     - simulate the total-persistence series directly (fast; isolates the
           inference step: detrending + permutation + HFD)
  signal - simulate 18-channel signals and run the full pipeline
           (windows -> Rips persistence -> total persistence -> test)

    python experiments/p1_size.py --level tp                 # ~2-5 min on a laptop
    python experiments/p1_size.py --level signal             # ~5-15 min (H0,H1 only)
    python experiments/p1_size.py --level both --quick       # smoke test, ~1-2 min
    python experiments/p1_size.py --level signal --maxdim 2  # include H2 (slow)
    python experiments/p1_size.py --level signal --per_window_std   # amplitude-normalized windows

Outputs (results/p1/): p1_<level>.csv and p1_summary_<level>.md
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import dtda  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_WIN = 416          # same number of windows as the EEG
SW, P = 120, 18      # window length (samples) and channels, as in the anchor
ALPHA = 0.05


# --------------------------------------------------------------------- null models
def ar1(n, phi, rng, burn=500):
    e = rng.standard_normal(n + burn)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, len(e)):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def arfima_d(n, d, rng, trunc=3000):
    """ARFIMA(0,d,0) via truncated MA(inf) filter; d in (0, .5) gives long memory."""
    k = np.arange(1, trunc)
    psi = np.concatenate([[1.0], np.cumprod((k - 1 + d) / k)])
    e = rng.standard_normal(n + trunc)
    return np.convolve(e, psi, mode="full")[trunc:trunc + n]


TP_NULLS = {
    "iid":            lambda n, r: 10 + r.standard_normal(n),
    "AR1(0.5)":       lambda n, r: 10 + ar1(n, 0.5, r),
    "AR1(0.9)":       lambda n, r: 10 + ar1(n, 0.9, r),
    "slowAR1(0.98)":  lambda n, r: 10 + ar1(n, 0.98, r),
    "longmem(d=0.3)": lambda n, r: 10 + arfima_d(n, 0.3, r),
}


def sig_iid(T, r):
    return r.standard_normal((T, P))


def sig_var(T, r, phi=0.9, rho=0.5):
    """VAR(1): each channel AR(0.9) at 100 Hz, innovations equicorrelated (rho)."""
    L = np.linalg.cholesky((1 - rho) * np.eye(P) + rho * np.ones((P, P)))
    e = r.standard_normal((T + 500, P)) @ L.T
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, len(e)):
        x[t] = phi * x[t - 1] + e[t]
    return x[500:]


def sig_sv(T, r):
    """Stationary stochastic volatility: VAR(1) signal times a slowly varying
    amplitude exp(h_t), h an AR(1) at window scale (phi=0.98 per 1.2 s window)."""
    x = sig_var(T, r)
    nw = T // SW + 2
    h = 0.4 * ar1(nw, 0.98, r) * np.sqrt(1 - 0.98 ** 2)   # sd(h) ~ 0.4
    amp = np.exp(np.interp(np.arange(T), np.arange(nw) * SW, h))
    return x * amp[:, None]


SIG_NULLS = {"iid": sig_iid, "VAR1": sig_var, "stoch.vol (VAR1 x slow amplitude)": sig_sv}


# ------------------------------------------------------------------- test configs
def configs(n):
    L = dtda.default_block_len(n)
    return [("trailing", "window", L), ("trailing", "block", L), ("trailing", "block", 20),
            ("global", "window", L), ("global", "block", L), ("global", "block", 20)]


def run_tests(tp_matrix, B, seed):
    """tp_matrix: (n, ndims). Returns {(config_idx, dim): p}."""
    n = tp_matrix.shape[0]
    out = {}
    for ci, (det, perm, L) in enumerate(configs(n)):
        for k in range(tp_matrix.shape[1]):
            rng = np.random.default_rng([seed, ci, k])
            d = dtda.detrend(tp_matrix[:, k], det, 50)
            out[(ci, k)] = dtda.hfd_test(d, 10, B, perm, L, rng).p_value
    return out


def job_tp(args):
    name, rep, B, seed = args
    rng = np.random.default_rng([seed, rep, list(TP_NULLS).index(name)])
    tp = TP_NULLS[name](N_WIN, rng)[:, None]
    return name, run_tests(tp, B, seed * 7919 + rep)


def job_signal(args):
    name, rep, B, seed, maxdim, pwstd = args
    rng = np.random.default_rng([seed, rep, 100 + list(SIG_NULLS).index(name)])
    T = (N_WIN + 1) * SW
    x = SIG_NULLS[name](T, rng)
    x = (x - x.mean(0)) / x.std(0)        # anchor: global per-channel z-score
    tp = dtda.window_total_persistence(x, SW, maxdim, per_window_standardize=pwstd)
    return name, run_tests(tp, B, seed * 7919 + rep)


# ---------------------------------------------------------------------- reporting
def summarize(results, ndims, level, args, elapsed):
    names = list(dict.fromkeys(r[0] for r in results))
    cfg = configs(N_WIN)
    rows = []
    for name in names:
        ps = [r[1] for r in results if r[0] == name]
        R = len(ps)
        for ci, (det, perm, L) in enumerate(cfg):
            for k in range(ndims):
                rej = np.mean([p[(ci, k)] <= ALPHA for p in ps])
                se = np.sqrt(rej * (1 - rej) / R) if 0 < rej < 1 else np.sqrt(ALPHA * (1 - ALPHA) / R)
                rows.append(dict(level=level, null=name, detrend=det,
                                 perm=perm if perm == "window" else f"block L={L}",
                                 dim=f"H{k}" if level.startswith("signal") else "TP", reps=R,
                                 size=rej, mc_se=se, gate_pass=abs(rej - ALPHA) <= 0.015))
            # Holm across dims (family-wise rejection) for the signal level
            if ndims > 1:
                # Holm rejects at least one hypothesis iff min p <= alpha/m
                fw = np.mean([min(p[(ci, k)] for k in range(ndims)) <= ALPHA / ndims for p in ps])
                rows.append(dict(level=level, null=name, detrend=det,
                                 perm=perm if perm == "window" else f"block L={L}",
                                 dim="any (Holm)", reps=R, size=fw,
                                 mc_se=np.sqrt(max(fw * (1 - fw), ALPHA * (1 - ALPHA)) / R),
                                 gate_pass=abs(fw - ALPHA) <= 0.015))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"p1_{level}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    dims = sorted(set(r["dim"] for r in rows), key=lambda s: (s.startswith("any"), s))
    cols = [(det, perm if perm == "window" else f"block L={L}") for det, perm, L in cfg]
    md = [f"# P1 size check - level: {level}\n",
          f"reps={args.reps}, B={args.B}, alpha={ALPHA}, n_windows={N_WIN}, seed={args.seed}, "
          f"runtime={elapsed/60:.1f} min. Cells: rejection rate under NO change (target 5%). "
          f"** = outside the 3.5-6.5% gate.\n"]
    for dim in dims:
        md.append(f"\n## {dim}\n\n| null | " + " | ".join(f"{d} / {p}" for d, p in cols) + " |")
        md.append("|---|" + "---|" * len(cols))
        for name in names:
            cells = []
            for d, p in cols:
                r = next(x for x in rows if x["null"] == name and x["dim"] == dim
                         and x["detrend"] == d and x["perm"] == p)
                cells.append(f"{100*r['size']:.1f}%{'' if r['gate_pass'] else ' **'}")
            md.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(md) + "\n"
    with open(os.path.join(args.out, f"p1_summary_{level}.md"), "w") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["tp", "signal", "both"], default="tp")
    ap.add_argument("--reps", type=int, default=None, help="default: 500 (tp), 100 (signal)")
    ap.add_argument("--B", type=int, default=199)
    ap.add_argument("--maxdim", type=int, default=1, help="signal level: 1 = H0,H1 (fast); 2 adds H2")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true", help="few reps, for a smoke test")
    ap.add_argument("--per_window_std", action="store_true",
                    help="signal level: rescale each window to unit variance before PH (amplitude fix)")
    ap.add_argument("--nulls", nargs="*", default=None, help="signal level: subset, e.g. --nulls stoch.vol")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "p1"))
    args = ap.parse_args()
    levels = ["tp", "signal"] if args.level == "both" else [args.level]

    for level in levels:
        reps = args.reps or (500 if level == "tp" else 100)
        if args.quick:
            reps = 40 if level == "tp" else 8
        a = argparse.Namespace(**{**vars(args), "reps": reps})
        if level == "tp":
            jobs = [(n, r, args.B, args.seed) for n in TP_NULLS for r in range(reps)]
            fn, ndims = job_tp, 1
        else:
            jobs = [(n, r, args.B, args.seed, args.maxdim, args.per_window_std) for n in SIG_NULLS
                    if args.nulls is None or n.split()[0] in args.nulls for r in range(reps)]
            fn, ndims = job_signal, args.maxdim + 1
        print(f"[P1:{level}] {len(jobs)} simulated datasets, B={args.B}, workers={args.workers} ...", flush=True)
        t0 = time.time()
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(fn, jobs, chunksize=1 if level == "signal" else 4)):
                results.append(res)
                if (i + 1) % max(1, len(jobs) // 10) == 0:
                    print(f"   {i+1}/{len(jobs)} done ({time.time()-t0:.0f} s)", flush=True)
        summarize(results, ndims, level + ("_perwindowstd" if level == "signal" and args.per_window_std else ""),
                  a, time.time() - t0)
    try:
        from figs import p1_figures
        p1_figures()
    except Exception as e:
        print("P1 figures skipped:", e)


if __name__ == "__main__":
    main()
