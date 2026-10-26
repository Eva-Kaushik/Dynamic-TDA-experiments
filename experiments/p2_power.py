"""P2: size and power of better change statistics, against non-topological baselines.

Part A - size (no change) on the P1 total-persistence nulls, now including the
         EEG-like weak dependence AR(1) phi=0.17 (lag-1 autocorrelation of the
         anchor EEG's pre-seizure H0 total persistence).
Part B - power on 18-channel synthetic signals with a change at 60% of the record:
   * "lock"      topology change: a fraction s of channels switch from independent
                 to a common (phase-locked) oscillation. Every channel's own
                 spectrum and variance are unchanged; only the cross-channel
                 geometry changes (the point cloud collapses towards a loop).
   * "amplitude" nuisance change: all channels scaled by a after the change, no
                 change in structure. A topology test should NOT fire here once
                 amplitude is normalized.
   Tests run on H0/H1 total persistence, raw and per-window normalized, and on
   two non-topological baselines computed from the same windows:
     amp-baseline  = log mean window variance
     corr-baseline = mean |off-diagonal correlation| (linear connectivity)
     pca2-baseline = variance share of the top-2 principal components (linear
                     low-dimensionality; a loop in R^18 lives in a 2-D plane)

    python experiments/p2_power.py               # A + B, ~10 min on an M-series Mac
    python experiments/p2_power.py --part A
    python experiments/p2_power.py --quick

Outputs: results/p2/*.csv, results/p2/p2_summary_A.md / _B.md, figures/p2/*.png
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
from p1_size import TP_NULLS, ar1, arfima_d  # noqa: E402

N_WIN, SW, P, FS = 416, 120, 18, 100
TAU = int(0.6 * N_WIN)          # change at window 249 (~300 s)
ALPHA = 0.05
OUT = os.path.join(dtda.RES_DIR, "p2")

SIZE_NULLS = {
    "iid": TP_NULLS["iid"],
    "AR1(0.17) EEG-like": lambda n, r: 10 + ar1(n, 0.17, r),
    "AR1(0.5)": TP_NULLS["AR1(0.5)"],
    "AR1(0.9)": TP_NULLS["AR1(0.9)"],
    "longmem(d=0.3)": lambda n, r: 10 + arfima_d(n, 0.3, r),
}
TEST_NAMES = list(dtda.TESTS)


# ----------------------------------------------------------------- Part A
def job_size(a):
    name, rep, B, seed = a
    rng = np.random.default_rng([seed, rep, list(SIZE_NULLS).index(name)])
    x = SIZE_NULLS[name](N_WIN, rng)
    return name, {t: dtda.TESTS[t](x, np.random.default_rng([seed, rep, i, 7]), B)
                  for i, t in enumerate(TEST_NAMES)}


# ----------------------------------------------------------------- Part B
def simulate(kind, level, rng, T=(N_WIN + 1) * SW):
    """18 channels of 10 Hz oscillations with random-walk phase + VAR(1) noise."""
    t = np.arange(T) / FS
    dphi = 0.05 * rng.standard_normal((T, P))
    phi_ind = np.cumsum(dphi, axis=0) + rng.uniform(0, 2 * np.pi, P)
    phi_com = np.cumsum(0.05 * rng.standard_normal(T)) + rng.uniform(0, 2 * np.pi)
    offsets = np.linspace(0, 2 * np.pi, P, endpoint=False)
    phi = phi_ind.copy()
    tc = TAU * SW
    if kind == "lock" and level > 0:
        m = int(round(level * P))
        ch = rng.choice(P, m, replace=False)
        phi[tc:, ch] = phi_com[tc:, None] + offsets[ch]
    x = np.sin(2 * np.pi * 10 * t[:, None] + phi)
    noise = np.empty((T, P))
    e = 0.5 * rng.standard_normal((T, P))
    noise[0] = e[0]
    for i in range(1, T):
        noise[i] = 0.6 * noise[i - 1] + e[i]
    x = x + noise
    if kind == "amplitude" and level != 1:
        x[tc:] *= level
    return (x - x.mean(0)) / x.std(0)


def window_features(x):
    amp, corr, pca2 = [], [], []
    for s, e in dtda.window_bounds(x.shape[0], SW):
        W = x[s:e]
        amp.append(np.log(W.var(axis=0).mean()))
        C = np.corrcoef(W.T)
        corr.append(np.abs(C[np.triu_indices(P, 1)]).mean())
        ev = np.linalg.eigvalsh(C)[::-1]
        pca2.append(ev[:2].sum() / ev.sum())
    return np.array(amp), np.array(corr), np.array(pca2)


def job_power(a):
    kind, level, rep, B, seed = a
    rng = np.random.default_rng([seed, rep, int(level * 100), 0 if kind == "lock" else 1])
    x = simulate(kind, level, rng)
    series = {}
    tp_raw = dtda.window_total_persistence(x, SW, 1)
    tp_pw = dtda.window_total_persistence(x, SW, 1, per_window_standardize=True)
    for k in range(2):
        series[f"TP H{k} raw"] = tp_raw[:, k]
        series[f"TP H{k} norm"] = tp_pw[:, k]
    series["amp-baseline"], series["corr-baseline"], series["pca2-baseline"] = window_features(x)
    res = {}
    for si, (sname, s) in enumerate(series.items()):
        for ti, t in enumerate(TEST_NAMES):
            res[(sname, t)] = dtda.TESTS[t](s, np.random.default_rng([seed, rep, si, ti]), B)
        res[(sname, "tau_hat")] = dtda.argmax_changepoint(s)
    return kind, level, res


# ----------------------------------------------------------------- reporting
def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "both"], default="both")
    ap.add_argument("--reps_size", type=int, default=500)
    ap.add_argument("--reps_power", type=int, default=100)
    ap.add_argument("--B", type=int, default=199)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps_size, args.reps_power = 40, 6
    dtda.sn_null(N_WIN)  # build the SN null once before forking
    md = ["# P2 - size and power of change statistics\n"]
    parts_md = {}

    if args.part in ("A", "both"):
        jobs = [(n, r, args.B, args.seed) for n in SIZE_NULLS for r in range(args.reps_size)]
        print(f"[P2:A] size: {len(jobs)} series ...", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            out = list(ex.map(job_size, jobs, chunksize=8))
        rows = []
        for n in SIZE_NULLS:
            ps = [o[1] for o in out if o[0] == n]
            for t in TEST_NAMES:
                rej = float(np.mean([p[t] <= ALPHA for p in ps]))
                rows.append(dict(null=n, test=t, reps=len(ps), size=rej))
        write_csv(rows, os.path.join(OUT, "p2_size.csv"))
        md += [f"\n## A. Size under no change (target 5%; reps={args.reps_size}, {time.time()-t0:.0f} s)\n",
               "| null | " + " | ".join(TEST_NAMES) + " |", "|---|" + "---|" * len(TEST_NAMES)]
        for n in SIZE_NULLS:
            md.append(f"| {n} | " + " | ".join(
                f"{100*next(r['size'] for r in rows if r['null']==n and r['test']==t):.1f}%" for t in TEST_NAMES) + " |")
        from figs import heatmap
        M = np.array([[next(r["size"] for r in rows if r["null"] == n and r["test"] == t) for t in TEST_NAMES]
                      for n in SIZE_NULLS])
        heatmap(M * 100, list(SIZE_NULLS), TEST_NAMES, "P2-A: false-alarm rate (%) with no change; target 5%",
                dtda.fig_path("p2", "p2_size_heatmap.png"), center=5)
        parts_md["A"], md = md, []

    if args.part in ("B", "both"):
        conds = [("lock", 0.0), ("lock", 0.25), ("lock", 0.5), ("lock", 1.0),
                 ("amplitude", 1.25), ("amplitude", 1.5), ("amplitude", 2.0)]
        jobs = [(k, l, r, args.B, args.seed) for k, l in conds for r in range(args.reps_power)]
        print(f"[P2:B] power: {len(jobs)} simulated recordings ...", flush=True)
        t0 = time.time()
        out = []
        with ProcessPoolExecutor(args.workers) as ex:
            for i, o in enumerate(ex.map(job_power, jobs, chunksize=1)):
                out.append(o)
                if (i + 1) % max(1, len(jobs) // 10) == 0:
                    print(f"   {i+1}/{len(jobs)} ({time.time()-t0:.0f} s)", flush=True)
        series_names = [k[0] for k in out[0][2] if k[1] == "tau_hat"]
        rows = []
        for kind, level in conds:
            rs = [o[2] for o in out if o[0] == kind and o[1] == level]
            for sname in series_names:
                for t in TEST_NAMES:
                    rej = [r[(sname, t)] <= ALPHA for r in rs]
                    err = [abs(r[(sname, "tau_hat")] - TAU) for r, d in zip(rs, rej) if d]
                    rows.append(dict(change=kind, level=level, series=sname, test=t, reps=len(rs),
                                     reject=float(np.mean(rej)),
                                     median_loc_err_windows=float(np.median(err)) if err else ""))
        write_csv(rows, os.path.join(OUT, "p2_power.csv"))
        focus = ["HFD, block perm L=20", "self-normalized CUSUM", "LRV CUSUM (Bartlett)", "anchor (HFD, window perm)"]
        md.append(f"\n## B. Rejection rate with a change at window {TAU} (reps={args.reps_power}, {time.time()-t0:.0f} s)\n")
        md.append("lock s = fraction of channels that phase-lock (topology change; s=0 is no change). "
                  "amplitude a = all channels scaled by a (nuisance change). Higher is better for lock s>0; "
                  "for amplitude a topology test should stay near 5% after normalization.\n")
        for t in focus:
            md.append(f"\n### {t}\n\n| series | " + " | ".join(f"{k} {l}" for k, l in conds) + " |")
            md.append("|---|" + "---|" * len(conds))
            for sname in series_names:
                md.append(f"| {sname} | " + " | ".join(
                    f"{100*next(r['reject'] for r in rows if r['change']==k and r['level']==l and r['series']==sname and r['test']==t):.0f}%"
                    for k, l in conds) + " |")
        md.append("\nMedian localization error (windows of 1.2 s) of the CUSUM argmax when the "
                  "self-normalized CUSUM rejects, lock s=1:\n")
        for sname in series_names:
            r = next(r for r in rows if r["change"] == "lock" and r["level"] == 1.0 and r["series"] == sname
                     and r["test"] == "self-normalized CUSUM")
            md.append(f"- {sname}: {r['median_loc_err_windows']}")
        from figs import power_curves
        power_curves(rows, conds, series_names, focus, dtda.fig_path("p2", "p2_power_curves.png"))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
            for a_, (kind, lvl) in zip(ax, [("lock", 1.0), ("amplitude", 2.0)]):
                x = simulate(kind, lvl, np.random.default_rng(1))
                a_.plot(np.arange(x.shape[0])[:60000] / FS, x[:60000, :3], lw=.3)
                a_.axvline(TAU * SW / FS, color="k", ls=":")
                a_.set_title(f"example: {kind} {lvl} (3 of 18 channels; dotted = change)")
            ax[-1].set_xlabel("time (s)")
            fig.tight_layout()
            fig.savefig(dtda.fig_path("p2", "p2_example_signals.png"), dpi=120)
            plt.close(fig)
        except Exception as e:
            print("example plot skipped:", e)

        parts_md["B"], md = md, []
    for part, lines in parts_md.items():
        with open(os.path.join(OUT, f"p2_summary_{part}.md"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
