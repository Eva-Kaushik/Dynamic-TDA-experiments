"""P4: how much do the EEG conclusions depend on tuning choices?  (Gap G12.)

Sweeps, on the anchor EEG:
  window length SW in {80, 100, 120, 160, 200} samples (0.8-2.0 s), raw and
      per-window normalized, for the anchor test, HFD+block, and the
      self-normalized CUSUM;
  k_max in {5, 8, 10, 15, 20}  (HFD tests, SW=120);
  trailing-mean span W in {10, 25, 50, 100} (anchor test, SW=120);
  block length L in {10, 20, 40, 60} (HFD+block and CUSUM+block, SW=120).

    python experiments/p4_sensitivity.py           # ~5-10 min on an M-series Mac
    python experiments/p4_sensitivity.py --maxdim 1   # faster, H0/H1 only

Outputs: results/p4/p4_*.csv, results/p4/p4_summary.md, figures/p4/*.png
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

OUT = os.path.join(dtda.RES_DIR, "p4")
DATA = os.path.join(dtda.REPO_ROOT, "Epileptic Seizure EEG Dataset")
SWS = [80, 100, 120, 160, 200]


def tp_cached(sw, pw, maxdim):
    path = os.path.join(dtda.RES_DIR, "_cache", f"tp_sw{sw}_md{maxdim}_{'pw' if pw else 'raw'}.npy")
    if os.path.exists(path):
        return np.load(path)
    ts = dtda.load_eeg(DATA)
    tp = dtda.window_total_persistence(ts, sw, maxdim, per_window_standardize=pw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, tp)
    return tp


def job_tp(a):
    sw, pw, maxdim = a
    return sw, pw, tp_cached(sw, pw, maxdim)


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxdim", type=int, default=2)
    ap.add_argument("--B", type=int, default=999)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(sw, pw, args.maxdim) for sw in SWS[::-1] for pw in (False, True)]
    print(f"[P4] total persistence for {len(jobs)} (window, normalization) settings ...", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        tps = {(sw, pw): tp for sw, pw, tp in ex.map(job_tp, jobs)}
    print(f"     done in {time.time()-t0:.0f} s", flush=True)
    dims = range(args.maxdim + 1)
    rng = lambda *k: np.random.default_rng([args.seed, *k])  # noqa: E731

    # 1. window length
    rows_sw = []
    for (sw, pw), tp in sorted(tps.items()):
        for k in dims:
            x = tp[:, k]
            rows_sw.append(dict(sw=sw, seconds=sw / dtda.FS, normalized=pw, dim=f"H{k}", n=len(x),
                                anchor=dtda.anchor_test(x, args.B, rng=rng(sw, pw, k, 0)),
                                hfd_block=dtda.hfd_block_test(x, args.B, rng=rng(sw, pw, k, 1)),
                                sn=dtda.sn_test(x),
                                cp_s=dtda.argmax_changepoint(x) * sw / dtda.FS))
    write_csv(rows_sw, os.path.join(OUT, "p4_window.csv"))

    # 2. k_max, 3. W, 4. block length (SW=120)
    rows_k, rows_w, rows_l = [], [], []
    for pw in (False, True):
        tp = tps[(120, pw)]
        for k in dims:
            x = tp[:, k]
            for km in (5, 8, 10, 15, 20):
                rows_k.append(dict(k_max=km, normalized=pw, dim=f"H{k}",
                                   anchor=dtda.anchor_test(x, args.B, k_max=km, rng=rng(km, pw, k, 2)),
                                   hfd_block=dtda.hfd_block_test(x, args.B, k_max=km, rng=rng(km, pw, k, 3))))
            for W in (10, 25, 50, 100):
                rows_w.append(dict(W=W, normalized=pw, dim=f"H{k}",
                                   anchor=dtda.anchor_test(x, args.B, W=W, rng=rng(W, pw, k, 4))))
            for L in (10, 20, 40, 60):
                rows_l.append(dict(L=L, normalized=pw, dim=f"H{k}",
                                   hfd_block=dtda.hfd_block_test(x, args.B, block_len=L, rng=rng(L, pw, k, 5)),
                                   cusum_block=dtda.block_perm_cusum_test(x, args.B, L, rng=rng(L, pw, k, 6))))
    write_csv(rows_k, os.path.join(OUT, "p4_kmax.csv"))
    write_csv(rows_w, os.path.join(OUT, "p4_W.csv"))
    write_csv(rows_l, os.path.join(OUT, "p4_blocklen.csv"))

    # summary
    md = [f"# P4 - sensitivity on the anchor EEG (B={args.B}, {time.time()-t0:.0f} s)\n",
          "Cells are p-values. anchor = trailing mean + window permutation (the published test); "
          "hfd_block = global mean + block permutation L=20; sn = self-normalized CUSUM.\n"]

    def table(rows, key, cols, title):
        md.append(f"\n## {title}\n")
        for pw in (False, True):
            md.append(f"\n**{'normalized' if pw else 'raw'}**\n")
            vals = sorted(set(r[key] for r in rows))
            md.append(f"| test | dim | " + " | ".join(f"{key}={v}" for v in vals) + " |")
            md.append("|---|---|" + "---|" * len(vals))
            for c in cols:
                for k in dims:
                    md.append(f"| {c} | H{k} | " + " | ".join(
                        f"{next(r[c] for r in rows if r[key]==v and r['normalized']==pw and r['dim']==f'H{k}'):.3f}"
                        for v in vals) + " |")

    table(rows_sw, "sw", ["anchor", "hfd_block", "sn"], "Window length (samples; 100 = 1 s)")
    table(rows_k, "k_max", ["anchor", "hfd_block"], "Higuchi k_max (SW=120)")
    table(rows_w, "W", ["anchor"], "Trailing-mean span W (anchor test, SW=120)")
    table(rows_l, "L", ["hfd_block", "cusum_block"], "Block length L (SW=120)")
    md.append("\nCUSUM change-point estimate (s) by window length, normalized: " + ", ".join(
        f"SW={r['sw']} H{r['dim'][1]}: {r['cp_s']:.0f}" for r in rows_sw if r["normalized"]))
    with open(os.path.join(OUT, "p4_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    # figure: p-values vs window length
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(dims), figsize=(4.2 * len(dims), 6), sharey=True)
    for i, pw in enumerate((False, True)):
        for k in dims:
            ax = axes[i, k]
            for c, m in (("anchor", "o"), ("hfd_block", "s"), ("sn", "^")):
                ys = [next(r[c] for r in rows_sw if r["sw"] == sw and r["normalized"] == pw and r["dim"] == f"H{k}")
                      for sw in SWS]
                ax.semilogy([s / 100 for s in SWS], np.maximum(ys, 1e-4), marker=m, label=c)
            ax.axhline(0.05, color="grey", ls=":", lw=1)
            ax.set_title(f"H{k} {'normalized' if pw else 'raw'}", fontsize=9)
            ax.set_xlabel("window length (s)")
    axes[0, 0].set_ylabel("p-value")
    axes[1, 0].set_ylabel("p-value")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dtda.fig_path("p4", "p4_pvalues_vs_window.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
