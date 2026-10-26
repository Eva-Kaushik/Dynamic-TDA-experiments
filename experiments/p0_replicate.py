"""P0: replicate the anchor's EEG result from the terminal, then re-test it.

    python experiments/p0_replicate.py                 # full run (B=9999)
    python experiments/p0_replicate.py --B 999         # quicker
    python experiments/p0_replicate.py --per_window_std   # does the change survive amplitude normalization?

Outputs (results/p0/):
    tp_eeg.npy        cached total persistence (H0,H1,H2) per 1.2 s window
    p0_results.csv    HFD + lower-tail p-value for every (detrend, permutation, dim)
    p0_summary.md     readable table + replication check + change-point estimate
    p0_eeg.png        TP series, deviations and CUSUMs
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import dtda  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR_HFD = (1.0764366389939508, 1.1258100503908508, 1.212368772187317)  # notebook output


def holm(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "Epileptic Seizure EEG Dataset"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "p0"))
    ap.add_argument("--sw", type=int, default=120)
    ap.add_argument("--W", type=int, default=50)
    ap.add_argument("--k_max", type=int, default=10)
    ap.add_argument("--B", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--recompute", action="store_true", help="ignore cached total persistence")
    ap.add_argument("--per_window_std", action="store_true",
                    help="amplitude-confound check: rescale each window to unit variance before PH (G3/G9)")
    args = ap.parse_args()
    if args.per_window_std and args.out.endswith(os.path.join("results", "p0")):
        args.out += "_perwindowstd"
    os.makedirs(args.out, exist_ok=True)

    # ---- 1. total persistence per window (cached; H2 takes ~1-2 min)
    cache = os.path.join(args.out, f"tp_eeg_sw{args.sw}{'_pwstd' if args.per_window_std else ''}.npy")
    if os.path.exists(cache) and not args.recompute:
        tp = np.load(cache)
        print(f"[P0] loaded cached total persistence {tp.shape} from {cache}")
    else:
        ts = dtda.load_eeg(args.data)
        print(f"[P0] EEG loaded: T={ts.shape[0]} samples ({ts.shape[0]/dtda.FS:.0f} s), p={ts.shape[1]} channels")
        t0 = time.time()
        tp = dtda.window_total_persistence(ts, sw=args.sw, maxdim=2, per_window_standardize=args.per_window_std)
        np.save(cache, tp)
        print(f"[P0] total persistence for {tp.shape[0]} windows in {time.time()-t0:.0f} s")
    n = tp.shape[0]

    # ---- 2. replication check (anchor: trailing mean, W=50, k_max=10)
    rep = [dtda.hfd_of_cusum(dtda.detrend(tp[:, k], "trailing", args.W), args.k_max) for k in range(3)]
    print("\n[P0] Replication of anchor HFDs (trailing mean, W=50, k_max=10)")
    rep_ok = True
    for k in range(3):
        diff = abs(rep[k] - ANCHOR_HFD[k])
        rep_ok &= diff < 1e-6
        print(f"   H{k}: ours {rep[k]:.6f}   anchor {ANCHOR_HFD[k]:.6f}   |diff| {diff:.1e}")
    if args.per_window_std:
        print("   -> (not comparable: per-window normalization changes the pipeline on purpose)")
    else:
        print("   ->", "EXACT MATCH" if rep_ok else "MISMATCH (check package versions / channels)")

    # ---- 3. inference grid
    L_auto = dtda.default_block_len(n)
    configs = [("trailing", "window", None), ("trailing", "block", L_auto), ("trailing", "block", 20),
               ("global", "window", None), ("global", "block", L_auto), ("global", "block", 20)]
    rows = []
    print(f"\n[P0] Permutation tests, B={args.B}, lower tail (n={n} windows, auto block length={L_auto})")
    for ci, (det, perm, L) in enumerate(configs):
        ps = []
        for k in range(3):
            rng = np.random.default_rng([args.seed, ci, k])
            d = dtda.detrend(tp[:, k], det, args.W)
            r = dtda.hfd_test(d, args.k_max, args.B, perm, L or L_auto, rng)
            ps.append(r.p_value)
            rows.append(dict(detrend=det, perm=perm, block_len=L or "", dim=f"H{k}", hfd=r.hfd,
                             null_mean=r.null_mean, null_sd=r.null_sd, p=r.p_value))
        for row, padj in zip(rows[-3:], holm(ps)):
            row["p_holm"] = float(padj)
        tag = f"{det:8s} {perm:6s}{'' if L is None else f' L={L}':6s}"
        print("   " + tag + "  " + "  ".join(f"H{k}: HFD={rows[-3+k]['hfd']:.3f} p={rows[-3+k]['p']:.4f}" for k in range(3)))

    # ---- 4. where is the change? (classical CUSUM argmax on global-mean-centred TP)
    cps = {f"H{k}": dtda.cusum_changepoint(tp[:, k]) for k in range(3)}
    cp_sec = {k: v * args.sw / dtda.FS for k, v in cps.items()}
    print("\n[P0] CUSUM change-point estimate (window index -> seconds):",
          ", ".join(f"{k}: {cps[k]} -> {cp_sec[k]:.0f} s" for k in cps))

    # ---- 5. save
    import csv
    with open(os.path.join(args.out, "p0_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "p0_meta.json"), "w") as f:
        json.dump(dict(vars(args), n_windows=n, replication=rep, replication_exact=bool(rep_ok),
                       changepoint_window=cps, changepoint_sec=cp_sec, auto_block_len=L_auto), f, indent=2)

    with open(os.path.join(args.out, "p0_summary.md"), "w") as f:
        f.write(f"# P0 - anchor replication and re-test{' (per-window amplitude normalization)' if args.per_window_std else ''}\n\n")
        f.write(f"Windows: {n} x {args.sw/dtda.FS:.1f} s; W={args.W}; k_max={args.k_max}; B={args.B}; seed={args.seed}\n\n")
        f.write("## Replication\n\n| dim | ours | anchor |\n|---|---|---|\n")
        for k in range(3):
            f.write(f"| H{k} | {rep[k]:.6f} | {ANCHOR_HFD[k]:.6f} |\n")
        f.write("\n**Not comparable (per-window normalization)**\n\n" if args.per_window_std
                else f"\n**{'Exact match' if rep_ok else 'Mismatch'}**\n\n")
        f.write("## Lower-tail p-values (Holm-adjusted across H0-H2 in brackets)\n\n")
        f.write("| detrend | permutation | H0 | H1 | H2 |\n|---|---|---|---|---|\n")
        for i in range(0, len(rows), 3):
            r = rows[i:i + 3]
            perm = r[0]["perm"] + (f" (L={r[0]['block_len']})" if r[0]["block_len"] else "")
            f.write(f"| {r[0]['detrend']} | {perm} | " + " | ".join(
                f"{x['p']:.4f} [{x['p_holm']:.4f}] (HFD {x['hfd']:.3f})" for x in r) + " |\n")
        f.write("\n## CUSUM change-point estimate\n\n" + ", ".join(f"{k}: window {cps[k]} = {cp_sec[k]:.0f} s" for k in cps) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(n) * args.sw / dtda.FS
        fig, ax = plt.subplots(3, 3, figsize=(14, 7), sharex=True)
        for k in range(3):
            ax[0, k].plot(t, tp[:, k], lw=.8); ax[0, k].set_title(f"H{k} total persistence")
            ax[1, k].plot(t, dtda.detrend(tp[:, k], "trailing", args.W), lw=.8, label="trailing W=50")
            ax[1, k].plot(t, dtda.detrend(tp[:, k], "global"), lw=.8, alpha=.7, label="global mean")
            ax[1, k].set_title("deviations D_t")
            ax[2, k].plot(t, np.cumsum(dtda.detrend(tp[:, k], "trailing", args.W)), lw=.8)
            ax[2, k].plot(t, np.cumsum(dtda.detrend(tp[:, k], "global")), lw=.8, alpha=.7)
            ax[2, k].axvline(cp_sec[f"H{k}"], color="k", ls=":", lw=1)
            ax[2, k].set_title("CUSUM S_t (dotted: CUSUM change-point)")
            ax[2, k].set_xlabel("time (s)")
        ax[1, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "p0_eeg.png"), dpi=130)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(f"\n[P0] wrote {args.out}/p0_summary.md, p0_results.csv, p0_eeg.png")


if __name__ == "__main__":
    main()
