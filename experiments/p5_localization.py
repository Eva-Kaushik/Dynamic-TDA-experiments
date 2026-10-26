"""P5: where is the change, and how sure are we?  (Gap G8.)  Anchor EEG.

  * single change: CUSUM argmax, with a 95% moving-block bootstrap confidence
    interval (residuals around the fitted two-mean model, block length 20)
  * multiple changes: binary segmentation with the self-normalized CUSUM
    (alpha 0.05, min segment 25 windows = 30 s)
  * error against the clinical onset used in the review (363 s)

    python experiments/p5_localization.py          # ~1-3 min (reuses P4/P0 caches)

Outputs: results/p5/p5_localization.csv, results/p5/p5_summary.md, figures/p5/*.png
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import dtda  # noqa: E402
from p4_sensitivity import tp_cached  # noqa: E402

OUT = os.path.join(dtda.RES_DIR, "p5")
SW, FS, ONSET_S = 120, dtda.FS, 363


def bootstrap_ci(x, B=999, L=20, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n = len(x)
    tau = dtda.argmax_changepoint(x)
    fit = np.r_[np.full(tau, x[:tau].mean()), np.full(n - tau, x[tau:].mean())]
    e = x - fit
    taus = []
    for _ in range(B):
        starts = rng.integers(0, n - L + 1, int(np.ceil(n / L)))
        eb = np.concatenate([e[s:s + L] for s in starts])[:n]
        taus.append(dtda.argmax_changepoint(fit + eb))
    lo, hi = np.percentile(taus, [2.5, 97.5])
    return tau, lo, hi


def binseg(x, offset=0, alpha=0.05, min_seg=25, depth=0, max_depth=4):
    if len(x) < 2 * min_seg or depth > max_depth:
        return []
    if dtda.sn_test_any_n(x) > alpha:
        return []
    k = dtda.argmax_changepoint(x)
    if k < min_seg or len(x) - k < min_seg:
        return []
    return (binseg(x[:k], offset, alpha, min_seg, depth + 1, max_depth) + [offset + k] +
            binseg(x[k:], offset + k, alpha, min_seg, depth + 1, max_depth))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxdim", type=int, default=2)
    ap.add_argument("--B", type=int, default=999)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    store = {}
    for pw in (False, True):
        tp = tp_cached(SW, pw, args.maxdim)
        for k in range(tp.shape[1]):
            x = tp[:, k]
            tau, lo, hi = bootstrap_ci(x, args.B, rng=np.random.default_rng([args.seed, k, pw]))
            cps = binseg(x)
            s = lambda w: w * SW / FS  # noqa: E731
            rows.append(dict(normalized=pw, dim=f"H{k}", cp_s=s(tau), ci_lo_s=s(lo), ci_hi_s=s(hi),
                             error_vs_onset_s=s(tau) - ONSET_S,
                             onset_in_ci=bool(s(lo) <= ONSET_S <= s(hi) + SW / FS),
                             binseg_changes_s=";".join(f"{s(c):.0f}" for c in cps)))
            store[(pw, k)] = (x, tau, lo, hi, cps)
    with open(os.path.join(OUT, "p5_localization.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = ["# P5 - localization on the anchor EEG\n",
          f"Onset reference: {ONSET_S} s. CI = 95% moving-block bootstrap (L=20, B={args.B}). "
          "Binary segmentation: self-normalized CUSUM, alpha=0.05, min segment 30 s.\n",
          "| normalized | dim | change (s) | 95% CI (s) | error vs onset (s) | onset in CI | binary segmentation (s) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['normalized']} | {r['dim']} | {r['cp_s']:.0f} | [{r['ci_lo_s']:.0f}, {r['ci_hi_s']:.0f}] | "
                  f"{r['error_vs_onset_s']:+.0f} | {r['onset_in_ci']} | {r['binseg_changes_s'] or '-'} |")
    with open(os.path.join(OUT, "p5_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nd = args.maxdim + 1
    fig, axes = plt.subplots(2, nd, figsize=(4.6 * nd, 5.5), sharex=True)
    for i, pw in enumerate((False, True)):
        for k in range(nd):
            x, tau, lo, hi, cps = store[(pw, k)]
            t = np.arange(len(x)) * SW / FS
            ax = axes[i, k]
            ax.plot(t, x, lw=.6, color="0.3")
            ax.axvspan(lo * SW / FS, hi * SW / FS, color="tab:blue", alpha=.25, label="95% CI")
            ax.axvline(tau * SW / FS, color="tab:blue", lw=1.2, label="CUSUM change")
            for c in cps:
                ax.axvline(c * SW / FS, color="tab:green", ls="--", lw=.9)
            ax.axvline(ONSET_S, color="r", ls=":", lw=1.2, label="onset 363 s")
            ax.set_title(f"H{k} {'normalized' if pw else 'raw'}", fontsize=9)
            if i == 1:
                ax.set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("P5: change location (green dashed = binary segmentation)", fontsize=10)
    fig.tight_layout()
    fig.savefig(dtda.fig_path("p5", "p5_localization.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
