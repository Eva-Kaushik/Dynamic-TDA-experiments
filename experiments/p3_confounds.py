"""P3: is the seizure-time topology change more than amplitude, rhythm and
linear structure?  (Gaps G9, G10, G11.)  All on the anchor EEG.

(a) Surrogates (G10). A pre-ictal segment and an ictal segment are each replaced
    by multivariate surrogates that keep every auto- and cross-spectrum
    (Prichard-Theiler phase randomization) and, for IAAFT, also each channel's
    value distribution. If the real ictal-minus-pre-ictal increase in total
    persistence is no larger than the surrogates' increase, the "topological"
    change is explained by linear (spectral/amplitude) changes.
(b) Summaries (G9). Total persistence p=1 (anchor) vs p=2, max persistence,
    persistent entropy, first-landscape L2 norm.
(c) Embeddings (G11). Time points as points (anchor) vs channels as points with
    correlation distance (TADA-like) vs delay embedding of the first PC.

    python experiments/p3_confounds.py                # ~10-15 min on an M-series Mac
    python experiments/p3_confounds.py --part a --n_surr 99
    python experiments/p3_confounds.py --part bc

Outputs: results/p3/*.csv, results/p3/p3_summary.md, figures/p3/*.png
"""
import argparse
import csv
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import dtda  # noqa: E402

OUT = os.path.join(dtda.RES_DIR, "p3")
DATA = os.path.join(dtda.REPO_ROOT, "Epileptic Seizure EEG Dataset")
SW, FS = 120, dtda.FS
ONSET_S = 363          # seizure onset used in the review (s)
PRE = (240, 340)       # pre-ictal segment (s)
ICT = (365, 425)       # ictal segment (s)


def seg_tp(X, maxdim, pw):
    return dtda.window_total_persistence(X, SW, maxdim, per_window_standardize=pw).mean(axis=0)


def job_surr(a):
    kind, i, maxdim, seed = a
    ts = dtda.load_eeg(DATA)
    rng = np.random.default_rng([seed, i, 0 if kind == "PT" else 1])
    out = {}
    for name, (s, e) in (("pre", PRE), ("ict", ICT)):
        X = ts[s * FS:e * FS]
        Y = dtda.phase_randomize_mv(X, rng) if kind == "PT" else dtda.iaaft_mv(X, rng)
        for pw in (False, True):
            out[(name, pw)] = seg_tp(Y, maxdim, pw)
    return kind, i, out


def diagrams_cached(ts, pw, maxdim=2):
    path = os.path.join(dtda.RES_DIR, "_cache", f"dgms_sw{SW}_md{maxdim}_{'pw' if pw else 'raw'}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    d = dtda.window_diagrams(ts, SW, maxdim, per_window_standardize=pw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(d, f)
    return d


def tests_on(series, B, seed):
    rng = np.random.default_rng(seed)
    return dict(sn=dtda.sn_test(series), lrv=dtda.lrv_cusum_test(series),
                hfd_block=dtda.hfd_block_test(series, B, rng=rng),
                cp_s=dtda.argmax_changepoint(series) * SW / FS)


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "bc", "all"], default="all")
    ap.add_argument("--n_surr", type=int, default=39, help="surrogates per type (39 -> min two-sided p=0.05)")
    ap.add_argument("--maxdim", type=int, default=2)
    ap.add_argument("--B", type=int, default=999)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ts = dtda.load_eeg(DATA)
    dtda.sn_null(len(dtda.window_bounds(ts.shape[0], SW)))
    md = ["# P3 - confounds: surrogates, summaries, embeddings (anchor EEG)\n"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------ (a) surrogates
    if args.part in ("a", "all"):
        t0 = time.time()
        real = {}
        for name, (s, e) in (("pre", PRE), ("ict", ICT)):
            for pw in (False, True):
                real[(name, pw)] = seg_tp(ts[s * FS:e * FS], args.maxdim, pw)
        jobs = [(k, i, args.maxdim, args.seed) for k in ("PT", "IAAFT") for i in range(args.n_surr)]
        print(f"[P3:a] {len(jobs)} surrogate pairs (pre + ictal) ...", flush=True)
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(job_surr, jobs, chunksize=1))
        rows = []
        for kind in ("PT", "IAAFT"):
            sur = [r[2] for r in res if r[0] == kind]
            for pw in (False, True):
                for k in range(args.maxdim + 1):
                    d_real = real[("ict", pw)][k] - real[("pre", pw)][k]
                    d_sur = np.array([s[("ict", pw)][k] - s[("pre", pw)][k] for s in sur])
                    lv_real = real[("ict", pw)][k]
                    lv_sur = np.array([s[("ict", pw)][k] for s in sur])
                    rows.append(dict(
                        surrogate=kind, normalized=pw, dim=f"H{k}",
                        real_pre=real[("pre", pw)][k], real_ict=lv_real, real_delta=d_real,
                        sur_delta_mean=d_sur.mean(), sur_delta_sd=d_sur.std(ddof=1),
                        z_delta=(d_real - d_sur.mean()) / (d_sur.std(ddof=1) + 1e-12),
                        p_delta_two_sided=min(1.0, 2 * min(1 + np.sum(d_sur >= d_real),
                                                           1 + np.sum(d_sur <= d_real)) / (len(d_sur) + 1)),
                        sur_ict_mean=lv_sur.mean(),
                        z_ict_level=(lv_real - lv_sur.mean()) / (lv_sur.std(ddof=1) + 1e-12),
                        p_ict_level_two_sided=min(1.0, 2 * min((1 + np.sum(lv_sur >= lv_real)),
                                                               (1 + np.sum(lv_sur <= lv_real))) / (len(lv_sur) + 1))))
        write_csv(rows, os.path.join(OUT, "p3a_surrogates.csv"))
        md += [f"\n## (a) Surrogate test of the ictal change (G10) - {args.n_surr} surrogates per type, "
               f"pre {PRE} s vs ictal {ICT} s, {time.time()-t0:.0f} s\n",
               "Delta = mean total persistence (ictal) - (pre-ictal). p_delta: does the real Delta differ from "
               "the Delta of surrogates that keep the spectra (and, for IAAFT, the amplitude distributions)? "
               "z_ict: is the ictal level itself beyond its own surrogates (nonlinearity within the seizure)?\n",
               "| surrogate | normalized | dim | real Delta | surrogate Delta (mean +/- sd) | z | p_delta (2-sided) | z_ict | p_ict (2-sided) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['surrogate']} | {r['normalized']} | {r['dim']} | {r['real_delta']:.3g} | "
                      f"{r['sur_delta_mean']:.3g} +/- {r['sur_delta_sd']:.2g} | {r['z_delta']:.1f} | "
                      f"{r['p_delta_two_sided']:.3f} | {r['z_ict_level']:.1f} | {r['p_ict_level_two_sided']:.3f} |")
        # figure
        fig, axes = plt.subplots(2, args.maxdim + 1, figsize=(4 * (args.maxdim + 1), 6))
        for i, pw in enumerate((False, True)):
            for k in range(args.maxdim + 1):
                ax = axes[i, k]
                for j, kind in enumerate(("PT", "IAAFT")):
                    sur = [r[2] for r in res if r[0] == kind]
                    d_sur = [s[("ict", pw)][k] - s[("pre", pw)][k] for s in sur]
                    ax.scatter(np.full(len(d_sur), j) + np.random.default_rng(0).uniform(-.1, .1, len(d_sur)),
                               d_sur, s=12, alpha=.6, label=f"{kind} surrogates")
                ax.axhline(real[("ict", pw)][k] - real[("pre", pw)][k], color="r", lw=2, label="real EEG")
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["PT", "IAAFT"])
                ax.set_title(f"H{k} {'normalized' if pw else 'raw'}: ictal - pre-ictal TP", fontsize=9)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dtda.fig_path("p3", "p3a_surrogates.png"), dpi=140)
        plt.close(fig)

    # ------------------------------------------------- (b) summaries, (c) embeddings
    if args.part in ("bc", "all"):
        t0 = time.time()
        print("[P3:b] persistence summaries (diagrams cached after first run) ...", flush=True)
        rows = []
        kinds = ["tp1", "tp2", "maxpers", "entropy", "landscape1"]
        series_store = {}
        for pw in (False, True):
            dg = diagrams_cached(ts, pw)
            for kind in kinds:
                S = dtda.summaries_from_diagrams(dg, kind)
                for k in range(S.shape[1]):
                    t = tests_on(S[:, k], args.B, [args.seed, kinds.index(kind), k, pw])
                    rows.append(dict(part="b summary", variant=kind, normalized=pw, dim=f"H{k}", **t))
                    series_store[(kind, pw, k)] = S[:, k]
        print("[P3:c] embeddings ...", flush=True)
        emb = {"corr-graph (channels as points)": lambda pw: dtda.corr_graph_tp(ts, SW, 1),
               "delay embedding of PC1 (d=3, tau=50 ms)": lambda pw: dtda.delay_embed_tp(
                   ts, SW, 3, 5, 1, per_window_standardize=pw)}
        for name, fn in emb.items():
            for pw in ((True,) if name.startswith("corr") else (False, True)):
                S = fn(pw)
                for k in range(S.shape[1]):
                    t = tests_on(S[:, k], args.B, [args.seed, 99, k, pw, len(name)])
                    rows.append(dict(part="c embedding", variant=name, normalized=pw, dim=f"H{k}", **t))
                    series_store[(name, pw, k)] = S[:, k]
        write_csv(rows, os.path.join(OUT, "p3bc_summaries_embeddings.csv"))
        md += [f"\n## (b) Persistence summaries and (c) embeddings - p-values ({time.time()-t0:.0f} s)\n",
               "sn = self-normalized CUSUM, lrv = long-run-variance CUSUM, hfd_block = HFD with global "
               "mean + block permutation (L=20). cp = CUSUM change-point estimate (onset in review: 363 s). "
               "'normalized' = per-window amplitude normalization before persistence (corr-graph is scale-free).\n",
               "| part | variant | normalized | dim | sn | lrv | hfd_block | cp (s) |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['part']} | {r['variant']} | {r['normalized']} | {r['dim']} | {r['sn']:.4f} | "
                      f"{r['lrv']:.2g} | {r['hfd_block']:.4f} | {r['cp_s']:.0f} |")
        keys = [k for k in series_store if k[1]]
        fig, axes = plt.subplots(len(keys) // 3 + 1, 3, figsize=(14, 2.1 * (len(keys) // 3 + 1)))
        t_axis = np.arange(len(next(iter(series_store.values())))) * SW / FS
        for ax, key in zip(axes.flat, keys):
            ax.plot(t_axis, series_store[key], lw=.6)
            ax.axvline(ONSET_S, color="r", ls=":", lw=1)
            ax.set_title(f"{key[0]} H{key[2]} (normalized)", fontsize=8)
        for ax in list(axes.flat)[len(keys):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(dtda.fig_path("p3", "p3bc_series_normalized.png"), dpi=120)
        plt.close(fig)

    with open(os.path.join(OUT, f"p3_summary{'' if args.part == 'all' else '_' + args.part}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
