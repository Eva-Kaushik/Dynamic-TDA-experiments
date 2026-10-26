"""P6: external validation on a CHB-MIT subset (PhysioNet, open access).  (Gap G13.)

For each patient: up to N seizure recordings and one seizure-free recording.
Each recording is cut to 500 s exactly like the anchor EEG (seizure onset at
363 s; seizure-free: 1200-1700 s), the 18 standard bipolar channels are kept
(18 channels, like the anchor), resampled 256 -> 100 Hz, z-scored, and run
through the same pipeline. Methods (decision = Holm across H0-H2 at 5%):

  anchor        raw total persistence, trailing mean + window permutation (published)
  hfd_block     per-window normalized TP, global mean + block permutation L=20
  sn            per-window normalized TP, self-normalized CUSUM
  amp-baseline  log window variance, self-normalized CUSUM (no topology at all)

A good topological test fires on seizure segments and stays quiet on
seizure-free ones; if amp-baseline does as well, topology adds little.

    python experiments/p6_chbmit.py                         # chb01, chb03, chb05 (~400 MB download)
    python experiments/p6_chbmit.py --patients chb01 chb02 chb03 chb05 chb08 --per_patient 3
    python experiments/p6_chbmit.py --maxdim 1              # faster

Downloads go to data/chbmit/ (git-ignored). Source: https://physionet.org/content/chbmit/1.0.0/
Outputs: results/p6/p6_results.csv, results/p6/p6_summary.md, figures/p6/*.png
"""
import argparse
import csv
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(__file__))
import dtda  # noqa: E402

BASE = "https://physionet.org/files/chbmit/1.0.0"
DATA_DIR = os.path.join(dtda.REPO_ROOT, "data", "chbmit")
OUT = os.path.join(dtda.RES_DIR, "p6")
CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ"]
PRE_S, POST_S, SEG_S = 363, 137, 500   # same geometry as the anchor EEG
FREE_START = 1200
SW, FS = 120, 100


# ----------------------------------------------------------------- download
def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "dynamic-tda-experiments"})
    print(f"   downloading {url}", flush=True)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, path)
    return path


def parse_summary(text):
    """-> {file: [(start_s, end_s), ...]} from chbXX-summary.txt."""
    out, cur = {}, None
    starts = []
    for line in text.splitlines():
        m = re.match(r"\s*File Name:\s*(\S+)", line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        if cur is None:
            continue
        m = re.match(r"\s*Seizure\s*\d*\s*Start Time:\s*(\d+)", line)
        if m:
            starts.append(int(m.group(1)))
            continue
        m = re.match(r"\s*Seizure\s*\d*\s*End Time:\s*(\d+)", line)
        if m and starts:
            out[cur].append((starts.pop(0), int(m.group(1))))
    return out


# ----------------------------------------------------------------- EDF reader
def read_edf(path, labels=CHANNELS):
    """Minimal EDF reader -> (data (T, len(labels)) in physical units, fs)."""
    with open(path, "rb") as f:
        hdr = f.read(256)
        n_rec = int(hdr[236:244])
        dur = float(hdr[244:252])
        ns = int(hdr[252:256])
        sh = f.read(256 * ns)

        def field(off, width):
            return [sh[off + i * width: off + (i + 1) * width].decode("ascii", "ignore").strip() for i in range(ns)]
        lab = field(0, 16)
        o = 16 * ns + 80 * ns + 8 * ns
        pmin = np.array(field(o, 8), float); o += 8 * ns
        pmax = np.array(field(o, 8), float); o += 8 * ns
        dmin = np.array(field(o, 8), float); o += 8 * ns
        dmax = np.array(field(o, 8), float); o += 8 * ns
        o += 80 * ns
        spr = np.array(field(o, 8), int)
        raw = np.frombuffer(f.read(), dtype="<i2")
    rec_len = spr.sum()
    n_rec = min(n_rec, len(raw) // rec_len) if n_rec > 0 else len(raw) // rec_len
    raw = raw[:n_rec * rec_len].reshape(n_rec, rec_len)
    offs = np.r_[0, np.cumsum(spr)]
    up = [l.upper() for l in lab]
    cols, fs = [], None
    for want in labels:
        idx = next((i for i, l in enumerate(up) if l == want or l.startswith(want + "-")), None)
        if idx is None:
            raise KeyError(f"{os.path.basename(path)}: channel {want} not found")
        sig = raw[:, offs[idx]:offs[idx + 1]].reshape(-1).astype(float)
        gain = (pmax[idx] - pmin[idx]) / (dmax[idx] - dmin[idx])
        cols.append((sig - dmin[idx]) * gain + pmin[idx])
        fs = spr[idx] / dur
    return np.stack(cols, axis=1), fs


def prepare(path, start_s):
    X, fs = read_edf(path)
    X = X[int(start_s * fs): int((start_s + SEG_S) * fs)]
    if X.shape[0] < int(SEG_S * fs) - 1:
        raise ValueError("segment runs past the end of the file")
    if int(round(fs)) != FS:
        from math import gcd
        g = gcd(int(round(fs)), FS)
        X = resample_poly(X, FS // g, int(round(fs)) // g, axis=0)
    X = X[:SEG_S * FS]
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


# ----------------------------------------------------------------- analysis
def analyze(a):
    try:
        return _analyze(a)
    except Exception as e:  # a bad file should not kill the whole run
        print(f"   skipped {a[0]}: {e}", flush=True)
        return None


def _analyze(a):
    rec_id, path, start_s, has_seizure, maxdim, B, seed = a
    X = prepare(path, start_s)
    tp_raw = dtda.window_total_persistence(X, SW, maxdim)
    tp_pw = dtda.window_total_persistence(X, SW, maxdim, per_window_standardize=True)
    amp = np.array([np.log(X[s:e].var(0).mean()) for s, e in dtda.window_bounds(X.shape[0], SW)])
    rng = lambda *k: np.random.default_rng([seed, *k])  # noqa: E731
    res = dict(recording=rec_id, seizure=has_seizure, start_s=start_s)
    ps = {"anchor": [dtda.anchor_test(tp_raw[:, k], B, rng=rng(k, 0)) for k in range(maxdim + 1)],
          "hfd_block": [dtda.hfd_block_test(tp_pw[:, k], B, rng=rng(k, 1)) for k in range(maxdim + 1)],
          "sn": [dtda.sn_test(tp_pw[:, k]) for k in range(maxdim + 1)],
          "amp-baseline": [dtda.sn_test(amp)]}
    for m, p in ps.items():
        adj = dtda.holm(p)
        res[f"{m}_detect"] = bool(adj.min() <= 0.05)
        res[f"{m}_p_min_holm"] = float(adj.min())
        series = {"anchor": tp_raw, "hfd_block": tp_pw, "sn": tp_pw}.get(m)
        x = amp if series is None else series[:, int(np.argmin(p))]
        res[f"{m}_cp_err_s"] = (dtda.argmax_changepoint(x) * SW / FS - PRE_S) if has_seizure else ""
    return res, tp_pw, amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="+", default=["chb01", "chb03", "chb05"])
    ap.add_argument("--per_patient", type=int, default=2, help="seizure recordings per patient")
    ap.add_argument("--maxdim", type=int, default=2)
    ap.add_argument("--B", type=int, default=999)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--data_dir", default=DATA_DIR)
    args = ap.parse_args()
    t0 = time.time()

    jobs = []
    for pat in args.patients:
        sp = fetch(f"{BASE}/{pat}/{pat}-summary.txt", os.path.join(args.data_dir, pat, f"{pat}-summary.txt"))
        with open(sp) as f:
            seiz = parse_summary(f.read())
        chosen = 0
        for fn, sz in seiz.items():
            if chosen >= args.per_patient or not sz:
                continue
            on = sz[0][0]
            start = on - PRE_S
            if start < 0:      # need 363 s of recording before the onset
                continue
            jobs.append((f"{fn} (seizure @{on}s)", os.path.join(args.data_dir, pat, fn), start, True))
            chosen += 1
        free = next((fn for fn, sz in seiz.items() if not sz), None)
        if free:
            jobs.append((f"{free} (seizure-free)", os.path.join(args.data_dir, pat, free), FREE_START, False))
    print(f"[P6] {len(jobs)} recordings from {', '.join(args.patients)}", flush=True)
    for rec_id, path, _, _ in jobs:
        try:
            fetch(f"{BASE}/{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}", path)
        except Exception as e:
            print(f"   download failed for {rec_id}: {e}", flush=True)
    jobs = [j for j in jobs if os.path.exists(j[1])]
    print(f"[P6] data ready ({time.time()-t0:.0f} s); analysing ...", flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        out = [o for o in ex.map(analyze, [(r, p, s, h, args.maxdim, args.B, args.seed) for r, p, s, h in jobs])
               if o is not None]
    rows = [o[0] for o in out]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "p6_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    methods = ["anchor", "hfd_block", "sn", "amp-baseline"]
    md = [f"# P6 - CHB-MIT subset ({', '.join(args.patients)}; {len(rows)} recordings; {time.time()-t0:.0f} s)\n",
          "Detect = Holm-adjusted p <= 0.05 across H0-H2. cp err = CUSUM change estimate minus "
          "annotated onset (s), seizure segments only.\n",
          "| recording | " + " | ".join(f"{m} detect (p) | {m} cp err" for m in methods) + " |",
          "|---|" + "---|---|" * len(methods)]
    for r in rows:
        cells = []
        for m in methods:
            cells.append(f"{'YES' if r[m+'_detect'] else 'no'} ({r[m+'_p_min_holm']:.3f})")
            cells.append(f"{r[m+'_cp_err_s']:+.0f}" if r["seizure"] else "-")
        md.append(f"| {r['recording']} | " + " | ".join(cells) + " |")
    sz = [r for r in rows if r["seizure"]]
    fr = [r for r in rows if not r["seizure"]]
    md.append("\n| method | sensitivity (seizure segments detected) | false alarms (seizure-free flagged) | median abs cp error (s) |")
    md.append("|---|---|---|---|")
    for m in methods:
        errs = [abs(r[m + "_cp_err_s"]) for r in sz if r[m + "_detect"]]
        md.append(f"| {m} | {sum(r[m+'_detect'] for r in sz)}/{len(sz)} | {sum(r[m+'_detect'] for r in fr)}/{len(fr)} | "
                  f"{np.median(errs):.0f} |" if errs else f"| {m} | {sum(r[m+'_detect'] for r in sz)}/{len(sz)} | "
                  f"{sum(r[m+'_detect'] for r in fr)}/{len(fr)} | - |")
    with open(os.path.join(OUT, "p6_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(out)
    fig, axes = plt.subplots(n, 2, figsize=(12, 1.7 * n), squeeze=False)
    for i, (r, tp_pw, amp) in enumerate(out):
        t = np.arange(len(amp)) * SW / FS
        axes[i, 0].plot(t, tp_pw[:, 1], lw=.6)
        axes[i, 0].set_title(f"{r['recording']}: H1 TP (normalized)", fontsize=8)
        axes[i, 1].plot(t, amp, lw=.6, color="tab:orange")
        axes[i, 1].set_title("log window variance (amp-baseline)", fontsize=8)
        for ax in axes[i]:
            if r["seizure"]:
                ax.axvline(PRE_S, color="r", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(dtda.fig_path("p6", "p6_recordings.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
