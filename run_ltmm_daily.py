"""
LTMM Experiment 3: daily-life gait from the 3-day recordings
============================================================

Why: 1-minute lab walks plateau at AUC ~0.7 (Experiments 1-2). The original
LTMM study (Weiss et al. 2013) and Askhatova et al. 2026 (TCN, AUC 0.749) both
use the 3-day free-living recordings, which contain far more walking.

PRE-REGISTERED PROTOCOL (fixed before running on real data)
  Sampling : 12 slices x 30 min per recording, evenly spaced (6 h/subject),
             fetched with HTTP range requests (~26 MB/subject, ~1.9 GB total)
  Walking  : label-free detector in agent/daily_gait.py (bouts >= 20 s)
  Features : per-subject medians of v2 gait features over bouts + walking
             minutes per hour + median bout length (a priori directions)
  Models   : D0 daily-life index (untrained)
             D1 combined lab + daily index (untrained)          <- primary
             D2 logistic regression, daily features (20x5 CV)
             D3 logistic regression, lab + daily, SelectKBest k=6 inside folds (20x5 CV)
             M0 lab v2 index on the same subjects (reference)
  Chance   : label-permutation tests for D1 (1000x) and D3 (200x)
  Reporting: all models reported; one run; no changes after seeing results.

Usage:
  py run_ltmm_daily.py                 # download slices (resumable) + analyse
  py run_ltmm_daily.py --workers 6     # more parallel downloads
Disk: ~2 GB in data/ltmm/daily_slices/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent.daily_gait import DAILY_DIRECTION, daily_features_for_chunks, daily_index
from agent.gait_features import RISK_DIRECTION_V2, gait_instability_index_v2
from run_ltmm_validation import auc, bootstrap_ci

warnings.filterwarnings("ignore")
BASE_URL = "https://physionet.org/files/ltmm/1.0.0"
OUT_DIR = "evaluation_charts"
SEED = 2026
UA = {"User-Agent": "MedXAI-LTMM-daily-validation"}


# ── download ───────────────────────────────────────────────────────────────
def _get(url: str, headers: dict | None = None, timeout: int = 120):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def list_records(data_dir: str) -> list[str]:
    path = os.path.join(data_dir, "RECORDS")
    if not os.path.exists(path):
        with _get(f"{BASE_URL}/RECORDS") as r, open(path, "wb") as f:
            f.write(r.read())
    names = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]
    return [n for n in names if "/" not in n and n[:2].upper() in ("CO", "FL")]


def _read_ranges(url: str, ranges: list[tuple[int, int]]) -> list[bytes]:
    """Fetch byte ranges [start, end) with HTTP Range; falls back to streaming if Range is ignored."""
    out = []
    for s, e in ranges:
        for attempt in range(4):
            try:
                with _get(url, {"Range": f"bytes={s}-{e - 1}"}) as r:
                    if r.status == 206:
                        out.append(r.read())
                        break
                    # server ignored Range: stream and cut (slow but correct)
                    buf, pos = bytearray(), 0
                    while pos < e:
                        block = r.read(1 << 20)
                        if not block:
                            break
                        lo, hi = max(s - pos, 0), min(e - pos, len(block))
                        if lo < hi:
                            buf += block[lo:hi]
                        pos += len(block)
                    out.append(bytes(buf))
                    break
            except Exception as ex:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
    return out


def fetch_subject(name: str, data_dir: str, n_chunks: int, chunk_min: float) -> str:
    import wfdb
    sdir = os.path.join(data_dir, "daily_slices")
    os.makedirs(sdir, exist_ok=True)
    out_path = os.path.join(sdir, f"{name}_{n_chunks}x{int(chunk_min)}m.npz")
    if os.path.exists(out_path):
        return f"{name}: cached"
    hea = os.path.join(sdir, f"{name}.hea")
    if not os.path.exists(hea):
        with _get(f"{BASE_URL}/{name}.hea") as r, open(hea, "wb") as f:
            f.write(r.read())
    h = wfdb.rdheader(os.path.join(sdir, name))
    if not hasattr(h, "file_name") or h.file_name is None:
        return f"{name}: SKIPPED (multi-segment header)"
    if any(str(fmt) != "16" for fmt in h.fmt):
        return f"{name}: SKIPPED (format {set(h.fmt)} not supported)"
    fs, sig_len = float(h.fs), int(h.sig_len)
    frames = int(chunk_min * 60 * fs)
    if sig_len < frames:
        return f"{name}: SKIPPED (recording too short)"
    starts = np.linspace(0, sig_len - frames, n_chunks).astype(int)

    files = {}
    for i, fn in enumerate(h.file_name):
        files.setdefault(fn, []).append(i)
    raw = np.zeros((n_chunks, frames, h.n_sig), dtype=np.int16)
    for fn, idxs in files.items():
        fb = 2 * len(idxs)
        off = int(h.byte_offset[idxs[0]] or 0) if h.byte_offset else 0
        ranges = [(off + int(s) * fb, off + (int(s) + frames) * fb) for s in starts]
        blobs = _read_ranges(f"{BASE_URL}/{fn}", ranges)
        for c, blob in enumerate(blobs):
            arr = np.frombuffer(blob, dtype="<i2")
            n_fr = len(arr) // len(idxs)
            arr = arr[: n_fr * len(idxs)].reshape(n_fr, len(idxs))
            raw[c, :n_fr, idxs] = arr.T
    np.savez_compressed(out_path, raw=raw, gain=np.array(h.adc_gain, float), baseline=np.array(h.baseline, float),
                        sig_name=np.array(h.sig_name), fs=fs, starts=starts, sig_len=sig_len)
    return f"{name}: ok ({sig_len / fs / 3600:.1f} h recording, {n_chunks} x {chunk_min:.0f} min sampled)"


def load_chunks(path: str) -> tuple[list[dict], float]:
    d = np.load(path, allow_pickle=False)
    raw, gain, base, names, fs = d["raw"], d["gain"], d["baseline"], [str(s).lower() for s in d["sig_name"]], float(d["fs"])
    chunks = []
    for c in range(raw.shape[0]):
        phys = (raw[c].astype(float) - base) / gain
        sig = {n: phys[:, i] for i, n in enumerate(names)}
        chunks.append({"v": sig.get("v-acceleration"), "ml": sig.get("ml-acceleration"),
                       "ap": sig.get("ap-acceleration"), "yaw": sig.get("yaw-velocity")})
    return chunks, fs


# ── models ─────────────────────────────────────────────────────────────────
def make_lr(k=None):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    steps = [SimpleImputer(strategy="median"), StandardScaler()]
    if k:
        steps.append(SelectKBest(f_classif, k=k))
    steps.append(LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))
    return make_pipeline(*steps)


def repeated_cv(fn, X, y, repeats=20):
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold
    aucs, first = [], None
    for r in range(repeats):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, y):
            m = clone(fn()).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(auc(oof, y))
        first = oof if first is None else first
    return float(np.mean(aucs)), float(np.std(aucs)), first


def perm_p_untrained(score, y, n=1000):
    rng = np.random.default_rng(SEED)
    real = auc(score, y)
    null = np.array([auc(score, rng.permutation(y)) for _ in range(n)])
    return real, float((np.sum(null >= real) + 1) / (n + 1)), null


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join("data", "ltmm"))
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--chunk-min", type=float, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.data_dir, exist_ok=True)

    names = list_records(a.data_dir) if not a.no_download else sorted(
        {f.split("_")[0] for f in os.listdir(os.path.join(a.data_dir, "daily_slices")) if f.endswith(".npz")})
    print(f"3-day records: {len(names)} | sampling {a.chunks} x {a.chunk_min:.0f} min per recording "
          f"(~{a.chunks * a.chunk_min * 60 * 100 * 12 / 1e6:.0f} MB each before compression)")
    if not a.no_download:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(fetch_subject, n, a.data_dir, a.chunks, a.chunk_min): n for n in names}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    msg = f.result()
                except Exception as e:
                    msg = f"{futs[f]}: FAILED ({e})"
                print(f"  [{i}/{len(names)}] {msg}  ({time.time() - t0:.0f}s)")

    # ── daily-life features (cached) ──
    cache_path = os.path.join(a.data_dir, f"daily_features_{a.chunks}x{int(a.chunk_min)}m.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    rows = []
    print("\nDetecting walking bouts and extracting daily-life gait ...")
    for n in names:
        p = os.path.join(a.data_dir, "daily_slices", f"{n}_{a.chunks}x{int(a.chunk_min)}m.npz")
        if not os.path.exists(p):
            continue
        if n not in cache:
            chunks, fs = load_chunks(p)
            cache[n] = daily_features_for_chunks(chunks, fs)
            json.dump(cache, open(cache_path, "w"), default=float)
        rows.append({"subject": n.upper(), "faller": int(n.upper().startswith("FL")), **cache[n]})
    if len(rows) < 20:
        print("Too few subjects processed.")
        sys.exit(1)

    y = np.array([r["faller"] for r in rows])
    nb = np.array([r["d_n_bouts"] for r in rows], float)
    wm = np.array([r["d_walking_min_per_hour"] for r in rows], float)
    print(f"Subjects: {len(rows)} ({int(y.sum())} fallers, {int((1 - y).sum())} non-fallers)")
    print(f"QC (label-free): bouts/subject median {np.median(nb):.0f} (range {nb.min():.0f}-{nb.max():.0f}); "
          f"walking {np.nanmedian(wm):.1f} min per sampled hour; subjects with < 3 bouts: {int(np.sum(nb < 3))}")

    print(f"\n{'Daily-life feature':<26}{'Fallers':>10}{'Non-f.':>10}{'AUC':>7}")
    uni = {}
    for f, d in DAILY_DIRECTION.items():
        x = np.array([r.get(f, np.nan) for r in rows], float)
        uni[f] = {"auc": auc(d * x, y), "med_f": float(np.nanmedian(x[y == 1])), "med_nf": float(np.nanmedian(x[y == 0]))}
        print(f"{f:<26}{uni[f]['med_f']:>10.3f}{uni[f]['med_nf']:>10.3f}{uni[f]['auc']:>7.2f}")

    # ── lab features for the same subjects ──
    from run_ltmm_validation import load_features
    lab = {r["subject"]: r["v2"] for r in load_features(a.data_dir)}
    both = [r for r in rows if r["subject"] in lab]
    yb = np.array([r["faller"] for r in both])
    for r in both:
        for k in RISK_DIRECTION_V2:
            r[f"lab_{k}"] = lab[r["subject"]].get(k, np.nan)
    comb_dir = {**{f"lab_{k}": v for k, v in RISK_DIRECTION_V2.items()}, **DAILY_DIRECTION}
    print(f"\nSubjects with both lab walk and 3-day data: {len(both)}")

    res = {}
    d0 = daily_index(rows)
    res["D0"] = {"name": "Daily-life index (untrained)", "n": len(rows), "auc": auc(d0, y), "ci": bootstrap_ci(d0, y), "trained": False}
    m0 = gait_instability_index_v2([lab[r["subject"]] for r in both])
    res["M0"] = {"name": "Lab v2 index, same subjects (reference)", "n": len(both), "auc": auc(m0, yb), "ci": bootstrap_ci(m0, yb), "trained": False}
    d1 = daily_index(both, comb_dir)
    real, p_d1, null_d1 = perm_p_untrained(d1, yb)
    res["D1"] = {"name": "Combined lab + daily index (untrained) [primary]", "n": len(both), "auc": real,
                 "ci": bootstrap_ci(d1, yb), "trained": False, "perm_p": p_d1}

    Xd = np.array([[r.get(f, np.nan) for f in DAILY_DIRECTION] for r in rows], float)
    m, s, oof_d2 = repeated_cv(lambda: make_lr(None), Xd, y)
    res["D2"] = {"name": "LogReg: daily features", "n": len(rows), "auc": m, "sd": s, "trained": True}
    Xc = np.array([[r.get(f, np.nan) for f in comb_dir] for r in both], float)
    m, s, oof_d3 = repeated_cv(lambda: make_lr(6), Xc, yb)
    from sklearn.model_selection import StratifiedKFold, permutation_test_score
    sc, _, p_d3 = permutation_test_score(make_lr(6), Xc, yb, scoring="roc_auc",
                                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                         n_permutations=200, random_state=SEED, n_jobs=-1)
    res["D3"] = {"name": "LogReg: lab + daily (k=6)", "n": len(both), "auc": m, "sd": s, "trained": True, "perm_p": float(p_d3)}

    print("\n" + "=" * 96)
    for k in ("M0", "D0", "D1", "D2", "D3"):
        r = res[k]
        spread = f"95% CI {r['ci'][0]:.2f}-{r['ci'][1]:.2f}" if not r["trained"] else f"± {r['sd']:.2f} (20x5 CV)"
        extra = f"  perm p = {r['perm_p']:.3f}" if "perm_p" in r else ""
        print(f"{k}  {r['name']:<50} n={r['n']:<3} AUC {r['auc']:.2f}  {spread}{extra}")
    print("-" * 96)
    print("References: lab-walk index 0.69 (n=73) | TUG 0.68 | Askhatova et al. 2026 TCN 0.749 (n=63)")
    print("All pre-registered models reported. Retrospective labels: discrimination, not prospective prediction.")
    print("=" * 96)

    # charts
    os.makedirs(OUT_DIR, exist_ok=True)
    order = ["M0", "D0", "D1", "D2", "D3"]
    plt.figure(figsize=(9.5, 5.2), dpi=150)
    vals = [res[k]["auc"] for k in order]
    errs = [[v - res[k]["ci"][0] if not res[k]["trained"] else res[k]["sd"] for k, v in zip(order, vals)],
            [res[k]["ci"][1] - v if not res[k]["trained"] else res[k]["sd"] for k, v in zip(order, vals)]]
    bars = plt.bar([f"{k}\n{res[k]['name'].split('(')[0].strip()}" for k in order], vals, yerr=errs, capsize=4,
                   color=["#9B9B9B", "#4A90E2", "#D0021B", "#F5A623", "#F5A623"], edgecolor="black")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    plt.axhline(0.68, color="#D0021B", ls=":", lw=1.4, label="TUG, clinical (0.68)")
    plt.axhline(0.749, color="black", ls="--", lw=1.2, label="Askhatova et al. 2026 TCN (0.75)")
    plt.axhline(0.5, color="grey", ls="-.", lw=1)
    plt.ylim(0.35, 1.0); plt.xticks(fontsize=7.5); plt.ylabel("AUC (fallers vs non-fallers)", fontweight="bold")
    plt.title("LTMM Experiment 3: Adding Daily-Life Walking (all pre-registered models)", fontweight="bold")
    plt.legend(fontsize=8, loc="upper left"); plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/ltmm_daily_models.png"); plt.close()

    from sklearn.metrics import roc_curve
    plt.figure(figsize=(6.4, 6), dpi=150)
    for sc_, yy, lab_, col, ls in ((d1, yb, f"D1 combined index ({res['D1']['auc']:.2f})", "#D0021B", "-"),
                                   (d0, y, f"D0 daily-life index ({res['D0']['auc']:.2f})", "#4A90E2", "-"),
                                   (m0, yb, f"M0 lab index ({res['M0']['auc']:.2f})", "#9B9B9B", ":")):
        mk = ~np.isnan(sc_)
        fpr, tpr, _ = roc_curve(yy[mk], sc_[mk]); plt.plot(fpr, tpr, color=col, ls=ls, lw=2, label=lab_)
    plt.plot([0, 1], [0, 1], color="grey", ls="-.", lw=1)
    plt.xlabel("False positive rate", fontweight="bold"); plt.ylabel("True positive rate", fontweight="bold")
    plt.title("Lab vs Daily-Life Gait (untrained indices)", fontweight="bold"); plt.legend(fontsize=8, loc="lower right")
    plt.grid(linestyle="--", alpha=0.4); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/ltmm_daily_roc.png"); plt.close()

    with open("ltmm_daily_results.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": f"{a.chunks} x {a.chunk_min} min slices; pre-registered D0-D3 + M0",
                   "results": res, "univariate": uni, "subjects": rows}, f, indent=2, default=float)
    print(f"\n[SAVED] {OUT_DIR}/ltmm_daily_models.png, ltmm_daily_roc.png, ltmm_daily_results.json")


if __name__ == "__main__":
    main()
