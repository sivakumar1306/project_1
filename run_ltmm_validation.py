"""
Real-data validation of the Motion layer on the PhysioNet LTMM database.
=========================================================================

Question: do the Motion layer's signal-processing gait features, combined
with a priori (literature) directions and NO training on fall labels,
separate older adults with a fall history (>= 2 falls in the past year)
from non-fallers?

Data: LTMM v1.0.0 (Weiss et al., Neurorehabil Neural Repair 2013; Hausdorff,
PhysioNet), LabWalks = 1-minute lab walks, lower-back 3D accelerometer +
gyroscope, 100 Hz. Only the small LabWalks files are downloaded (~5 MB);
the 20.8 GB 3-day recordings are not needed.

Reports
  - Per-feature AUC with the a priori risk direction (Mann-Whitney p-value)
  - Knowledge-driven Gait Instability Index AUC (+ bootstrap 95% CI)
  - Data-driven comparison: logistic regression, repeated stratified 5-fold CV
  - Clinical comparison: Timed Up and Go (TUG) AUC, if the clinical
    spreadsheet can be matched to subjects

Caveats (state them on the slide):
  - Labels are RETROSPECTIVE fall history, so this is discrimination, not
    prospective prediction.
  - Sensor is on the lower back, not a finger; this validates the signal-
    processing method, not ring hardware.
  - n is small (~73), so confidence intervals are wide.

Usage:
  py -m pip install wfdb openpyxl scikit-learn
  py run_ltmm_validation.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent.gait_features import (EXPLORATORY, RISK_DIRECTION, RISK_DIRECTION_V2, extract_gait_features,
                                 extract_gait_features_v2, gait_instability_index, gait_instability_index_v2)

BASE_URL = "https://physionet.org/files/ltmm/1.0.0"
CONTROLS = [f"co{n:03d}_base" for n in list(range(1, 12)) + list(range(13, 37)) + [40, 41, 42]]
FALLERS = [f"fl{n:03d}_base" for n in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22,
                                       23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]]
FEATURE_LABELS = {
    "cadence": "Cadence (steps/min)", "step_time_cv": "Step-time variability (CV %)",
    "stride_time_cv": "Stride-time variability (CV %)",
    "step_regularity": "Step regularity", "stride_regularity": "Stride regularity",
    "harmonic_ratio_ap": "Harmonic ratio (AP)", "ml_rms_ratio": "ML sway ratio (exploratory)",
}
OUT_DIR = "evaluation_charts"


# ── download ───────────────────────────────────────────────────────────────
def _fetch(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "MedXAI-LTMM-validation"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())


def download(data_dir: str) -> None:
    lw = os.path.join(data_dir, "LabWalks")
    names = CONTROLS + FALLERS
    print(f"Downloading {len(names)} LTMM lab-walk records to {lw} (skips files already present) ...")
    for i, name in enumerate(names, 1):
        for ext in (".hea", ".dat"):
            try:
                _fetch(f"{BASE_URL}/LabWalks/{name}{ext}", os.path.join(lw, name + ext))
            except Exception as e:
                print(f"  [WARN] {name}{ext}: {e}")
        if i % 10 == 0:
            print(f"  {i}/{len(names)}")
    try:
        _fetch(f"{BASE_URL}/ClinicalDemogData_COFL.xlsx", os.path.join(data_dir, "ClinicalDemogData_COFL.xlsx"))
    except Exception as e:
        print(f"  [WARN] clinical spreadsheet: {e}")


# ── features ───────────────────────────────────────────────────────────────
def load_features(data_dir: str) -> list[dict]:
    import wfdb
    rows = []
    for name in CONTROLS + FALLERS:
        path = os.path.join(data_dir, "LabWalks", name)
        if not os.path.exists(path + ".hea"):
            continue
        try:
            rec = wfdb.rdrecord(path)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
            continue
        sig = {n.lower(): rec.p_signal[:, i] for i, n in enumerate(rec.sig_name)}
        v, ml, ap = sig.get("v-acceleration"), sig.get("ml-acceleration"), sig.get("ap-acceleration")
        if v is None or ml is None or ap is None:
            print(f"  [SKIP] {name}: missing acceleration channels {rec.sig_name}")
            continue
        yaw = sig.get("yaw-velocity")
        f1 = extract_gait_features(v, ml, ap, float(rec.fs))
        f2 = extract_gait_features_v2(v, ml, ap, float(rec.fs), yaw=yaw)
        meta = {"subject": name.split("_")[0].upper(), "faller": int(name.startswith("fl")),
                "duration_s": round(len(v) / rec.fs, 1)}
        rows.append({"v1": f1, "v2": f2, **meta})
    return rows


# ── stats ──────────────────────────────────────────────────────────────────
def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    m = ~np.isnan(scores)
    if len(set(labels[m])) < 2:
        return float("nan")
    return float(roc_auc_score(labels[m], scores[m]))


def bootstrap_ci(scores, labels, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(labels))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(labels[b])) < 2:
            continue
        a = auc(scores[b], labels[b])
        if not np.isnan(a):
            vals.append(a)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def data_driven_auc(rows, labels, feats=None, repeats=10, seed=0):
    """Logistic regression, repeated stratified 5-fold CV (fitted on labels)."""
    feats = list(feats or RISK_DIRECTION)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows], dtype=float)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    aucs, first_pred = [], None
    for r in range(repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + r)
        p = cross_val_predict(model, X, labels, cv=cv, method="predict_proba")[:, 1]
        aucs.append(auc(p, labels))
        if first_pred is None:
            first_pred = p
    return float(np.mean(aucs)), float(np.std(aucs)), first_pred


CLINICAL_TESTS = {  # column -> +1 if higher = worse, -1 if lower = worse
    "TUG": +1,    # Timed Up and Go, seconds
    "BERG": -1,   # Berg Balance Scale, points
}


def load_clinical(data_dir: str, subjects: list[str]):
    """
    Read clinical tests from ClinicalDemogData_COFL.xlsx.
    Layout (LTMM v1.0.0): sheets 'Controls' and 'Fallers', subject number in
    column '#', test scores in columns such as 'TUG' and 'BERG'.
    Subject ids are rebuilt as CO### / FL### to match the lab-walk files.
    Returns ({test: array aligned to subjects}, message).
    """
    path = os.path.join(data_dir, "ClinicalDemogData_COFL.xlsx")
    if not os.path.exists(path):
        return {}, "clinical spreadsheet not downloaded"
    try:
        import pandas as pd
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        return {}, f"could not read spreadsheet ({e}); run: py -m pip install openpyxl pandas"

    values = {t: {} for t in CLINICAL_TESTS}
    for sname, df in sheets.items():
        low = sname.strip().lower()
        prefix = "CO" if low.startswith("control") else ("FL" if low.startswith("faller") else None)
        if prefix is None or "#" not in df.columns:
            continue
        cols = {str(c).strip().upper(): c for c in df.columns}
        for _, row in df.iterrows():
            m = re.search(r"(\d+)", str(row["#"]))
            if not m:
                continue
            sid = f"{prefix}{int(m.group(1)):03d}"
            for test in CLINICAL_TESTS:
                col = cols.get(test)
                if col is None:
                    continue
                try:
                    v = float(row[col])
                    if not np.isnan(v):
                        values[test][sid] = v
                except Exception:
                    pass

    out, msgs = {}, []
    for test, d in values.items():
        if d:
            out[test] = np.array([d.get(s, np.nan) for s in subjects], dtype=float)
            msgs.append(f"{test}: {int(np.sum(~np.isnan(out[test])))} subjects matched")
    return out, "; ".join(msgs) if msgs else "no clinical test columns matched"


# ── main ───────────────────────────────────────────────────────────────────
def _feature_table(rows, labels, directions, title):
    from scipy.stats import mannwhitneyu
    res = {}
    print(f"\n{title}")
    print(f"{'Feature':<34}{'Fallers (median)':>18}{'Non-fallers':>14}{'AUC':>7}{'p':>9}")
    for f, d in directions.items():
        x = np.array([r.get(f, np.nan) for r in rows], dtype=float)
        a_ = auc(d * x, labels)
        m = ~np.isnan(x)
        try:
            p = float(mannwhitneyu(x[m & (labels == 1)], x[m & (labels == 0)]).pvalue)
        except Exception:
            p = float("nan")
        res[f] = {"auc": a_, "p": p, "median_fallers": float(np.nanmedian(x[labels == 1])),
                  "median_nonfallers": float(np.nanmedian(x[labels == 0])), "direction": d}
        print(f"{FEATURE_LABELS.get(f, f):<34}{res[f]['median_fallers']:>18.3f}{res[f]['median_nonfallers']:>14.3f}{a_:>7.2f}{p:>9.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join("data", "ltmm"))
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()

    if not a.no_download:
        download(a.data_dir)
    rows = load_features(a.data_dir)
    if len(rows) < 20:
        print(f"Only {len(rows)} usable records found in {a.data_dir}; check the download.")
        sys.exit(1)

    labels = np.array([r["faller"] for r in rows])
    subjects = [r["subject"] for r in rows]
    r1 = [r["v1"] for r in rows]
    r2 = [r["v2"] for r in rows]
    n_f, n_c = int(labels.sum()), int((1 - labels).sum())
    print(f"\nUsable subjects: {len(rows)} ({n_f} fallers, {n_c} non-fallers)")

    # ── 1. signal-quality check (label-free) ──
    cv1 = np.array([x["step_time_cv"] for x in r1], float)
    cv2 = np.array([x["stride_time_cv"] for x in r2], float)
    sf = np.array([x["straight_fraction"] for x in r2], float)
    rj = np.array([x["rejected_fraction"] for x in r2], float)
    ca = np.array([x["cadence_agreement_pct"] for x in r2], float)
    print("\nSIGNAL-QUALITY CHECK (no labels used)")
    print(f"  v1 step-time CV   median {np.nanmedian(cv1):5.2f} %  (IQR {np.nanpercentile(cv1,25):.2f}-{np.nanpercentile(cv1,75):.2f})")
    print(f"  v2 stride-time CV median {np.nanmedian(cv2):5.2f} %  (IQR {np.nanpercentile(cv2,25):.2f}-{np.nanpercentile(cv2,75):.2f})")
    print(f"  v2 straight-walking fraction {np.nanmedian(sf):.2f} | rejected intervals {np.nanmedian(rj)*100:.1f} % | "
          f"cadence agreement (steps vs autocorrelation) {np.nanmedian(ca):.1f} % difference")

    # ── 2. discrimination ──
    pf1 = _feature_table(r1, labels, {**RISK_DIRECTION, **EXPLORATORY}, "v1 FEATURES (original detector)")
    pf2 = _feature_table(r2, labels, {**RISK_DIRECTION_V2, **EXPLORATORY}, "v2 FEATURES (turn removal + McCamley detection + stride CV)")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        idx1, idx2 = gait_instability_index(r1), gait_instability_index_v2(r2)
    no_seg = [r["subject"] for r, x in zip(rows, idx2) if np.isnan(x)]
    fb = [r["subject"] for r in rows if r["v2"].get("turn_fallback")]
    coverage = 1 - len(no_seg) / len(rows)
    print(f"\n  v2 coverage: {len(rows) - len(no_seg)}/{len(rows)} subjects ({coverage*100:.0f} %; pre-set target >= 95 %)")
    if fb:
        print(f"  v2: {len(fb)} subject(s) had no straight segment >= 4 s; whole walk analysed (flagged): {fb}")
    if no_seg:
        print(f"  v2: {len(no_seg)} subject(s) could not be analysed and are excluded from v2 AUCs: {no_seg}")
    a1, a2 = auc(idx1, labels), auc(idx2, labels)
    ci1, ci2 = bootstrap_ci(idx1, labels), bootstrap_ci(idx2, labels)
    dd1 = data_driven_auc(r1, labels, RISK_DIRECTION)
    dd2 = data_driven_auc(r2, labels, RISK_DIRECTION_V2)

    clinical, clin_msg = load_clinical(a.data_dir, subjects)
    clinical_auc = {}
    for test, arr in clinical.items():
        m = ~np.isnan(arr)
        clinical_auc[test] = {"auc": auc(CLINICAL_TESTS[test] * arr, labels), "n": int(m.sum()),
                              "v1_index_same_subjects": auc(idx1[m], labels[m]),
                              "v2_index_same_subjects": auc(idx2[m], labels[m])}

    print("\n" + "=" * 86)
    print(f"v1 Gait Instability Index (no training):  AUC {a1:.2f} (95% CI {ci1[0]:.2f}-{ci1[1]:.2f})")
    print(f"v2 Gait Instability Index (no training):  AUC {a2:.2f} (95% CI {ci2[0]:.2f}-{ci2[1]:.2f})")
    same = ~np.isnan(idx2)
    print(f"v1 index on the same {int(same.sum())} subjects as v2:  AUC {auc(idx1[same], labels[same]):.2f}")
    print(f"v1 logistic regression (5-fold CV x10):   AUC {dd1[0]:.2f} ± {dd1[1]:.2f}")
    print(f"v2 logistic regression (5-fold CV x10):   AUC {dd2[0]:.2f} ± {dd2[1]:.2f}")
    names = {"TUG": "Timed Up and Go (clinical)", "BERG": "Berg Balance Scale (clinical)"}
    for test, r in clinical_auc.items():
        print(f"{names[test] + ':':<42}AUC {r['auc']:.2f}  (n={r['n']}; v1 index {r['v1_index_same_subjects']:.2f}, "
              f"v2 index {r['v2_index_same_subjects']:.2f} on same subjects)")
    if not clinical_auc:
        print(f"Clinical comparison skipped: {clin_msg}")
    print("=" * 86)
    print("Both versions are reported. v2 changes were chosen on signal-quality grounds, not to raise AUC.")
    print("Labels are retrospective fall history (>=2 falls/yr): discrimination, not prospective prediction.")

    # ── charts ──
    os.makedirs(OUT_DIR, exist_ok=True)
    from sklearn.metrics import roc_curve
    plt.figure(figsize=(6.8, 6.2), dpi=150)
    for sc, lab_, col, ls in [(idx2, f"v2 Gait Index, no training (AUC {a2:.2f})", "#4A90E2", "-"),
                              (idx1, f"v1 Gait Index, no training (AUC {a1:.2f})", "#9B9B9B", "-")]:
        mk = ~np.isnan(sc)
        fpr, tpr, _ = roc_curve(labels[mk], sc[mk]); plt.plot(fpr, tpr, lw=2.4 if col == "#4A90E2" else 1.6, color=col, ls=ls, label=lab_)
    fpr, tpr, _ = roc_curve(labels, dd2[2]); plt.plot(fpr, tpr, lw=1.8, ls="--", color="#F5A623", label=f"v2 logistic regression, CV (AUC {dd2[0]:.2f})")
    for test, col in (("TUG", "#D0021B"), ("BERG", "#7B61FF")):
        if test in clinical:
            m = ~np.isnan(clinical[test])
            fpr, tpr, _ = roc_curve(labels[m], CLINICAL_TESTS[test] * clinical[test][m])
            plt.plot(fpr, tpr, lw=1.8, ls=":", color=col, label=f"{names[test]} (AUC {clinical_auc[test]['auc']:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, ls="-.", label="Chance (0.50)")
    plt.xlabel("False positive rate", fontweight="bold"); plt.ylabel("True positive rate", fontweight="bold")
    plt.title(f"LTMM Real Data: Fallers vs Non-fallers (n={len(rows)})", fontweight="bold")
    plt.legend(fontsize=7.5, loc="lower right"); plt.grid(linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/ltmm_roc.png"); plt.close()

    feats = list(RISK_DIRECTION_V2)
    plt.figure(figsize=(8.8, 4.8), dpi=150)
    vals = [pf2[f]["auc"] for f in feats] + [a2]
    labs = [FEATURE_LABELS.get(f, f) for f in feats] + ["v2 Gait Instability Index"]
    bars = plt.barh(labs[::-1], vals[::-1], color=(["#9B9B9B"] * len(feats) + ["#4A90E2"])[::-1], edgecolor="black")
    for b, v in zip(bars, vals[::-1]):
        plt.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontweight="bold")
    plt.axvline(0.5, color="black", ls=":", lw=1); plt.xlim(0.3, 1.0)
    plt.xlabel("AUC (a priori risk direction; 0.5 = chance)", fontweight="bold")
    plt.title("v2 Motion-Layer Gait Features on LTMM (no training on labels)", fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.4); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/ltmm_feature_auc.png"); plt.close()

    plt.figure(figsize=(6.2, 4.8), dpi=150)
    bp = plt.boxplot([cv1[~np.isnan(cv1)], cv2[~np.isnan(cv2)]], patch_artist=True, widths=0.5)
    plt.xticks([1, 2], ["v1 step-time CV", "v2 stride-time CV"])
    for patch, c in zip(bp["boxes"], ["#9B9B9B", "#4A90E2"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    plt.ylabel("Gait variability (CV %)", fontweight="bold")
    plt.title("Signal-Quality Fix: Variability Before vs After", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.4); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/ltmm_quality_fix.png"); plt.close()

    plt.figure(figsize=(5.5, 5), dpi=150)
    data = [idx2[(labels == 0) & ~np.isnan(idx2)], idx2[(labels == 1) & ~np.isnan(idx2)]]
    bp = plt.boxplot(data, patch_artist=True, widths=0.5)
    plt.xticks([1, 2], [f"Non-fallers (n={n_c})", f"Fallers (n={n_f})"])
    for patch, c in zip(bp["boxes"], ["#7ED321", "#D0021B"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, d in enumerate(data, 1):
        plt.scatter(np.random.default_rng(i).normal(i, 0.05, len(d)), d, s=12, color="black", alpha=0.6, zorder=3)
    plt.ylabel("v2 Gait Instability Index (higher = less stable)", fontweight="bold")
    plt.title("Gait Instability Index by Group", fontweight="bold"); plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/ltmm_index_boxplot.png"); plt.close()

    with open("ltmm_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "PhysioNet LTMM v1.0.0 LabWalks", "n": len(rows), "n_fallers": n_f, "n_nonfallers": n_c,
            "quality": {"v1_step_cv_median": float(np.nanmedian(cv1)), "v2_stride_cv_median": float(np.nanmedian(cv2)),
                        "v2_straight_fraction_median": float(np.nanmedian(sf)), "v2_rejected_fraction_median": float(np.nanmedian(rj)),
                        "v2_cadence_agreement_pct_median": float(np.nanmedian(ca))},
            "v1": {"per_feature": pf1, "index_auc": a1, "index_ci95": ci1, "lr_cv_auc": dd1[0], "lr_cv_sd": dd1[1]},
            "v2": {"per_feature": pf2, "index_auc": a2, "index_ci95": ci2, "lr_cv_auc": dd2[0], "lr_cv_sd": dd2[1]},
            "clinical": {"results": clinical_auc, "note": clin_msg},
            "subjects": [{"subject": r["subject"], "faller": r["faller"], "v1": r["v1"], "v2": r["v2"]} for r in rows],
        }, f, indent=2, default=float)
    print(f"\n[SAVED] {OUT_DIR}/ltmm_roc.png, ltmm_feature_auc.png, ltmm_quality_fix.png, ltmm_index_boxplot.png, ltmm_results.json")


if __name__ == "__main__":
    main()
