"""
LTMM Experiment 2: can the Motion layer's AUC be raised LEGITIMATELY?
====================================================================

PRE-REGISTERED PROTOCOL (fixed before this script was ever run on real data)
  Data     : LTMM LabWalks (same 73 subjects as run_ltmm_validation.py)
  Labels   : faller = >= 2 falls in past year (retrospective)
  Models   : exactly the 7 below, fixed hyperparameters, NO tuning on results
     M0  v2 Gait Instability Index               untrained (reference, AUC 0.69 in Exp. 1)
     M1  v3 Index = v2 + turn metrics            untrained, literature directions
     M2  Logistic regression, v2 features        trained
     M3  Logistic regression, v3 gait + turn     trained, SelectKBest(k=8) inside folds
     M4  Logistic regression, M3 + age + sex     trained, SelectKBest(k=8) inside folds
     M5  Random forest, v3 gait + turn           trained (300 trees, depth 3, leaf 5)
     M6  Random forest, M5 + age + sex           trained
  CV       : 20 x repeated stratified 5-fold, split BY SUBJECT (one record per
             subject, so no subject can appear in train and test)
  Chance   : permutation test (200 label shuffles) for the pre-designated
             primary trained model M4
  Reporting: ALL models are reported whatever they score. One run; results
             are not used to go back and change features or settings.

Reference points: TUG 0.68 (this data, n=69); Askhatova et al. MethodsX 2026
TCN AUC 0.749 (subject-wise CV, 1-min + 3-day data + demographics, n=63).

Usage:  py run_ltmm_experiment2.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent.gait_features import (RISK_DIRECTION_V2, RISK_DIRECTION_V3, extract_gait_features_v3,
                                 gait_instability_index_v2, gait_instability_index_v3)
from run_ltmm_validation import CONTROLS, FALLERS, auc, bootstrap_ci, download, load_clinical

warnings.filterwarnings("ignore")
OUT_DIR = "evaluation_charts"
SEED = 2026
N_REPEATS = 20
N_PERM = 200

V2_FEATS = list(RISK_DIRECTION_V2)
EXTRA_FEATS = ["step_asymmetry_pct", "jerk_v_norm", "jerk_ap_norm", "spectral_entropy_v",
               "spectral_entropy_ml", "corr_v_ap", "rms_v_g", "ml_rms_ratio"]
TURN_FEATS = ["turn_duration_s", "turn_peak_velocity", "steps_per_180", "turn_count"]
V3_FEATS = V2_FEATS + EXTRA_FEATS + TURN_FEATS
DEMO_FEATS = ["age", "sex_female"]


def load_rows(data_dir: str) -> list[dict]:
    import wfdb
    rows = []
    for name in CONTROLS + FALLERS:
        path = os.path.join(data_dir, "LabWalks", name)
        if not os.path.exists(path + ".hea"):
            continue
        rec = wfdb.rdrecord(path)
        sig = {n.lower(): rec.p_signal[:, i] for i, n in enumerate(rec.sig_name)}
        v, ml, ap, yaw = (sig.get(k) for k in ("v-acceleration", "ml-acceleration", "ap-acceleration", "yaw-velocity"))
        if v is None:
            continue
        f = extract_gait_features_v3(v, ml, ap, float(rec.fs), yaw=yaw)
        f.update({"subject": name.split("_")[0].upper(), "faller": int(name.startswith("fl"))})
        rows.append(f)
    return rows


def load_demographics(data_dir: str, subjects: list[str]) -> dict[str, np.ndarray]:
    import re
    import pandas as pd
    path = os.path.join(data_dir, "ClinicalDemogData_COFL.xlsx")
    out = {"age": {}, "sex_female": {}}
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        print(f"  [WARN] demographics unavailable: {e}")
        return {k: np.full(len(subjects), np.nan) for k in out}
    for sname, df in sheets.items():
        low = sname.strip().lower()
        prefix = "CO" if low.startswith("control") else ("FL" if low.startswith("faller") else None)
        if prefix is None or "#" not in df.columns:
            continue
        gcol = next((c for c in df.columns if str(c).lower().startswith("gender")), None)
        acol = next((c for c in df.columns if str(c).strip().lower() == "age"), None)
        for _, r in df.iterrows():
            m = re.search(r"(\d+)", str(r["#"]))
            if not m:
                continue
            sid = f"{prefix}{int(m.group(1)):03d}"
            try:
                if acol is not None:
                    out["age"][sid] = float(r[acol])
            except Exception:
                pass
            try:
                if gcol is not None:
                    out["sex_female"][sid] = float(r[gcol])   # both sheets: 1 = female, 0 = male
            except Exception:
                pass
    return {k: np.array([d.get(s, np.nan) for s in subjects], float) for k, d in out.items()}


def make_lr(k: int | None):
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


def make_rf():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=5,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1))


def repeated_cv_auc(model_fn, X, y, repeats=N_REPEATS):
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold
    aucs, oof_first = [], None
    for r in range(repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + r)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y):
            m = clone(model_fn())
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(auc(oof, y))
        if oof_first is None:
            oof_first = oof
    return float(np.mean(aucs)), float(np.std(aucs)), oof_first


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join("data", "ltmm"))
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    data_dir = a.data_dir
    if not a.no_download:
        download(data_dir)
    print("\nExtracting v3 features (gait + turns + extra descriptors) ...")
    rows = load_rows(data_dir)
    if len(rows) < 20:
        print("Not enough records; run run_ltmm_validation.py first.")
        sys.exit(1)
    y = np.array([r["faller"] for r in rows])
    subjects = [r["subject"] for r in rows]
    demo = load_demographics(data_dir, subjects)
    for k, arr in demo.items():
        for r, v in zip(rows, arr):
            r[k] = v
    print(f"Subjects: {len(rows)} ({int(y.sum())} fallers, {int((1 - y).sum())} non-fallers); "
          f"age available for {int(np.sum(~np.isnan(demo['age'])))}")

    # turn feature summary (label-free)
    tc = np.array([r["turn_count"] for r in rows], float)
    print(f"Turns detected per walk: median {np.median(tc):.0f} (range {tc.min():.0f}-{tc.max():.0f}); "
          f"walks with >=1 turn: {int(np.sum(tc > 0))}/{len(rows)}")

    # univariate table (a priori directions for index features)
    print(f"\n{'Feature':<24}{'Fallers':>10}{'Non-f.':>10}{'AUC*':>7}   (*AUC with a priori direction where defined, else raw)")
    uni = {}
    for f in V3_FEATS + DEMO_FEATS:
        x = np.array([r.get(f, np.nan) for r in rows], float)
        d = RISK_DIRECTION_V3.get(f, +1)
        a_ = auc(d * x, y)
        uni[f] = {"auc": a_, "median_fallers": float(np.nanmedian(x[y == 1])), "median_nonfallers": float(np.nanmedian(x[y == 0]))}
        print(f"{f:<24}{uni[f]['median_fallers']:>10.3f}{uni[f]['median_nonfallers']:>10.3f}{a_:>7.2f}")

    def X_of(feats):
        return np.array([[r.get(f, np.nan) for f in feats] for r in rows], float)

    results = {}
    idx0 = gait_instability_index_v2(rows)
    idx1 = gait_instability_index_v3(rows)
    for key, name, sc in (("M0", "v2 Gait Index (untrained)", idx0), ("M1", "v3 Index: gait + turns (untrained)", idx1)):
        lo, hi = bootstrap_ci(sc, y)
        results[key] = {"name": name, "auc": auc(sc, y), "sd": 0.0, "ci95": [lo, hi], "trained": False, "oof": sc}

    specs = [
        ("M2", "LogReg: v2 features", lambda: make_lr(None), V2_FEATS),
        ("M3", "LogReg: v3 gait + turns (k=8)", lambda: make_lr(8), V3_FEATS),
        ("M4", "LogReg: v3 + age + sex (k=8) [primary]", lambda: make_lr(8), V3_FEATS + DEMO_FEATS),
        ("M5", "RandomForest: v3 gait + turns", make_rf, V3_FEATS),
        ("M6", "RandomForest: v3 + age + sex", make_rf, V3_FEATS + DEMO_FEATS),
    ]
    for key, name, fn, feats in specs:
        print(f"  running {key} {name} ...")
        m, s, oof = repeated_cv_auc(fn, X_of(feats), y)
        results[key] = {"name": name, "auc": m, "sd": s, "trained": True, "oof": oof, "features": feats}

    # permutation test for the pre-designated primary trained model (M4)
    print(f"  permutation test for M4 ({N_PERM} label shuffles) ...")
    from sklearn.model_selection import StratifiedKFold, permutation_test_score
    X4 = X_of(V3_FEATS + DEMO_FEATS)
    score, perm_scores, pval = permutation_test_score(
        make_lr(8), X4, y, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        n_permutations=N_PERM, random_state=SEED, n_jobs=-1)
    results["M4"]["permutation"] = {"auc_single_cv": float(score), "p_value": float(pval),
                                    "null_mean": float(np.mean(perm_scores)), "null_95th": float(np.percentile(perm_scores, 95))}

    # selected-feature frequency for M4 (interpretability)
    from sklearn.base import clone
    sel_counts = {f: 0 for f in V3_FEATS + DEMO_FEATS}
    total = 0
    for r in range(N_REPEATS):
        for tr, _ in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X4, y):
            m = clone(make_lr(8)).fit(X4[tr], y[tr])
            mask = m.named_steps["selectkbest"].get_support()
            for f, s in zip(V3_FEATS + DEMO_FEATS, mask):
                sel_counts[f] += int(s)
            total += 1
    sel_freq = {f: c / total for f, c in sorted(sel_counts.items(), key=lambda kv: -kv[1])}

    print("\n" + "=" * 92)
    print(f"{'Model':<46}{'AUC':>7}{'':>4}{'Training':>12}")
    print("-" * 92)
    for key in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
        r = results[key]
        spread = f"95% CI {r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}" if not r["trained"] else f"± {r['sd']:.2f} (20x5 CV)"
        print(f"{key + '  ' + r['name']:<46}{r['auc']:>7.2f}   {spread:<22}{'trained' if r['trained'] else 'none'}")
    print("-" * 92)
    pm = results["M4"]["permutation"]
    print(f"M4 permutation test: AUC {pm['auc_single_cv']:.2f} vs shuffled-label null mean {pm['null_mean']:.2f} "
          f"(95th pct {pm['null_95th']:.2f}), p = {pm['p_value']:.3f}")
    print("References: TUG 0.68 (this data) | Askhatova et al. 2026 TCN 0.749 (1-min + 3-day + demographics)")
    print("=" * 92)
    print("M4 most frequently selected features:", ", ".join(f"{f} ({p:.0%})" for f, p in list(sel_freq.items())[:8]))
    print("All pre-registered models are reported. Retrospective labels: discrimination, not prospective prediction.")

    # charts
    os.makedirs(OUT_DIR, exist_ok=True)
    order = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
    plt.figure(figsize=(10, 5.4), dpi=150)
    vals = [results[k]["auc"] for k in order]
    errs = [0 if not results[k]["trained"] else results[k]["sd"] for k in order]
    cols = ["#9B9B9B", "#4A90E2", "#F5A623", "#F5A623", "#D0021B", "#7ED321", "#7ED321"]
    bars = plt.bar([f"{k}\n{results[k]['name'].split(':')[0]}" for k in order], vals, yerr=errs, capsize=4, color=cols, edgecolor="black")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold")
    plt.axhline(0.68, color="#D0021B", ls=":", lw=1.5, label="Timed Up and Go, clinical (0.68)")
    plt.axhline(0.749, color="black", ls="--", lw=1.2, label="Askhatova et al. 2026 TCN (0.75)")
    plt.axhline(0.5, color="grey", ls="-.", lw=1, label="Chance")
    plt.ylim(0.4, 1.0); plt.ylabel("AUC (subject-wise)", fontweight="bold"); plt.xticks(fontsize=7.5)
    plt.title("LTMM Experiment 2: Pre-registered Models (all reported)", fontweight="bold")
    plt.legend(fontsize=8, loc="upper left"); plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/ltmm_exp2_models.png"); plt.close()

    from sklearn.metrics import roc_curve
    plt.figure(figsize=(6.6, 6), dpi=150)
    for key, col, ls in (("M1", "#4A90E2", "-"), ("M4", "#D0021B", "-"), ("M6", "#7ED321", "--"), ("M0", "#9B9B9B", ":")):
        sc = results[key]["oof"]; mk = ~np.isnan(sc)
        fpr, tpr, _ = roc_curve(y[mk], sc[mk])
        plt.plot(fpr, tpr, color=col, ls=ls, lw=2, label=f"{key} {results[key]['name']} ({results[key]['auc']:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, ls="-.")
    plt.xlabel("False positive rate", fontweight="bold"); plt.ylabel("True positive rate", fontweight="bold")
    plt.title("LTMM Experiment 2: ROC (out-of-fold)", fontweight="bold"); plt.legend(fontsize=7, loc="lower right")
    plt.grid(linestyle="--", alpha=0.4); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/ltmm_exp2_roc.png"); plt.close()

    plt.figure(figsize=(6.4, 4.6), dpi=150)
    plt.hist(perm_scores, bins=25, color="#9B9B9B", edgecolor="black", alpha=0.8, label="Shuffled labels (null)")
    plt.axvline(pm["auc_single_cv"], color="#D0021B", lw=2.5, label=f"Real labels: {pm['auc_single_cv']:.2f} (p = {pm['p_value']:.3f})")
    plt.xlabel("AUC", fontweight="bold"); plt.ylabel("Count", fontweight="bold")
    plt.title("Permutation Test: Is M4 Better Than Chance?", fontweight="bold"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/ltmm_exp2_permutation.png"); plt.close()

    with open("ltmm_experiment2_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "protocol": "pre-registered; 7 models; 20x5 repeated stratified subject-wise CV; permutation test on M4",
            "n": len(rows), "models": {k: {kk: vv for kk, vv in v.items() if kk != "oof"} for k, v in results.items()},
            "univariate": uni, "m4_selection_frequency": sel_freq,
        }, f, indent=2, default=float)
    print(f"\n[SAVED] {OUT_DIR}/ltmm_exp2_models.png, ltmm_exp2_roc.png, ltmm_exp2_permutation.png, ltmm_experiment2_results.json")


if __name__ == "__main__":
    main()
