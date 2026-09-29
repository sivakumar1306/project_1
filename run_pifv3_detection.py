"""
Experiment 4 - Trained ML fall DETECTION on PIFv3 (real wearable data)
======================================================================
Dataset: Dhaliwal, Sharma & Kaur, "Physiological Features and Inertial Features
Based Dataset: PIFv3", Mendeley Data (CC BY 4.0), https://data.mendeley.com/datasets/phb9y6cp5c/3
Comparison: Dhaliwal et al., "PIF dataset: a comprehensive dataset of physiological
and inertial features for recognition of human activities", Multimedia Tools and
Applications, 2024 (SVM/KNN/RF/DT; >95% accuracy reported on IMU data).

DATA FACTS (from inspecting the files, before any modelling)
- Per participant: PID_X.csv (IMU + ECG/EMG/GSR, ~8 Hz, label column "Features")
  and PID_X_BPM.csv (Sys, Dia, HR, SpO2, label "features").
- HR/SpO2/BP are spot readings: constant across the whole fall protocol for a
  participant, so they carry no within-person information about falls.
- The fall protocol (Standing, 6 staged falls, Sitting, Walking) lasts seconds per
  activity; the emotion/hand tasks (Sad, Anxiety, ... Fist) last minutes, seated.

PRE-REGISTERED ANALYSIS (fixed before running)
  PRIMARY   : falls vs daily activities (Standing, Sitting, Walking)
  SECONDARY : falls vs all other activities (incl. seated emotion/hand tasks)
  Windows   : 2 s, 50% overlap, within one labelled activity run
  Feature sets: MOTION (acc+gyro stats + magnitudes), PHYSIO (HR, SpO2, Sys, Dia),
                MOTION+PHYSIO
  Models    : Random Forest (300 trees, balanced) and Logistic Regression (balanced)
  Protocols : RANDOM (StratifiedKFold, windows shuffled; same person in train & test)
              SUBJECT-WISE (GroupKFold by participant; tested on unseen people)
  Metrics   : accuracy, balanced accuracy, F1 (fall), ROC-AUC. Everything reported.

This is fall DETECTION on staged lab falls. It supports the biometric fall-event
signal in the safety fusion; it is NOT the daily fall-risk score.
Usage: py run_pifv3_detection.py [--data-dir data/pifv3]
"""
import argparse, glob, json, os, re, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 2026
OUT_DIR = "evaluation_charts"
MOTION_KEYS = ["accx", "accy", "accz", "gyrox", "gyroy", "gyroz"]
PHYSIO_KEYS = ["hr", "spo2", "sys", "dia"]
ADL = {"standing", "sitting", "walking"}


def _norm(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def _secs(s):
    """'mm:ss.f' -> seconds, unwrapped to be monotonic."""
    out = []
    for v in s.astype(str):
        m = re.match(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$", str(v))
        out.append(int(m.group(1)) * 60 + float(m.group(2)) if m else np.nan)
    x = np.array(out, float)
    off, prev = 0.0, None
    for i in range(len(x)):
        if np.isnan(x[i]):
            continue
        if prev is not None and x[i] + off < prev - 1800:   # hour wrap
            off += 3600
        x[i] += off
        prev = x[i]
    return x


def _read(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [_norm(c) for c in df.columns]
    return df


def load(data_dir):
    mains = [f for f in glob.glob(os.path.join(data_dir, "**", "PID_*.csv"), recursive=True)
             if "bpm" not in os.path.basename(f).lower()]
    frames = []
    for f in sorted(mains):
        pid_num = int(re.search(r"PID_(\d+)", os.path.basename(f)).group(1))
        m = _read(f)
        lab = next((c for c in m.columns if c in ("features", "activities", "label")), None)
        if lab is None:
            continue
        d = pd.DataFrame({"label": m[lab].astype(str).str.strip(), "t": _secs(m["datetime"])})
        for k in MOTION_KEYS:
            if k in m:
                d[k] = pd.to_numeric(m[k], errors="coerce")
        bpm_path = f[:-4] + "_BPM.csv"
        if os.path.exists(bpm_path):
            b = _read(bpm_path)
            b = pd.DataFrame({"t": _secs(b["datetime"]), **{k: pd.to_numeric(b[k], errors="coerce")
                                                            for k in PHYSIO_KEYS if k in b}}).dropna(subset=["t"])
            d = d.dropna(subset=["t"]).sort_values("t")
            d = pd.merge_asof(d, b.sort_values("t"), on="t", direction="nearest", tolerance=2.0)
        d["pid"] = f"P{pid_num:02d}"
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def windows(df, win, step):
    rows = []
    for pid, g in df.groupby("pid", sort=False):
        g = g.reset_index(drop=True)
        run = (g["label"] != g["label"].shift()).cumsum()
        for _, r in g.groupby(run, sort=False):
            lab = r["label"].iloc[0]
            if lab.lower() in ("nan", "none", ""):
                continue
            for s in range(0, max(0, len(r) - win) + 1, step):
                w = r.iloc[s:s + win]
                if len(w) < win:
                    continue
                f = {"pid": pid, "label": lab, "fall": int("fall" in lab.lower()),
                     "adl": int(lab.lower() in ADL)}
                for k in MOTION_KEYS + PHYSIO_KEYS:
                    if k not in w:
                        continue
                    x = w[k].to_numpy(float); x = x[~np.isnan(x)]
                    if len(x) == 0:
                        continue
                    f[f"{k}_mean"] = x.mean(); f[f"{k}_std"] = x.std()
                    f[f"{k}_min"] = x.min(); f[f"{k}_max"] = x.max(); f[f"{k}_range"] = np.ptp(x)
                for name, ks in (("accmag", MOTION_KEYS[:3]), ("gyrmag", MOTION_KEYS[3:])):
                    if all(k in w for k in ks):
                        mag = np.sqrt(sum(np.nan_to_num(w[k].to_numpy(float)) ** 2 for k in ks))
                        f[f"{name}_mean"] = mag.mean(); f[f"{name}_std"] = mag.std()
                        f[f"{name}_max"] = mag.max(); f[f"{name}_range"] = np.ptp(mag)
                rows.append(f)
    return pd.DataFrame(rows)


def make_model(kind):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "RF":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                    random_state=SEED, n_jobs=-1))
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))


def evaluate(X, y, groups, kind, protocol):
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    split = (GroupKFold(n_splits=5).split(X, y, groups) if protocol == "SUBJECT-WISE"
             else StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    prob = np.zeros(len(y))
    for tr, te in split:
        m = clone(make_model(kind)).fit(X[tr], y[tr])
        prob[te] = m.predict_proba(X[te])[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {"accuracy": round(float(accuracy_score(y, pred)), 3),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 3),
            "f1_fall": round(float(f1_score(y, pred, zero_division=0)), 3),
            "auc": round(float(roc_auc_score(y, prob)), 3)}


def run_analysis(feats, name):
    y = feats["fall"].to_numpy(); g = feats["pid"].to_numpy()
    cols = [c for c in feats.columns if c not in ("pid", "label", "fall", "adl")]
    motion = [c for c in cols if c.split("_")[0] in MOTION_KEYS + ["accmag", "gyrmag"]]
    physio = [c for c in cols if c.split("_")[0] in PHYSIO_KEYS]
    print(f"\n{name}: {len(feats):,} windows ({int(y.sum()):,} fall / {int((1-y).sum()):,} non-fall), {feats['pid'].nunique()} participants")
    res = {}
    for set_name, fc in (("MOTION", motion), ("PHYSIO", physio), ("MOTION+PHYSIO", motion + physio)):
        if not fc:
            continue
        X = feats[fc].to_numpy(float)
        for kind in ("RF", "LR"):
            for protocol in ("RANDOM", "SUBJECT-WISE"):
                k = f"{set_name} | {kind} | {protocol}"
                r = res[k] = evaluate(X, y, g, kind, protocol)
                print(f"  {k:<38} acc {r['accuracy']:.3f}  bal-acc {r['balanced_accuracy']:.3f}  F1(fall) {r['f1_fall']:.3f}  AUC {r['auc']:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join("data", "pifv3"))
    ap.add_argument("--win-sec", type=float, default=2.0)
    a = ap.parse_args()
    df = load(a.data_dir)
    if df is None or df.empty:
        print(f"No PID_*.csv files under {a.data_dir}. Download from https://data.mendeley.com/datasets/phb9y6cp5c/3")
        sys.exit(1)
    dt = df.groupby("pid")["t"].diff(); dt = dt[(dt > 0) & (dt < 2)]
    fs = float(1 / dt.median())
    win = max(4, int(round(a.win_sec * fs)))
    print(f"Rows {len(df):,} | participants {df['pid'].nunique()} | ~{fs:.1f} Hz | window {win} samples (~{a.win_sec}s, 50% overlap)")
    # physiology variability check (label-free)
    fp = df[df["label"].str.lower().str.contains("fall|standing|sitting|walking")]
    nun = fp.groupby("pid")[["hr", "spo2", "sys", "dia"]].nunique().median()
    print("Median distinct HR/SpO2/Sys/Dia values per participant during the fall protocol:", nun.to_dict())

    feats = windows(df, win, max(1, win // 2))
    primary = feats[(feats["fall"] == 1) | (feats["adl"] == 1)].reset_index(drop=True)
    results = {"primary_falls_vs_daily_activities": run_analysis(primary, "PRIMARY  falls vs Standing/Sitting/Walking"),
               "secondary_falls_vs_all": run_analysis(feats, "SECONDARY falls vs all activities")}

    print("\n" + "=" * 92)
    print("Honest protocol = SUBJECT-WISE. Comparison: Dhaliwal et al. 2024 report >95% accuracy on IMU data.")
    print("Staged lab falls; fall DETECTION (not risk prediction). All pre-registered results reported.")
    print("=" * 92)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT_DIR, exist_ok=True)
    R = results["primary_falls_vs_daily_activities"]
    order = [s for s in ("MOTION", "PHYSIO", "MOTION+PHYSIO") if f"{s} | RF | RANDOM" in R]
    x = np.arange(len(order)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    for i, (p, col) in enumerate((("RANDOM", "#9B9B9B"), ("SUBJECT-WISE", "#4A90E2"))):
        vals = [R[f"{s} | RF | {p}"]["balanced_accuracy"] for s in order]
        bars = ax.bar(x + (i - .5) * w, vals, w, color=col, edgecolor="black",
                      label=f"Random Forest, {'random split (same people in train/test)' if p == 'RANDOM' else 'subject-wise (unseen people)'}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(.5, color="black", ls=":", lw=1, label="Chance")
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylim(.3, 1.05)
    ax.set_ylabel("Balanced accuracy", fontweight="bold")
    ax.set_title("PIFv3 Fall Detection (falls vs daily activities, 32 participants)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", ls="--", alpha=.4)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/pifv3_detection.png"); plt.close(fig)
    json.dump({"dataset": "PIFv3 (Mendeley phb9y6cp5c v3), CC BY 4.0", "participants": int(df["pid"].nunique()),
               "sampling_hz_est": round(fs, 2), "window_sec": a.win_sec,
               "physio_distinct_values_during_fall_protocol_median": {k: float(v) for k, v in nun.items()},
               "results": results}, open("pifv3_results.json", "w"), indent=2)
    print(f"\n[SAVED] {OUT_DIR}/pifv3_detection.png, pifv3_results.json")


if __name__ == "__main__":
    main()
