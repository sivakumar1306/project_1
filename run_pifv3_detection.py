"""
Experiment 4 - Trained ML fall DETECTION on PIFv3 (real wearable data)

Dataset: PIFv3, Dhaliwal, Sharma & Kaur, Mendeley Data (CC BY 4.0)
https://data.mendeley.com/datasets/phb9y6cp5c/3
Comparison: Dhaliwal et al., Multimedia Tools and Applications, 2024
(SVM/KNN/RF/DT; >95% accuracy reported on IMU data)

PRE-REGISTERED QUESTIONS (fixed before running)
  Q1 How well does a trained model detect falls (fall vs non-fall windows)?
  Q2 Do ring-type physiological signals (HR, SpO2, BP) help beyond motion?
     Feature sets: MOTION, PHYSIO, MOTION+PHYSIO
  Q3 Does the evaluation protocol matter?
     RANDOM split (same person in train and test) vs
     SUBJECT-WISE split (tested on unseen people) - the honest one.
  Models: Random Forest (300 trees), Logistic Regression; fixed settings.
  All results reported.

This is fall DETECTION on staged lab falls (supports the biometric
fall-event signal), NOT the daily fall-risk score.
Usage: py run_pifv3_detection.py
"""
import argparse, glob, json, os, re, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
SEED = 2026
OUT_DIR = "evaluation_charts"
MOTION_KEYS = ["accx", "accy", "accz", "gyrox", "gyroy", "gyroz"]
PHYSIO_KEYS = ["hr", "spo2", "sys", "dia"]


def _norm(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def load_subject_csvs(data_dir):
    import pandas as pd
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    files = [f for f in files if "bpm" not in os.path.basename(f).lower()]
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print(f"  [SKIP] {f}: {e}")
            continue
        cols = {_norm(c): c for c in df.columns}
        label_col = next((cols[k] for k in cols if k.startswith("activit") or k == "label"), None)
        if label_col is None:
            continue
        m = re.search(r"(\d+)", os.path.basename(f)) or re.search(r"(\d+)", os.path.basename(os.path.dirname(f)))
        pid = f"P{int(m.group(1)):02d}" if m else os.path.basename(f)
        keep = {"label": df[label_col].astype(str).str.strip()}
        for k in MOTION_KEYS + PHYSIO_KEYS:
            if k in cols:
                keep[k] = pd.to_numeric(df[cols[k]], errors="coerce")
        tcol = next((cols[k] for k in cols if k.startswith("datetime") or k in ("time", "timestamp")), None)
        if tcol is not None:
            keep["t"] = pd.to_datetime(df[tcol], errors="coerce")
        sub = pd.DataFrame(keep)
        sub["pid"] = pid
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else None


def window_features(df, win, step):
    rows = []
    for pid, g in df.groupby("pid", sort=False):
        g = g.reset_index(drop=True)
        run_id = (g["label"] != g["label"].shift()).cumsum()
        for _, run in g.groupby(run_id, sort=False):
            lab = run["label"].iloc[0]
            if not lab or lab.lower() in ("nan", "none"):
                continue
            n = len(run)
            starts = range(0, n - win + 1, step) if n >= win else [0]
            for s in starts:
                w = run.iloc[s:s + win]
                if len(w) < max(4, win // 2):
                    continue
                feat = {"pid": pid, "label": lab, "fall": int("fall" in lab.lower())}
                acc = [k for k in ("accx", "accy", "accz") if k in w]
                gyr = [k for k in ("gyrox", "gyroy", "gyroz") if k in w]
                for k in acc + gyr + [k for k in PHYSIO_KEYS if k in w]:
                    x = w[k].to_numpy(dtype=float)
                    x = x[~np.isnan(x)]
                    if len(x) == 0:
                        for st in ("mean", "std", "min", "max", "range"):
                            feat[f"{k}_{st}"] = np.nan
                        continue
                    feat[f"{k}_mean"] = x.mean(); feat[f"{k}_std"] = x.std()
                    feat[f"{k}_min"] = x.min(); feat[f"{k}_max"] = x.max()
                    feat[f"{k}_range"] = x.max() - x.min()
                for name, keys in (("accmag", acc), ("gyrmag", gyr)):
                    if len(keys) == 3:
                        mag = np.sqrt(sum(np.nan_to_num(w[k].to_numpy(dtype=float)) ** 2 for k in keys))
                        feat[f"{name}_mean"] = mag.mean(); feat[f"{name}_std"] = mag.std()
                        feat[f"{name}_max"] = mag.max(); feat[f"{name}_range"] = mag.max() - mag.min()
                rows.append(feat)
    return rows


def make_model(kind):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "RF":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                    random_state=SEED, n_jobs=-1))
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))


def evaluate(X, y, groups, kind, protocol):
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    if protocol == "SUBJECT-WISE":
        splits = GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups)
    else:
        splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    prob = np.zeros(len(y))
    for tr, te in splits:
        if len(set(y[tr])) < 2:
            continue
        m = clone(make_model(kind)).fit(X[tr], y[tr])
        prob[te] = m.predict_proba(X[te])[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1_fall": float(f1_score(y, pred, zero_division=0)),
            "auc": float(roc_auc_score(y, prob))}


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join("data", "pifv3"))
    ap.add_argument("--win-sec", type=float, default=2.0)
    a = ap.parse_args()

    df = load_subject_csvs(a.data_dir)
    if df is None or df.empty:
        print(f"No CSVs with an 'Activities' column under {a.data_dir}. "
              "Download from https://data.mendeley.com/datasets/phb9y6cp5c/3")
        sys.exit(1)

    fs = 10.0
    if "t" in df and df["t"].notna().sum() > 100:
        dt = df.groupby("pid")["t"].diff().dt.total_seconds()
        dt = dt[(dt > 0) & (dt < 5)]
        if len(dt):
            fs = float(1.0 / dt.median())
    win = max(4, int(round(a.win_sec * fs)))
    print(f"Rows: {len(df):,} | participants: {df['pid'].nunique()} | ~{fs:.1f} Hz | window {win} rows (~{a.win_sec}s)")
    print("Labels:", ", ".join(f"{k} ({v})" for k, v in df["label"].value_counts().head(20).items()))
    print("Signals found:", [k for k in MOTION_KEYS + PHYSIO_KEYS if k in df])

    feats = pd.DataFrame(window_features(df, win, max(1, win // 2)))
    if feats.empty or feats["fall"].nunique() < 2:
        print("Could not build fall and non-fall windows; check the label column.")
        sys.exit(1)
    y = feats["fall"].to_numpy(); groups = feats["pid"].to_numpy()
    print(f"Windows: {len(feats):,} ({int(y.sum()):,} fall / {int((1 - y).sum()):,} non-fall) "
          f"from {feats['pid'].nunique()} participants\n")

    cols = [c for c in feats.columns if c not in ("pid", "label", "fall")]
    motion = [c for c in cols if c.split("_")[0] in MOTION_KEYS + ["accmag", "gyrmag"]]
    physio = [c for c in cols if c.split("_")[0] in PHYSIO_KEYS]
    sets = {"MOTION": motion, "PHYSIO": physio, "MOTION+PHYSIO": motion + physio}

    results = {}
    for set_name, fcols in sets.items():
        if not fcols:
            print(f"  [SKIP] {set_name}: no columns"); continue
        X = feats[fcols].to_numpy(dtype=float)
        for kind in ("RF", "LR"):
            for protocol in ("RANDOM", "SUBJECT-WISE"):
                key = f"{set_name} | {kind} | {protocol}"
                r = results[key] = evaluate(X, y, groups, kind, protocol)
                print(f"  {key:<40} acc {r['accuracy']:.3f}  bal-acc {r['balanced_accuracy']:.3f}  "
                      f"F1(fall) {r['f1_fall']:.3f}  AUC {r['auc']:.3f}")

    print("\n" + "=" * 90)
    print("Honest protocol = SUBJECT-WISE (tested on people the model never saw).")
    print("Comparison: Dhaliwal et al. 2024 report >95% accuracy on IMU data.")
    print("=" * 90)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT_DIR, exist_ok=True)
    order = [s for s in ("MOTION", "PHYSIO", "MOTION+PHYSIO") if f"{s} | RF | RANDOM" in results]
    x = np.arange(len(order)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    for i, (protocol, col) in enumerate((("RANDOM", "#9B9B9B"), ("SUBJECT-WISE", "#4A90E2"))):
        vals = [results[f"{s} | RF | {protocol}"]["balanced_accuracy"] for s in order]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=col, edgecolor="black",
                      label=f"Random Forest, {protocol.lower()} split")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.5, color="black", ls=":", lw=1, label="Chance")
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("Balanced accuracy (fall vs non-fall)", fontweight="bold")
    ax.set_title("PIFv3 Fall Detection: Sensors and Evaluation Protocol", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/pifv3_detection.png"); plt.close(fig)

    json.dump({"dataset": "PIFv3 (Mendeley phb9y6cp5c v3)", "participants": int(feats["pid"].nunique()),
               "windows": int(len(feats)), "fall_windows": int(y.sum()), "sampling_hz_est": fs,
               "results": results}, open("pifv3_results.json", "w"), indent=2)
    print(f"\n[SAVED] {OUT_DIR}/pifv3_detection.png, pifv3_results.json")


if __name__ == "__main__":
    main()