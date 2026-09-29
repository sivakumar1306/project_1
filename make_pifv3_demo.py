"""
Build the live-demo file for the web page: train the MOTION Random Forest on
participants P01-P28 and predict windows from the 4 HELD-OUT participants
(P29-P32) the model never saw. Writes pifv3_demo_samples.json (small, no raw
dataset needed by the web page). Same features/settings as run_pifv3_detection.py.
"""
import json, os, sys
import numpy as np
from run_pifv3_detection import MOTION_KEYS, load, make_model, windows

data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "pifv3")
df = load(data_dir)
fs = 10.0
win = 20
feats = windows(df, win, win // 2)
feats = feats[(feats["fall"] == 1) | (feats["adl"] == 1)].reset_index(drop=True)
pids = sorted(feats["pid"].unique())
held = pids[-4:]
motion = [c for c in feats.columns if c.split("_")[0] in MOTION_KEYS + ["accmag", "gyrmag"]]
tr = feats[~feats["pid"].isin(held)]; te = feats[feats["pid"].isin(held)].reset_index(drop=True)
model = make_model("RF").fit(tr[motion].to_numpy(float), tr["fall"].to_numpy())
prob = model.predict_proba(te[motion].to_numpy(float))[:, 1]
pred = (prob >= 0.5).astype(int)
acc = float((pred == te["fall"].to_numpy()).mean())
print(f"Train participants: {len(pids) - 4} | held-out: {held} | held-out windows {len(te)} | accuracy {acc:.3f}")

# raw magnitude series for plotting: rebuild windows in the same order
rows = []
for pid in held:
    g = df[df["pid"] == pid].reset_index(drop=True)
    run = (g["label"] != g["label"].shift()).cumsum()
    for _, r in g.groupby(run, sort=False):
        lab = r["label"].iloc[0]
        if not ("fall" in lab.lower() or lab.lower() in ("standing", "sitting", "walking")):
            continue
        for s in range(0, max(0, len(r) - win) + 1, win // 2):
            w = r.iloc[s:s + win]
            if len(w) < win:
                continue
            acc_mag = np.sqrt(sum(np.nan_to_num(w[k].to_numpy(float)) ** 2 for k in MOTION_KEYS[:3]))
            gyr_mag = np.sqrt(sum(np.nan_to_num(w[k].to_numpy(float)) ** 2 for k in MOTION_KEYS[3:]))
            rows.append({"pid": pid, "label": lab, "acc": acc_mag.round(3).tolist(), "gyr": gyr_mag.round(2).tolist(),
                         "hr": None if np.isnan(w["hr"].mean()) else round(float(w["hr"].mean()), 1),
                         "spo2": None if np.isnan(w["spo2"].mean()) else round(float(w["spo2"].mean()), 1)})
assert len(rows) == len(te), (len(rows), len(te))
for i, r in enumerate(rows):
    r["fall_probability"] = round(float(prob[i]), 3); r["predicted"] = "FALL" if pred[i] else "NO FALL"
    r["is_fall"] = bool(te["fall"].iloc[i]); r["correct"] = bool(pred[i] == te["fall"].iloc[i])

rng = np.random.default_rng(2026)
chosen = []
for pid in held:
    f_idx = [i for i, r in enumerate(rows) if r["pid"] == pid and r["is_fall"]]
    n_idx = [i for i, r in enumerate(rows) if r["pid"] == pid and not r["is_fall"]]
    chosen += list(rng.choice(f_idx, size=min(2, len(f_idx)), replace=False))
    chosen += list(rng.choice(n_idx, size=min(1, len(n_idx)), replace=False))
samples = []
for k, i in enumerate(sorted(chosen, key=lambda j: (rows[j]["pid"], j))):
    r = dict(rows[i]); r["sample_id"] = f"S{k+1:02d}"
    samples.append(r)
out = {"model": "Random Forest (300 trees, balanced), MOTION features, 2 s windows",
       "trained_on_participants": len(pids) - 4, "held_out_participants": held,
       "held_out_windows": int(len(te)), "held_out_accuracy": round(acc, 3),
       "note": "Samples come only from held-out participants the model never saw during training.",
       "samples": samples}
json.dump(out, open("pifv3_demo_samples.json", "w"), indent=1)
print(f"Samples: {len(samples)} ({sum(s['is_fall'] for s in samples)} falls) | correct {sum(s['correct'] for s in samples)}/{len(samples)}")
print("[SAVED] pifv3_demo_samples.json", os.path.getsize("pifv3_demo_samples.json") // 1024, "KB")
