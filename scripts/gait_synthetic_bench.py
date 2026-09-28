"""
Synthetic ground-truth bench for the gait detectors (no labels, no real data).
Simulates lower-back walking with the artefacts seen in corridor lab walks:
  - turns every ~18 s (yaw ~80 deg/s, irregular slower steps)
  - normal left/right step asymmetry
  - a secondary loading peak ~0.12 s after each initial contact
and compares v1 / v2 variability estimates with the TRUE stride-time CV.
"""
from __future__ import annotations
import numpy as np


def simulate_walk(true_stride_cv=0.02, asym=0.04, seed=0, fs=100, dur=80, turns=True):
    rng = np.random.default_rng(seed)
    base_step = rng.uniform(0.50, 0.60)
    t, ics, straight_strides, in_turn = 0.5, [], [], []
    turn_windows = [(s, s + 3.0) for s in np.arange(15, dur, 18)] if turns else []
    k = 0
    while t < dur - 1:
        turning = any(a <= t <= b for a, b in turn_windows)
        stride = 2 * base_step * (1 + rng.normal(0, true_stride_cv))
        if turning:
            stride *= rng.uniform(1.1, 1.4)
        s1 = stride / 2 * (1 + asym / 2) if k % 2 == 0 else stride / 2 * (1 - asym / 2)
        s2 = stride - s1
        ics.append(t); in_turn.append(turning); t += s1
        ics.append(t); in_turn.append(turning); t += s2
        if not turning:
            straight_strides.append(stride)
        k += 1
    n = int(dur * fs); tt = np.arange(n) / fs
    v = np.ones(n); ap = np.zeros(n); ml = np.zeros(n); yaw = rng.normal(0, 4, n)
    for i, ic in enumerate(ics):
        amp = 0.35 * (0.6 if in_turn[i] else 1.0) * (1 + rng.normal(0, 0.08))
        v += amp * np.exp(-((tt - ic) / 0.035) ** 2) + 0.5 * amp * np.exp(-((tt - ic - 0.12) / 0.05) ** 2)
        ap += 0.25 * amp * np.exp(-((tt - ic - 0.05) / 0.06) ** 2)
        ml += (0.08 if i % 2 else -0.08) * np.exp(-((tt - ic - 0.15) / 0.12) ** 2)
    for a, b in turn_windows:
        m = (tt >= a) & (tt <= b)
        yaw[m] += 80 * np.sin(np.pi * (tt[m] - a) / (b - a))
    v += rng.normal(0, 0.03, n); ap += rng.normal(0, 0.03, n); ml += rng.normal(0, 0.03, n)
    true_cv = float(np.std(straight_strides, ddof=1) / np.mean(straight_strides) * 100)
    return v, ml, ap, yaw, fs, true_cv


def run_bench(n=30):
    from agent.gait_features import extract_gait_features, extract_gait_features_v2
    rows = []
    for s in range(n):
        cv = [0.015, 0.03, 0.05][s % 3]
        v, ml, ap, yaw, fs, true_cv = simulate_walk(true_stride_cv=cv, seed=s)
        f1 = extract_gait_features(v, ml, ap, fs)
        f2 = extract_gait_features_v2(v, ml, ap, fs, yaw=yaw)
        rows.append((true_cv, f1["step_time_cv"], f2["stride_time_cv"], f2["straight_fraction"], f2["rejected_fraction"]))
    return np.array(rows, dtype=float)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r = run_bench()
    err1, err2 = np.abs(r[:, 1] - r[:, 0]), np.abs(r[:, 2] - r[:, 0])
    print(f"True stride CV (mean):       {r[:,0].mean():.2f} %")
    print(f"v1 step-time CV (mean):      {r[:,1].mean():.2f} %   mean abs error {err1.mean():.2f}  corr w/ truth {np.corrcoef(r[:,0], r[:,1])[0,1]:.2f}")
    print(f"v2 stride-time CV (mean):    {r[:,2].mean():.2f} %   mean abs error {err2.mean():.2f}  corr w/ truth {np.corrcoef(r[:,0], r[:,2])[0,1]:.2f}")
    print(f"v2 straight fraction {r[:,3].mean():.2f} | rejected intervals {r[:,4].mean():.3f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("evaluation_charts", exist_ok=True)
    plt.figure(figsize=(6, 5.5), dpi=150)
    plt.scatter(r[:, 0], r[:, 1], color="#9B9B9B", label=f"v1 step-time CV (error {err1.mean():.1f} pts)", s=28)
    plt.scatter(r[:, 0], r[:, 2], color="#4A90E2", label=f"v2 stride-time CV (error {err2.mean():.1f} pts)", s=28)
    lim = [0, max(r[:, 1].max(), 12)]
    plt.plot(lim, lim, "k:", lw=1, label="Perfect agreement")
    plt.xlabel("True stride-time CV (%)", fontweight="bold"); plt.ylabel("Estimated CV (%)", fontweight="bold")
    plt.title("Detector Validation on Synthetic Ground Truth\n(walks with turns + L/R asymmetry)", fontweight="bold")
    plt.legend(fontsize=8); plt.grid(linestyle="--", alpha=0.4); plt.tight_layout()
    plt.savefig("evaluation_charts/gait_detector_bench.png")
    print("[SAVED] evaluation_charts/gait_detector_bench.png")
