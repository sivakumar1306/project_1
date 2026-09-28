"""
Offline analysis of the deterministic fall-risk engine (Review-II results).
No LLM calls and no Supabase: runs on the synthetic demo cohort, so the
charts are reproducible anywhere.

Outputs (evaluation_charts/):
  fall_risk_trend.png          21-day FRS trajectory per demo user
  cycle_ablation.png           physiology-layer risk with vs without cycle-aware baselines
  evidence_weights.png         layer weights derived from published risk ratios
  contribution_breakdown.png   additive attribution for the highest-risk users
and fall_engine_analysis.json with all numbers.

Run:  python run_fall_engine_analysis.py
"""

import json
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent.fall_risk import (BASE_LAYER_WEIGHTS, EVIDENCE_RATIOS, LAYER_NAMES, compute_fall_risk,
                             compute_fall_risk_series)
from agent.correlations import compute_correlations
from scripts.fall_demo_data import DEMO_USERS, history_for

OUT = "evaluation_charts"
FLAG_THRESHOLD = 30   # physiology-layer risk at/above this is shown as "elevated physiological strain"
SHORT = {"stable": "Arun (low risk)", "luteal_strain": "Meera (luteal + poor recovery)", "stale": "Ravi (stale data)",
         "recent_fall": "Lakshmi (recent fall)", "healthy_luteal": "Priya (healthy luteal)"}
COLORS = {"stable": "#4A90E2", "luteal_strain": "#D0021B", "stale": "#9B9B9B", "recent_fall": "#F5A623", "healthy_luteal": "#7ED321"}


def main():
    os.makedirs(OUT, exist_ok=True)
    now = datetime.now(timezone.utc)
    today = now.date()
    hist = {u["scenario"]: history_for(u, today, now) for u in DEMO_USERS}
    summary = {"date": today.isoformat(), "weights": {}, "users": {}, "cycle_ablation": {}}

    # ── weights table ──
    print("\nEVIDENCE-DERIVED LAYER WEIGHTS")
    print(f"{'Layer':<14}{'Ratio':>8}{'ln(ratio)':>11}{'weight':>9}")
    for k in ("FH", "MO", "P", "RC"):
        ln = float(np.log(EVIDENCE_RATIOS[k]))
        summary["weights"][LAYER_NAMES[k]] = {"ratio": EVIDENCE_RATIOS[k], "ln": round(ln, 3), "weight": round(BASE_LAYER_WEIGHTS[k], 3)}
        print(f"{LAYER_NAMES[k]:<14}{EVIDENCE_RATIOS[k]:>8.2f}{ln:>11.3f}{BASE_LAYER_WEIGHTS[k]:>9.2f}")

    plt.figure(figsize=(7.5, 4.8), dpi=150)
    keys = ["FH", "MO", "P", "RC"]
    bars = plt.bar([LAYER_NAMES[k] for k in keys], [BASE_LAYER_WEIGHTS[k] for k in keys],
                   color=["#F5A623", "#4A90E2", "#D0021B", "#7ED321"], edgecolor="black", alpha=0.85)
    for b, k in zip(bars, keys):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{BASE_LAYER_WEIGHTS[k]:.2f}\n(ratio {EVIDENCE_RATIOS[k]})", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.ylim(0, 0.55)
    plt.ylabel("Layer weight λ (sums to 1)", fontweight="bold")
    plt.title("Evidence-Derived Layer Weights: λ ∝ ln(published risk ratio)", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(f"{OUT}/evidence_weights.png"); plt.close()

    # ── per-user today + trend ──
    print("\nTODAY'S SCORES")
    plt.figure(figsize=(10, 5.5), dpi=150)
    for sc, h in hist.items():
        r = compute_fall_risk(h, today)
        series = compute_fall_risk_series(h, today, 21)
        cors = compute_correlations(h["daily"])
        summary["users"][SHORT[sc]] = {
            "frs": r["frs"], "tier": r["tier"], "coverage_q": r["coverage_q"], "stale": r["stale"],
            "layers": {LAYER_NAMES[k]: round(v["risk"], 1) for k, v in r["layers"].items()},
            "top_contributors": [{"label": c["label"], "points": round(c["points"], 1)} for c in r["contributions"][:4]],
            "cycle": r["cycle"], "correlations": [{"pair": f"{c['x']}->{c['y']}", "r": round(c["r"], 2), "n": c["n"]} for c in cors],
            "series": series,
        }
        cyc = f" | cycle: {r['cycle']['phase']}, without correction {r['cycle'].get('frs_without_correction')}" if r["cycle"] else ""
        print(f"  {SHORT[sc]:<32} FRS {r['frs']:>3}  {r['tier']:<9} Q {r['coverage_q']:>3}%{' STALE' if r['stale'] else ''}{cyc}")
        xs = list(range(-len(series) + 1, 1))
        plt.plot(xs, [p["frs"] for p in series], marker="o", ms=3, lw=2, label=SHORT[sc], color=COLORS[sc])
    plt.axhspan(0, 40, color="#7ED321", alpha=0.08); plt.axhspan(40, 70, color="#F5A623", alpha=0.08); plt.axhspan(70, 100, color="#D0021B", alpha=0.08)
    plt.text(-20.5, 36, "LOW", fontsize=8, color="#417505"); plt.text(-20.5, 66, "MODERATE", fontsize=8, color="#b8741a"); plt.text(-20.5, 96, "HIGH", fontsize=8, color="#a0001b")
    plt.ylim(0, 100); plt.xlabel("Day (0 = today)", fontweight="bold"); plt.ylabel("Fall Risk Score (0-100)", fontweight="bold")
    plt.title("21-Day Fall-Risk Trajectory per Demo User (deterministic engine)", fontweight="bold")
    plt.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.93)); plt.grid(linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(f"{OUT}/fall_risk_trend.png"); plt.close()

    # ── cycle ablation ──
    print("\nCYCLE-AWARE BASELINE ABLATION (luteal days, physiology-layer risk)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150, sharey=True)
    for ax, sc in zip(axes, ["healthy_luteal", "luteal_strain"]):
        on = compute_fall_risk_series(hist[sc], today, 21, True)
        off = compute_fall_risk_series(hist[sc], today, 21, False)
        xs = list(range(-20, 1))
        luteal = [p["phase"] == "luteal" for p in on]
        # exclude Meera's last 3 days (genuine acute strain) from the false-flag count
        genuine = [sc == "luteal_strain" and i >= len(on) - 3 for i in range(len(on))]
        idx = [i for i in range(len(on)) if luteal[i] and not genuine[i]]
        m_on = float(np.mean([on[i]["physiology_risk"] for i in idx])) if idx else 0.0
        m_off = float(np.mean([off[i]["physiology_risk"] for i in idx])) if idx else 0.0
        f_on = sum(on[i]["physiology_risk"] >= FLAG_THRESHOLD for i in idx)
        f_off = sum(off[i]["physiology_risk"] >= FLAG_THRESHOLD for i in idx)
        summary["cycle_ablation"][SHORT[sc]] = {
            "normal_luteal_days": len(idx), "mean_physio_risk_with_correction": round(m_on, 1),
            "mean_physio_risk_without_correction": round(m_off, 1),
            "reduction_pct": round((1 - m_on / m_off) * 100, 1) if m_off else 0.0,
            f"days_flagged_ge_{FLAG_THRESHOLD}_with": f_on, f"days_flagged_ge_{FLAG_THRESHOLD}_without": f_off,
            "acute_days_with_correction": [round(on[i]["physiology_risk"]) for i in range(len(on)) if genuine[i]],
        }
        print(f"  {SHORT[sc]:<32} normal luteal days {len(idx):>2} | mean risk {m_off:5.1f} -> {m_on:5.1f} "
              f"| flagged(>={FLAG_THRESHOLD}) {f_off} -> {f_on}")
        for i, l in enumerate(luteal):
            if l:
                ax.axvspan(xs[i] - 0.5, xs[i] + 0.5, color="#F8E1E7", alpha=0.7, lw=0)
        ax.plot(xs, [p["physiology_risk"] for p in off], "--", color="#9B9B9B", lw=2, label="Without cycle correction")
        ax.plot(xs, [p["physiology_risk"] for p in on], "-", color=COLORS[sc], lw=2.4, label="With cycle-aware baseline")
        ax.axhline(FLAG_THRESHOLD, color="black", lw=0.8, ls=":")
        ax.set_title(SHORT[sc], fontweight="bold"); ax.set_xlabel("Day (0 = today); shaded = luteal phase")
        ax.grid(linestyle="--", alpha=0.4); ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Physiology-layer risk (0-100)", fontweight="bold")
    fig.suptitle("Cycle-Aware Baselines Suppress Normal Luteal-Phase Changes but Keep Genuine Strain", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/cycle_ablation.png"); plt.close(fig)

    # ── attribution ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    for ax, sc in zip(axes, ["luteal_strain", "recent_fall"]):
        r = compute_fall_risk(hist[sc], today)
        cs = [c for c in r["contributions"] if c["points"] >= 0.5][:6][::-1]
        ax.barh([c["label"] for c in cs], [c["points"] for c in cs], color=COLORS[sc], edgecolor="black", alpha=0.85)
        for i, c in enumerate(cs):
            ax.text(c["points"] + 0.4, i, f"+{c['points']:.1f}", va="center", fontsize=9, fontweight="bold")
        ax.set_title(f"{SHORT[sc]}: FRS {r['frs']} ({r['tier']})", fontweight="bold")
        ax.set_xlabel("Points added to the fall-risk score"); ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.set_xlim(0, max(c["points"] for c in cs) * 1.25)
    fig.suptitle("Additive Attribution: Contributions Sum Exactly to the Score", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/contribution_breakdown.png"); plt.close(fig)

    with open("fall_engine_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[SAVED] {OUT}/evidence_weights.png, fall_risk_trend.png, cycle_ablation.png, contribution_breakdown.png")
    print("[SAVED] fall_engine_analysis.json")


if __name__ == "__main__":
    main()
