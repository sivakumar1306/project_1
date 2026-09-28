"""
Build the Review-II deck from the VIT template.

Writes (default into docs/review2/):
  SLIDES.md               slide text + speaker notes
  NUMBERS.md              every number shown on a slide, with its source file / key
  Review-II_MedXAI.pptx   docs/review2/Review-2_Template_.pptx filled in with python-pptx

Every result number is read at build time from the results JSON files in the
repo root (or from code constants), through `L.num(...)`, which records its
source in the ledger. Nothing is typed in by hand. Missing values render as
{{MISSING: ...}}; unverified citations are marked [TO VERIFY].

Run:  python scripts/build_review2_deck.py [--out docs/review2]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TEMPLATE = os.path.join(ROOT, "docs", "review2", "Review-2_Template_.pptx")
CHARTS = os.path.join(ROOT, "evaluation_charts")
NAVY, GREY_TXT = "1D2F82", "595959"


# ── number ledger ─────────────────────────────────────────────────────────────
class Ledger:
    def __init__(self, root: str):
        self.root, self.rows, self._cache = root, [], {}

    def load(self, fname: str):
        if fname not in self._cache:
            path = os.path.join(self.root, fname)
            self._cache[fname] = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None
        return self._cache[fname]

    def num(self, fname: str, keys: list, fmt: str = "{:.3f}", label: str = "") -> str:
        """Value at keys inside fname, formatted; {{MISSING}} if absent. Recorded in the ledger."""
        d = self.load(fname)
        try:
            for k in keys:
                d = d[k]
            if d is None or (isinstance(d, float) and math.isnan(d)):
                raise KeyError
            s = fmt.format(d)
        except (KeyError, IndexError, TypeError):
            s = "{{MISSING: " + f"{fname}:{'/'.join(map(str, keys))}" + "}}"
        self.rows.append((s, fname, "/".join(map(str, keys)), label, ("key", fname, list(keys), fmt)))
        return s

    def derived(self, value, fmt: str, source: str, how: str, label: str = "") -> str:
        s = fmt.format(value) if value is not None else "{{MISSING: " + how + "}}"
        self.rows.append((s, source, how, label, ("derived",)))
        return s


# ── facts computed from the results files ────────────────────────────────────
def agent_facts(L: Ledger) -> dict:
    recs = L.load("fall_evaluation_results.json") or []
    f = {"n_cases": len({r["case"] for r in recs}), "n_runs": len({r["run"] for r in recs}), "v": {}}
    src = "fall_evaluation_results.json"
    for v in "ABCD":
        rv = [r for r in recs if r["version"] == v]
        em = [r for r in rv if r["expected_emergency"]]
        ne = [r for r in rv if not r["expected_emergency"]]
        st = [r for r in rv if r["category"] == "stale-data"]
        f["v"][v] = {
            "emerg": L.derived(f"{sum(r['emergency_detected'] for r in em)}/{len(em)}", "{}", src,
                               f"version={v}: emergency_detected over expected_emergency records", f"{v} emergencies caught"),
            "fa": L.derived(f"{sum(r['emergency_detected'] for r in ne)}/{len(ne)}", "{}", src,
                            f"version={v}: emergency_detected over non-emergency records", f"{v} false alarms"),
            "stale": L.derived(f"{sum(r['mentions_stale'] for r in st)}/{len(st)}", "{}", src,
                               f"version={v}, category=stale-data: mentions_stale", f"{v} stale disclosed"),
            "grounding": L.num("fall_evaluation_summary.json", [v, "grounding"], "{:.3f}", f"{v} strict grounding"),
            "latency": L.num("fall_evaluation_summary.json", [v, "latency"], "{:.2f}", f"{v} mean latency (s)"),
        }
    d = [r for r in recs if r["version"] == "D"]
    sv = [r["self_verification"] for r in d if (r.get("self_verification") or {}).get("triggered")]
    f["sv_n"] = L.derived(f"{len(sv)}/{len(d)}", "{}", src, "version=D: self_verification.triggered", "D self-verification fired")
    f["sv_cases"] = [r["case"] for r in d if (r.get("self_verification") or {}).get("triggered")]
    f["sv_before"] = L.derived(sum(s["score_before"] for s in sv) / len(sv) if sv else None, "{:.2f}", src,
                               "version=D: mean self_verification.score_before where triggered", "SV before")
    f["sv_after"] = L.derived(sum(s["score_after"] for s in sv) / len(sv) if sv else None, "{:.2f}", src,
                              "version=D: mean self_verification.score_after where triggered", "SV after")
    confs = [r["router_meta"]["confidence"] for r in d if (r.get("router_meta") or {}).get("confidence") is not None]
    f["conf_lo"] = L.derived(min(confs) if confs else None, "{:.2f}", src, "version=D: min router_meta.confidence", "router conf min")
    f["conf_hi"] = L.derived(max(confs) if confs else None, "{:.2f}", src, "version=D: max router_meta.confidence", "router conf max")
    f["widened"] = sum(1 for r in d if (r.get("router_meta") or {}).get("widened"))
    L.derived(f["widened"], "{}", src, "version=D: count router_meta.widened", "widening fired")
    ne = [len(r.get("streams") or []) for r in d if not r["emergency_detected"]]
    f["streams_d"] = L.derived(sum(ne) / len(ne) if ne else None, "{:.2f}", src,
                               "version=D: mean len(streams) on non-emergency replies", "D streams/reply")
    from agent.router import ALL_STREAMS
    f["streams_all"] = L.derived(len(ALL_STREAMS), "{}", "agent/router.py", "len(ALL_STREAMS) (Version C fetches all)", "all streams")
    bio = [r for r in d if r["category"] == "biometric-only"]
    f["bio_case"] = bio[0]["case"] if bio else None
    f["bio_llm"] = bool(bio and (bio[0].get("safety_signals") or {}).get("llm_triggered"))
    f["d_emerg_lat"] = ", ".join(L.derived(r["latency"], "{:.3f}", src, f"version=D case={r['case']}: latency", "D emergency latency")
                                 for r in d if r["emergency_detected"])
    f["n_emerg_cases"] = len({r["case"] for r in recs if r["expected_emergency"]})
    c1 = next((r for r in d if r["user"] == "luteal_strain" and r["category"] == "fall-risk"), None)
    f["demo1_g"] = L.derived(c1["strict_grounding"] if c1 else None, "{:.2f}", src,
                             f"version=D case={c1['case'] if c1 else '?'} (luteal_strain, fall-risk): strict_grounding", "demo step 1 grounding")
    return f


def bench_facts(L: Ledger) -> dict:
    import numpy as np
    from scripts.gait_synthetic_bench import run_bench
    r = run_bench()
    e1, e2 = np.abs(r[:, 1] - r[:, 0]).mean(), np.abs(r[:, 2] - r[:, 0]).mean()
    src = "scripts/gait_synthetic_bench.py::run_bench() (computed at build time)"
    return {"e1": L.derived(e1, "{:.2f}", src, "mean |v1 step CV - true CV|", "bench v1 error"),
            "e2": L.derived(e2, "{:.2f}", src, "mean |v2 stride CV - true CV|", "bench v2 error"),
            "r1": L.derived(np.corrcoef(r[:, 0], r[:, 1])[0, 1], "{:.2f}", src, "corr(true, v1)", "bench v1 r"),
            "r2": L.derived(np.corrcoef(r[:, 0], r[:, 2])[0, 1], "{:.2f}", src, "corr(true, v2)", "bench v2 r")}


# ── slide content ─────────────────────────────────────────────────────────────
def build_slides(L: Ledger) -> list[dict]:
    import agent.fall_risk as fr
    E, X1, X2, X3 = "fall_engine_analysis.json", "ltmm_results.json", "ltmm_experiment2_results.json", "ltmm_daily_results.json"
    A = agent_facts(L)
    B = bench_facts(L)
    V = A["v"]
    auc = lambda f, k: L.num(f, k, "{:.3f}", "/".join(map(str, k)))
    ci = lambda f, k: f"{L.num(f, k + [0], '{:.3f}')}–{L.num(f, k + [1], '{:.3f}')}"
    users = (L.load(E) or {}).get("users", {})
    ukey = lambda prefix: next((u for u in users if u.startswith(prefix)), prefix)
    code = lambda name, fmt="{}": L.derived(getattr(fr, name), fmt, "agent/fall_risk.py", name, name)
    tiers = fr.TIERS
    t_lo = L.derived(tiers[0][0], "{}", "agent/fall_risk.py", "TIERS[0] upper bound", "LOW < x")
    t_mid = L.derived(tiers[1][0], "{}", "agent/fall_risk.py", "TIERS[1] upper bound", "MODERATE < x")
    n1 = L.num(X1, ["n"], "{}")
    nf, nnf = L.num(X1, ["n_fallers"], "{}"), L.num(X1, ["n_nonfallers"], "{}")
    w = {k: L.num(E, ["weights", name, "weight"], "{:.3f}") for k, name in
         (("FH", "Fall history"), ("MO", "Motion"), ("P", "Physiology"), ("RC", "Recovery"))}
    ratio = {k: L.num(E, ["weights", name, "ratio"], "{:g}") for k, name in
             (("FH", "Fall history"), ("MO", "Motion"), ("P", "Physiology"), ("RC", "Recovery"))}
    ln = {k: L.num(E, ["weights", name, "ln"], "{:.3f}") for k, name in
          (("FH", "Fall history"), ("MO", "Motion"), ("P", "Physiology"), ("RC", "Recovery"))}
    pri, mee = "Priya (healthy luteal)", "Meera (luteal + poor recovery)"
    abl = lambda who, k, fmt="{:.1f}": L.num(E, ["cycle_ablation", who, k], fmt)
    tug_n = L.num(X1, ["clinical", "results", "TUG", "n"], "{}")
    import inspect
    import agent.correlations as co
    import agent.gait_features as gf

    def src_lit(value, relpath, needle, how):
        """A literal from code or a code comment, verified to be present in that file at build time."""
        text = open(os.path.join(ROOT, relpath), encoding="utf-8").read()
        return L.derived(value if needle in text else None, "{:g}" if isinstance(value, float) else "{}", relpath, how, how)
    c_n = L.derived(co.MIN_N, "{}", "agent/correlations.py", "MIN_N")
    c_r = L.derived(co.MIN_ABS_R, "{:g}", "agent/correlations.py", "MIN_ABS_R")
    c_p = L.derived(co.ALPHA, "{:g}", "agent/correlations.py", "ALPHA")
    bio_win = L.derived(inspect.signature(fr.get_recent_fall_event).parameters["window_minutes"].default, "{}",
                        "agent/fall_risk.py", "get_recent_fall_event(window_minutes=...) default")
    mad_k = src_lit(1.4826, "agent/fall_risk.py", "1.4826 * mad", "_robust_baseline: 1.4826 * MAD")
    bp = inspect.signature(gf._bandpass).parameters
    bp_lo = L.derived(bp["lo"].default, "{:g}", "agent/gait_features.py", "_bandpass lo default")
    bp_hi = L.derived(bp["hi"].default, "{:g}", "agent/gait_features.py", "_bandpass hi default")
    turn_thr = L.derived(gf.TURN_THRESHOLD_DPS, "{:g}", "agent/gait_features.py", "TURN_THRESHOLD_DPS")
    turn_m = L.derived(gf.TURN_MARGIN_S, "{:g}", "agent/gait_features.py", "TURN_MARGIN_S")
    pl_lo = src_lit(0.6, "agent/gait_features.py", "(iv > 0.6 * med) & (iv < 1.5 * med)", "extract_gait_features_v2 plausibility lower")
    pl_hi = src_lit(1.5, "agent/gait_features.py", "(iv > 0.6 * med) & (iv < 1.5 * med)", "extract_gait_features_v2 plausibility upper")
    ask = src_lit(0.749, "run_ltmm_experiment2.py", "TCN AUC 0.749", "literature value quoted in module docstring (Askhatova et al. 2026)")
    flag_thr = src_lit(30, "run_fall_engine_analysis.py", "FLAG_THRESHOLD = 30", "FLAG_THRESHOLD (physiology risk flag)")
    stale_days = src_lit(4, "scripts/fall_demo_data.py", 'last_day_offset = 4 if sc == "stale"', "stale demo user: last_day_offset")
    n_acute = L.derived(len((L.load(E) or {}).get("cycle_ablation", {}).get(mee, {}).get("acute_days_with_correction", [])) or None,
                        "{}", E, f"len(cycle_ablation/{mee}/acute_days_with_correction)")
    n_rep = src_lit(20, "run_ltmm_experiment2.py", "N_REPEATS = 20", "N_REPEATS (Exp 2)")
    src_lit(20, "run_ltmm_daily.py", "def repeated_cv(fn, X, y, repeats=20)", "repeated_cv repeats (Exp 3)")
    from scripts.fall_demo_data import DAYS, DEMO_USERS
    n_users = L.derived(len(DEMO_USERS), "{}", "scripts/fall_demo_data.py", "len(DEMO_USERS)", "demo users")
    n_days = L.derived(DAYS, "{}", "scripts/fall_demo_data.py", "DAYS", "demo days")
    S = []

    # 1 ── title
    S.append({"kind": "title", "template": 1,
              "project_title": "MedXAI: Explainable, Safety-Aware Fall-Risk Prediction from Smart-Ring Data",
              "notes": ("Title slide. Names, register numbers, guide and the guide's signature are left as the template's "
                        "placeholders: fill them in, get the signature, scan this slide and keep the scan as slide 1.")})

    # 2 ── introduction
    S.append({"kind": "bullets", "template": 2, "title": "Introduction & Problem Recap", "body_size": 16, "body": [
        ("h", "Problem recap"),
        ("b", "Fall systems detect a fall after it happens; they do not predict or explain risk"),
        ("b", "Health chatbots state health numbers they cannot be trusted with"),
        ("b", "Text-only safety checks miss a fallen user who types “I'm fine”"),
        ("h", "Objectives"),
        ("b", "Predict daily fall risk from ring data with a deterministic, explainable engine"),
        ("b", "Let sensors, engine and LLM check each other before anything reaches the user"),
        ("b", "Validate on real data against a clinical gold standard (Timed Up and Go)"),
    ], "notes": (
        "Open with the hook. An elderly woman falls at home. Her ring detects the fall. A minute later she types "
        "“I'm fine, just a bit shaken”. A text-only assistant reads that and tells her she is fine. Ours does not: the "
        "ring's fall event overrides the calm text and the reply is an emergency message. I will prove that live at the "
        f"end (demo, step 3), and it is scenario {A['bio_case']} in our evaluation, where only Version D caught it.\n\n"
        "The problem has three parts: existing fall systems react after the fall; chatbots are not trustworthy with "
        "health numbers; and safety checks only read text. Our objectives follow directly: an explainable deterministic "
        "fall-risk engine, an agent where sensors, engine and LLM check each other, and real-data validation against "
        "the clinical Timed Up and Go test.")})

    # 3 ── refinements
    S.append({"kind": "bullets", "template": 2, "title": "Refinements after Review-I Feedback", "body": [], "tables": [{
        "x": 0.7, "y": 1.45, "cols": [3.6, 5.2, 3.1], "size": 13, "row_h": 0.52,
        "header": ["Promised / requested at Review-I", "Delivered", "Where"],
        "rows": [
            ["Trend-deviation signal-processing layer", "Personal median/MAD baselines, z-score deviations, dead-zone strain", "agent/fall_risk.py"],
            ["Self-verification loop", "Strict number check vs patient data + one corrective regeneration", "agent/graph.py"],
            ["Personal correlation detection", f"Pearson r on the user's own history (n ≥ {c_n}, |r| ≥ {c_r}, p < {c_p})", "agent/correlations.py"],
            ["Router confidence scoring", "Confidence + safety-asymmetric widening of the data fetch", "agent/router.py"],
            ["Safety-critical application", "Fall prediction: fall-risk score, fall events, fall card", "agent/fall_risk.py, graph.py"],
            ["Deeper signal processing (guide)", "Gait pipeline v2: turn removal, McCamley contacts, stride CV", "agent/gait_features.py"],
        ]}],
        "notes": ("All four modules promised at Review-I are delivered, each with offline unit tests: the trend-deviation "
                  "signal-processing layer, the self-verification loop, personal correlation detection and router "
                  "confidence scoring. Following the feedback, we chose fall prediction as the safety-critical "
                  "application, and, as our guide asked, we went deeper into signal processing: a gait pipeline "
                  "validated on real IMU data (PhysioNet LTMM), shown in the results.")})

    # 4–6 ── literature
    lit_new = [
        ["11", "Deandrea et al., 2010", "Risk factors for falls in community-dwelling older people / Epidemiology",
         "Systematic review + meta-analysis", f"History of falls OR {ratio['FH']}, gait problems OR {ratio['MO']}. Gap: population-level risk, no personal daily trend"],
        ["12", "Hausdorff et al., 2001", "Gait variability and fall risk… 1-year prospective study / Arch Phys Med Rehabil",
         "Prospective cohort, stride-time variability", "Stride-time variability predicts future falls. Gap: lab measurement, not continuous wearable"],
        ["13", "[TO VERIFY: authors], JAMDA", "Orthostatic hypotension and falls meta-analysis / J Am Med Dir Assoc",
         "Meta-analysis", f"Orthostatic hypotension OR {ratio['P']} for falls. Gap: one-off BP test, not daily physiology"],
        ["14", "SWAN [TO VERIFY: authors], 2024", "Sleep and falls (SWAN cohort) / Innovation in Aging",
         "Cohort study", f"Poor sleep RR {ratio['RC']} for falls. Gap: questionnaire sleep, not ring-measured"],
        ["15", "Mishra, Snyder et al., 2020", "Pre-symptomatic detection of COVID-19 from smartwatch data / Nat Biomed Eng",
         "Personal-baseline deviation detection", "Deviations from a personal baseline flag illness early. Gap: not falls, no explanation"],
        ["16", "[TO VERIFY: authors], 2022", "Oura ring across the menstrual cycle / Int J Women's Health",
         "Ring skin temperature, HR, HRV by cycle phase", "Luteal phase shifts temperature and resting HR. Gap: not used in risk baselines"],
        ["17", "[TO VERIFY], 2026", "Wearable HRV across the menstrual cycle: living systematic review / Sports Med",
         "Living systematic review", "HRV varies with cycle phase. Gap: wearable scores ignore phase"],
        ["18", "Askhatova et al., 2026", "[TO VERIFY: title] / MethodsX",
         "Temporal CNN on LTMM, subject-wise CV", f"AUC {ask} using 1-min + 3-day data + demographics. Gap: black box, no explanation"],
        ["19", "Weiss et al., 2013", "Gait quality during daily life and fall risk (3-day recordings) / Neurorehabil Neural Repair",
         "LTMM study: 3-day lower-back accelerometer", "Daily-life gait differs in fallers. Gap: no personal agent or explanation"],
        ["20", "McCamley et al., 2012", "Initial and final contact from lower-trunk inertial data / Gait & Posture",
         "Gaussian-derivative initial-contact detection", "Accurate step timing from one trunk IMU. Gap: detection only"],
        ["21", "Moe-Nilssen & Helbostad, 2004 [TO VERIFY]", "Gait cycle characteristics by trunk accelerometry / J Biomech",
         "Autocorrelation step/stride regularity", "Regularity from one sensor. Gap: method only, no risk model"],
        ["22", "Drover et al., 2017", "Faller classification from turn and straight-walking features / Sensors",
         "Turn vs straight-walking accelerometer features", "Turn features help classify fallers. Gap: not replicated on our data"],
    ]
    hdr = ["No.", "Author(s) & Year", "Title / Source", "Method / Approach", "Findings & Research Gap"]
    cols = [0.5, 1.9, 3.3, 2.5, 3.93]
    S.append({"kind": "bullets", "template": 3, "title": "Literature Review (1/3): Review-I Papers", "body": [],
              "tables": [{"x": 0.6, "y": 1.45, "cols": [0.5, 2.6, 3.0, 2.4, 3.63], "size": 11, "row_h": 0.4, "header": hdr,
                          "rows": [[str(i), "{{Review-I paper %d}}" % i, "", "", ""] for i in range(1, 11)]}],
              "footnote": "Papers 1–10: the ten Review-I papers (to be pasted). Papers 11–22: added for Review-II.",
              "notes": "The ten papers from Review-I stay in the table; they covered health agents, RAG and grounding. "
                       "Kriti will paste them here and in the references."})
    for part, rows in ((2, lit_new[:6]), (3, lit_new[6:])):
        S.append({"kind": "bullets", "template": 3, "title": f"Literature Review ({part}/3): Fall Risk & Gait", "body": [],
                  "tables": [{"x": 0.6, "y": 1.45, "cols": cols, "size": 11, "row_h": 0.76, "header": hdr, "rows": rows}],
                  "footnote": "[TO VERIFY] = bibliographic detail still to be checked against the paper.",
                  "notes": ("Rows 11–14 give the evidence for our layer weights (fall history, gait, orthostatic "
                            "hypotension, sleep). Row 15 is the personal-baseline idea. Rows 16–17 motivate cycle-aware baselines."
                            if part == 2 else
                            "Rows 18–22 are the gait and LTMM papers. Askhatova 2026 is the fair comparison for our LTMM "
                            "results because it also uses subject-wise cross-validation. McCamley and Moe-Nilssen are the "
                            "signal-processing methods we implemented. Drover 2017 motivated our turn features; on our "
                            "data they did not help (Experiment 2, model M1).")})

    # 7 ── research gaps
    S.append({"kind": "bullets", "template": 2, "title": "Research Gaps Identified", "body": [], "tables": [{
        "x": 0.7, "y": 1.6, "cols": [5.6, 6.3], "size": 15, "row_h": 0.9,
        "header": ["Gap in existing work", "How MedXAI addresses it"],
        "rows": [
            ["Fall systems detect falls but do not predict or explain risk", "Daily fall-risk score with per-feature contributions"],
            ["Best fall-risk models are black boxes", "Deterministic engine; the LLM only explains its numbers"],
            ["Health agents check safety from text only", "Tri-modal fusion: keyword + LLM + ring fall event"],
            ["No cycle-aware baselines in wearable risk scores", "Phase-matched baselines for HRV, resting HR, temperature"],
        ]}],
        "notes": "Four gaps come out of the literature, and each maps to one of our contributions on the next slide."})

    # 8 ── novelty
    S.append({"kind": "bullets", "template": 2, "title": "Novelty: Existing → Ours → Proof", "body": [], "tables": [{
        "x": 0.6, "y": 1.35, "cols": [2.35, 2.6, 3.3, 3.88], "size": 12, "row_h": 0.95,
        "header": ["Contribution", "Existing", "Ours", "Proof (our data)"],
        "rows": [
            ["1. Tri-modal safety fusion", "Keyword or LLM reads the text only",
             f"Keyword + LLM + ring fall event (last {bio_win} min), fused",
             f"Emergencies caught: D {V['D']['emerg']} (incl. biometric-only), C {V['C']['emerg']}, A {V['A']['emerg']}, B {V['B']['emerg']}"],
            ["2. Compute-then-explain, closed-loop grounding", "LLM generates health numbers",
             "Engine computes; LLM explains; every number checked, one corrective retry",
             f"Strict grounding D {V['D']['grounding']} vs A {V['A']['grounding']}, B {V['B']['grounding']}; self-verification {A['sv_before']} → {A['sv_after']}"],
            ["3. Evidence-derived weights", "Hand-tuned or learned black-box weights",
             "λ ∝ ln(published risk ratio), renormalised by coverage",
             f"FH {w['FH']} (OR {ratio['FH']}), MO {w['MO']} (OR {ratio['MO']}), P {w['P']} (OR {ratio['P']}), RC {w['RC']} (RR {ratio['RC']})"],
            ["4. Cycle-aware baselines", "One baseline for every day",
             "Phase-matched baseline for HRV, resting HR, temperature",
             f"Healthy luteal user: physiology risk {abl(pri, 'mean_physio_risk_without_correction')} → {abl(pri, 'mean_physio_risk_with_correction')}; "
             f"strained user: flagged days {abl(mee, 'days_flagged_ge_30_without', '{}')} → {abl(mee, 'days_flagged_ge_30_with', '{}')}"],
        ]}],
        "footnote": (f"Additional safeguard (unit-tested): safety-asymmetric routing. It did not fire in the evaluation "
                     f"(router confidence {A['conf_lo']}–{A['conf_hi']})."),
        "notes": (
            f"Four core contributions, each with proof from our own results files.\n"
            f"1. Tri-modal safety fusion: in {A['n_cases']} scenarios with {A['n_emerg_cases']} emergencies, D caught "
            f"{V['D']['emerg']}, including scenario {A['bio_case']}, where the text was calm and only the ring's fall event "
            f"showed the emergency; the LLM classifier {'also flagged it' if A['bio_llm'] else 'did not flag it'}. C caught "
            f"{V['C']['emerg']}; A and B caught {V['A']['emerg']} and {V['B']['emerg']}. No version raised a false alarm "
            f"(D {V['D']['fa']}).\n"
            f"2. Compute-then-explain: strict grounding is the share of numbers in a reply that appear in the user's real "
            f"data, on personal queries. D {V['D']['grounding']}, C {V['C']['grounding']}, A {V['A']['grounding']}, "
            f"B {V['B']['grounding']}. Self-verification fired {A['sv_n']} times and fixed that reply from {A['sv_before']} "
            f"to {A['sv_after']}.\n"
            f"3. Weights: λ is proportional to the natural log of published odds/risk ratios, so fall history counts most. "
            f"No training data needed; the ratios come from different studies, which we state as a limitation.\n"
            f"4. Cycle-aware baselines: for the healthy luteal user, mean physiology-layer risk on "
            f"{abl(pri, 'normal_luteal_days', '{}')} normal luteal days drops from {abl(pri, 'mean_physio_risk_without_correction')} to "
            f"{abl(pri, 'mean_physio_risk_with_correction')}; for the strained user, days flagged at ≥ {flag_thr} drop from "
            f"{abl(mee, 'days_flagged_ge_30_without', '{}')} to {abl(mee, 'days_flagged_ge_30_with', '{}')} while her genuine "
            f"acute strain days stay high. Honest caveat: for the healthy user one day stays flagged "
            f"({abl(pri, 'days_flagged_ge_30_without', '{}')} → {abl(pri, 'days_flagged_ge_30_with', '{}')}).\n"
            f"Safety-asymmetric routing is an extra safeguard. It did not fire in the evaluation because the router was "
            f"confident ({A['conf_lo']}–{A['conf_hi']}), so we only claim it through unit tests.")})

    # 9 ── architecture
    S.append({"kind": "image", "template": 4, "title": "System Design & Architecture",
              "images": [{"path": os.path.join(ROOT, "docs", "review2", "architecture_slide.png"), "box": (0.5, 1.2, 12.33, 4.3)}],
              "callouts": [("Engine computes", "Scores come from agent/fall_risk.py, never from the LLM"),
                           ("LLM only explains", "Every number is checked against patient data before sending"),
                           ("Ring can override chat", f"An uncancelled ring fall event (last {bio_win} min) forces an emergency reply")],
              "notes": ("The request goes through six stages. (1) Router, LLM safety classifier and the ring fall-event "
                        "lookup run in parallel. (2) Tri-modal fusion decides if this is an emergency; if so we reply "
                        "immediately. (3) Otherwise we fetch only the data streams the router selected. (4) The deterministic "
                        "engine computes the score, tier, confidence and contributors. (5) The LLM explains those numbers as "
                        "facts, rationale and action. (6) Every number in the reply is checked against the patient data; if "
                        "one is missing we regenerate once. Three things to remember: the engine computes, the LLM only "
                        "explains, and the ring can override the chat. The full function-level diagram is the backup slide "
                        "at the end and in docs/ARCHITECTURE.md.")})

    # 10 ── implementation: engine
    S.append({"kind": "bullets", "template": 5, "title": "Implementation (1/2): Fall-Risk Engine",
              "body_box": (0.7, 1.35, 6.2, 5.4), "body_size": 14, "body": [
                  ("h", "Personal baseline and deviation"),
                  ("b", f"Median / MAD of the previous {code('BASELINE_DAYS')} days (≥ {code('MIN_BASELINE_POINTS')} days)"),
                  ("b", f"z = (today − median) / ({mad_k}·MAD), in the risk direction"),
                  ("b", f"Strain 0–100: dead zone below z = {code('DEAD_ZONE_Z', '{:g}')}, full at z = {code('FULL_STRAIN_Z', '{:g}')}"),
                  ("h", "Four layers → score"),
                  ("b", "Layer risk = weighted mean strain of its features"),
                  ("b", "FRS = Σ λ′_L · risk_L,  λ′_L = λ_L·c_L / Σ λ_j·c_j  (c = data coverage)"),
                  ("b", f"Tier: LOW < {t_lo} ≤ MODERATE < {t_mid} ≤ HIGH; contributions add up to FRS"),
              ],
              "tables": [{"x": 7.2, "y": 1.45, "cols": [1.9, 1.3, 1.1, 1.2], "size": 13, "row_h": 0.5,
                          "header": ["Layer", "Ratio", "ln", "λ"],
                          "rows": [["Fall history", f"OR {ratio['FH']}", ln["FH"], w["FH"]],
                                   ["Motion (gait)", f"OR {ratio['MO']}", ln["MO"], w["MO"]],
                                   ["Physiology (OH)", f"OR {ratio['P']}", ln["P"], w["P"]],
                                   ["Recovery (sleep)", f"RR {ratio['RC']}", ln["RC"], w["RC"]]]}],
              "footnote_at": (7.2, 4.15, 5.5),
              "footnote": "λ = ln(ratio) / Σ ln(ratio). Sources: Deandrea 2010 [11], JAMDA [13], SWAN [14].",
              "notes": (f"Every feature (HRV, resting HR, SpO2, skin temperature, sleep duration and score, stress, steps) is "
                        f"compared with the user's own robust baseline: median and MAD over the previous {code('BASELINE_DAYS')} "
                        f"days. The z-score goes through a dead zone so normal day-to-day noise adds nothing, and saturates at "
                        f"z = {code('FULL_STRAIN_Z', '{:g}')}. Layers average their features' strain; the four layers are combined "
                        f"with evidence weights λ, renormalised by how much data each layer actually has (coverage), so a "
                        f"missing sensor is not silently read as 'no risk'. Because the score is a weighted sum, each feature's "
                        f"contribution in points adds up exactly to the score, which is what the LLM is allowed to explain. "
                        f"Weights: fall history {w['FH']}, motion {w['MO']}, physiology {w['P']}, recovery {w['RC']}. "
                        f"Exact formulas: docs/METHODS.md §1.")})

    # 11 ── implementation: signal processing + code
    S.append({"kind": "bullets", "template": 5, "title": "Implementation (2/2): Gait & Safety Fusion",
              "body_box": (0.7, 1.35, 5.8, 5.4), "body_size": 14, "body": [
                  ("h", "Gait pipeline v2 (lower-back IMU, 100 Hz)"),
                  ("b", f"Band-pass {bp_lo}–{bp_hi} Hz; remove turns (yaw > {turn_thr} °/s ± {turn_m} s)"),
                  ("b", "Initial contacts: McCamley method, person-specific step period"),
                  ("b", f"Reject intervals outside {pl_lo}–{pl_hi} × median"),
                  ("b", "Stride-time CV, step/stride regularity, harmonic ratio, cadence"),
                  ("b", "Untrained index: mean of direction-signed z-scores, no labels used"),
              ],
              "code": {"box": (6.8, 1.45, 6.0, 3.9), "size": 10.5, "text": (
                  "def check_emergency_fused(message: str,\n"
                  "        llm_res: tuple[bool, str, float],\n"
                  "        biometric_event: dict | None = None\n"
                  "        ) -> tuple[bool, str, dict]:\n"
                  "    ...\n"
                  "    # Third signal (Review-II): physiological evidence\n"
                  "    # independent of the text. An uncancelled fall detected\n"
                  "    # by the ring/phone in the last 30 minutes escalates\n"
                  "    # ANY message, even a calm-sounding one (\"I'm fine\").\n"
                  "    biometric_triggered = bool(biometric_event)\n\n"
                  "    is_emergency = bool(effective_kw_triggered) \\\n"
                  "        or llm_triggered or biometric_triggered")},
              "footnote_at": (6.8, 5.45, 6.0),
              "footnote": "agent/tools.py · check_emergency_fused (excerpt)",
              "notes": ("The guide asked for deeper signal processing. Version 1 of the gait detector gave implausibly high "
                        "variability, so version 2 removes turns using the gyroscope, detects initial contacts with the "
                        "McCamley method, rejects implausible intervals and uses stride-time CV as in Hausdorff 2001. Every "
                        "threshold is in docs/METHODS.md §2. On the right is the core of the safety fusion: an uncancelled "
                        f"ring fall event in the last {bio_win} minutes is enough on its own to make the reply an emergency, "
                        "whatever the text says.")})

    # 12 ── results: engine
    cohort_rows = []
    for prefix, note in (("Arun", ""), ("Meera", "cycle"), ("Ravi", "stale"), ("Lakshmi", ""), ("Priya", "cycle")):
        k = ukey(prefix)
        extra = ""
        if note == "cycle":
            extra = f"cycle-aware; {L.num(E, ['users', k, 'cycle', 'frs_without_correction'], '{}')} without"
        elif note == "stale":
            extra = "data flagged STALE"
        cohort_rows.append([k, L.num(E, ["users", k, "frs"], "{}"), L.num(E, ["users", k, "tier"], "{}"),
                            L.num(E, ["users", k, "coverage_q"], "{}") + "%", extra])
    S.append({"kind": "image", "template": 6, "title": "Results & Analysis (75%): Engine",
              "tables": [{"x": 0.6, "y": 1.3, "cols": [2.9, 0.75, 1.35, 0.75, 2.05], "size": 12, "row_h": 0.45,
                          "header": ["User (synthetic)", "FRS", "Tier", "Q", "Note"], "rows": cohort_rows}],
              "images": [{"path": os.path.join(CHARTS, "cycle_ablation.png"), "box": (0.6, 4.2, 7.8, 2.75)}],
              "side": [("h", "Cycle-aware ablation"),
                       ("b", f"Healthy luteal: risk {abl(pri, 'mean_physio_risk_without_correction')} → {abl(pri, 'mean_physio_risk_with_correction')} "
                             f"({abl(pri, 'normal_luteal_days', '{}')} days, −{abl(pri, 'reduction_pct')}%)"),
                       ("b", f"Strained luteal: {abl(mee, 'mean_physio_risk_without_correction')} → {abl(mee, 'mean_physio_risk_with_correction')}; "
                             f"flags {abl(mee, 'days_flagged_ge_30_without', '{}')} → {abl(mee, 'days_flagged_ge_30_with', '{}')}"),
                       ("b", f"Genuine acute strain is kept (last {n_acute} days stay high)"),
                       ("b", "Synthetic cohort: shows the engine behaves as designed")],
              "side_box": (9.0, 1.3, 3.8, 5.5),
              "notes": (f"{n_users} synthetic demo users, {n_days} days each, generated with fixed seeds (engine run dated "
                        f"{L.num(E, ['date'], '{}')}). The tiers come out as designed: the recent-faller is HIGH, the "
                        f"luteal user with poor recovery MODERATE, the others LOW. The stale user is flagged and her coverage "
                        f"drops to {L.num(E, ['users', ukey('Ravi'), 'coverage_q'], '{}')}%. The chart is the cycle-aware ablation: "
                        f"grey is without correction, colour with. Normal luteal changes are suppressed, but the strained user's "
                        f"real acute strain stays high. This is synthetic data, so it demonstrates correct behaviour, not "
                        f"clinical accuracy; that is what the next slide is for.")})

    # 13 ── results: LTMM exp 1
    S.append({"kind": "image", "template": 6, "title": "Results: Real-Data Validation (LTMM)",
              "images": [{"path": os.path.join(CHARTS, "ltmm_roc.png"), "box": (0.5, 1.2, 5.7, 5.7)},
                         {"path": os.path.join(CHARTS, "gait_detector_bench.png"), "box": (6.3, 1.2, 3.1, 2.85)}],
              "tables": [{"x": 9.6, "y": 1.3, "cols": [2.0, 1.23], "size": 12, "row_h": 0.5,
                          "header": ["Faller vs non-faller", "AUC"],
                          "rows": [["v2 gait index, untrained", auc(X1, ["v2", "index_auc"])],
                                   ["  95% CI", ci(X1, ["v2", "index_ci95"])],
                                   ["Timed Up and Go", auc(X1, ["clinical", "results", "TUG", "auc"])],
                                   ["Berg Balance Scale", auc(X1, ["clinical", "results", "BERG", "auc"])],
                                   ["Cadence alone (v2)", auc(X1, ["v2", "per_feature", "cadence", "auc"])]]}],
              "side": [("b", f"n = {n1} ({nf} fallers, {nnf} non-fallers); TUG/Berg on {tug_n}"),
                       ("b", f"Detector error {B['e1']} → {B['e2']} CV points (r {B['r1']} → {B['r2']})"),
                       ("b", f"Real data: stride CV median {L.num(X1, ['quality', 'v1_step_cv_median'], '{:.2f}')}% → "
                             f"{L.num(X1, ['quality', 'v2_stride_cv_median'], '{:.2f}')}%")],
              "side_box": (6.3, 4.25, 6.5, 2.5), "side_size": 14,
              "notes": (f"This is real data: {n1} older adults from PhysioNet LTMM, one-minute lab walks with a lower-back IMU, "
                        f"labelled by fall history ({nf} fallers). Our untrained v2 gait index reaches AUC "
                        f"{auc(X1, ['v2', 'index_auc'])} (95% CI {ci(X1, ['v2', 'index_ci95'])}), with no training on labels. "
                        f"The clinical gold standard, Timed Up and Go, gets {auc(X1, ['clinical', 'results', 'TUG', 'auc'])} on the "
                        f"{tug_n} subjects who have it, and our index on those same subjects gets "
                        f"{auc(X1, ['clinical', 'results', 'TUG', 'v2_index_same_subjects'])}. So: comparable to the clinical test, "
                        f"not better; the confidence intervals overlap. Berg is {auc(X1, ['clinical', 'results', 'BERG', 'auc'])}. "
                        f"The small chart is the synthetic ground-truth bench: v1 overestimated variability by {B['e1']} points, "
                        f"v2 by {B['e2']} (correlation with truth {B['r1']} → {B['r2']}). On the real walks, median variability "
                        f"fell from {L.num(X1, ['quality', 'v1_step_cv_median'], '{:.2f}')}% (implausible) to "
                        f"{L.num(X1, ['quality', 'v2_stride_cv_median'], '{:.2f}')}%. Caveats: labels are retrospective, and the "
                        f"sensor is on the lower back, not a ring.")})

    # 14 ── results: rigor
    m2 = lambda k, key="auc": auc(X2, ["models", k, key])
    exp2_rows = [[k, (L.load(X2) or {}).get("models", {}).get(k, {}).get("name", k).replace(" [primary]", " ★"),
                  m2(k)] for k in ("M0", "M1", "M2", "M3", "M4", "M5", "M6")]
    m3 = lambda k: auc(X3, ["results", k, "auc"])
    exp3_rows = [[k, (L.load(X3) or {}).get("results", {}).get(k, {}).get("name", k).replace(" [primary]", " ★"),
                  L.num(X3, ["results", k, "n"], "{}"), m3(k)] for k in ("M0", "D0", "D1", "D2", "D3")]
    S.append({"kind": "image", "template": 7, "title": "Results (contd.): Rigor",
              "tables": [{"x": 0.6, "y": 1.3, "cols": [0.6, 3.9, 0.9], "size": 11, "row_h": 0.4,
                          "header": ["Exp 2", f"Lab walks, n = {L.num(X2, ['n'], '{}')}, pre-registered", "AUC"], "rows": exp2_rows},
                         {"x": 6.4, "y": 1.3, "cols": [0.6, 4.0, 0.6, 0.9], "size": 11, "row_h": 0.4,
                          "header": ["Exp 3", "Daily life, 3-day recordings", "n", "AUC"], "rows": exp3_rows}],
              "side": [("b", "Pre-registered, every model reported: more features and complex models did not beat M0"),
                       ("b", f"Daily-life gait did not improve on the same subjects: best D3 {m3('D3')} vs lab M0 {m3('M0')}"),
                       ("b", f"Cadence is the consistent marker: lab AUC {auc(X1, ['v2', 'per_feature', 'cadence', 'auc'])}, "
                             f"daily life {auc(X3, ['univariate', 'd_cadence', 'auc'])}"),
                       ("b", f"Askhatova 2026: {ask} with 3-day data + deep network (black box)")],
              "side_box": (6.4, 4.05, 6.4, 2.8), "side_size": 13,
              "footnote_at": (0.6, 4.65, 5.5),
              "footnote": f"★ primary model. Trained models: mean over {n_rep}×5 subject-wise CV.",
              "notes": (f"We tried hard to beat our own number, with the protocol fixed in advance. Experiment 2: seven models "
                        f"on the lab walks. The untrained v2 index M0 scores {m2('M0')}. Adding turn features (M1) gives "
                        f"{m2('M1')}. The best trained model is logistic regression on the five v2 features, M2, "
                        f"{m2('M2')} ± {L.num(X2, ['models', 'M2', 'sd'], '{:.3f}')}. The pre-registered primary, M4, gets "
                        f"{m2('M4')}; its permutation test is p = {L.num(X2, ['models', 'M4', 'permutation', 'p_value'], '{:.3f}')} on a "
                        f"single split. Random forests get {m2('M5')} and {m2('M6')}: more complexity did worse on "
                        f"{L.num(X2, ['n'], '{}')} subjects. Experiment 3: daily-life walking from the 3-day recordings. The "
                        f"primary combined index D1 gets {m3('D1')} (permutation p = {L.num(X3, ['results', 'D1', 'perm_p'], '{:.3f}')}), "
                        f"below the lab index on the same {L.num(X3, ['results', 'M0', 'n'], '{}')} subjects ({m3('M0')}). "
                        f"Daily-life alone: D0 {m3('D0')}, D2 {m3('D2')}. We report all of it. That is why the headline number "
                        f"is trustworthy: it survived our own attempts to replace it. Cadence is the one marker that holds in "
                        f"both settings. Askhatova 2026 reports {ask} with subject-wise CV, using 3-day data, demographics and a "
                        f"deep network; our index is untrained and fully explainable.")})

    # 15 ── results: agent
    agent_rows = []
    for v, name in (("A", "A · Plain LLM"), ("B", "B · Plain RAG"), ("C", "C · Fetch-all + keyword safety"), ("D", "D · Ours")):
        agent_rows.append([name, V[v]["emerg"], V[v]["fa"], V[v]["grounding"], V[v]["stale"]])
    S.append({"kind": "image", "template": 7, "title": "Results (contd.): Agent Versions A–D",
              "tables": [{"x": 0.6, "y": 1.3, "cols": [3.2, 1.55, 1.3, 1.45, 1.3], "size": 12, "row_h": 0.48,
                          "header": ["Version", "Emergencies caught", "False alarms", "Strict grounding", "Stale data disclosed"],
                          "rows": agent_rows}],
              "images": [{"path": os.path.join(CHARTS, "fall_emergency_detection.png"), "box": (0.6, 3.95, 5.95, 3.0)},
                         {"path": os.path.join(CHARTS, "fall_grounding.png"), "box": (6.85, 3.95, 5.95, 3.0)}],
              "side": [("b", f"{A['n_cases']} scenarios, {A['n_runs']} run each"),
                       ("b", f"D fetched {A['streams_d']} streams per reply; C fetches all {A['streams_all']}"),
                       ("b", f"Self-verification fired {A['sv_n']}: {A['sv_before']} → {A['sv_after']}"),
                       ("b", f"Latency D {V['D']['latency']} s vs C {V['C']['latency']} s (batch run, rate-limit retries): not faster")],
              "side_box": (9.7, 1.3, 3.1, 2.6), "side_size": 12,
              "notes": (f"Same {A['n_cases']} scenarios for all four versions, one run each, so the rates are small counts: "
                        f"{A['n_emerg_cases']} emergencies and {A['n_cases'] - A['n_emerg_cases']} non-emergencies. "
                        f"D caught {V['D']['emerg']}, C {V['C']['emerg']}, A and B {V['A']['emerg']}. C missed the biometric-only "
                        f"case because it only reads text. No version raised a false alarm, including on “I fell asleep on the "
                        f"couch”. Strict grounding: D {V['D']['grounding']}, C {V['C']['grounding']}, A {V['A']['grounding']}, "
                        f"B {V['B']['grounding']}; A and B invent numbers because they have no data. Only D told the stale-data user "
                        f"that the data was old ({V['D']['stale']}). D fetched {A['streams_d']} data streams per non-emergency reply "
                        f"on average versus all {A['streams_all']} for C. Latency: D averaged {V['D']['latency']} s and C "
                        f"{V['C']['latency']} s in this batch run, which includes rate-limit retries and D's extra router, safety "
                        f"and verification calls, so we do not claim D is faster. Safety widening did not fire because router "
                        f"confidence was {A['conf_lo']}–{A['conf_hi']} on every query; it is covered by unit tests only.")})

    # 16 ── live demo
    S.append({"kind": "image", "template": 7, "title": "Live Demo",
              "tables": [{"x": 0.6, "y": 1.35, "cols": [0.6, 1.7, 3.9, 5.9], "size": 13, "row_h": 0.7,
                          "header": ["#", "User", "Message", "What to point out"],
                          "rows": [["1", "Meera", "What's my fall risk today?", f"Score, tier, top contributors, cycle adjustment; grounding {A['demo1_g']} in the evaluation"],
                                   ["2", "Meera", "I fell asleep on the couch", "No false alarm (guarded fall pattern)"],
                                   ["3", "/user lakshmi", "I'm fine, just a bit shaken", "EMERGENCY from the ring fall event; LLM alone said no"],
                                   ["4", "/user ravi", "What's my fall risk today?", "Discloses stale data, lower confidence"]]}],
              "side": [("b", "Run: py demo_display.py (switch users with /user <name>)"),
                       ("b", "Backup: recorded video of the same four steps")],
              "side_box": (0.6, 5.0, 12.0, 1.2), "side_size": 14,
              "notes": ("Before the review: run `py seed_fall_demo.py` in the morning (data is generated relative to today), "
                        "then `py seed_fall_demo.py --fall-now` about 5 minutes before the demo, because the ring signal only "
                        "looks back 30 minutes. Keep a backup video of these four steps. "
                        "Step 1: Meera asks her fall risk; point to the score, the contributors, the cycle-adjustment line and "
                        "that every number is grounded. Step 2: “I fell asleep on the couch” must not raise an alarm. "
                        "Step 3: switch to Lakshmi and type “I'm fine, just a bit shaken”: the reply is an emergency because of "
                        "the ring fall event, even though the text is calm and the LLM classifier alone did not flag it. This "
                        "closes the story from the first slide. Step 4: Ravi's data is " + stale_days + " days old; the reply says the estimate is "
                        "less reliable and asks him to sync. Scores on the day can differ from the table on slide 12 because "
                        "the demo data is generated relative to the review date.")})

    # 17 ── challenges
    S.append({"kind": "bullets", "template": 8, "title": "Challenges & Remaining Work", "body_box": (7.3, 1.35, 5.5, 5.4),
              "body_size": 14, "body": [
                  ("h", "Limitations we state ourselves"),
                  ("b", "Retrospective fall labels: discrimination, not prediction"),
                  ("b", "Lower-back sensor, not the ring"),
                  ("b", f"Small sample: n = {n1} lab walks"),
                  ("b", "Synthetic demo cohort for engine and agent tests"),
                  ("b", "Evidence weights approximate (studies not jointly adjusted)"),
                  ("b", "No authentication yet"),
              ],
              "tables": [{"x": 0.6, "y": 1.45, "cols": [2.9, 3.6], "size": 12, "row_h": 0.72,
                          "header": ["Challenge", "Solution"],
                          "rows": [["API rate limits (429)", "Retry with backoff; batched evaluation pauses"],
                                   ["LLM hallucinated numbers", "Strict grounding + one self-verification retry"],
                                   ["Keyword false alarms (“fell asleep”)", "LLM override + guarded fall patterns"],
                                   ["Implausible gait variability (v1)", "Turn removal + McCamley detector (v2)"],
                                   ["No dataset with ring + falls + chat", "Validate each layer on its own data"]]}],
              "notes": ("Challenges we solved, left: rate limits, hallucinated numbers, keyword false alarms, an implausible "
                        "gait detector, and the lack of a combined dataset, which we handled by validating each layer "
                        "separately. Right: the limitations we state before anyone asks. The labels are retrospective; the "
                        "sensor is a lower-back IMU, not the ring; the sample is small; the engine and agent are tested on a "
                        "synthetic cohort; the evidence weights come from different studies; and there is no authentication "
                        "yet.")})

    # 18 ── remaining work
    S.append({"kind": "bullets", "template": 8, "title": "Remaining Work (25%) & Timeline", "body": [],
              "tables": [{"x": 0.7, "y": 1.45, "cols": [2.6, 6.6, 2.7], "size": 14, "row_h": 0.62,
                          "header": ["Dates (proposed)", "Work item", "Output"],
                          "rows": [["01–07 Oct 2026", "Port the v2 gait algorithm to phone / ring IMU", "Motion layer from real gait"],
                                   ["08–14 Oct 2026", "Validate the cycle layer on mcPHASES", "Cycle-layer results"],
                                   ["08–14 Oct 2026", "Learn layer weights from data; compare with evidence weights", "Weight comparison"],
                                   ["15–21 Oct 2026", "Flutter integration incl. the fall_risk card", "App build"],
                                   ["15–21 Oct 2026", "Authentication: verified JWT + row-level security", "Secured API"],
                                   ["22–27 Oct 2026", "Final report and slides", "Report"],
                                   ["28 Oct 2026", "Review-III", "—"]]}],
              "notes": ("The remaining 25%, scheduled up to Review-III on 28 October 2026. First, move the gait algorithm from "
                        "the lower-back dataset to the phone or ring IMU, so the Motion layer uses real gait rather than step "
                        "counts. Then validate the cycle layer on mcPHASES and learn the layer weights from data to compare "
                        "with our evidence-derived weights. Then Flutter integration including the fall-risk card, and "
                        "authentication. The last week is the final report. Dates are our proposal.")})

    # 19–20 ── references
    refs = [f"[{i}] {{{{Review-I paper {i}: IEEE reference}}}}" for i in range(1, 11)] + [
        "[11] S. Deandrea, E. Lucenteforte, F. Bravi, R. Foschi, C. La Vecchia, and E. Negri, “Risk factors for falls in community-dwelling older people: A systematic review and meta-analysis,” Epidemiology, vol. 21, no. 5, pp. 658–668, 2010.",
        "[12] J. M. Hausdorff, D. A. Rios, and H. K. Edelberg, “Gait variability and fall risk in community-living older adults: A 1-year prospective study,” Arch. Phys. Med. Rehabil., vol. 82, no. 8, pp. 1050–1056, 2001.",
        "[13] [TO VERIFY: authors], “Orthostatic hypotension and falls: A systematic review and meta-analysis,” J. Am. Med. Dir. Assoc., [TO VERIFY: vol., pp., year].",
        "[14] [TO VERIFY: authors], “[TO VERIFY: title] (sleep and falls, SWAN cohort),” Innov. Aging, 2024.",
        "[15] T. Mishra et al., “Pre-symptomatic detection of COVID-19 from smartwatch data,” Nat. Biomed. Eng., vol. 4, no. 12, pp. 1208–1220, 2020.",
    ]
    refs2 = [
        "[16] [TO VERIFY: authors], “[TO VERIFY: title] (Oura ring measurements across the menstrual cycle),” Int. J. Women's Health, 2022.",
        "[17] [TO VERIFY: authors], “[TO VERIFY: title] (wearable HRV across the menstrual cycle: living systematic review),” Sports Med., 2026.",
        "[18] [TO VERIFY: initials] Askhatova et al., “[TO VERIFY: title] (temporal CNN fall-risk classification on LTMM),” MethodsX, 2026.",
        "[19] A. Weiss, M. Brozgol, M. Dorfman, T. Herman, S. Shema, N. Giladi, and J. M. Hausdorff, “Does the evaluation of gait quality during daily life provide insight into fall risk? A novel approach using 3-day accelerometer recordings,” Neurorehabil. Neural Repair, vol. 27, no. 8, pp. 742–752, 2013.",
        "[20] J. McCamley, M. Donati, E. Grimpampi, and C. Mazzà, “An enhanced estimate of initial contact and final contact instants of time using lower trunk inertial sensor data,” Gait Posture, vol. 36, no. 2, pp. 316–318, 2012.",
        "[21] R. Moe-Nilssen and J. L. Helbostad, “Estimation of gait cycle characteristics by trunk accelerometry,” J. Biomech., vol. 37, no. 1, pp. 121–126, 2004. [TO VERIFY]",
        "[22] D. Drover, J. Howcroft, J. Kofman, and E. D. Lemaire, “Faller classification in older adults using wearable sensors based on turn and straight-walking accelerometer-based features,” Sensors, vol. 17, no. 6, Art. no. 1321, 2017.",
        "[23] PhysioNet, “Long Term Movement Monitoring Database (LTMM), v1.0.0,” [TO VERIFY: IEEE entry and DOI].",
    ]
    for part, rr in ((1, refs), (2, refs2)):
        S.append({"kind": "refs", "template": 9, "title": f"References ({part}/2)", "refs": rr,
                  "notes": ("IEEE format. [1]–[10] are the Review-I papers, to be pasted. Entries marked [TO VERIFY] must be "
                            "checked against the paper before submission; the volume and page details of the others should "
                            "also be checked once." if part == 1 else
                            "Every reference is cited in the literature table (rows 11–22) or the methods; [23] is the "
                            "dataset used in the real-data validation.")})

    # 21 ── thank you
    S.append({"kind": "thanks", "template": 10, "notes": "Thank the panel; open for questions (see docs/review2/QA_PREP.md)."})

    # 22–23 ── backup
    for part in (1, 2):
        S.append({"kind": "image", "template": 4, "title": f"Backup: Full Architecture ({part}/2)",
                  "images": [{"path": os.path.join(ROOT, "docs", "review2", f"architecture_full_{part}.png"), "box": (0.4, 1.3, 12.53, 5.5)}],
                  "notes": ("Backup: the complete function-level pipeline from docs/ARCHITECTURE.md, rendered left to right and "
                            "split in two. Part 1: request entry, parallel stage 1, safety fusion, selective fetch. "
                            if part == 1 else "Backup part 2: fall-risk engine, grounded generation, strict grounding and "
                            "self-verification, reply and card.")})

    # ── Q&A preparation (same numbers, same ledger) ──
    qa = [
        ("“This already exists.”",
         "Fall detection exists, but it reacts after the fall; health chatbots exist, but they read text only and generate "
         "their own numbers. What is new is the coupling. (1) The ring's fall event can override a calm message: in our "
         f"evaluation only D caught the biometric-only case (emergencies caught: D {V['D']['emerg']}, C {V['C']['emerg']}, "
         f"A {V['A']['emerg']}, B {V['B']['emerg']}). (2) The score is computed by a deterministic engine and every number the "
         f"LLM says is checked (strict grounding D {V['D']['grounding']} vs A {V['A']['grounding']}, B {V['B']['grounding']}). "
         "(3) Cycle-aware personal baselines. The individual building blocks are established methods, which we cite."),
        ("“You used an existing formula.”",
         "Yes, deliberately: the gait features follow Hausdorff 2001, McCamley 2012 and Moe-Nilssen & Helbostad 2004 because "
         "they are validated. Our contribution is at system level: evidence-derived layer weights (λ ∝ ln risk ratio), "
         "cycle-aware baselines, compute-then-explain with a closed grounding loop, and tri-modal safety fusion. The score is "
         "a weighted sum on purpose, so each feature's contribution adds up exactly to the score the user sees."),
        ("“Is it more accurate?”",
         f"On real data it is comparable to the clinical test, not better. Untrained gait index AUC {auc(X1, ['v2', 'index_auc'])} "
         f"(95% CI {ci(X1, ['v2', 'index_ci95'])}, n = {n1}); Timed Up and Go {auc(X1, ['clinical', 'results', 'TUG', 'auc'])} on the "
         f"{tug_n} subjects who have it, where our index gets {auc(X1, ['clinical', 'results', 'TUG', 'v2_index_same_subjects'])}. "
         f"The best trained model (M2) reaches {m2('M2')}. Askhatova 2026 reports {ask} using 3-day data, demographics and a deep "
         "network. Our claim is comparable discrimination with no training on labels and full explainability, not higher accuracy."),
        ("“Some LTMM papers report 0.98.”",
         f"We would need the specific paper to comment [TO VERIFY if the panel names one]. With only {n1} subjects, very high "
         "AUCs usually come from splitting windows or recordings rather than people, so the same person is in training and "
         "test data. We split by subject and fixed the models in advance; the subject-wise reference we cite (Askhatova 2026) "
         f"reports {ask}."),
        ("“Why should we trust it?”",
         f"Because we reported everything, including what failed. The protocols were pre-registered; all "
         f"{len((L.load(X2) or {}).get('models', {}))} Experiment 2 models and all {len((L.load(X3) or {}).get('results', {}))} "
         f"Experiment 3 models are reported. Adding daily-life gait did not help (best D3 {m3('D3')} vs lab M0 "
         f"{m3('M0')} on the same subjects), and we say so. Permutation tests check against chance; the engine is deterministic "
         "and unit-tested; every number in a reply is checked against the data; and we state our limitations up front."),
        ("“Why not test on the ring?”",
         "We did not find a public dataset with ring IMU data and fall labels. LTMM (lower-back IMU, fall history) is the closest "
         "labelled real data, so it validates the signal-processing method, not the ring hardware. Porting the gait pipeline "
         "to the phone or ring IMU is the first Review-III task."),
        ("“What dataset did you train on?”",
         "Nothing in the engine is trained: the layer weights come from published risk ratios and the gait index is label-free. "
         f"The trained comparison models used only LTMM with subject-wise cross-validation ({n1} lab walks; "
         f"{L.num(X3, ['results', 'D0', 'n'], '{}')} subjects with 3-day data). The agent evaluation uses a synthetic cohort of "
         f"{n_users} users × {n_days} days, for testing, not training. The LLM is used as-is, not fine-tuned."),
        ("“Why is latency higher?”",
         f"In the batch evaluation D averaged {V['D']['latency']} s and C {V['C']['latency']} s. The batch timing includes "
         "API rate-limit retries, and D makes more LLM calls: router and safety classifier in parallel, the grounded answer, "
         "and occasionally a verification retry. We do not claim D is faster. On emergencies, D's replies took "
         f"{A['d_emerg_lat']} s."),
        ("“What about authentication and privacy?”",
         "Not implemented yet, and we say so: the API trusts the user id in the request and the backend uses the Supabase "
         "service-role key. Review-III plan: verify the Supabase JWT on every request, take the user id from the token, and "
         "query with the user's own token so row-level security applies. The demo uses synthetic users only."),
    ]
    return S, qa


# ── python-pptx rendering ─────────────────────────────────────────────────────
def render_pptx(slides: list[dict], out_path: str) -> None:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import MSO_ANCHOR
    from pptx.opc.constants import RELATIONSHIP_TYPE as RT
    from pptx.oxml.ns import qn
    from pptx.util import Emu, Inches, Pt
    from PIL import Image

    prs = Presentation(TEMPLATE)
    tmpl = list(prs.slides)
    table_proto = deepcopy(next(sh for sh in tmpl[2].shapes if sh.has_table)._element)
    # The template's notes master has no placeholders; its notes pages carry their own body
    # placeholder, so copy that one into each new notes page.
    notes_proto = deepcopy(tmpl[3].notes_slide.notes_placeholder._element)

    def duplicate(src):
        new = prs.slides.add_slide(src.slide_layout)
        for sh in list(new.shapes):
            sh._element.getparent().remove(sh._element)
        bg = src._element.cSld.find(qn("p:bg"))
        if bg is not None:
            new._element.cSld.insert(0, deepcopy(bg))
        rid_map = {rid: new.part.relate_to(rel.target_part, RT.IMAGE)
                   for rid, rel in src.part.rels.items() if rel.reltype == RT.IMAGE}
        for el in src.shapes._spTree.iterchildren():
            if el.tag in (qn("p:nvGrpSpPr"), qn("p:grpSpPr")):
                continue
            e = deepcopy(el)
            for blip in e.iter(qn("a:blip")):
                if blip.get(qn("r:embed")) in rid_map:
                    blip.set(qn("r:embed"), rid_map[blip.get(qn("r:embed"))])
            new.shapes._spTree.append(e)
        return new

    def by_name(slide, name):
        return next((sh for sh in slide.shapes if sh.name == name), None)

    def set_text(shape, text):
        p = shape.text_frame.paragraphs[0]
        for extra in shape.text_frame.paragraphs[1:]:
            extra._p.getparent().remove(extra._p)
        runs = p.runs
        runs[0].text = text
        for r in runs[1:]:
            r._r.getparent().remove(r._r)

    def fill_body(shape, items, size):
        txBody = shape.text_frame._txBody
        paras = txBody.findall(qn("a:p"))
        proto_p, proto_r = paras[0], paras[0].find(qn("a:r"))
        for p in paras:
            txBody.remove(p)
        for kind, text in items:
            p = deepcopy(proto_p)
            for r in p.findall(qn("a:r")):
                p.remove(r)
            pPr = p.find(qn("a:pPr"))
            if kind == "h":
                for tag in ("a:buChar", "a:buSzPct"):
                    for b in pPr.findall(qn(tag)):
                        pPr.remove(b)
                pPr.set("marL", "0"); pPr.set("indent", "0")
                pPr.append(pPr.makeelement(qn("a:buNone"), {}))
            r = deepcopy(proto_r)
            r.find(qn("a:t")).text = text
            rPr = r.find(qn("a:rPr"))
            rPr.set("sz", str(int(size * 100)))
            if kind == "h":
                rPr.set("b", "1")
                fill = rPr.find(qn("a:solidFill"))
                fill.find(qn("a:srgbClr")).set("val", NAVY)
            end = p.find(qn("a:endParaRPr"))
            (end.addprevious(r) if end is not None else p.append(r))
            txBody.append(p)

    def add_table(slide, spec):
        gf = deepcopy(table_proto)
        tbl = gf.find(".//" + qn("a:tbl"))
        grid = tbl.find(qn("a:tblGrid"))
        gc_proto = deepcopy(grid[0])
        for ext in gc_proto.findall(qn("a:extLst")):
            gc_proto.remove(ext)
        for g in list(grid):
            grid.remove(g)
        for wdt in spec["cols"]:
            g = deepcopy(gc_proto); g.set("w", str(int(Inches(wdt)))); grid.append(g)
        trs = tbl.findall(qn("a:tr"))
        hdr_tc, body_tc = deepcopy(trs[0].find(qn("a:tc"))), deepcopy(trs[1].find(qn("a:tc")))
        for tr in trs:
            tbl.remove(tr)

        def cell(proto, text, size, bold=None):
            tc = deepcopy(proto)
            txBody = tc.find(qn("a:txBody"))
            ps = txBody.findall(qn("a:p"))
            p_proto, r_proto = ps[0], ps[0].find(qn("a:r"))
            for p in ps:
                txBody.remove(p)
            for line in str(text).split("\n"):
                p = deepcopy(p_proto)
                for r in p.findall(qn("a:r")):
                    p.remove(r)
                r = deepcopy(r_proto)
                r.find(qn("a:t")).text = line
                r.find(qn("a:rPr")).set("sz", str(int(size * 100)))
                if bold:
                    r.find(qn("a:rPr")).set("b", "1")
                end = p.find(qn("a:endParaRPr"))
                if end is not None:
                    end.set("sz", str(int(size * 100)))
                (end.addprevious(r) if end is not None else p.append(r))
                txBody.append(p)
            return tc

        size, rh, hh = spec["size"], spec["row_h"], spec.get("hdr_h", 0.45)
        for i, row in enumerate([spec["header"]] + spec["rows"]):
            tr = tbl.makeelement(qn("a:tr"), {"h": str(int(Inches(hh if i == 0 else rh)))})
            for j, txt in enumerate(row):
                tr.append(cell(hdr_tc if i == 0 else body_tc, txt, size, bold=(i > 0 and j == 0 and len(row) > 2 and spec.get("bold_first", True) and not txt.isdigit())))
            tbl.append(tr)
        xfrm = gf.find(qn("p:xfrm"))
        xfrm.find(qn("a:off")).set("x", str(int(Inches(spec["x"])))); xfrm.find(qn("a:off")).set("y", str(int(Inches(spec["y"]))))
        xfrm.find(qn("a:ext")).set("cx", str(int(Inches(sum(spec["cols"])))))
        xfrm.find(qn("a:ext")).set("cy", str(int(Inches(hh + rh * len(spec["rows"])))))
        cNvPr = gf.find(".//" + qn("p:cNvPr"))
        cNvPr.set("id", str(slide.shapes._next_shape_id)); cNvPr.set("name", f"Table {slide.shapes._next_shape_id}")
        slide.shapes._spTree.append(gf)

    def add_box(slide, box, items, size, font="Calibri", fill=None, color="000000"):
        x, y, w, h = box
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.05 if fill is None else 0.12)
        if fill:
            tb.fill.solid(); tb.fill.fore_color.rgb = RGBColor.from_string(fill)
        first = True
        for kind, text in items:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            r = p.add_run(); r.text = text
            r.font.size, r.font.name = Pt(size), font
            r.font.color.rgb = RGBColor.from_string(NAVY if kind == "h" else color)
            r.font.bold = kind == "h"
            p.space_after = Pt(6)
            if kind == "b":
                pPr = p._p.get_or_add_pPr()
                pPr.set("marL", "177800"); pPr.set("indent", "-177800")
                bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
                pPr.append(bu)
        return tb

    def add_image(slide, path, box):
        x, y, w, h = box
        iw, ih = Image.open(path).size
        scale = min(w / iw, h / ih)
        pw, ph = iw * scale, ih * scale
        slide.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))

    for i, spec in enumerate(slides, 1):
        s = duplicate(tmpl[spec["template"] - 1])
        num = by_name(s, "Text 0")
        if num is not None:
            set_text(num, str(i))
        kind = spec["kind"]
        if kind == "title":
            set_text(by_name(s, "Text 3"), spec["project_title"])
        elif kind == "thanks":
            pass
        else:
            set_text(by_name(s, "Text 1"), spec["title"])
        body = by_name(s, "Text 2") if kind in ("bullets",) else None
        if kind == "bullets":
            if spec.get("body"):
                if spec.get("body_box"):
                    x, y, w, h = spec["body_box"]
                    body.left, body.top, body.width, body.height = Inches(x), Inches(y), Inches(w), Inches(h)
                fill_body(body, spec["body"], spec.get("body_size", 16))
            else:
                body._element.getparent().remove(body._element)
        if kind == "refs":
            box = by_name(s, "Text 2")
            fill_body(box, [("b", r) for r in spec["refs"]], 12 if len(spec["refs"]) > 9 else 13)
            hint = by_name(s, "Text 3")
            hint._element.getparent().remove(hint._element)
        if spec["template"] == 3:                         # lit-review slides: drop the template's own table + hint
            for sh in list(s.shapes):
                if sh.has_table or sh.name == "Text 2":
                    sh._element.getparent().remove(sh._element)
        for t in spec.get("tables", []):
            add_table(s, t)
        for im in spec.get("images", []):
            add_image(s, im["path"], im["box"])
        if spec.get("callouts"):
            n = len(spec["callouts"])
            cw, gap, x0 = (12.13 - 0.3 * (n - 1)) / n, 0.3, 0.6
            for k, (head, text) in enumerate(spec["callouts"]):
                add_box(s, (x0 + k * (cw + gap), 5.7, cw, 1.1), [("h", head), ("t", text)], 14, fill="EEF1FA")
        if spec.get("side"):
            add_box(s, spec["side_box"], spec["side"], spec.get("side_size", 14))
        if spec.get("code"):
            c = spec["code"]
            add_box(s, c["box"], [("t", line) for line in c["text"].split("\n")], c["size"], font="Courier New", fill="F2F2F2")
        if spec.get("footnote"):
            x, y, w = spec.get("footnote_at", (0.6, 6.62, 12.1))
            add_box(s, (x, y, w, 0.45), [("t", spec["footnote"])], 11, color=GREY_TXT)
        ns = s.notes_slide
        if ns.notes_text_frame is None:
            ph = deepcopy(notes_proto)
            ph.find(".//" + qn("p:cNvPr")).set("id", str(ns.shapes._next_shape_id))
            ns.shapes._spTree.append(ph)
        ns.notes_text_frame.text = spec.get("notes", "")

    # drop the original template slides (they stay in docs/review2/Review-2_Template_.pptx)
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst)[:len(tmpl)]:
        prs.part.drop_rel(sldId.rId)
        sldIdLst.remove(sldId)
    prs.save(out_path)


# ── markdown ──────────────────────────────────────────────────────────────────
def render_md(slides: list[dict]) -> str:
    out = ["# MedXAI — Review-II slides", "",
           "_Generated by `scripts/build_review2_deck.py` from the results files; do not edit by hand. "
           "Numbers and their sources: [`NUMBERS.md`](NUMBERS.md)._", ""]
    for i, s in enumerate(slides, 1):
        title = {"title": s.get("project_title", "Title"), "thanks": "Thank You"}.get(s["kind"], s.get("title", ""))
        out += [f"## Slide {i} — {title}", ""]
        if s["kind"] == "title":
            out += ["- Student Name(s): Name — Reg. No. _(template placeholder)_",
                    "- Guide: Dr. Guide Name, School _(template placeholder)_",
                    "- Guide's signature with date _(template placeholder)_",
                    "- Programme / School: B.Tech — SENSE, VIT Chennai", "- Review & Date: Review-II · 30.09.2026 (Wednesday)", ""]
        if s["kind"] == "thanks":
            out += ["Thank You · Project-I (2026) · SENSE · VIT Chennai", ""]
        for kind, text in s.get("body", []) + s.get("side", []):
            out.append(f"**{text}**" if kind == "h" else f"- {text}")
        if s.get("body") or s.get("side"):
            out.append("")
        for c in s.get("callouts", []):
            out.append(f"- **{c[0]}:** {c[1]}")
        if s.get("callouts"):
            out.append("")
        for t in s.get("tables", []):
            out.append("| " + " | ".join(t["header"]) + " |")
            out.append("|" + "|".join("---" for _ in t["header"]) + "|")
            out += ["| " + " | ".join(str(c).replace("\n", " ") for c in r) + " |" for r in t["rows"]]
            out.append("")
        for im in s.get("images", []):
            rel = os.path.relpath(im["path"], os.path.join(ROOT, "docs", "review2"))
            out += [f"![{os.path.basename(im['path'])}]({rel})", ""]
        if s.get("code"):
            out += ["```python", s["code"]["text"], "```", ""]
        for r in s.get("refs", []):
            out.append(f"- {r}")
        if s.get("refs"):
            out.append("")
        if s.get("footnote"):
            out += [f"_{s['footnote']}_", ""]
        out += ["**Speaker notes:**", "", s.get("notes", ""), "", "---", ""]
    return "\n".join(out)


def render_qa(qa: list[tuple[str, str]]) -> str:
    out = ["# Review-II — Q&A preparation", "",
           "_Generated by `scripts/build_review2_deck.py`; numbers match the slides (sources in [`NUMBERS.md`](NUMBERS.md))._", ""]
    for q, a in qa:
        out += [f"### {q}", "", a, ""]
    return "\n".join(out)


def render_ledger(L: Ledger) -> str:
    out = ["# Numbers used on the Review-II slides", "",
           "_Generated by `scripts/build_review2_deck.py`: every number shown on a slide or in the speaker notes, "
           "with the file and key it was read from. Duplicates removed._", "",
           "| Value | Source file | Key / how computed |", "|---|---|---|"]
    seen = set()
    for s, f, k, *_ in L.rows:
        if (s, f, k) not in seen:
            seen.add((s, f, k))
            esc = lambda x: str(x).replace("|", "\\|")
            out.append(f"| {esc(s)} | `{esc(f)}` | `{esc(k)}` |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "review2"))
    ap.add_argument("--no-pptx", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    L = Ledger(ROOT)
    slides, qa = build_slides(L)
    with open(os.path.join(a.out, "SLIDES.md"), "w", encoding="utf-8") as f:
        f.write(render_md(slides))
    with open(os.path.join(a.out, "QA_PREP.md"), "w", encoding="utf-8") as f:
        f.write(render_qa(qa))
    with open(os.path.join(a.out, "NUMBERS.md"), "w", encoding="utf-8") as f:
        f.write(render_ledger(L))
    if not a.no_pptx:
        render_pptx(slides, os.path.join(a.out, "Review-II_MedXAI.pptx"))
    print(f"[SAVED] {len(slides)} slides -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
