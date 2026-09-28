"""
Review-II results summary.

Reads whichever results files exist in the repo root and writes
docs/RESULTS_SUMMARY.md (also printed to stdout):

  fall_engine_analysis.json        run_fall_engine_analysis.py
  ltmm_results.json                run_ltmm_validation.py      (Exp 1)
  ltmm_experiment2_results.json    run_ltmm_experiment2.py     (Exp 2)
  ltmm_daily_results.json          run_ltmm_daily.py           (Exp 3)
  fall_evaluation_summary.json     run_fall_evaluation.py
  fall_evaluation_results.json     run_fall_evaluation.py

Every number in the output is read from one of those files (or, for the
evidence-weight table only, recomputed from agent/fall_risk.py when the
analysis file is missing). A missing file produces a "not yet run" row,
never an estimate.

Run:  python scripts/make_review2_summary.py [--root .] [--out docs/RESULTS_SUMMARY.md]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "engine": "fall_engine_analysis.json",
    "exp1": "ltmm_results.json",
    "exp2": "ltmm_experiment2_results.json",
    "exp3": "ltmm_daily_results.json",
    "eval_summary": "fall_evaluation_summary.json",
    "eval_results": "fall_evaluation_results.json",
}
PRODUCERS = {
    "engine": "run_fall_engine_analysis.py",
    "exp1": "run_ltmm_validation.py",
    "exp2": "run_ltmm_experiment2.py",
    "exp3": "run_ltmm_daily.py",
    "eval_summary": "run_fall_evaluation.py",
    "eval_results": "run_fall_evaluation.py",
}
NOT_RUN = "not yet run"


# ── helpers ────────────────────────────────────────────────────────────────
def load(root: str, key: str):
    path = os.path.join(root, FILES[key])
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)          # Python's json accepts the NaN the scripts may write


def num(v, nd: int = 3) -> str:
    """Format a number read from a results file; never invents a value."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        return f"{v:.{nd}f}"
    return str(v)


def ci(v, nd: int = 3) -> str:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f"{num(v[0], nd)} – {num(v[1], nd)}"
    return "n/a"


def table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def not_run_row(key: str, width: int) -> list:
    return [f"_{NOT_RUN}_ (`{FILES[key]}` missing; run `{PRODUCERS[key]}`)"] + [""] * (width - 1)


def _first(d: dict, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return None


# ── sections ───────────────────────────────────────────────────────────────
def section_weights(engine) -> list[str]:
    out = ["## 1. Evidence-derived layer weights", "",
           "λ_L = ln(ratio_L) / Σ ln(ratio). Source: `agent/fall_risk.py` (`EVIDENCE_RATIOS`, `BASE_LAYER_WEIGHTS`).", ""]
    headers = ["Layer", "Published ratio", "ln(ratio)", "Weight λ"]
    rows = []
    if engine and engine.get("weights"):
        for layer, w in engine["weights"].items():
            rows.append([layer, num(w.get("ratio"), 2), num(w.get("ln"), 3), num(w.get("weight"), 3)])
        out += table(headers, rows) + ["", f"Read from `{FILES['engine']}`.", ""]
        return out
    # Fallback: the weights are constants in the code, so they can be recomputed exactly.
    try:
        sys.path.insert(0, ROOT)
        from agent.fall_risk import BASE_LAYER_WEIGHTS, EVIDENCE_RATIOS, LAYER_NAMES
        for k in ("FH", "MO", "P", "RC"):
            rows.append([LAYER_NAMES[k], num(float(EVIDENCE_RATIOS[k]), 2),
                         num(math.log(EVIDENCE_RATIOS[k]), 3), num(BASE_LAYER_WEIGHTS[k], 3)])
        out += table(headers, rows) + ["", f"`{FILES['engine']}` missing: recomputed from `agent/fall_risk.py` constants.", ""]
    except Exception as e:  # pragma: no cover - only when the package can't be imported
        out += table(headers, [not_run_row("engine", 4)]) + ["", f"(could not import agent.fall_risk: {e})", ""]
    return out


def section_cohort(engine) -> list[str]:
    out = ["## 2. Demo cohort scores (synthetic, deterministic engine)", ""]
    headers = ["User", "FRS", "Tier", "Coverage Q", "Stale", "Cycle phase", "FRS without cycle correction",
               "Physiology risk (with → without correction)"]
    if not engine or not engine.get("users"):
        return out + table(headers, [not_run_row("engine", len(headers))]) + [""]
    rows = []
    for name, u in engine["users"].items():
        cyc = u.get("cycle") or {}
        phys_with = (u.get("layers") or {}).get("Physiology")
        if cyc.get("applied") and "frs_without_correction" in cyc:
            cyc_frs = num(cyc.get("frs_without_correction"))
            phys = f"{num(phys_with, 1)} → {num(cyc.get('physiology_risk_without_correction'), 1)}"
        else:
            cyc_frs, phys = "—", num(phys_with, 1)
        rows.append([name, num(u.get("frs")), u.get("tier", "n/a"), num(u.get("coverage_q")),
                     num(u.get("stale")), cyc.get("phase", "—") if cyc else "—", cyc_frs, phys])
    out += table(headers, rows)
    out += ["", f"Engine run date in file: `{engine.get('date', 'n/a')}`. The cohort is generated relative to the run date, "
            "so re-running on another day can change these values.", ""]

    # layer breakdown + top contributors
    headers2 = ["User", "Physiology", "Recovery", "Motion", "Fall history", "Top contributors (points)"]
    rows2 = []
    for name, u in engine["users"].items():
        L = u.get("layers") or {}
        tops = ", ".join(f"{c.get('label')} +{num(c.get('points'), 1)}" for c in (u.get("top_contributors") or [])
                         if isinstance(c.get("points"), (int, float)) and c.get("points") > 0)
        rows2.append([name, num(L.get("Physiology"), 1), num(L.get("Recovery"), 1), num(L.get("Motion"), 1),
                      num(L.get("Fall history"), 1), tops or "—"])
    out += ["### Layer risk (0–100) and top contributors", ""] + table(headers2, rows2) + [""]
    return out


def section_ablation(engine) -> list[str]:
    out = ["## 3. Cycle-aware baseline ablation (normal luteal days)", ""]
    headers = ["User", "Normal luteal days", "Mean physiology risk without", "Mean physiology risk with",
               "Reduction %", "Days flagged without", "Days flagged with", "Acute days (with correction)"]
    abl = (engine or {}).get("cycle_ablation")
    if not abl:
        return out + table(headers, [not_run_row("engine", len(headers))]) + [""]
    rows = []
    thr = None
    for name, a in abl.items():
        f_with = next((v for k, v in a.items() if k.startswith("days_flagged_ge_") and k.endswith("_with")), None)
        f_without = next((v for k, v in a.items() if k.startswith("days_flagged_ge_") and k.endswith("_without")), None)
        for k in a:
            if k.startswith("days_flagged_ge_"):
                thr = k[len("days_flagged_ge_"):].split("_")[0]
        acute = a.get("acute_days_with_correction") or []
        rows.append([name, num(a.get("normal_luteal_days")), num(a.get("mean_physio_risk_without_correction"), 1),
                     num(a.get("mean_physio_risk_with_correction"), 1), num(a.get("reduction_pct"), 1),
                     num(f_without), num(f_with), ", ".join(str(x) for x in acute) or "—"])
    out += table(headers, rows)
    if thr:
        out += ["", f"A day is 'flagged' when physiology-layer risk ≥ {thr} (threshold from `run_fall_engine_analysis.py`)."]
    out += [""]
    return out


def section_exp1(d) -> list[str]:
    out = ["## 4. LTMM Experiment 1 — untrained Gait Instability Index (lab walks)", ""]
    headers = ["Measure", "AUC", "95% CI / spread", "n"]
    if not d:
        return out + table(headers, [not_run_row("exp1", 4)]) + [""]
    n = d.get("n")
    v1, v2 = d.get("v1") or {}, d.get("v2") or {}
    rows = [
        ["v1 index (no training)", num(v1.get("index_auc")), ci(v1.get("index_ci95")), num(n)],
        ["v2 index (no training)", num(v2.get("index_auc")), ci(v2.get("index_ci95")), num(n)],
        ["v1 logistic regression (5-fold CV ×10)", num(v1.get("lr_cv_auc")), f"± {num(v1.get('lr_cv_sd'))}", num(n)],
        ["v2 logistic regression (5-fold CV ×10)", num(v2.get("lr_cv_auc")), f"± {num(v2.get('lr_cv_sd'))}", num(n)],
    ]
    clin = ((d.get("clinical") or {}).get("results")) or {}
    names = {"TUG": "Timed Up and Go (clinical)", "BERG": "Berg Balance Scale (clinical)"}
    for test in ("TUG", "BERG"):
        c = clin.get(test)
        if c:
            rows.append([names[test], num(c.get("auc")),
                         f"v1 index {num(c.get('v1_index_same_subjects'))}, v2 index {num(c.get('v2_index_same_subjects'))} on same subjects",
                         num(c.get("n"))])
        else:
            rows.append([names[test], "n/a", "not matched in clinical spreadsheet", "—"])
    out += [f"Dataset: {d.get('dataset', 'n/a')}; n = {num(n)} ({num(d.get('n_fallers'))} fallers, "
            f"{num(d.get('n_nonfallers'))} non-fallers). Labels are retrospective (≥ 2 falls in the past year).", ""]
    out += table(headers, rows) + [""]

    q = d.get("quality") or {}
    out += ["### Signal-quality check (label-free)", ""]
    out += table(["Quality metric", "Median"], [
        ["v1 step-time CV (%)", num(q.get("v1_step_cv_median"), 2)],
        ["v2 stride-time CV (%)", num(q.get("v2_stride_cv_median"), 2)],
        ["v2 straight-walking fraction", num(q.get("v2_straight_fraction_median"), 2)],
        ["v2 rejected-interval fraction", num(q.get("v2_rejected_fraction_median"), 3)],
        ["v2 cadence agreement, steps vs autocorrelation (% difference)", num(q.get("v2_cadence_agreement_pct_median"), 2)],
    ]) + [""]

    for ver in ("v1", "v2"):
        pf = (d.get(ver) or {}).get("per_feature") or {}
        if not pf:
            continue
        rows = [[f, "higher" if r.get("direction", 0) > 0 else "lower", num(r.get("median_fallers")),
                 num(r.get("median_nonfallers")), num(r.get("auc")), num(r.get("p"), 4)] for f, r in pf.items()]
        out += [f"### {ver} per-feature AUC (a priori risk direction)", ""]
        out += table(["Feature", "Worse when", "Median fallers", "Median non-fallers", "AUC", "Mann–Whitney p"], rows) + [""]
    return out


def section_exp2(d) -> list[str]:
    out = ["## 5. LTMM Experiment 2 — pre-registered models (all reported)", ""]
    headers = ["Model", "Description", "AUC", "Spread", "Trained"]
    if not d:
        return out + table(headers, [not_run_row("exp2", len(headers))]) + [""]
    models = d.get("models") or {}
    rows = []
    for key in sorted(models):
        m = models[key]
        spread = f"95% CI {ci(m.get('ci95'))}" if not m.get("trained") else f"± {num(m.get('sd'))} SD over 20×5 CV"
        rows.append([key, m.get("name", ""), num(m.get("auc")), spread, num(m.get("trained"))])
    out += [f"Protocol: {d.get('protocol', 'n/a')}; n = {num(d.get('n'))}.", ""]
    out += table(headers, rows) + [""]
    for key, m in sorted(models.items()):
        p = m.get("permutation")
        if p:
            out += [f"**{key} permutation test:** AUC (single 5-fold CV) {num(p.get('auc_single_cv'))} vs shuffled-label "
                    f"null mean {num(p.get('null_mean'))} (95th percentile {num(p.get('null_95th'))}), "
                    f"p = {num(p.get('p_value'), 4)}.", "",
                    f"Why two AUCs for {key}: the permutation test (`permutation_test_score`) refits the model on "
                    "one stratified 5-fold split and averages the per-fold ROC AUC, so its "
                    f"{num(p.get('auc_single_cv'))} depends on that single split. The table value "
                    f"({num(m.get('auc'))}) pools out-of-fold predictions and averages over 20 repeated 5-fold "
                    "splits; it is less split-dependent and is **the reported estimate** for "
                    f"{key}. The permutation test is used only to ask whether {key} beats chance.", ""]
    sel = d.get("m4_selection_frequency") or {}
    if sel:
        top = list(sel.items())[:8]
        out += ["M4 most frequently selected features (share of CV folds): "
                + ", ".join(f"{f} {num(v, 2)}" for f, v in top), ""]
    return out


def section_exp3(d) -> list[str]:
    out = ["## 6. LTMM Experiment 3 — daily-life walking (3-day recordings)", ""]
    headers = ["Model", "Description", "n", "AUC", "Spread", "Permutation p"]
    if not d:
        out += table(headers, [not_run_row("exp3", len(headers))])
        out += ["", "Placeholders for the slide until the run finishes: "
                "`{{LTMM_DAILY_N}}`, `{{LTMM_DAILY_D1_AUC}}`, `{{LTMM_DAILY_PERM_P}}`.", ""]
        return out
    # run_ltmm_daily.py writes {"protocol", "results": {M0, D0..D3: {name, n, auc, ci | sd, trained, perm_p}}, ...}
    models = d.get("results") if isinstance(d.get("results"), dict) else None
    if not models:
        out += ["`ltmm_daily_results.json` found but it has no `results` object; top-level scalar values:", ""]
        rows = [[k, num(v)] for k, v in d.items() if isinstance(v, (int, float, str, bool))]
        return out + table(["Key", "Value"], rows or [["(none)", ""]]) + [""]
    order = [k for k in ("M0", "D0", "D1", "D2", "D3") if k in models] + sorted(set(models) - {"M0", "D0", "D1", "D2", "D3"})
    rows = []
    for key in order:
        m = models[key]
        spread = f"95% CI {ci(m.get('ci'))}" if not m.get("trained") else f"± {num(m.get('sd'))} SD over 20×5 CV"
        rows.append([key, m.get("name", ""), num(m.get("n")), num(m.get("auc")), spread,
                     num(m["perm_p"], 4) if "perm_p" in m else "—"])
    out += [f"Protocol: {d.get('protocol', 'n/a')}.", ""] + table(headers, rows) + [""]
    notes = []
    if "perm_p" in models.get("D1", {}):
        notes.append("D1 permutation p: the untrained index is re-scored against shuffled labels, so it tests the "
                     "same AUC shown in the table.")
    if "perm_p" in models.get("D3", {}):
        notes.append("D3 permutation p comes from a single stratified 5-fold split (`permutation_test_score`); "
                     "the D3 AUC in the table is the 20×5 repeated-CV mean, which is the reported estimate.")
    m0, same = models.get("M0"), [k for k in ("D1", "D3") if k in models]
    if m0 and same and all(models[k].get("n") == m0.get("n") for k in same):
        best = max(same, key=lambda k: models[k].get("auc", float("-inf")))
        verdict = "did not improve on" if models[best]["auc"] <= m0["auc"] else "scored above"
        notes.insert(0, f"On the same {num(m0.get('n'))} subjects, the best model that adds daily-life gait ({best}, AUC "
                        f"{num(models[best]['auc'])}) {verdict} the lab-walk index alone (M0, AUC {num(m0['auc'])}). "
                        "All pre-registered models are reported above.")
    out += [f"- {n}" for n in notes] + ([""] if notes else [])
    uni = d.get("univariate") or {}
    if uni:
        rows = [[f, num(u.get("med_f")), num(u.get("med_nf")), num(u.get("auc"))] for f, u in uni.items()]
        out += ["### Daily-life per-feature AUC (a priori risk direction)", ""]
        out += table(["Feature", "Median fallers", "Median non-fallers", "AUC"], rows) + [""]
    return out


def _count(rs, pred) -> str:
    return f"{sum(1 for r in rs if pred(r))}/{len(rs)}"


def section_agent(summary, records) -> list[str]:
    out = ["## 7. Agent Versions A–D on fall scenarios (live, `run_fall_evaluation.py`)", ""]
    labels = {"A": "A — Plain LLM", "B": "B — Plain RAG", "C": "C — Baseline (fetch-all, keyword safety)",
              "D": "D — Coupled Routing-Safety"}
    headers = ["Version", "Emergency recall", "False-positive rate", "Strict grounding", "Stale disclosure", "Mean latency (s)"]
    if not summary:
        out += table(headers, [not_run_row("eval_summary", len(headers))]) + [""]
    if not records:
        if summary:
            out += table(headers, [[labels.get(v, v), num(s.get("recall"), 2), num(s.get("fpr"), 2),
                                    num(s.get("grounding"), 2), num(s.get("stale_disclosed"), 2),
                                    num(s.get("latency"), 2)] for v, s in summary.items()]) + [""]
        return out + [f"Per-reply records: _{NOT_RUN}_ (`{FILES['eval_results']}` missing).", ""]

    versions = sorted({r.get("version") for r in records})
    cases = sorted({r.get("case") for r in records})
    runs = sorted({r.get("run") for r in records})
    n_em = len({r["case"] for r in records if r.get("expected_emergency")})
    out += [f"{len(cases)} scenarios × {len(versions)} versions × {len(runs)} run(s) = {len(records)} replies; "
            f"{n_em} scenarios are emergencies, {len(cases) - n_em} are not. With one run per scenario, "
            "each rate below is a count out of a handful of replies, so counts are shown next to rates.", ""]
    rows = []
    for v in versions:
        rv = [r for r in records if r.get("version") == v]
        em = [r for r in rv if r.get("expected_emergency")]
        ne = [r for r in rv if not r.get("expected_emergency")]
        st = [r for r in rv if r.get("category") == "stale-data"]
        s = (summary or {}).get(v, {})
        rows.append([labels.get(v, v),
                     f"{_count(em, lambda r: r.get('emergency_detected'))} ({num(s.get('recall'), 2)})",
                     f"{_count(ne, lambda r: r.get('emergency_detected'))} ({num(s.get('fpr'), 2)})",
                     num(s.get("grounding"), 2),
                     _count(st, lambda r: r.get("mentions_stale")),
                     num(s.get("latency"), 2)])
    out += table(["Version", "Emergencies caught", "False alarms", "Strict grounding (personal queries)",
                  "Stale data disclosed", "Mean latency (s)"], rows) + [""]
    out += ["Strict grounding and latency are read from `fall_evaluation_summary.json`; counts are computed from "
            "`fall_evaluation_results.json`.", ""]

    d = [r for r in records if r.get("version") == "D"]
    notes = []
    if d:
        per_case = []
        for r in sorted(d, key=lambda r: r["case"]):
            if not r.get("expected_emergency"):
                continue
            sig = r.get("safety_signals") or {}
            fired = [name for name, key in (("keyword", "keyword_triggered"), ("LLM", "llm_triggered"),
                                            ("biometric", "biometric_triggered")) if sig.get(key)]
            per_case.append(f"case {r['case']} ({r.get('category')}): {', '.join(fired) or 'none'}")
        if per_case:
            notes.append("D safety signals that fired on the emergency scenarios: " + "; ".join(per_case) + ".")
        sv = [r.get("self_verification") or {} for r in d]
        fired = [s for s in sv if s.get("triggered")]
        notes.append(f"D self-verification fired on {len(fired)} of {len(d)} replies"
                     + (f"; strict grounding {num(sum(s['score_before'] for s in fired) / len(fired), 2)} → "
                        f"{num(sum(s['score_after'] for s in fired) / len(fired), 2)}" if fired else "") + ".")
        confs = [(r.get("router_meta") or {}).get("confidence") for r in d]
        confs = [c for c in confs if isinstance(c, (int, float))]
        widened = sorted({r["case"] for r in d if (r.get("router_meta") or {}).get("widened")})
        if confs:
            notes.append(f"Safety widening fired on {'case ids ' + str(widened) if widened else 'no query'}: router "
                         f"confidence was {num(min(confs), 2)}–{num(max(confs), 2)} on all {len(confs)} routed queries, "
                         "above the 0.75 widening threshold. Widening is covered by unit tests only.")
        non_em = [len(r.get("streams") or []) for r in d if not r.get("emergency_detected")]
        if non_em:
            notes.append(f"D fetched {num(sum(non_em) / len(non_em), 2)} data streams per non-emergency reply on average "
                         "(C always fetches everything).")
    lat = {v: (summary or {}).get(v, {}).get("latency") for v in ("C", "D")}
    cmp_ = ""
    if all(isinstance(x, (int, float)) for x in lat.values()):
        cmp_ = f" On this run D's mean ({num(lat['D'], 2)} s) is {'higher' if lat['D'] > lat['C'] else 'not higher'} than C's ({num(lat['C'], 2)} s)."
    notes.append("Latency is from a batch run: it includes API rate-limit retries and, for D, the extra router, safety "
                 "and self-verification LLM calls. Do not read it as D being faster." + cmp_)
    return out + [f"- {n}" for n in notes] + [""]


# ── main ───────────────────────────────────────────────────────────────────
def build(root: str) -> str:
    data = {k: load(root, k) for k in FILES}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = ["# MedXAI Review-II — Results Summary", "",
             f"_Generated by `scripts/make_review2_summary.py` on {stamp}. Do not edit by hand; re-run the script._", "",
             "Every number below is read from a results file in the repository. Missing files are shown as "
             f"“{NOT_RUN}”, never estimated.", "", "### Results files", ""]
    lines += table(["File", "Status", "Produced by"],
                   [[f"`{FILES[k]}`", "present" if data[k] is not None else f"**{NOT_RUN}**", f"`{PRODUCERS[k]}`"]
                    for k in FILES]) + [""]
    lines += section_weights(data["engine"])
    lines += section_cohort(data["engine"])
    lines += section_ablation(data["engine"])
    lines += section_exp1(data["exp1"])
    lines += section_exp2(data["exp2"])
    lines += section_exp3(data["exp3"])
    lines += section_agent(data["eval_summary"], data["eval_results"])
    lines += ["## Caveats", "",
              "- Sections 2, 3 and 7 use the synthetic 5-user demo cohort (`scripts/fall_demo_data.py`), not real patients.",
              "- LTMM labels are retrospective fall history: discrimination, not prospective prediction. Sensor is lower-back, not a ring.",
              "- Section 7 uses live LLM calls; values vary between runs.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=ROOT, help="directory containing the results JSON files")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "RESULTS_SUMMARY.md"))
    a = ap.parse_args(argv)
    md = build(a.root)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"\n[SAVED] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
