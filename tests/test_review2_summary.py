"""
Tests for scripts/make_review2_summary.py (offline, no network).
Run:  python -m pytest tests/test_review2_summary.py -q
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.make_review2_summary import FILES, build, main


def _write(dirpath, name, obj):
    with open(os.path.join(dirpath, name), "w", encoding="utf-8") as f:
        json.dump(obj, f)


def test_empty_root_marks_every_file_not_yet_run(tmp_path):
    md = build(str(tmp_path))
    for fname in FILES.values():
        assert f"`{fname}`" in md
    assert md.count("**not yet run**") == len(FILES)
    assert "{{LTMM_DAILY_D1_AUC}}" in md
    # the weight table is the one section allowed to fall back to code constants
    assert "recomputed from `agent/fall_risk.py`" in md
    assert "| Fall history | 2.80 | 1.030 | 0.402 |" in md


def test_values_are_copied_not_invented(tmp_path):
    _write(tmp_path, FILES["exp1"], {
        "dataset": "X", "n": 10, "n_fallers": 4, "n_nonfallers": 6,
        "quality": {"v1_step_cv_median": 1.2345},
        "v1": {"index_auc": 0.61234, "index_ci95": [0.5, 0.7]},
        "v2": {"index_auc": 0.71234, "index_ci95": [0.6, 0.8], "lr_cv_auc": 0.7, "lr_cv_sd": 0.01},
        "clinical": {"results": {}},
    })
    md = build(str(tmp_path))
    assert "| v1 index (no training) | 0.612 | 0.500 – 0.700 | 10 |" in md
    assert "| v2 index (no training) | 0.712 | 0.600 – 0.800 | 10 |" in md
    # missing values render as n/a, not as a number
    assert "| v1 logistic regression (5-fold CV ×10) | n/a | ± n/a | 10 |" in md
    assert "| Timed Up and Go (clinical) | n/a |" in md


def test_daily_results_use_run_ltmm_daily_schema(tmp_path):
    # shape written by run_ltmm_daily.py::main (values here are test fixtures, not results)
    _write(tmp_path, FILES["exp3"], {"protocol": "12 x 30.0 min slices; pre-registered D0-D3 + M0", "results": {
        "D0": {"name": "Daily-life index (untrained)", "n": 71, "auc": 0.66, "ci": [0.5, 0.8], "trained": False},
        "D1": {"name": "Combined [primary]", "n": 70, "auc": 0.7, "ci": [0.55, 0.82], "trained": False, "perm_p": 0.002},
        "D3": {"name": "LogReg: lab + daily (k=6)", "n": 70, "auc": 0.68, "sd": 0.03, "trained": True, "perm_p": 0.04},
    }})
    md = build(str(tmp_path))
    assert "| D0 | Daily-life index (untrained) | 71 | 0.660 | 95% CI 0.500 – 0.800 | — |" in md
    assert "| D1 | Combined [primary] | 70 | 0.700 | 95% CI 0.550 – 0.820 | 0.0020 |" in md
    assert "| D3 | LogReg: lab + daily (k=6) | 70 | 0.680 | ± 0.030 SD over 20×5 CV | 0.0400 |" in md
    assert md.index("| D0 |") < md.index("| D1 |") < md.index("| D3 |")
    assert "20×5 repeated-CV mean, which is the reported estimate" in md
    assert "{{LTMM_DAILY_D1_AUC}}" not in md


def test_m4_permutation_auc_is_explained_as_secondary(tmp_path):
    _write(tmp_path, FILES["exp2"], {"n": 73, "models": {"M4": {
        "name": "LogReg [primary]", "auc": 0.65, "sd": 0.03, "trained": True,
        "permutation": {"auc_single_cv": 0.68, "p_value": 0.03, "null_mean": 0.51, "null_95th": 0.66}}}})
    md = build(str(tmp_path))
    assert "| M4 | LogReg [primary] | 0.650 |" in md
    assert "(0.650) pools out-of-fold predictions" in md and "**the reported estimate** for M4" in md


def test_agent_summary_and_nan_grounding(tmp_path):
    _write(tmp_path, FILES["eval_summary"], {
        "A": {"recall": 0.5, "fpr": 0.0, "grounding": float("nan"), "stale_disclosed": 0.0, "latency": 1.5},
    })
    md = build(str(tmp_path))
    assert "| A — Plain LLM | 0.50 | 0.00 | n/a | 0.00 | 1.50 |" in md


def test_main_writes_markdown(tmp_path):
    out = tmp_path / "docs" / "RESULTS_SUMMARY.md"
    assert main(["--root", str(tmp_path), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# MedXAI Review-II")


def _rec(case, version, expected, detected, **kw):
    return {"case": case, "version": version, "run": 1, "category": kw.pop("category", "x"),
            "expected_emergency": expected, "emergency_detected": detected, **kw}


def test_agent_section_reports_counts_and_data_derived_notes(tmp_path):
    # record shape written by run_fall_evaluation.py::main (fixture values, not results)
    records = [
        _rec(1, "C", True, True), _rec(2, "C", True, False), _rec(3, "C", False, False),
        _rec(4, "C", False, False, category="stale-data", mentions_stale=False),
        _rec(1, "D", True, True, safety_signals={"keyword_triggered": ["i fell"], "llm_triggered": True}),
        _rec(2, "D", True, True, category="biometric-only", safety_signals={"biometric_triggered": True}),
        _rec(3, "D", False, False, streams=["fall_risk"], router_meta={"confidence": 0.93, "widened": False},
             self_verification={"triggered": True, "score_before": 0.5, "score_after": 1.0}),
        _rec(4, "D", False, False, category="stale-data", mentions_stale=True, streams=["fall_risk"],
             router_meta={"confidence": 0.97, "widened": False}, self_verification={"triggered": False}),
    ]
    _write(tmp_path, FILES["eval_results"], records)
    _write(tmp_path, FILES["eval_summary"], {
        "C": {"recall": 0.5, "fpr": 0.0, "grounding": 0.9, "stale_disclosed": 0.0, "latency": 10.0},
        "D": {"recall": 1.0, "fpr": 0.0, "grounding": 1.0, "stale_disclosed": 1.0, "latency": 12.0}})
    md = build(str(tmp_path))
    assert "4 scenarios × 2 versions × 1 run(s) = 8 replies; 2 scenarios are emergencies, 2 are not." in md
    assert "| C — Baseline (fetch-all, keyword safety) | 1/2 (0.50) | 0/2 (0.00) | 0.90 | 0/1 | 10.00 |" in md
    assert "| D — Coupled Routing-Safety | 2/2 (1.00) | 0/2 (0.00) | 1.00 | 1/1 | 12.00 |" in md
    assert "case 1 (x): keyword, LLM; case 2 (biometric-only): biometric." in md
    assert "D self-verification fired on 1 of 4 replies; strict grounding 0.50 → 1.00." in md
    assert "Safety widening fired on no query: router confidence was 0.93–0.97 on all 2 routed queries" in md
    assert "Do not read it as D being faster. On this run D's mean (12.00 s) is higher than C's (10.00 s)." in md


def test_daily_section_says_whether_daily_gait_beat_the_lab_index(tmp_path):
    base = {"M0": {"name": "lab", "n": 60, "auc": 0.66, "ci": [0.5, 0.8], "trained": False},
            "D1": {"name": "comb", "n": 60, "auc": 0.64, "ci": [0.5, 0.8], "trained": False, "perm_p": 0.03},
            "D3": {"name": "lr", "n": 60, "auc": 0.65, "sd": 0.03, "trained": True}}
    _write(tmp_path, FILES["exp3"], {"results": base, "univariate": {"d_cadence": {"auc": 0.6, "med_f": 100, "med_nf": 110}}})
    md = build(str(tmp_path))
    assert "(D3, AUC 0.650) did not improve on the lab-walk index alone (M0, AUC 0.660)" in md
    assert "| d_cadence | 100 | 110 | 0.600 |" in md
    base["D3"]["auc"] = 0.70
    _write(tmp_path, FILES["exp3"], {"results": base})
    assert "(D3, AUC 0.700) scored above the lab-walk index alone" in build(str(tmp_path))
