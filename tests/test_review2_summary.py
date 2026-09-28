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
