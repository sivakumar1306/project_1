# Review-II: Fall Prediction in MedXAI

## What was added

| Contribution | Where |
|---|---|
| Deterministic fall-risk engine: personal baselines (median/MAD, 28 days), trend-deviation z-scores, 4 layers, evidence-derived weights (λ ∝ ln risk ratio), coverage-based renormalisation, additive attribution, coverage score Q | `agent/fall_risk.py` |
| Cycle-aware baselines: phase-matched baseline for HRV, resting HR, temperature (literature offsets as fallback) | `agent/fall_risk.py` |
| Personal correlation detection (Pearson, n ≥ 14, \|r\| ≥ 0.4, p < 0.05) | `agent/correlations.py` |
| Router confidence scoring + safety-asymmetric widening, new `fall_risk` stream | `agent/router.py` |
| Tri-modal safety fusion: keyword + LLM + biometric fall event; guarded fall phrases ("I fell asleep" is not an emergency) | `agent/tools.py` |
| Closed-loop self-verification: strict grounding vs PATIENT DATA, one corrective regeneration | `agent/graph.py` |
| Fall-risk chat card (`type: "fall_risk"`), fall prompt rules | `agent/graph.py` |
| Fall tables | `schema_fall.sql` |
| Demo cohort (5 users × 35 days) | `scripts/fall_demo_data.py`, `seed_fall_demo.py` |
| Offline engine analysis + charts | `run_fall_engine_analysis.py` |
| Live A–D evaluation + charts | `run_fall_evaluation.py` |
| Tests (offline) | `tests/` |
| Results summary, architecture, methods | `scripts/make_review2_summary.py`, `docs/` |

Also fixed: `agent/tools.py` used `timedelta` without importing it, so the cycle "estimated next period" was always "unknown".

## Run order

1. Supabase SQL editor: run `schema_fall.sql` once.
2. `python seed_fall_demo.py` — seeds the 5 demo users. **Re-run on the morning of the review**, because data is generated relative to today.
3. `python -m pytest tests -q` — all tests should pass (34 at the time of writing; no network needed).
4. `python run_fall_engine_analysis.py` — offline charts: evidence weights, 21-day trends, cycle ablation, attribution.
5. `python run_fall_evaluation.py` — live A–D evaluation (uses Groq). Add `--runs 3` if quota allows.
   Then `python scripts/make_review2_summary.py` to regenerate `docs/RESULTS_SUMMARY.md` from whatever results files exist.
6. Right before the live demo: `python seed_fall_demo.py --fall-now` (the biometric signal only looks back 30 minutes).

## Demo users

| User | id | Expected |
|---|---|---|
| Arun, low risk | `11111111-1111-4111-8111-000000000001` | LOW |
| Meera, luteal + poor recovery, 1 past fall | `...0002` | MODERATE, cycle-aware baseline applied |
| Ravi, stale data (4 days) | `...0003` | LOW, flagged STALE, lower confidence |
| Lakshmi, 3 falls, 2 near-falls, recent fall | `...0004` | HIGH; biometric emergency after `--fall-now` |
| Priya, healthy luteal | `...0005` | LOW, correction lowers physiology strain |

## Known limitations (say these before the panel does)

- Motion layer uses daily activity deviation + near-fall events; true gait variability from raw IMU is Review-III.
- Evidence ratios come from different studies (OR/RR mixed, not mutually adjusted); weights are an evidence-informed approximation to be learned from data in Review-III.
- Results use a synthetic cohort; the engine is validated per layer, not end-to-end on real falls.
- Resting HR uses the daily minimum HR as a proxy.
- The Flutter chat needs a renderer for the new `fall_risk` card type (Review-III UI integration).

## Real-data validation: PhysioNet LTMM (Motion layer)

`py -m pip install wfdb openpyxl` then `py run_ltmm_validation.py`.
Downloads the 73 one-minute lab walks (~5 MB, lower-back IMU, 100 Hz) plus the clinical spreadsheet into `data/ltmm/`.
Extracts gait features (`agent/gait_features.py`: cadence, step-time CV, step/stride regularity, harmonic ratio), combines them into a Gait Instability Index with literature-fixed directions and **no training on labels**, and reports AUC for fallers (≥2 falls/yr) vs non-fallers, compared with a cross-validated logistic regression and the clinical Timed Up and Go test.
Caveats: retrospective labels (discrimination, not prospective prediction); lower-back sensor, not a ring; n ≈ 73.

### v2 detector (signal-quality refinement)
v1 produced step-time CV of ~8-9 % in both groups, implausibly high for steady walking. v2 adds turn removal (yaw gyroscope), McCamley-style initial-contact detection with a person-specific step period, physiological plausibility checks, and stride-time CV (Hausdorff 2001) instead of step-time CV. On a synthetic ground-truth bench with turns and left/right asymmetry (`py scripts/gait_synthetic_bench.py`), v1 overestimated variability by 6.8 points (r = 0.61 with truth) and v2 by 0.5 points (r ≈ 0.97–0.98 (varies slightly with numpy version)). `run_ltmm_validation.py` reports v1 and v2 side by side; both are reported regardless of which scores higher.

## Live demo page

A browser page with the fall-risk score and the AI assistant, for the five synthetic demo users.
Backend: `routers/fall_risk.py`, `routers/demo.py`; page: `static/demo.html` (no build step, no external scripts).

**Prerequisites**
1. `schema_fall.sql` has been run once in the Supabase SQL editor.
2. `py seed_fall_demo.py` has been run **the same day** (the data is generated relative to today).
3. `.env` contains `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` and `GROQ_API_KEY`. Without an LLM key the score cards still work and ring/keyword emergencies are still detected, but ordinary chat answers show a friendly error.

**Start (Windows PowerShell, from the repo folder)**
```powershell
py -m pip install -r requirements.txt
py -m uvicorn main:app --reload
```
Open http://127.0.0.1:8000/demo

**Demo script**
1. **Meera** (selected by default): point at the score, tier, the four layers with their weights, *Why this score* and the cycle note. Click **"Why is my fall risk high?"**: the answer only uses the listed contributors; open *How this answer was produced* to show the streams, grounding score and self-verification.
2. Still Meera: click **"I fell asleep on the couch last night"**: no alarm.
3. Switch to **Lakshmi**: click **Simulate ring fall event** (logs one uncancelled fall, valid for 30 minutes; nothing is sent to anyone), then **"I'm fine, just a bit shaken"**: the reply is an EMERGENCY, and the pipeline panel shows the *Ring fall event* signal (in the evaluation, the LLM classifier alone did not flag this message).
4. Switch to **Ravi**: the yellow banner says the ring data is days old and the estimate is less reliable.

The demo endpoints (`/api/v1/demo/*`) only accept the five demo user ids; any other id gets 403.
