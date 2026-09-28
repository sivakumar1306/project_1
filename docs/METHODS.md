# MedXAI Review-II — Methods (as implemented)

Every formula below is transcribed from the code; the file and function are given for each block.
Constants are quoted with their names so they can be checked with `grep`. Results are in
[`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md); the pipeline is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

Notation: *k* = feature, *L* = layer ∈ {P, RC, MO, FH}, *x_k* = current value, *d₀* = the evaluation
date ("today").

---

## 1. Deterministic fall-risk engine — `agent/fall_risk.py`

### 1.1 Inputs — `assemble_history`, `fetch_fall_history`

| Feature *k* | Source | Layer | Within-layer weight *w_k* | Risk direction | `min_scale` |
|---|---|---|---|---|---|
| `hrv` | `user_hrv.avg_hrv` | P | 0.35 | low | 3.0 ms |
| `rhr` | `user_hr.min_hr` (fallback `avg_hr`) — daily minimum as resting-HR proxy | P | 0.25 | high | 1.5 bpm |
| `spo2` | `user_spo2.avg_spo2` | P | 0.25 | low | 1.0 % |
| `temp` | daily mean of `user_temp.value_c` | P | 0.15 | high | 0.1 °C |
| `sleep_min` | `user_sleep.total_duration` (values > 1440 treated as seconds, ÷ 60) | RC | 0.40 | low | 20 min |
| `sleep_score` | `user_sleep.sleep_score` | RC | 0.30 | low | 3 points |
| `stress` | daily mean of `user_stress.stress_value` | RC | 0.30 | high | 3 |
| `steps` | `user_steps.steps` | MO | 0.50 | low | 500 steps |
| near-falls | `user_fall_events` | MO | `NEAR_FALL_WEIGHT` = 0.50 | — | — |
| falls | `user_fall_events` | FH | — | — | — |

`fetch_fall_history` loads 45 days of daily rows, up to 6 cycle logs and 366 days of fall events.
Within-layer weights sum to 1.0 in every layer (P: 0.35+0.25+0.25+0.15; RC: 0.40+0.30+0.30;
MO: 0.50+0.50).

### 1.2 Personal baseline (median / MAD) — `compute_fall_risk`, `_robust_baseline`

For each feature:

1. Current value *x_k* = most recent non-null value on a date *d_k* ≤ *d₀*. If none, or
   *d₀* − *d_k* > `MISSING_AFTER_DAYS` = 7 days, the feature is **missing**.
2. History *H_k* = values on dates in [*d_k* − 28, *d_k*) (`BASELINE_DAYS` = 28; the window is anchored
   on the feature's own latest date, not on *d₀*). If |*H_k*| < `MIN_BASELINE_POINTS` = 7 the feature
   is **missing** ("insufficient baseline").
3. m_k = median(H_k), MAD_k = median(|h − m_k|), s_k = max(1.4826 · MAD_k, min_scale_k).

### 1.3 Cycle-aware baseline — `cycle_phase`, `_cycle_eligible`, `compute_fall_risk`

Applied only if the profile `gender` starts with "f" **and** cycle logs exist, and only to
`PHYSIO_CYCLE_FEATURES` = (`hrv`, `rhr`, `temp`).

Phase of a date *d* (`cycle_phase`): take the latest logged `period_start` ≤ *d* (dates before the first
log are projected backwards in steps of that cycle's length). cycle day = (*d* − start) + 1; if
cycle day > cycle_length + 10 the phase is unknown. With ov = max(cycle_length − 14, period_length + 2):

| Phase | Condition |
|---|---|
| menstrual | day ≤ period_length |
| follicular | day < ov − 1 |
| ovulatory | ov − 1 ≤ day ≤ ov + 1 |
| luteal | otherwise |

Baseline selection when eligible and today's phase is known:

- **phase-matched:** H_k ← {h ∈ H_k : phase(date(h)) = phase(*d₀*)} if at least `MIN_PHASE_POINTS` = 5
  such days exist;
- **luteal offset** (only if today is luteal and fewer than 5 same-phase days): m_k is computed from the
  full H_k and then shifted by `LUTEAL_OFFSETS`: temp m + 0.3 °C, rhr m + 2.0 bpm, hrv m × (1 − 0.07);
- otherwise the ordinary personal baseline.

`cycle_phase` for the comparison is evaluated at *d₀* even when *d_k* < *d₀* (see Limitations).

### 1.4 Deviation and dead-zone strain mapping — `compute_fall_risk`, `_strain_from_z`

z_k = (x_k − m_k) / s_k,  z_risk = −z_k if direction is "low", else z_k

strain_k = 100 · clip( (z_risk − 0.5) / (3.0 − 0.5), 0, 1 )

with `DEAD_ZONE_Z` = 0.5 (|z| below this is day-to-day noise) and `FULL_STRAIN_Z` = 3.0.
SpO₂ absolute soft threshold: if x_spo2 < 95, strain = max(strain, min(100, (95 − x) / 5 · 100)).

### 1.5 Event components — `compute_fall_risk`

Event age is measured from the end of *d₀* (23:59:59.999999 UTC); events in the future are ignored.

- falls₃₆₅ = #{`event_type` = "fall", not `user_cancelled`, age ≤ 365 d}
- N₁₄ = #{(`event_type` = "near_fall" **or** cancelled), age ≤ 14 d} (a cancelled fall counts as a near-fall)
- near-fall strain = min(100, 50 · N₁₄)
- Fall-history risk R_FH = 0 if falls₃₆₅ = 0, 60 if = 1, 100 if ≥ 2

### 1.6 Layer aggregation — `compute_fall_risk`

For L ∈ {P, RC, MO}, with present features *k* ∈ L:

- present weight p_L = Σ w_k (+ 0.50 for MO: the near-fall component is always "present")
- total weight W_L = Σ over all features of L (+ 0.50 for MO) — equal to 1.0 for every layer
- coverage c_L = p_L / W_L
- R_L = [Σ w_k · strain_k (+ 0.50 · near-fall strain for MO)] / p_L, and 0 if p_L = 0

For FH: R_FH as above, c_FH = 1.

### 1.7 Evidence weights and adaptive renormalisation — `EVIDENCE_RATIOS`, `BASE_LAYER_WEIGHTS`

λ_L = ln(ratio_L) / Σ_j ln(ratio_j), ratios FH 2.8, MO 2.1, P 1.73, RC 1.27.

Adaptive (coverage-adjusted) weight: λ′_L = λ_L · c_L / Σ_j λ_j · c_j (0 if the denominator is 0).

### 1.8 Fall Risk Score and tier — `compute_fall_risk`, `tier_for`

FRS_raw = Σ_L λ′_L · R_L,  FRS = round(FRS_raw) (Python `round`, i.e. half-to-even)

Tier (`TIERS`): LOW if FRS < 40, MODERATE if FRS < 70, HIGH otherwise.

### 1.9 Coverage / confidence Q — `compute_fall_risk`

Q = 100 · Σ_L λ_L · c_L (base λ, not λ′). Let *a* = *d₀* − (latest date among present features).

- no present feature → Q = 0
- *a* > 1 → Q ← Q − min(40, 10 · (*a* − 1))
- Q ← clip(Q, 0, 100), reported rounded; `low_confidence` = Q < 60
- `stale` = no present feature, or *a* > `STALE_AFTER_DAYS` = 2

### 1.10 Additive attribution — `compute_fall_risk`

points_k = λ′_L · (w_k / p_L) · strain_k for every present feature;
near-falls: λ′_MO · (0.50 / p_MO) · near-fall strain; fall history: λ′_FH · R_FH.

Σ points = FRS_raw exactly (checked by `tests/test_fall_risk.py::test_contributions_sum_to_score`).
The context and card show contributors with ≥ 0.5 points (top 4 in the LLM context, top 3 on the card).

### 1.11 Trend, cycle counterfactual, caching — `get_fall_risk`, `compute_fall_risk_series`

- 7-day trend = round(mean FRS over the 7 days ending *d₀* − 1), each day computed only from data up
  to that day.
- For cycle-eligible users the engine recomputes without correction and reports
  `frs_without_correction` and `physiology_risk_without_correction`.
- Results are cached per (user, date) for 60 s and each fresh computation is appended to
  `user_fall_risk` (`_log_result`).

### 1.12 Personal correlations — `agent/correlations.py::compute_correlations`

Pairs (x → y, lag in days): sleep_min → hrv (0), stress → rhr (0), stress → sleep_score (1),
steps → sleep_score (1), sleep_min → steps (0). Window: 45 days before the latest date. Pearson r
(`scipy.stats.pearsonr`; fallback Fisher-z p = erfc(|atanh(r)·√(n−3)| / √2)). Reported only if
n ≥ 14, both series have ≥ 3 distinct values, |r| ≥ 0.40 and p < 0.05; sorted by |r|; top 3 go into
the LLM context.

---

## 2. Gait signal processing — `agent/gait_features.py`

Input: vertical (V), mediolateral (ML), anteroposterior (AP) acceleration in g, optional yaw angular
velocity (°/s), sampling rate *f_s* (LTMM: 100 Hz).

### 2.1 Common filters

- Band-pass: 4th-order Butterworth 0.5–min(15, 0.45 f_s) Hz, zero-phase (`filtfilt`) — `_bandpass`.
- Low-pass: 4th-order Butterworth, zero-phase — `_lowpass`.
- Autocorrelation: unbiased (divided by N − lag) and normalised to lag 0 — `_autocorr`.
- Harmonic ratio: Hann-windowed FFT magnitude of the de-meaned AP signal; for k = 1…20 harmonics of
  the stride frequency (below Nyquist) take the nearest-bin amplitude; HR = Σ even / Σ odd —
  `_harmonic_ratio`. (Computed on the whole segment, not averaged per stride.)
- Trimming: first and last `TRIM_S` = 3 s removed only if the record is longer than 4·3 s + 10 s.

### 2.2 v1 — `extract_gait_features`, `gait_instability_index`

- Steps: peaks of the 3 Hz low-passed band-passed V, min distance 0.35 s, prominence 0.25·SD; step
  intervals kept if 0.3–1.2 s. With ≥ 6 intervals: cadence = 60 / mean, step-time CV =
  SD(ddof=1) / mean · 100.
- Regularity: step lag = argmax of the V autocorrelation in [0.3, 0.9) s; stride lag = argmax in
  [1.6, 2.4) × step lag; step / stride regularity = autocorrelation at those lags.
- HR_AP at stride frequency f_s / stride lag. ML/V RMS ratio (exploratory, not in the index).

### 2.3 v2 pipeline — `extract_gait_features_v2`, `straight_segments`, `_step_period`, `detect_initial_contacts`

1. **Orientation:** if median(V) < 0 the V axis is flipped.
2. **Turn removal** (`straight_segments`): yaw − median(yaw), low-pass 1.5 Hz; |yaw| > 25 °/s
   (`TURN_THRESHOLD_DPS`) = turning; the turning mask is dilated by 1.0 s (`TURN_MARGIN_S`) on each side;
   straight segments ≥ 4 s (`MIN_SEGMENT_S`) are kept and intersected with the trim window. If none
   remain but the trimmed walk is ≥ 4 s, the whole walk is analysed and `turn_fallback` is set.
3. **Person-specific step period** (`_step_period`): sp = argmax of the segment's V autocorrelation in
   [0.3, 0.9) s.
4. **Initial contacts, McCamley-style** (`detect_initial_contacts`): I(t) = Σ (V_bp − mean) / f_s;
   if the segment is > 3 s, I ← I − lowpass_0.5 Hz(I) (drift removal); D = −(Gaussian first
   derivative of I, σ = 0.05 s); ICs = local minima of D, found as peaks of −D with minimum distance
   max(1, 0.6·sp·f_s) samples and prominence 0.3·SD(D).
5. **Plausibility:** with ≥ 4 step intervals in a segment, an interval is valid if
   0.6·median < Δ < 1.5·median. `rejected_fraction` = 1 − valid / candidate intervals.
6. **Stride-time CV:** stride = sum of two consecutive valid step intervals (same foot to same foot);
   with ≥ 4 strides, CV = SD(ddof=1) / mean · 100 (Hausdorff 2001).
7. **Cadence** = 60 / median(valid step intervals) (needs ≥ 6); `cadence_agreement_pct` =
   |cadence − 60/mean(sp)| / (60/mean(sp)) · 100.
8. **Regularity:** step lag = round(sp·f_s), stride lag = argmax in [1.6, 2.4) × step lag; step /
   stride regularity and HR_AP (at f_s / stride lag) are averaged over segments weighted by segment
   length.

### 2.4 v3 additions — `extract_turn_features`, `extract_gait_features_v3`

- Turns: contiguous |yaw| > 15 °/s (`TURN_EDGE_DPS`) with peak ≥ 30 °/s (`TURN_MIN_PEAK_DPS`) and
  duration 0.5–8 s. Per walk: mean turn duration, mean peak velocity, mean angle (|Σ yaw| / f_s) and
  steps per 180° (ICs inside the turn / angle · 180, turns ≥ 45° only).
- Descriptors: step asymmetry (mean |odd − even| valid step intervals / mean · 100, ≥ 8 intervals),
  normalised jerk (RMS(ΔV·f_s) / RMS(V), same for AP), spectral entropy of V and ML over 0.3–15 Hz
  normalised by ln(#bins), corr(V, AP), RMS(V).

### 2.5 Gait Instability Index (label-free) — `gait_instability_index[_v2|_v3]`

Each feature is z-scored over the pooled sample (NaN-aware mean and SD, ddof = 0; SD 0 → 1),
multiplied by its a priori risk direction (`RISK_DIRECTION`, `RISK_DIRECTION_V2`, `RISK_DIRECTION_V3`:
cadence −, variability +, regularity −, HR_AP −, turn duration +, turn peak velocity −, steps/180° +),
and averaged over the available features. No fall labels are used.

### 2.6 Daily-life walking detection (Experiment 3) — `agent/daily_gait.py` *(not in repository)*

`agent/daily_gait.py` and `run_ltmm_daily.py` are not committed, so their thresholds cannot be
transcribed yet. Placeholders to fill from the code once it is pushed:

| Quantity | Value | Where |
|---|---|---|
| Walking-bout activity threshold | `{{DAILY_GAIT_ACTIVITY_THRESHOLD}}` | `agent/daily_gait.py::{{FUNCTION}}` |
| Minimum bout duration | `{{DAILY_GAIT_MIN_BOUT_S}}` | `agent/daily_gait.py::{{FUNCTION}}` |
| Step-frequency band for walking | `{{DAILY_GAIT_STEP_FREQ_BAND}}` | `agent/daily_gait.py::{{FUNCTION}}` |
| Regularity threshold for walking | `{{DAILY_GAIT_REGULARITY_MIN}}` | `agent/daily_gait.py::{{FUNCTION}}` |
| Per-subject aggregation of bouts | `{{DAILY_GAIT_AGGREGATION}}` | `agent/daily_gait.py::{{FUNCTION}}` |

---

## 3. Statistical evaluation — `run_ltmm_validation.py`, `run_ltmm_experiment2.py`

- Labels: LTMM subject prefix `FL` = faller (≥ 2 falls in the previous year, retrospective), `CO` =
  non-faller; one lab walk per subject (73 records listed in `CONTROLS` + `FALLERS`).
- AUC: `sklearn.metrics.roc_auc_score` after dropping NaN scores (`auc`); per-feature AUC uses the a
  priori direction; Mann–Whitney U p-values.
- 95 % CI: 2000 bootstrap resamples of subjects (seed 7), percentile 2.5 / 97.5 (`bootstrap_ci`).
- Exp 1 trained comparator: median-impute → standardise → logistic regression, stratified 5-fold CV
  repeated 10× (seeds 0–9), mean ± SD of AUC (`data_driven_auc`).
- Clinical comparators: TUG (higher = worse) and Berg (lower = worse) from
  `ClinicalDemogData_COFL.xlsx` (`load_clinical`).
- Exp 2 (pre-registered): 7 fixed models M0–M6; logistic regression C = 0.5, balanced class weights,
  `SelectKBest(f_classif, k=8)` inside the pipeline for M3/M4; random forest 300 trees, depth 3,
  min leaf 5, balanced; 20 × stratified 5-fold CV (seed 2026 + r), AUC of pooled out-of-fold
  predictions per repeat, mean ± SD. Permutation test for M4: 200 label shuffles with a single
  stratified 5-fold CV (`permutation_test_score`, mean per-fold ROC AUC).

---

## 4. Agent-side rules — `agent/router.py`, `agent/tools.py`, `agent/graph.py`

- Safety widening: `CONFIDENCE_THRESHOLD` = 0.75, `SAFETY_STREAMS` = fall_risk, current_hr, spo2, hrv
  (`apply_safety_widening`).
- Fusion thresholds: LLM triggers at confidence ≥ 0.7; LLM overrides a keyword hit only when it says
  "not an emergency" with confidence ≥ 0.85; biometric window 30 min (`check_emergency_fused`,
  `get_recent_fall_event`).
- Strict grounding score = (#numbers in the reply found in PATIENT DATA with digit boundaries) /
  (#numbers checked), allow-list {"112"}; 1.0 when no numbers (`compute_grounding_score_strict`).
  One corrective regeneration, accepted only if it parses and does not lower the score.

---

## 5. Limitations

1. **Retrospective labels.** LTMM faller status is self-reported fall history over the previous year,
   so the LTMM experiments measure discrimination, not prospective prediction.
2. **Sensor placement.** LTMM uses a lower-back IMU. The results validate the signal-processing
   method, not a finger-worn ring, whose motion signal is dominated by hand movement.
3. **Small sample.** n = 73 subjects (35 fallers, 38 non-fallers; 69 with TUG/Berg), so AUC confidence
   intervals are wide (see `RESULTS_SUMMARY.md` §4–5) and differences between models of a few
   hundredths are within noise.
4. **Synthetic demo cohort.** The engine scores, cycle ablation and A–D agent evaluation use five
   generated users (`scripts/fall_demo_data.py`) whose couplings (e.g. luteal temp +0.35 °C,
   rhr +3 bpm, HRV −10 %) were written by us. They show the engine behaves as designed; they are not
   evidence of clinical validity. The engine has not been validated end-to-end on real falls.
5. **Approximate evidence weights.** The four ratios come from different studies, mix odds ratios and
   a risk ratio, and are not mutually adjusted; λ ∝ ln(ratio) is an evidence-informed prior, not a
   fitted model. The within-layer weights *w_k*, dead-zone (0.5), full-strain z (3.0), FH mapping
   (0/60/100), near-fall strain (50 per event) and tier cut-offs (40/70) are design choices.
6. **Motion layer in the backend is not IMU gait.** The live Motion layer uses daily step-count
   deviation plus near-fall events. The validated gait features (`agent/gait_features.py`) are not
   wired into `compute_fall_risk`, yet the MO weight is derived from the gait-problems odds ratio.
7. **Resting HR proxy.** Resting HR is the daily minimum HR (`min_hr`), not a measured resting value.
8. **Coverage Q is optimistic.** FH coverage is always 1 and MO coverage is at least 0.5 (near-fall
   component always counts as observed), so before the staleness penalty Q ≥ 100 · (λ_FH + 0.5 · λ_MO)
   ≈ 55 whenever any feature is present, and "no near-fall events recorded" is treated as observed
   zero risk.
9. **Cycle phase at d₀.** For a stale value (*d_k* < *d₀*) the phase-matched baseline uses today's
   phase rather than the phase on the day the value was measured.
10. **Grounding metric scope.** Strict grounding checks numbers only; it cannot detect a wrong
    qualitative claim, and on general-knowledge questions (no streams fetched) it flags every
    legitimate number, which triggers the correction loop.
11. **Dates in UTC.** Engine "today" is the UTC date; for IST users the day boundary is 5 h 30 min off.
