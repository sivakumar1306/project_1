"""
Gait feature extraction for the Motion layer (signal processing).

Input: tri-axial acceleration (vertical, mediolateral, anteroposterior) in g
from a body-worn IMU during straight walking, sampled at fs Hz.

Pipeline
  1. Trim gait initiation / termination (first and last TRIM_S seconds)
  2. Zero-phase Butterworth band-pass (0.5-15 Hz) to remove gravity drift
     and high-frequency noise
  3. Step detection on the 3 Hz low-passed vertical signal (peak picking)
  4. Features, each with an A PRIORI risk direction taken from the gait
     literature (no labels are used to choose directions):
       cadence            steps/min                     lower  = worse
       step_time_cv       CV of step intervals (%)      higher = worse  (Hausdorff 2001)
       step_regularity    vertical autocorr @ step lag  lower  = worse  (Moe-Nilssen & Helbostad 2004)
       stride_regularity  vertical autocorr @ stride lag lower = worse
       harmonic_ratio_ap  even/odd harmonics, AP axis   lower  = worse  (gait smoothness)
       ml_rms_ratio       ML RMS / vertical RMS         higher = worse  (exploratory; not in index)

The Gait Instability Index is the mean of direction-adjusted z-scores of the
five literature-directed features. It is knowledge-driven: nothing is fitted
to fall labels.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

TRIM_S = 3.0

# feature -> +1 if higher values mean higher risk, -1 if lower values mean higher risk
RISK_DIRECTION = {
    "cadence": -1,
    "step_time_cv": +1,
    "step_regularity": -1,
    "stride_regularity": -1,
    "harmonic_ratio_ap": -1,
}
EXPLORATORY = {"ml_rms_ratio": +1}


def _bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 15.0, order: int = 4) -> np.ndarray:
    hi = min(hi, 0.45 * fs)
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def _lowpass(x: np.ndarray, fs: float, fc: float = 3.0, order: int = 4) -> np.ndarray:
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def _autocorr(x: np.ndarray) -> np.ndarray:
    """Unbiased, normalised autocorrelation (lag 0 = 1)."""
    x = x - x.mean()
    n = len(x)
    full = np.correlate(x, x, mode="full")[n - 1:]
    full = full / (n - np.arange(n))          # unbiased
    return full / full[0] if full[0] else full


def _harmonic_ratio(x: np.ndarray, fs: float, stride_f: float, n_harm: int = 20) -> float:
    """Even/odd harmonic amplitude ratio at multiples of the stride frequency."""
    n = len(x)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft((x - x.mean()) * win))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    even, odd = 0.0, 0.0
    for k in range(1, n_harm + 1):
        f = k * stride_f
        if f >= fs / 2:
            break
        amp = spec[np.argmin(np.abs(freqs - f))]
        if k % 2 == 0:
            even += amp
        else:
            odd += amp
    return even / odd if odd else float("nan")


def extract_gait_features(v: np.ndarray, ml: np.ndarray, ap: np.ndarray, fs: float) -> dict:
    """Return gait features for one walking bout (nan where not computable)."""
    trim = int(TRIM_S * fs)
    if len(v) > 4 * trim + int(10 * fs):
        v, ml, ap = v[trim:-trim], ml[trim:-trim], ap[trim:-trim]

    v_bp, ml_bp, ap_bp = (_bandpass(np.asarray(s, float), fs) for s in (v, ml, ap))

    # step detection on smoothed vertical acceleration
    v_lp = _lowpass(v_bp, fs, 3.0)
    prom = 0.25 * np.std(v_lp)
    peaks, _ = find_peaks(v_lp, distance=int(0.35 * fs), prominence=prom)
    intervals = np.diff(peaks) / fs
    intervals = intervals[(intervals > 0.3) & (intervals < 1.2)]

    out = {k: float("nan") for k in list(RISK_DIRECTION) + list(EXPLORATORY)}
    out["n_steps"] = int(len(intervals))
    if len(intervals) >= 6:
        out["cadence"] = 60.0 / float(np.mean(intervals))
        out["step_time_cv"] = float(np.std(intervals, ddof=1) / np.mean(intervals) * 100)

    # autocorrelation regularity (Moe-Nilssen & Helbostad 2004)
    ac = _autocorr(v_bp)
    lag = lambda s: int(s * fs)
    step_win = ac[lag(0.3):lag(0.9)]
    if len(step_win):
        step_lag = lag(0.3) + int(np.argmax(step_win))
        out["step_regularity"] = float(ac[step_lag])
        s_lo, s_hi = int(1.6 * step_lag), min(int(2.4 * step_lag), len(ac) - 1)
        if s_hi > s_lo:
            stride_lag = s_lo + int(np.argmax(ac[s_lo:s_hi]))
            out["stride_regularity"] = float(ac[stride_lag])
            stride_f = fs / stride_lag
            out["harmonic_ratio_ap"] = float(_harmonic_ratio(ap_bp, fs, stride_f))

    rms_v = float(np.sqrt(np.mean(v_bp ** 2)))
    out["ml_rms_ratio"] = float(np.sqrt(np.mean(ml_bp ** 2)) / rms_v) if rms_v else float("nan")
    return out


def gait_instability_index(rows: list[dict]) -> np.ndarray:
    """
    Knowledge-driven index: mean of direction-adjusted z-scores across subjects.
    z-scores use the pooled sample (label-free); missing features are skipped.
    """
    feats = list(RISK_DIRECTION)
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows], dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd * np.array([RISK_DIRECTION[f] for f in feats])
    return np.nanmean(Z, axis=1)


# ══════════════════════════════════════════════════════════════════════════
# Version 2 (Review-II refinement): signal-quality driven improvements.
# Motivation: v1 gave step-time CV of ~8-9 % in BOTH groups, implausibly high
# for steady walking, which points to measurement error rather than gait.
# Changes are justified on signal-processing grounds, not on fall labels:
#   1. Turn removal: yaw angular velocity marks turns; turning steps are
#      excluded because turns change step timing in everyone.
#   2. Initial-contact detection following McCamley et al. (Gait & Posture
#      2012): integrate vertical acceleration, differentiate with a Gaussian
#      kernel, initial contacts are the minima. Minimum spacing adapts to each
#      person's own step period from autocorrelation.
#   3. Physiological plausibility: intervals outside 0.6-1.5x the segment
#      median are rejected (missed/extra detections).
#   4. Stride-time CV (same foot to same foot) replaces step-time CV. This is
#      the measure used by Hausdorff et al. 2001 and is not inflated by normal
#      left/right step asymmetry.
# Quality metrics are reported so detection can be checked without labels.
# ══════════════════════════════════════════════════════════════════════════

from scipy.ndimage import gaussian_filter1d, binary_dilation

RISK_DIRECTION_V2 = {
    "cadence": -1,
    "stride_time_cv": +1,
    "step_regularity": -1,
    "stride_regularity": -1,
    "harmonic_ratio_ap": -1,
}
TURN_THRESHOLD_DPS = 25.0     # |yaw rate| above this (after 1.5 Hz low-pass) = turning
TURN_MARGIN_S = 1.0           # exclude this much either side of a turn
MIN_SEGMENT_S = 4.0           # shortest straight segment analysed (~3-4 strides)


def straight_segments(yaw: np.ndarray | None, n: int, fs: float) -> list[tuple[int, int]]:
    """Index ranges of straight walking (no turns). Whole signal if no gyro."""
    if yaw is None:
        return [(0, n)]
    y = _lowpass(np.asarray(yaw, float) - np.median(yaw), fs, 1.5)
    turning = np.abs(y) > TURN_THRESHOLD_DPS
    turning = binary_dilation(turning, iterations=int(TURN_MARGIN_S * fs))
    segs, start = [], None
    for i, t in enumerate(turning):
        if not t and start is None:
            start = i
        elif t and start is not None:
            segs.append((start, i)); start = None
    if start is not None:
        segs.append((start, n))
    return [(a, b) for a, b in segs if (b - a) >= MIN_SEGMENT_S * fs]


def _step_period(v_bp: np.ndarray, fs: float) -> float | None:
    ac = _autocorr(v_bp)
    lo, hi = int(0.3 * fs), int(0.9 * fs)
    if len(ac) <= hi:
        return None
    return (lo + int(np.argmax(ac[lo:hi]))) / fs


def detect_initial_contacts(v_bp: np.ndarray, fs: float, step_period: float) -> np.ndarray:
    """McCamley-style IC detection: minima of the Gaussian-differentiated integrated vertical acceleration."""
    integ = np.cumsum(v_bp - v_bp.mean()) / fs
    integ = integ - _lowpass(integ, fs, 0.5) if len(integ) > 3 * fs else integ   # remove drift
    diff = -gaussian_filter1d(integ, sigma=0.05 * fs, order=1)
    peaks, _ = find_peaks(-diff, distance=max(1, int(0.6 * step_period * fs)),
                          prominence=0.3 * np.std(diff))
    return peaks


def extract_gait_features_v2(v, ml, ap, fs: float, yaw=None) -> dict:
    v, ml, ap = (np.asarray(s, float) for s in (v, ml, ap))
    if np.median(v) < 0:          # sensor mounted upside-down: make "up" positive
        v = -v
    n = len(v)
    trim = int(TRIM_S * fs)
    lo_i, hi_i = (trim, n - trim) if n > 4 * trim + int(10 * fs) else (0, n)

    v_bp_all, ml_bp_all, ap_bp_all = (_bandpass(s, fs) for s in (v, ml, ap))
    segs = [(max(a, lo_i), min(b, hi_i)) for a, b in straight_segments(yaw, n, fs)]
    segs = [(a, b) for a, b in segs if (b - a) >= MIN_SEGMENT_S * fs]
    turn_fallback = False
    if not segs and (hi_i - lo_i) >= MIN_SEGMENT_S * fs:
        # no straight stretch long enough: analyse the whole walk (flagged) rather than drop the person
        segs, turn_fallback = [(lo_i, hi_i)], True

    out = {k: float("nan") for k in RISK_DIRECTION_V2}
    out.update({"turn_fallback": turn_fallback})
    out.update({"n_steps": 0, "n_segments": len(segs), "straight_fraction": 0.0,
                "rejected_fraction": float("nan"), "cadence_agreement_pct": float("nan"),
                "ml_rms_ratio": float("nan")})
    if not segs:
        return out
    out["straight_fraction"] = 0.0 if turn_fallback else float(sum(b - a for a, b in segs) / max(1, hi_i - lo_i))

    step_iv, stride_iv, n_cand = [], [], 0
    reg_step, reg_stride, hr, w, ac_periods = [], [], [], [], []
    ml_rms, v_rms = [], []
    for a, b in segs:
        vb, mlb, apb = v_bp_all[a:b], ml_bp_all[a:b], ap_bp_all[a:b]
        sp = _step_period(vb, fs)
        if sp is None:
            continue
        ac_periods.append(sp)
        ics = detect_initial_contacts(vb, fs, sp)
        iv = np.diff(ics) / fs
        n_cand += len(iv)
        if len(iv) >= 4:
            med = np.median(iv)
            ok = (iv > 0.6 * med) & (iv < 1.5 * med)
            step_iv.extend(iv[ok].tolist())
            for i in range(len(iv) - 1):                 # stride = two consecutive valid steps
                if ok[i] and ok[i + 1]:
                    stride_iv.append(iv[i] + iv[i + 1])
        ac = _autocorr(vb)
        sl = int(round(sp * fs))
        s_lo, s_hi = int(1.6 * sl), min(int(2.4 * sl), len(ac) - 1)
        if s_hi > s_lo:
            st = s_lo + int(np.argmax(ac[s_lo:s_hi]))
            reg_step.append(ac[sl]); reg_stride.append(ac[st])
            hr.append(_harmonic_ratio(apb, fs, fs / st)); w.append(b - a)
        ml_rms.append(np.sqrt(np.mean(mlb ** 2))); v_rms.append(np.sqrt(np.mean(vb ** 2)))

    out["n_steps"] = len(step_iv)
    if n_cand:
        out["rejected_fraction"] = float(1 - len(step_iv) / n_cand)
    if len(step_iv) >= 6:
        out["cadence"] = 60.0 / float(np.median(step_iv))
        ac_cad = 60.0 / float(np.average(ac_periods)) if ac_periods else float("nan")
        out["cadence_agreement_pct"] = float(abs(out["cadence"] - ac_cad) / ac_cad * 100) if ac_periods else float("nan")
    if len(stride_iv) >= 4:
        out["stride_time_cv"] = float(np.std(stride_iv, ddof=1) / np.mean(stride_iv) * 100)
    if w:
        out["step_regularity"] = float(np.average(reg_step, weights=w))
        out["stride_regularity"] = float(np.average(reg_stride, weights=w))
        out["harmonic_ratio_ap"] = float(np.average(hr, weights=w))
    if v_rms and sum(v_rms):
        out["ml_rms_ratio"] = float(np.mean(ml_rms) / np.mean(v_rms))
    return out


def gait_instability_index_v2(rows: list[dict]) -> np.ndarray:
    feats = list(RISK_DIRECTION_V2)
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows], dtype=float)
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd * np.array([RISK_DIRECTION_V2[f] for f in feats])
    return np.nanmean(Z, axis=1)


# ══════════════════════════════════════════════════════════════════════════
# Version 3 (Experiment 2): exploit turns instead of discarding them, and add
# interpretable gait descriptors similar to the handcrafted set in
# Askhatova et al., MethodsX 2026 (LTMM, TCN paper).
#
# Turn metrics (literature: turning discriminates fallers better than
# straight walking, e.g. Drover et al., Sensors 2017). A priori directions:
#   turn_duration_s        longer            = worse
#   turn_peak_velocity     slower (deg/s)    = worse
#   steps_per_180          more steps/180 deg = worse
# ══════════════════════════════════════════════════════════════════════════

TURN_MIN_PEAK_DPS = 30.0
TURN_EDGE_DPS = 15.0

RISK_DIRECTION_V3 = {
    **RISK_DIRECTION_V2,
    "turn_duration_s": +1,
    "turn_peak_velocity": -1,
    "steps_per_180": +1,
}


def extract_turn_features(yaw: np.ndarray | None, v: np.ndarray, fs: float) -> dict:
    out = {"turn_count": 0, "turn_duration_s": float("nan"), "turn_peak_velocity": float("nan"),
           "turn_angle_deg": float("nan"), "steps_per_180": float("nan")}
    if yaw is None:
        return out
    y = _lowpass(np.asarray(yaw, float) - np.median(yaw), fs, 1.5)
    ay = np.abs(y)
    above = ay > TURN_EDGE_DPS
    turns, start = [], None
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        elif not a and start is not None:
            turns.append((start, i)); start = None
    if start is not None:
        turns.append((start, len(ay)))
    turns = [(a, b) for a, b in turns if ay[a:b].max() >= TURN_MIN_PEAK_DPS and 0.5 * fs <= (b - a) <= 8 * fs]
    out["turn_count"] = len(turns)
    if not turns:
        return out

    v = np.asarray(v, float)
    if np.median(v) < 0:
        v = -v
    v_bp = _bandpass(v, fs)
    sp = _step_period(v_bp, fs) or 0.55
    ics = detect_initial_contacts(v_bp, fs, sp)

    durs, peaks, angles, spt = [], [], [], []
    for a, b in turns:
        durs.append((b - a) / fs)
        peaks.append(float(ay[a:b].max()))
        ang = float(np.abs(np.sum(y[a:b])) / fs)
        angles.append(ang)
        n_steps = int(np.sum((ics >= a) & (ics < b)))
        if ang >= 45:
            spt.append(n_steps / ang * 180.0)
    out["turn_duration_s"] = float(np.mean(durs))
    out["turn_peak_velocity"] = float(np.mean(peaks))
    out["turn_angle_deg"] = float(np.mean(angles))
    out["steps_per_180"] = float(np.mean(spt)) if spt else float("nan")
    return out


def _spectral_entropy(x: np.ndarray, fs: float, fmax: float = 15.0) -> float:
    f = np.fft.rfftfreq(len(x), 1 / fs)
    p = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) ** 2
    m = (f > 0.3) & (f < fmax)
    p = p[m]
    if p.sum() == 0:
        return float("nan")
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))


def extract_gait_features_v3(v, ml, ap, fs: float, yaw=None) -> dict:
    """v2 features + turn metrics + extra interpretable descriptors (for trained models)."""
    out = extract_gait_features_v2(v, ml, ap, fs, yaw=yaw)
    out.update(extract_turn_features(yaw, v, fs))

    v = np.asarray(v, float)
    if np.median(v) < 0:
        v = -v
    v_bp, ml_bp, ap_bp = (_bandpass(np.asarray(s, float), fs) for s in (v, ml, ap))
    trim = int(TRIM_S * fs)
    if len(v_bp) > 4 * trim + int(10 * fs):
        v_bp, ml_bp, ap_bp = v_bp[trim:-trim], ml_bp[trim:-trim], ap_bp[trim:-trim]

    # step asymmetry (left vs right alternate steps)
    sp = _step_period(v_bp, fs)
    if sp:
        ics = detect_initial_contacts(v_bp, fs, sp)
        iv = np.diff(ics) / fs
        med = np.median(iv) if len(iv) else np.nan
        iv = iv[(iv > 0.6 * med) & (iv < 1.5 * med)] if len(iv) else iv
        if len(iv) >= 8:
            odd, even = iv[0::2], iv[1::2]
            k = min(len(odd), len(even))
            out["step_asymmetry_pct"] = float(np.mean(np.abs(odd[:k] - even[:k])) / np.mean(iv) * 100)
    out.setdefault("step_asymmetry_pct", float("nan"))

    jerk = lambda s: float(np.sqrt(np.mean((np.diff(s) * fs) ** 2)))
    rms = lambda s: float(np.sqrt(np.mean(s ** 2)))
    out["jerk_v_norm"] = jerk(v_bp) / rms(v_bp) if rms(v_bp) else float("nan")
    out["jerk_ap_norm"] = jerk(ap_bp) / rms(ap_bp) if rms(ap_bp) else float("nan")
    out["spectral_entropy_v"] = _spectral_entropy(v_bp, fs)
    out["spectral_entropy_ml"] = _spectral_entropy(ml_bp, fs)
    out["corr_v_ap"] = float(np.corrcoef(v_bp, ap_bp)[0, 1])
    out["rms_v_g"] = rms(v_bp)
    return out


def gait_instability_index_v3(rows: list[dict]) -> np.ndarray:
    feats = list(RISK_DIRECTION_V3)
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows], dtype=float)
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd * np.array([RISK_DIRECTION_V3[f] for f in feats])
    with np.errstate(all="ignore"):
        return np.nanmean(Z, axis=1)
