"""
Daily-life gait from free-living recordings (lower-back IMU, 100 Hz).

Walking detection is label-free and fixed in advance:
  - 5 s windows, no overlap
  - band-passed vertical RMS >= 0.06 g            (the person is moving)
  - vertical autocorrelation peak >= 0.40 at a lag of 0.35-0.80 s (rhythmic stepping)
  - dominant vertical frequency in 1.3-2.6 Hz      (step frequency, ~80-155 steps/min)
  - a bout = >= 4 consecutive walking windows (>= 20 s); bouts are analysed up
    to 120 s so very long walks do not dominate
Per bout: the same v2 gait features used in the lab (turn removal, McCamley
initial contacts, stride-time CV, regularity, harmonic ratio).
Per subject: median of each feature over bouts (>= 3 bouts required), plus
amount of walking and typical bout length.

A priori risk directions (set before looking at labels):
  cadence -, stride_time_cv +, step_regularity -, stride_regularity -,
  harmonic_ratio_ap -, walking_min_per_hour - (less walking = worse),
  median_bout_s - (shorter bouts = worse)
"""

from __future__ import annotations

import numpy as np

from agent.gait_features import _autocorr, _bandpass, extract_gait_features_v2

WIN_S = 5.0
MIN_RMS_G = 0.06
MIN_AC = 0.40
STEP_F_LO, STEP_F_HI = 1.3, 2.6
MIN_BOUT_WINDOWS = 4
MAX_BOUT_S = 120.0
MIN_BOUTS = 3

DAILY_DIRECTION = {
    "d_cadence": -1,
    "d_stride_time_cv": +1,
    "d_step_regularity": -1,
    "d_stride_regularity": -1,
    "d_harmonic_ratio_ap": -1,
    "d_walking_min_per_hour": -1,
    "d_median_bout_s": -1,
}
_FEAT_MAP = {"cadence": "d_cadence", "stride_time_cv": "d_stride_time_cv", "step_regularity": "d_step_regularity",
             "stride_regularity": "d_stride_regularity", "harmonic_ratio_ap": "d_harmonic_ratio_ap"}


def walking_windows(v: np.ndarray, fs: float) -> np.ndarray:
    """Boolean per 5-s window: is this window walking?"""
    v = np.asarray(v, float)
    if np.median(v) < 0:
        v = -v
    w = int(WIN_S * fs)
    n_win = len(v) // w
    if n_win == 0:
        return np.zeros(0, bool)
    v_bp = _bandpass(v, fs)
    out = np.zeros(n_win, bool)
    lo, hi = int(0.35 * fs), int(0.80 * fs)
    freqs = np.fft.rfftfreq(w, 1 / fs)
    band = (freqs >= 0.5) & (freqs <= 4.0)
    for i in range(n_win):
        seg = v_bp[i * w:(i + 1) * w]
        if np.sqrt(np.mean(seg ** 2)) < MIN_RMS_G:
            continue
        ac = _autocorr(seg)
        if ac[lo:hi].max() < MIN_AC:
            continue
        spec = np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(w)))
        f_dom = freqs[band][np.argmax(spec[band])]
        if STEP_F_LO <= f_dom <= STEP_F_HI:
            out[i] = True
    return out


def detect_bouts(walk: np.ndarray, fs: float) -> list[tuple[int, int]]:
    """Sample index ranges of walking bouts (>= MIN_BOUT_WINDOWS consecutive windows)."""
    w = int(WIN_S * fs)
    bouts, start = [], None
    for i, flag in enumerate(list(walk) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= MIN_BOUT_WINDOWS:
                bouts.append((start * w, i * w))
            start = None
    return bouts


def daily_features_for_chunks(chunks: list[dict], fs: float) -> dict:
    """
    chunks: list of {"v","ml","ap","yaw"} arrays (each a sampled slice of the recording).
    Returns per-subject daily-life features (NaN if < MIN_BOUTS bouts).
    """
    per_bout = {k: [] for k in _FEAT_MAP}
    bout_lengths, walk_s, sampled_s = [], 0.0, 0.0
    for c in chunks:
        v = np.asarray(c["v"], float)
        sampled_s += len(v) / fs
        walk = walking_windows(v, fs)
        walk_s += float(walk.sum()) * WIN_S
        for a, b in detect_bouts(walk, fs):
            bout_lengths.append((b - a) / fs)
            b = min(b, a + int(MAX_BOUT_S * fs))
            yaw = c.get("yaw")
            f = extract_gait_features_v2(v[a:b], np.asarray(c["ml"])[a:b], np.asarray(c["ap"])[a:b], fs,
                                         yaw=None if yaw is None else np.asarray(yaw)[a:b])
            for k in _FEAT_MAP:
                if not np.isnan(f.get(k, np.nan)):
                    per_bout[k].append(f[k])

    out = {k: float("nan") for k in DAILY_DIRECTION}
    out.update({"d_n_bouts": len(bout_lengths), "d_sampled_hours": sampled_s / 3600, "d_walking_minutes": walk_s / 60})
    if sampled_s > 0:
        out["d_walking_min_per_hour"] = (walk_s / 60) / (sampled_s / 3600)
    if len(bout_lengths) >= MIN_BOUTS:
        out["d_median_bout_s"] = float(np.median(bout_lengths))
        for k, dk in _FEAT_MAP.items():
            if len(per_bout[k]) >= MIN_BOUTS:
                out[dk] = float(np.median(per_bout[k]))
    return out


def daily_index(rows: list[dict], directions: dict = DAILY_DIRECTION) -> np.ndarray:
    feats = list(directions)
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows], dtype=float)
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd * np.array([directions[f] for f in feats])
    with np.errstate(all="ignore"):
        return np.nanmean(Z, axis=1)
