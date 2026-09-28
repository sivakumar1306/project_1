"""
Personal correlation detection (Review-II)
==========================================

Finds relationships that hold for THIS user in their own ring history, e.g.
"on days after shorter sleep, your HRV is lower". Results feed the fall-risk
explanation as grounded, user-specific context.

Method: Pearson r over paired daily values (optionally lagged), significance
via scipy's exact test when available, otherwise the Fisher z approximation.
Only pairs with n >= MIN_N, |r| >= MIN_ABS_R and p < ALPHA are reported.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any

MIN_N = 14
MIN_ABS_R = 0.40
ALPHA = 0.05

# (x feature, y feature, lag_days, sentence if r>0, sentence if r<0)
# lag = 1 means x on day d is paired with y on day d+1.
PAIRS = [
    ("sleep_min", "hrv", 0,
     "on nights with shorter sleep, your HRV the next morning tends to be lower",
     "on nights with shorter sleep, your HRV the next morning tends to be higher"),
    ("stress", "rhr", 0,
     "on higher-stress days, your resting heart rate tends to be higher",
     "on higher-stress days, your resting heart rate tends to be lower"),
    ("stress", "sleep_score", 1,
     "higher stress tends to be followed by a better sleep score the next night",
     "higher stress tends to be followed by a worse sleep score the next night"),
    ("steps", "sleep_score", 1,
     "more active days tend to be followed by a better sleep score",
     "more active days tend to be followed by a worse sleep score"),
    ("sleep_min", "steps", 0,
     "after longer sleep you tend to be more active during the day",
     "after longer sleep you tend to be less active during the day"),
]


def _to_date(k: Any):
    if isinstance(k, date):
        return k
    try:
        return datetime.strptime(str(k)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _pearson(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(xs, ys)
        return float(r), float(p)
    except Exception:
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        if sxx == 0 or syy == 0:
            return 0.0, 1.0
        r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)
        r = max(-0.999999, min(0.999999, r))
        z = math.atanh(r) * math.sqrt(max(n - 3, 1))
        p = math.erfc(abs(z) / math.sqrt(2))
        return r, p


def compute_correlations(daily: dict, window_days: int = 45) -> list[dict]:
    """
    daily: {"YYYY-MM-DD": {"sleep_min":.., "hrv":.., ...}}
    Returns significant correlations sorted by |r| (strongest first).
    """
    by_date = {}
    for k, v in (daily or {}).items():
        d = _to_date(k)
        if d:
            by_date[d] = v
    if not by_date:
        return []
    end = max(by_date)
    start = end - timedelta(days=window_days)

    found = []
    for x_key, y_key, lag, pos_sentence, neg_sentence in PAIRS:
        xs, ys = [], []
        for d, row in by_date.items():
            if d < start:
                continue
            y_row = by_date.get(d + timedelta(days=lag))
            if y_row is None:
                continue
            x, y = row.get(x_key), y_row.get(y_key)
            if x is None or y is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
        if len(xs) < MIN_N:
            continue
        if len(set(xs)) < 3 or len(set(ys)) < 3:
            continue
        r, p = _pearson(xs, ys)
        if abs(r) >= MIN_ABS_R and p < ALPHA:
            sentence = pos_sentence if r > 0 else neg_sentence
            found.append({"x": x_key, "y": y_key, "lag": lag, "r": r, "p": p,
                          "n": len(xs), "sentence": sentence})
    found.sort(key=lambda c: abs(c["r"]), reverse=True)
    return found
