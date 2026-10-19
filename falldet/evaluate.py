"""Clip-level evaluation: sensitivity, false alarms, latency, per-scenario rates."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .pipeline import Split

CONSECUTIVE = 2          # windows above threshold needed to raise an alert
POST_FALL_GRACE = 3.0    # seconds after impact in which an alert still counts
REFRACTORY = 3.0         # seconds between distinct alert events


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def alert_times(p, t_end, tau):
    flag = p >= tau
    flag[1:] &= p[:-1] >= tau
    flag[0] = False
    events, last = [], -1e9
    for t, f in zip(t_end, flag):
        if f and t - last > REFRACTORY:
            events.append(float(t))
        if f:
            last = t
    return events


def clip_outcomes(split: Split, p: np.ndarray, tau: float):
    rows = []
    for ci, meta in enumerate(split.clips):
        m = split.clip_id == ci
        ev = alert_times(p[m].copy(), split.t_end[m], tau)
        if meta["label"]:
            lo, hi = meta["t_lob"] - 0.1, meta["fall_t1"] + POST_FALL_GRACE
            early = [t for t in ev if t < lo]
            hit = [t for t in ev if lo <= t <= hi]
            rows.append(dict(ci=ci, scenario=meta["scenario"], label=1, detected=bool(hit),
                             false_alerts=len(early) + len([t for t in ev if t > hi]),
                             latency=(hit[0] - meta["fall_t0"]) if hit else np.nan,
                             duration=meta["duration"]))
        else:
            rows.append(dict(ci=ci, scenario=meta["scenario"], label=0, detected=False,
                             false_alerts=len(ev), latency=np.nan, duration=meta["duration"]))
    return rows


def summarise(split: Split, p: np.ndarray, tau: float) -> dict:
    rows = clip_outcomes(split, p, tau)
    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    tp = sum(r["detected"] for r in pos)
    fa_clips = sum(r["false_alerts"] > 0 for r in neg)
    fa_events = sum(r["false_alerts"] for r in rows)
    hours = sum(r["duration"] for r in rows) / 3600
    lat = [r["latency"] for r in pos if r["detected"]]
    out = dict(
        domain=split.domain, n_clips=len(rows), n_fall=len(pos), n_nonfall=len(neg),
        sensitivity=tp / max(1, len(pos)), sens_ci=wilson(tp, len(pos)),
        fa_clip_rate=fa_clips / max(1, len(neg)), fa_ci=wilson(fa_clips, len(neg)),
        false_alerts_per_hour=fa_events / hours,
        median_latency_s=float(np.median(lat)) if lat else float("nan"),
        window_auc=float(roc_auc_score(split.y, p)) if len(set(split.y)) > 1 and p.std() > 0 else float("nan"),
        window_ap=float(average_precision_score(split.y, p)),
    )
    by = {}
    for r in rows:
        d = by.setdefault(r["scenario"], dict(n=0, alert=0))
        d["n"] += 1
        d["alert"] += int(r["detected"] if r["label"] else r["false_alerts"] > 0)
    out["per_scenario"] = {k: v["alert"] / v["n"] for k, v in by.items()}
    return out


def pick_threshold(split: Split, p: np.ndarray, max_fa_clip_rate: float = 0.03) -> float:
    """Smallest threshold whose clip-level false-alarm rate on `split` is <= target."""
    for tau in np.round(np.arange(0.05, 1.0, 0.01), 2):
        rows = clip_outcomes(split, p, float(tau))
        neg = [r for r in rows if not r["label"]]
        if sum(r["false_alerts"] > 0 for r in neg) / max(1, len(neg)) <= max_fa_clip_rate:
            return float(tau)
    return 0.99
