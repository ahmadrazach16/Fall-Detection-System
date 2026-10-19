"""Causal, scale-normalised window features from 2D keypoints.

Calibration uses the first second of a track (person walking / standing
upright) to obtain a body scale and a floor line, as a tracker would on entry.
All features are then computed from *past* frames only, so the same code runs
in a streaming setting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CALIB_SECONDS, STRIDE_SECONDS, WINDOW_SECONDS
from .simulate import JI

FEATURE_NAMES = [
    "hip_h_last", "hip_h_min", "hip_h_drop", "hip_h_ratio",
    "head_h_last", "head_h_min", "head_h_ratio",
    "ang_first", "ang_last", "ang_max", "ang_mean", "ang_delta",
    "asp_last", "asp_max", "asp_mean",
    "vy_hip_min", "vy_hip_mean_last", "vy_head_min", "ang_rate_max",
    "speed_mean", "speed_max", "speed_last", "t_since_vmin",
    "drop_x_speed", "head_below_hip_frac", "low_hip_frac",
]


@dataclass
class Calib:
    scale: float
    floor_y: float
    hip_h0: float
    head_h0: float


def calibrate(kp: np.ndarray, fps: float, seconds: float = CALIB_SECONDS) -> Calib:
    m = max(3, int(seconds * fps))
    k = kp[:m]
    hip = k[:, JI["mid_hip"]]
    neck = k[:, JI["neck"]]
    head = k[:, JI["head"]]
    scale = float(np.median(np.linalg.norm(neck - hip, axis=1)))
    scale = max(scale, 1.0)
    floor_y = float(np.median(np.maximum(k[:, JI["l_ank"], 1], k[:, JI["r_ank"], 1])))
    hip_h0 = float(np.median((floor_y - hip[:, 1]) / scale))
    head_h0 = float(np.median((floor_y - head[:, 1]) / scale))
    return Calib(scale, floor_y, max(hip_h0, 0.5), max(head_h0, 0.5))


def _causal_ma(x: np.ndarray, m: int) -> np.ndarray:
    if m <= 1:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - m + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def frame_signals(kp: np.ndarray, cal: Calib, fps: float) -> dict:
    s = cal.scale
    m = max(2, int(round(0.12 * fps)))
    lag = max(1, int(round(0.2 * fps)))
    hip, neck, head = kp[:, JI["mid_hip"]], kp[:, JI["neck"]], kp[:, JI["head"]]
    hip_h = _causal_ma((cal.floor_y - hip[:, 1]) / s, m)
    head_h = _causal_ma((cal.floor_y - head[:, 1]) / s, m)
    v = neck - hip
    ang = _causal_ma(np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1])), m)
    w = kp[:, :, 0].max(1) - kp[:, :, 0].min(1)
    h = kp[:, :, 1].max(1) - kp[:, :, 1].min(1)
    asp = _causal_ma(np.log((w + 1.0) / (h + 1.0)), m)

    def vel(x):
        out = np.zeros_like(x)
        out[lag:] = (x[lag:] - x[:-lag]) * fps / lag
        return out

    kpn = _causal_ma((kp / s).reshape(len(kp), -1), m)
    sp = np.zeros(len(kp))
    sp[lag:] = np.abs(kpn[lag:] - kpn[:-lag]).mean(1) * fps / lag
    return dict(hip_h=hip_h, head_h=head_h, ang=ang, asp=asp, vy_hip=vel(hip_h),
                vy_head=vel(head_h), ang_rate=np.abs(vel(ang)), speed=sp)


def window_features(sig: dict, cal: Calib, i_end: int, T: int, fps: float) -> np.ndarray:
    sl = slice(i_end - T + 1, i_end + 1)
    hip, head, ang, asp = sig["hip_h"][sl], sig["head_h"][sl], sig["ang"][sl], sig["asp"][sl]
    vh, vhd, ar, sp = sig["vy_hip"][sl], sig["vy_head"][sl], sig["ang_rate"][sl], sig["speed"][sl]
    last = max(2, int(round(0.6 * fps)))
    run_max = np.maximum.accumulate(hip)
    vmin_i = int(np.argmin(vh))
    return np.array([
        hip[-1], hip.min(), (run_max - hip).max(), hip[-1] / cal.hip_h0,
        head[-1], head.min(), head[-1] / cal.head_h0,
        ang[0], ang[-1], ang.max(), ang.mean(), ang[-1] - ang[0],
        asp[-1], asp.max(), asp.mean(),
        vh.min(), vh[-last:].mean(), vhd.min(), ar.max(),
        sp.mean(), sp.max(), sp[-last:].mean(), (T - 1 - vmin_i) / fps,
        -vh.min() * (run_max - hip).max(), float((head < hip).mean()),
        float((hip < 0.5 * cal.hip_h0).mean()),
    ], dtype=float)


def clip_windows(kp: np.ndarray, fps: float, window_s: float = WINDOW_SECONDS,
                 stride_s: float = STRIDE_SECONDS):
    """Yield (features matrix, window end times) for a whole clip."""
    cal = calibrate(kp, fps)
    sig = frame_signals(kp, cal, fps)
    T = int(round(window_s * fps))
    stride = max(1, int(round(stride_s * fps)))
    ends = np.arange(T - 1, len(kp), stride)
    X = np.stack([window_features(sig, cal, int(i), T, fps) for i in ends])
    return X, ends / fps
