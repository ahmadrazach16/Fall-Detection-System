"""Turn pose-estimator output from a real video into the track the detectors expect.

Pose comes from any COCO-17 estimator (for example Ultralytics YOLO pose). The
helpers here are pure NumPy, so they can be tested without a GPU or a model download.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np

from .config import WINDOW_SECONDS
from .evaluate import alert_times
from .features import calibrate, clip_windows, frame_signals
from .simulate import JI

# COCO-17 order: 0 nose, 1-4 eyes/ears, 5 l_sho, 6 r_sho, 7 l_elb, 8 r_elb, 9 l_wri, 10 r_wri,
# 11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ank, 16 r_ank
COCO = dict(nose=0, l_sho=5, r_sho=6, l_elb=7, r_elb=8, l_wri=9, r_wri=10, l_hip=11, r_hip=12,
            l_knee=13, r_knee=14, l_ank=15, r_ank=16)
_DIRECT = ["l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri", "l_hip", "r_hip",
           "l_knee", "r_knee", "l_ank", "r_ank"]


def coco_to_joints(kp17, conf17=None):
    """COCO-17 keypoints (..., 17, 2) -> the 15-joint layout (..., 15, 2) plus confidence (..., 15).

    head = nose, neck = mid-shoulders, mid_hip = mid-hips. A derived joint is as
    confident as its least confident parent.
    """
    kp17 = np.asarray(kp17, dtype=float)
    conf17 = np.ones(kp17.shape[:-1]) if conf17 is None else np.asarray(conf17, dtype=float)
    out = np.zeros(kp17.shape[:-2] + (15, 2))
    conf = np.zeros(kp17.shape[:-2] + (15,))
    out[..., JI["head"], :] = kp17[..., COCO["nose"], :]
    conf[..., JI["head"]] = conf17[..., COCO["nose"]]
    for name in _DIRECT:
        out[..., JI[name], :] = kp17[..., COCO[name], :]
        conf[..., JI[name]] = conf17[..., COCO[name]]
    for name, (a, b) in {"neck": ("l_sho", "r_sho"), "mid_hip": ("l_hip", "r_hip")}.items():
        out[..., JI[name], :] = (kp17[..., COCO[a], :] + kp17[..., COCO[b], :]) / 2
        conf[..., JI[name]] = np.minimum(conf17[..., COCO[a]], conf17[..., COCO[b]])
    return out, conf


def hold_last(kp, conf, thr=0.3):
    """Replace low-confidence joints by their last confident position (as the training noise model does)."""
    kp = np.array(kp, dtype=float)
    n = len(kp)
    idx = np.arange(n)
    for j in range(kp.shape[1]):
        ok = conf[:, j] >= thr
        good = np.where(ok)[0]
        if len(good) == 0:
            continue
        last = np.maximum.accumulate(np.where(ok, idx, -1))
        last = np.where(last < 0, good[0], last)
        kp[:, j] = kp[last, j]
    return kp


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def pick_person(boxes, prev_center=None):
    """Index of the detection to follow: the largest box first, then the one nearest the last position."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return None
    if prev_center is None:
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return int(np.argmax(area))
    centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    return int(np.argmin(np.linalg.norm(centres - np.asarray(prev_center), axis=1)))


def person_from_result(result, prev_center=None):
    """One Ultralytics pose result -> (kp17 (17,2), conf17 (17,), box centre) or None if nobody is found."""
    if result.keypoints is None or result.boxes is None:
        return None
    boxes = _np(result.boxes.xyxy)
    if len(boxes) == 0:
        return None
    k = pick_person(boxes, prev_center)
    kp = _np(result.keypoints.xy)[k]
    if kp.shape[0] != 17:
        raise ValueError(f"Expected a COCO-17 pose model, got {kp.shape[0]} keypoints.")
    conf = _np(result.keypoints.conf)[k] if result.keypoints.conf is not None else np.ones(17)
    b = boxes[k]
    return kp, conf, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def load_bundle(path):
    """Load a detector saved by scripts/run_experiments.py -> (model, tau)."""
    bundle = joblib.load(path)
    return bundle["model"], float(bundle["tau"])


@dataclass
class Analysis:
    fps: float
    tau: float
    t_end: np.ndarray            # window end times (s)
    prob: np.ndarray             # fall probability per window
    alerts: list                 # alert times (s)
    peak_prob: float
    peak_time: float
    t: np.ndarray                # per-frame time (s)
    hip_rel: np.ndarray          # hip height relative to the upright calibration
    torso_deg: np.ndarray        # torso angle from vertical
    warnings: list = field(default_factory=list)


def analyze_track(kp, fps, model, tau, detect_rate=1.0):
    """Run the detector over a 15-joint track (n, 15, 2) in image coordinates."""
    kp = np.asarray(kp, dtype=float)
    if len(kp) < int(round(WINDOW_SECONDS * fps)) + 1:
        raise ValueError(f"Video too short: the detector needs more than {WINDOW_SECONDS:.0f} s of footage.")
    X, t_end = clip_windows(kp, fps)
    prob = np.asarray(model.predict_proba(X), dtype=float)
    alerts = alert_times(prob.copy(), t_end, tau)
    cal = calibrate(kp, fps)
    sig = frame_signals(kp, cal, fps)
    warnings = []
    first = max(3, int(fps))
    if float(np.median(sig["ang"][:first])) > 35:
        warnings.append("The person does not look upright in the first second. The detector calibrates on "
                        "that second, so start the clip with the person standing or walking.")
    if detect_rate < 0.8:
        warnings.append(f"The pose model found a person in only {detect_rate:.0%} of frames. "
                        "Results are unreliable.")
    if not 10 <= fps <= 35:
        warnings.append(f"Frame rate {fps:.1f} fps is outside the 10-35 fps range seen in training.")
    peak = int(np.argmax(prob))
    return Analysis(fps=fps, tau=tau, t_end=t_end, prob=prob, alerts=alerts,
                    peak_prob=float(prob[peak]), peak_time=float(t_end[peak]),
                    t=np.arange(len(kp)) / fps, hip_rel=sig["hip_h"] / cal.hip_h0,
                    torso_deg=sig["ang"], warnings=warnings)
