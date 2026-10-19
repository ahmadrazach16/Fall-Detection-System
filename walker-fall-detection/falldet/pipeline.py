"""Dataset construction: simulate -> observe -> window -> label."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Domain
from .features import FEATURE_NAMES, clip_windows
from .simulate import observe, sample_scenarios, simulate_clip


@dataclass
class Split:
    X: np.ndarray          # (N, F) window features
    y: np.ndarray          # (N,) window labels
    clip_id: np.ndarray    # (N,)
    t_end: np.ndarray      # (N,) window end time in seconds
    clips: list            # per-clip metadata dicts
    domain: str


def window_labels(t_end, label, fall_t0, fall_t1):
    """A window is positive if it ends while the fall is >=60% through its
    descent, up to 2 s after impact (so the whole descent is inside it)."""
    if not label:
        return np.zeros(len(t_end), dtype=int)
    d = fall_t1 - fall_t0
    return ((t_end >= fall_t0 + 0.6 * d) & (t_end <= fall_t1 + 2.0)).astype(int)


def build_split(n_clips: int, dom: Domain, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    scenarios = sample_scenarios(n_clips, rng)
    Xs, ys, cid, te, metas = [], [], [], [], []
    for ci, sc in enumerate(scenarios):
        clip = simulate_clip(sc, dom, rng)
        kp, _ = observe(clip.kp2d, clip.fps, dom, rng)
        X, t_end = clip_windows(kp, clip.fps)
        Xs.append(X)
        ys.append(window_labels(t_end, clip.label, clip.fall_t0, clip.fall_t1))
        cid.append(np.full(len(X), ci))
        te.append(t_end)
        metas.append(dict(scenario=sc, label=clip.label, fps=clip.fps, t_lob=clip.t_lob,
                          fall_t0=clip.fall_t0, fall_t1=clip.fall_t1,
                          duration=float(clip.t[-1] + 1 / clip.fps), **clip.meta))
    assert Xs[0].shape[1] == len(FEATURE_NAMES)
    return Split(np.concatenate(Xs), np.concatenate(ys), np.concatenate(cid),
                 np.concatenate(te), metas, dom.name)
