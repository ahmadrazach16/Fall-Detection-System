"""Biomechanical simulator for walker-assisted older adults.

Each clip is a 3D skeleton (15 joints) plus a walker, driven by a small set of
physically-motivated time series:

* gait (hip / knee swing, forward speed, damped when the person stops),
* local posture (torso flexion, hip & knee flexion -> sitting, bending, crouching),
* a rigid "tipping" rotation of the whole body about the ankles (falls,
  stumbles, controlled lowering) with accelerating descent, impact bounce and
  ground contact,
* hands gripping the walker, released as the person loses balance,
* the walker itself tipping over.

A camera with configurable yaw / pitch / distance then projects the skeleton to
2D keypoints, which is what a real pose estimator would hand to the detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import Camera
from .config import (CLIP_SECONDS, NEGATIVE_SCENARIOS, POSITIVE_SCENARIOS,
                     SCENARIO_WEIGHTS, Domain)

JOINTS = ["head", "neck", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
          "mid_hip", "l_hip", "r_hip", "l_knee", "r_knee", "l_ank", "r_ank"]
JI = {n: i for i, n in enumerate(JOINTS)}
EDGES = [("head", "neck"), ("neck", "l_sho"), ("neck", "r_sho"), ("l_sho", "l_elb"),
         ("l_elb", "l_wri"), ("r_sho", "r_elb"), ("r_elb", "r_wri"), ("neck", "mid_hip"),
         ("mid_hip", "l_hip"), ("mid_hip", "r_hip"), ("l_hip", "l_knee"),
         ("l_knee", "l_ank"), ("r_hip", "r_knee"), ("r_knee", "r_ank")]
# walker: 0-3 feet (FL, FR, RL, RR), 4-7 tops (FL, FR, RL, RR)
WALKER_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (6, 7), (4, 6), (5, 7), (0, 1), (2, 3)]
FLOOR_CLEAR = 0.03   # joint centre-line clearance when lying (fraction of height)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


@dataclass
class Clip:
    scenario: str
    label: int                       # 1 if the clip contains a fall
    fps: float
    t: np.ndarray
    joints3d: np.ndarray             # (n, 15, 3)
    walker3d: np.ndarray             # (n, 8, 3)
    kp2d: np.ndarray                 # (n, 15, 2) clean projection
    camera: Camera
    t_lob: float = np.nan            # loss of balance begins
    fall_t0: float = np.nan          # descent begins
    fall_t1: float = np.nan          # impact
    meta: dict = field(default_factory=dict)


def _rotate(points, pivot, axis, angle):
    """Rodrigues rotation of points (n,K,3) about axis through pivot (n,3)."""
    p = points - pivot[:, None, :]
    c = np.cos(angle)[:, None, None]
    s = np.sin(angle)[:, None, None]
    a = axis[None, None, :]
    cross = np.cross(np.broadcast_to(a, p.shape), p)
    dot = (p * a).sum(-1, keepdims=True)
    return p * c + cross * s + a * dot * (1 - c) + pivot[:, None, :]


def _axis_for(phi):
    return np.array([np.sin(phi), 0.0, -np.cos(phi)])


def _local_joints(H, beta, h_l, h_r, k_l, k_r):
    n = len(beta)
    z = np.zeros(n)
    P = np.zeros((n, 3))
    s = np.array([0.0, 0.0, 1.0])
    u = np.stack([np.sin(beta), np.cos(beta), z], -1)
    neck = P + 0.30 * H * u
    head = neck + 0.13 * H * np.stack([np.sin(beta + 0.1), np.cos(beta + 0.1), z], -1)
    l_sho, r_sho = neck + 0.13 * H * s, neck - 0.13 * H * s
    l_hip, r_hip = P + 0.095 * H * s, P - 0.095 * H * s

    def leg(hip, h, k):
        knee = hip + 0.245 * H * np.stack([np.sin(h), -np.cos(h), z], -1)
        ank = knee + 0.246 * H * np.stack([np.sin(h - k), -np.cos(h - k), z], -1)
        return knee, ank

    l_knee, l_ank = leg(l_hip, h_l, k_l)
    r_knee, r_ank = leg(r_hip, h_r, k_r)
    hang = np.array([0.0, -0.30 * H, 0.0])
    l_wri, r_wri = l_sho + hang, r_sho + hang
    l_elb, r_elb = (l_sho + l_wri) / 2, (r_sho + r_wri) / 2
    J = np.stack([head, neck, l_sho, r_sho, l_elb, r_elb, l_wri, r_wri,
                  P, l_hip, r_hip, l_knee, r_knee, l_ank, r_ank], axis=1)
    J[:, :, 1] += (0.04 * H - np.minimum(l_ank[:, 1], r_ank[:, 1]))[:, None]
    return J


def _elbow(S, W, H, side):
    a = 0.166 * H
    d = W - S
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    dirn = d / (dist + 1e-9)
    dist_c = np.minimum(dist, 1.99 * a)
    mid = S + dirn * dist_c / 2
    hh = np.sqrt(np.maximum(a * a - (dist_c / 2) ** 2, 0.0))
    pref = np.array([-0.3, -0.5, side * 0.8])
    perp = pref - (pref * dirn).sum(-1, keepdims=True) * dirn
    perp = perp / (np.linalg.norm(perp, axis=-1, keepdims=True) + 1e-9)
    return mid + hh * perp


def _lowfreq(rng, n, scale, k=25):
    x = np.convolve(rng.normal(0, 1, n + 2 * k), np.ones(k) / k, mode="same")[k:-k]
    return scale * x / (x.std() + 1e-9)


def sample_camera(dom: Domain, rng) -> Camera:
    yaw = rng.uniform(*dom.yaw_deg)
    if dom.yaw_mirror:
        yaw *= rng.choice([-1, 1])
    return Camera(yaw=np.deg2rad(yaw), pitch=np.deg2rad(rng.uniform(*dom.pitch_deg)),
                  dist=rng.uniform(*dom.dist_m))


def simulate_clip(scenario: str, dom: Domain, rng, duration: float = CLIP_SECONDS,
                  t0: float | None = None) -> Clip:
    fps = float(rng.choice(dom.fps))
    n = int(round(duration * fps))
    t = np.arange(n) / fps
    H = rng.uniform(*dom.height_m)
    v0 = rng.uniform(*dom.speed_mps)
    gait_f = rng.uniform(0.75, 1.1)
    A, K = rng.uniform(0.22, 0.32), rng.uniform(0.35, 0.55)
    sgn = rng.choice([-1, 1])
    if t0 is None:
        t0 = rng.uniform(3.0, 4.0) if scenario == "controlled_lower" else rng.uniform(3.0, 5.5)

    def ramp(ts, d):
        return smoothstep((t - ts) / d)

    gait_w = np.ones(n)
    hip_f, knee_f = np.zeros(n), np.zeros(n)
    beta = 0.10 + _lowfreq(rng, n, 0.02)
    Theta, collapse, g = np.zeros(n), np.zeros(n), np.zeros(n)
    phi, collapse_gain = 0.0, 0.0
    t_lob = fall_t0 = fall_t1 = np.nan
    meta = {"t_event": float(t0)}

    if scenario == "walk":
        pass
    elif scenario == "stand":
        gait_w[:] = 0.0
        beta += 0.02 * np.sin(2 * np.pi * 0.3 * t)
    elif scenario == "sit_down":
        d = rng.uniform(1.5, 3.0)
        gait_w = 1 - ramp(t0 - 0.6, 0.5)
        q = ramp(t0, d)
        hip_f = q * rng.uniform(1.35, 1.5)
        knee_f = q * rng.uniform(1.35, 1.5)
        beta = beta + 0.35 * np.sin(np.pi * q)
        g = 0.8 * q
    elif scenario in ("bend_pick", "crouch_pick"):
        d1, d2, hold = rng.uniform(1.0, 1.7), rng.uniform(1.0, 1.7), rng.uniform(0.4, 1.4)
        gait_w = 1 - ramp(t0 - 0.6, 0.5)
        q = ramp(t0, d1) - ramp(t0 + d1 + hold, d2)
        if scenario == "bend_pick":
            hip_f, knee_f = q * 0.35, q * 0.7
            beta = beta + q * rng.uniform(0.8, 1.1)
        else:
            hip_f, knee_f = q * rng.uniform(1.2, 1.4), q * rng.uniform(2.0, 2.3)
            beta = beta + q * 0.55
        g = q
    elif scenario == "stumble_recover":
        phi = rng.choice([0.0, np.pi / 2 * sgn]) + rng.normal(0, 0.2)
        amp = rng.uniform(0.25, 0.6)
        Theta = amp * (ramp(t0, 0.5) - ramp(t0 + 0.6, 0.7))
        gait_w = np.clip(1 - ramp(t0 - 0.1, 0.3) + ramp(t0 + 1.2, 0.6), 0, 1)
        g = 0.5 * (ramp(t0, 0.4) - ramp(t0 + 0.8, 0.6))
        t_lob = t0
    elif scenario == "controlled_lower":
        d = rng.uniform(3.5, 5.5)
        phi = np.pi if rng.random() < 0.4 else sgn * np.pi / 2
        gait_w = 1 - ramp(t0 - 0.4, 0.4)
        Theta = rng.uniform(1.4, 1.55) * ramp(t0, d)
        collapse = 0.8 * ramp(t0, d)
        g = ramp(t0, 1.0)
    elif scenario in POSITIVE_SCENARIOS:
        if scenario == "fall_forward":
            phi, d, collapse_gain = rng.normal(0, 0.25), rng.uniform(0.45, 0.80), 0.3
        elif scenario == "fall_backward":
            phi, d, collapse_gain = np.pi + rng.normal(0, 0.25), rng.uniform(0.45, 0.80), 0.4
        elif scenario == "fall_side":
            phi, d, collapse_gain = sgn * np.pi / 2 + rng.normal(0, 0.25), rng.uniform(0.55, 0.95), 0.3
        else:  # slump
            phi, d, collapse_gain = rng.uniform(0, 2 * np.pi), rng.uniform(1.1, 1.8), 1.0
        th1 = rng.uniform(0.15, 0.30)
        th_f = rng.uniform(1.0, 1.5) if scenario == "slump" else rng.uniform(1.3, 1.6)
        t_d = t0 + 0.5
        u = np.clip((t - t_d) / d, 0, 1)
        s = u ** 2
        Theta = np.where(t < t_d, th1 * ramp(t0, 0.5), th1 + (th_f - th1) * s)
        te = t_d + d
        post = np.clip(t - te, 0, None)
        Theta = Theta + np.where(t > te, 0.05 * np.exp(-7 * post) * np.sin(20 * post), 0.0)
        collapse = collapse_gain * s
        gait_w = 1 - ramp(t0 - 0.1, 0.4)
        g = ramp(t0 + 0.1, 0.5)
        t_lob, fall_t0, fall_t1 = t0, t_d, te
        meta.update(descent=float(d))
    else:
        raise ValueError(scenario)

    phase = 2 * np.pi * gait_f * t
    kswing = lambda ph: K * gait_w * (0.5 + 0.5 * np.sin(ph + 1.0)) ** 2
    h_l = hip_f + A * gait_w * np.sin(phase) + collapse * 0.6
    h_r = hip_f + A * gait_w * np.sin(phase + np.pi) + collapse * 0.6 * 0.9
    k_l = knee_f + kswing(phase) + collapse * 1.2
    k_r = knee_f + kswing(phase + np.pi) + collapse * 1.2 * 0.8

    # position along the walking axis
    x = np.cumsum(v0 * gait_w) / fps
    x_stop = rng.uniform(-0.5, 0.5)
    iref = n // 2 if scenario == "walk" else min(int(t0 * fps), n - 1)
    x = x - x[iref] + x_stop
    zp = rng.uniform(-0.3, 0.3)

    J = _local_joints(H, beta, h_l, h_r, k_l, k_r)
    pivot = (J[:, JI["l_ank"]] + J[:, JI["r_ank"]]) / 2
    pivot[:, 1] = 0.0
    J = _rotate(J, pivot, _axis_for(phi), Theta)
    J[:, :, 1] += np.maximum(0.0, FLOOR_CLEAR * H - J[:, :, 1].min(1))[:, None]
    J[:, :, 0] += x[:, None]
    J[:, :, 2] += zp

    # walker
    xw, zw = x + 0.26 * H, np.full(n, zp)
    fx, rx, hz, wh = 0.12 * H, -0.12 * H, 0.22 * H, 0.55 * H
    base = np.array([[fx, 0, hz], [fx, 0, -hz], [rx, 0, hz], [rx, 0, -hz],
                     [fx, wh, hz], [fx, wh, -hz], [rx, wh, hz], [rx, wh, -hz]])
    W = base[None, :, :] + np.stack([xw, np.zeros(n), zw], -1)[:, None, :]
    if scenario in POSITIVE_SCENARIOS or scenario == "stumble_recover":
        ang = (phi + np.pi) % (2 * np.pi) - np.pi
        tipk = 0.9 if abs(ang) < 1.0 else (0.5 if abs(ang) < 2.1 else 0.0)
        wpivot = np.stack([xw, np.zeros(n), zw], -1)
        W = _rotate(W, wpivot, _axis_for(phi), tipk * Theta)
        W[:, :, 1] += np.maximum(0.0, -W[:, :, 1].min(1))[:, None]

    # hands: walker handles <-> hanging arms
    handles = W[:, [6, 7], :]
    gg = g[:, None]
    J[:, JI["l_wri"]] = (1 - gg) * handles[:, 0] + gg * J[:, JI["l_wri"]]
    J[:, JI["r_wri"]] = (1 - gg) * handles[:, 1] + gg * J[:, JI["r_wri"]]
    J[:, JI["l_elb"]] = _elbow(J[:, JI["l_sho"]], J[:, JI["l_wri"]], H, +1)
    J[:, JI["r_elb"]] = _elbow(J[:, JI["r_sho"]], J[:, JI["r_wri"]], H, -1)
    J[:, :, 1] = np.maximum(J[:, :, 1], 0.01 * H)   # limbs never penetrate the floor

    if rng.random() < 0.5:   # walk right-to-left instead
        J[:, :, 0] *= -1
        W[:, :, 0] *= -1

    cam = sample_camera(dom, rng)
    kp2d, _ = cam.project(J)
    meta.update(H=float(H), v0=float(v0), yaw_deg=float(np.rad2deg(cam.yaw)),
                pitch_deg=float(np.rad2deg(cam.pitch)), dist=float(cam.dist), phi=float(phi))
    return Clip(scenario=scenario, label=int(scenario in POSITIVE_SCENARIOS), fps=fps, t=t,
                joints3d=J, walker3d=W, kp2d=kp2d, camera=cam, t_lob=t_lob,
                fall_t0=fall_t0, fall_t1=fall_t1, meta=meta)


def observe(kp2d, fps, dom: Domain, rng):
    """Emulate a 2D pose estimator: jitter, occlusion blocks and dropouts.

    Missing joints are filled by hold-last-value, as most trackers do.
    Returns (keypoints (n,15,2), confidence (n,15)).
    """
    n, J_, _ = kp2d.shape
    sigma = rng.uniform(*dom.noise_px)
    kp = kp2d + rng.normal(0, sigma, kp2d.shape)
    conf = np.ones((n, J_))
    groups = [[JI["l_knee"], JI["r_knee"], JI["l_ank"], JI["r_ank"]],
              [JI["l_ank"], JI["r_ank"]],
              [JI["l_elb"], JI["r_elb"], JI["l_wri"], JI["r_wri"]],
              [JI["head"]]]
    for _ in range(int(rng.integers(dom.occlusion_blocks[0], dom.occlusion_blocks[1] + 1))):
        s = int(rng.uniform(0, n / fps - 0.5) * fps)
        e = min(n, s + int(rng.uniform(0.4, 2.0) * fps))
        conf[s:e, groups[int(rng.integers(len(groups)))]] = 0.0
    if dom.dropout_p > 0:
        conf[rng.random(conf.shape) < dom.dropout_p] = 0.0
    out = kp.copy()
    for j in range(J_):
        bad = conf[:, j] == 0
        if not bad.any():
            continue
        good_idx = np.where(~bad)[0]
        if len(good_idx) == 0:
            continue
        idx = np.arange(n)
        last = np.maximum.accumulate(np.where(~bad, idx, -1))
        last = np.where(last < 0, good_idx[0], last)
        out[:, j] = kp[last, j]
    return out, conf


def sample_scenarios(n: int, rng) -> list[str]:
    names = list(SCENARIO_WEIGHTS)
    w = np.array([SCENARIO_WEIGHTS[k] for k in names])
    counts = np.floor(w * n).astype(int)
    for i in np.argsort(-(w * n - counts))[: n - counts.sum()]:
        counts[i] += 1
    out = [nm for nm, c in zip(names, counts) for _ in range(c)]
    rng.shuffle(out)
    return out
