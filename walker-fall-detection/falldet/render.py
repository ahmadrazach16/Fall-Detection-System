"""Pixel renderer: draws a synthetic clip as a stylised mannequin + walker in a room.

This is intentionally a simple, fully original renderer (geometry only). It is
useful for optical-flow experiments, lighting augmentation and for visually
sanity-checking the simulator; photo-realistic footage would come from a
generative video model (see docs/video_generation_prompts.md).
"""
from __future__ import annotations

import cv2
import numpy as np

from .simulate import EDGES, JI, WALKER_EDGES, Clip

SKIN, CLOTH, LIMB = (150, 190, 235), (120, 150, 90), (95, 70, 60)


def _background(cam, w, h, brightness=1.0):
    img = np.zeros((h, w, 3), np.uint8)
    # floor / wall split from the projected floor line behind the subject
    p, _ = cam.project(np.array([[0.0, 0.0, 0.0]]))
    horizon = int(np.clip(p[0, 1] - 0.55 * cam.f * 1.3 / max(cam.dist, 1.0), 40, h - 80))
    wall = np.linspace((205, 200, 195), (185, 180, 175), horizon)[:, None, :]
    img[:horizon] = wall.astype(np.uint8)
    floor = np.linspace((150, 135, 120), (110, 98, 88), h - horizon)[:, None, :]
    img[horizon:] = floor.astype(np.uint8)
    cv2.line(img, (0, horizon), (w, horizon), (160, 155, 150), 2)
    return np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)


def render_frame(clip: Clip, i: int, brightness: float = 1.0, skeleton: bool = False,
                 noise: float = 0.0, rng=None) -> np.ndarray:
    cam = clip.camera
    w, h = cam.width, cam.height
    img = _background(cam, w, h, 1.0)
    J = clip.joints3d[i]
    uv, depth = cam.project(J)
    wuv, wdepth = cam.project(clip.walker3d[i])
    px_cm = cam.f / cam.dist / 100.0   # pixels per centimetre at the subject

    # ground shadow
    ground = J.copy()
    ground[:, 1] = 0.0
    guv, _ = cam.project(ground)
    sh = img.copy()
    for a, b in EDGES:
        cv2.line(sh, tuple(guv[JI[a]].astype(int)), tuple(guv[JI[b]].astype(int)), (60, 55, 50), max(3, int(9 * px_cm)))
    img = cv2.addWeighted(sh, 0.35, img, 0.65, 0)

    segs = []   # (depth, kind, a, b, thickness, colour)
    for a, b in WALKER_EDGES:
        segs.append((float((wdepth[a] + wdepth[b]) / 2), "w", wuv[a], wuv[b], max(2, int(2.5 * px_cm)), (170, 170, 175)))
    limb_t = max(3, int(5.5 * px_cm))
    for a, b in EDGES:
        ja, jb = JI[a], JI[b]
        if a == "neck" and b in ("l_sho", "r_sho"):
            col, t = CLOTH, limb_t
        elif a == "neck" and b == "mid_hip":
            col, t = CLOTH, int(limb_t * 2.4)
        elif "hip" in a or "hip" in b or "knee" in a or "knee" in b:
            col, t = LIMB, int(limb_t * 1.15)
        elif a == "head":
            col, t = SKIN, limb_t
        else:
            col, t = CLOTH if "sho" in a or "elb" in a else SKIN, limb_t
        segs.append((float((depth[ja] + depth[jb]) / 2), "p", uv[ja], uv[jb], t, col))
    for _, _, a, b, t, col in sorted(segs, key=lambda s: -s[0]):
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), col, t, cv2.LINE_AA)
    hu = uv[JI["head"]].astype(int)
    cv2.circle(img, tuple(hu), max(5, int(9.5 * px_cm)), SKIN, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(hu), max(5, int(9.5 * px_cm)), (90, 120, 170), 1, cv2.LINE_AA)

    img = np.clip(img.astype(np.float32) * brightness, 0, 255)
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        img = img + rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if skeleton:
        draw_skeleton(img, clip.kp2d[i])
    return img


def draw_skeleton(img, kp, color=(40, 200, 255)):
    for a, b in EDGES:
        cv2.line(img, tuple(kp[JI[a]].astype(int)), tuple(kp[JI[b]].astype(int)), color, 2, cv2.LINE_AA)
    for p in kp:
        cv2.circle(img, tuple(p.astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p.astype(int)), 3, (30, 30, 30), 1, cv2.LINE_AA)
    return img
