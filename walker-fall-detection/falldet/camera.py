"""Pinhole camera with yaw / pitch / distance, looking at a fixed point."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    yaw: float                      # radians; 0 = side view, pi/2 = frontal view
    pitch: float                    # radians; camera looks down
    dist: float                     # metres from look_at point
    f: float = 500.0                # focal length in pixels
    width: int = 640
    height: int = 480
    look_at: tuple = (0.0, 0.7, 0.0)

    def basis(self):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        right = np.array([cy, 0.0, -sy])
        fwd0 = np.array([-sy, 0.0, -cy])
        up0 = np.array([0.0, 1.0, 0.0])
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        fwd = fwd0 * cp - up0 * sp
        up = up0 * cp + fwd0 * sp
        return right, up, fwd

    def project(self, pts: np.ndarray):
        """pts (...,3) world -> (...,2) pixels, and depth (...)."""
        right, up, fwd = self.basis()
        r = np.asarray(pts, dtype=float) - np.asarray(self.look_at)
        x = r @ right
        y = r @ up
        z = np.maximum(self.dist + r @ fwd, 0.3)
        u = self.width / 2 + self.f * x / z
        v = self.height / 2 - self.f * y / z
        return np.stack([u, v], axis=-1), z
