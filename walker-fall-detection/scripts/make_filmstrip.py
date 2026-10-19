"""Render a contact sheet (still frames, no video) of one synthetic fall.

Usage: python scripts/make_filmstrip.py  ->  assets/fall_filmstrip.png
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from falldet.config import NARROW  # noqa: E402
from falldet.render import render_frame  # noqa: E402
from falldet.simulate import simulate_clip  # noqa: E402

DOM = replace(NARROW, name="sheet", yaw_deg=(22, 22), pitch_deg=(14, 14), dist_m=(4.4, 4.4))


def main(out: Path | None = None) -> Path:
    out = out or ROOT / "assets" / "fall_filmstrip.png"
    clip = simulate_clip("fall_forward", DOM, np.random.default_rng(13), duration=9.0, t0=4.2)
    times = [3.0, 4.4, 4.9, 5.2, 5.6, 7.5]
    tiles = []
    for t in times:
        i = min(int(round(t * clip.fps)), len(clip.t) - 1)
        img = render_frame(clip, i, skeleton=True)
        cv2.rectangle(img, (0, 0), (150, 30), (30, 25, 22), -1)
        cv2.putText(img, f"t = {clip.t[i]:.1f} s", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (240, 236, 232), 1, cv2.LINE_AA)
        img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA)
        tiles.append(img)
    sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    footer = np.full((34, sheet.shape[1], 3), (30, 25, 22), np.uint8)
    cv2.putText(footer, "Synthetic forward fall with walker (rendered from the simulator, no real footage)",
                (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 196, 190), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), np.vstack([sheet, footer]))
    return out


if __name__ == "__main__":
    print("wrote", main())
