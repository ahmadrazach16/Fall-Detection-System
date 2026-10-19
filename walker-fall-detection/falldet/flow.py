"""Dense optical-flow motion energy (Farneback) for rendered or real frames."""
from __future__ import annotations

import cv2
import numpy as np


def flow_energy(frames, size=(320, 240)):
    """Mean flow magnitude per frame (first entry is 0). Frames are BGR uint8."""
    prev, out = None, []
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is None:
            out.append(0.0)
        else:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            out.append(float(np.linalg.norm(fl, axis=2).mean()))
        prev = g
    return np.asarray(out)
