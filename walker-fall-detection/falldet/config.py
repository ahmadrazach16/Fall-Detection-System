"""Domain configurations for synthetic data generation.

A `Domain` describes *where the synthetic data comes from*: camera placement,
frame rate, keypoint noise and occlusion. Training on a narrow domain and
testing on a shifted one is how we measure robustness; training on a
randomised ("wide") domain is how we try to fix it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Domain:
    name: str = "narrow"
    yaw_deg: tuple = (-35.0, 35.0)      # camera yaw; 0 = side view, 90 = frontal view
    yaw_mirror: bool = False            # if True, yaw sign is sampled randomly
    pitch_deg: tuple = (8.0, 22.0)      # camera looks down by this angle
    dist_m: tuple = (4.0, 5.2)          # camera-to-subject distance
    fps: tuple = (25.0,)                # frame rate(s) to sample from
    noise_px: tuple = (0.8, 1.5)        # keypoint jitter sigma (pixels)
    occlusion_blocks: tuple = (0, 0)    # inclusive range of occlusion blocks / clip
    dropout_p: float = 0.0              # per-frame per-joint random dropout
    height_m: tuple = (1.50, 1.85)      # body height
    speed_mps: tuple = (0.25, 0.55)     # walker-assisted walking speed
    brightness: tuple = (0.8, 1.15)     # used by the pixel renderer only


NARROW = Domain("narrow")

WIDE = Domain(
    "wide",
    yaw_deg=(0.0, 90.0),
    yaw_mirror=True,
    pitch_deg=(5.0, 30.0),
    dist_m=(3.5, 6.0),
    fps=(15.0, 25.0, 30.0),
    noise_px=(0.5, 4.5),
    occlusion_blocks=(0, 2),
    dropout_p=0.01,
    brightness=(0.5, 1.3),
)

# ---- shifted test domains (all unseen by the NARROW-trained model) ----------
TEST_INDIST = replace(NARROW, name="in_distribution")
TEST_FRONTAL = replace(NARROW, name="frontal_view", yaw_deg=(60.0, 90.0), yaw_mirror=True)
TEST_NOISY = replace(
    NARROW, name="noisy_occluded", noise_px=(3.5, 5.0), occlusion_blocks=(1, 3), dropout_p=0.03
)
TEST_LOWFPS = replace(NARROW, name="low_fps_12", fps=(12.0,))
TEST_DEPLOY = Domain(
    "deployment_mix",
    yaw_deg=(0.0, 90.0),
    yaw_mirror=True,
    pitch_deg=(5.0, 30.0),
    dist_m=(3.5, 6.0),
    fps=(15.0,),
    noise_px=(2.0, 4.5),
    occlusion_blocks=(1, 2),
    dropout_p=0.02,
)

TEST_DOMAINS = [TEST_INDIST, TEST_FRONTAL, TEST_NOISY, TEST_LOWFPS, TEST_DEPLOY]

# ---- scenario catalogue ------------------------------------------------------
NEGATIVE_SCENARIOS = [
    "walk",              # walker-assisted walking
    "stand",             # standing still, swaying
    "sit_down",          # sitting on a chair (ADL)
    "bend_pick",         # bending to pick something up (hard negative)
    "crouch_pick",       # deep crouch (hard negative)
    "stumble_recover",   # loses balance, recovers (hard negative)
    "controlled_lower",  # deliberately lowers self to the floor (hard negative)
]
POSITIVE_SCENARIOS = ["fall_forward", "fall_backward", "fall_side", "slump"]

SCENARIO_WEIGHTS = {
    "walk": 0.14, "stand": 0.08, "sit_down": 0.10, "bend_pick": 0.10,
    "crouch_pick": 0.08, "stumble_recover": 0.08, "controlled_lower": 0.06,
    "fall_forward": 0.12, "fall_backward": 0.10, "fall_side": 0.08, "slump": 0.06,
}

CLIP_SECONDS = 10.0
WINDOW_SECONDS = 3.0
STRIDE_SECONDS = 0.2
CALIB_SECONDS = 1.0
