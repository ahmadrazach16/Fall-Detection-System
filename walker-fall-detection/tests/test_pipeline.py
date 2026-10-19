import numpy as np
import pytest

from falldet.camera import Camera
from falldet.config import NARROW, WIDE
from falldet.evaluate import alert_times, wilson
from falldet.features import FEATURE_NAMES, clip_windows
from falldet.models import RuleDetector
from falldet.pipeline import build_split, window_labels
from falldet.simulate import (JI, NEGATIVE_SCENARIOS, POSITIVE_SCENARIOS, observe,
                              sample_scenarios, simulate_clip)

ALL = NEGATIVE_SCENARIOS + POSITIVE_SCENARIOS


def test_camera_look_at_projects_to_image_centre():
    cam = Camera(yaw=0.3, pitch=0.2, dist=4.0)
    uv, z = cam.project(np.array([0.0, 0.7, 0.0]))
    assert np.allclose(uv, [320, 240], atol=1e-6)
    assert abs(z - 4.0) < 1e-6


@pytest.mark.parametrize("scenario", ALL)
def test_clip_shapes_and_finiteness(scenario):
    clip = simulate_clip(scenario, NARROW, np.random.default_rng(1))
    n = len(clip.t)
    assert clip.joints3d.shape == (n, 15, 3)
    assert clip.kp2d.shape == (n, 15, 2)
    assert np.isfinite(clip.joints3d).all() and np.isfinite(clip.kp2d).all()
    assert clip.joints3d[:, :, 1].min() >= 0.0          # nothing below the floor
    assert clip.label == int(scenario in POSITIVE_SCENARIOS)


@pytest.mark.parametrize("scenario", POSITIVE_SCENARIOS)
def test_falls_end_low_and_fast(scenario):
    clip = simulate_clip(scenario, NARROW, np.random.default_rng(2))
    hip_y = clip.joints3d[:, JI["mid_hip"], 1]
    assert hip_y[-1] < 0.5 * hip_y[0]
    assert clip.fall_t1 > clip.fall_t0 > clip.t_lob


def test_upright_clips_stay_upright():
    for sc in ("walk", "stand"):
        clip = simulate_clip(sc, NARROW, np.random.default_rng(3))
        hip_y = clip.joints3d[:, JI["mid_hip"], 1]
        assert hip_y.min() > 0.8 * hip_y.max()


def test_controlled_lowering_is_slower_than_a_fall():
    rng = np.random.default_rng(4)
    fall = simulate_clip("fall_forward", NARROW, rng)
    slow = simulate_clip("controlled_lower", NARROW, rng)
    vmax = lambda c: (-np.diff(c.joints3d[:, JI["mid_hip"], 1]) * c.fps).max()
    assert vmax(fall) > 2 * vmax(slow)


def test_observe_fills_occlusions_and_keeps_shape():
    dom = WIDE
    rng = np.random.default_rng(5)
    clip = simulate_clip("walk", dom, rng)
    kp, conf = observe(clip.kp2d, clip.fps, dom, rng)
    assert kp.shape == clip.kp2d.shape and np.isfinite(kp).all()
    assert conf.min() >= 0 and conf.max() <= 1


def test_features_are_finite_and_causal():
    rng = np.random.default_rng(6)
    clip = simulate_clip("fall_backward", NARROW, rng)
    kp, _ = observe(clip.kp2d, clip.fps, NARROW, rng)
    X, t_end = clip_windows(kp, clip.fps)
    assert X.shape[1] == len(FEATURE_NAMES) and np.isfinite(X).all()
    # changing the future must not change earlier windows
    kp2 = kp.copy()
    kp2[-30:] += 50.0
    X2, _ = clip_windows(kp2, clip.fps)
    assert np.allclose(X[:-6], X2[:-6])


def test_window_labels():
    t = np.arange(3.0, 10.0, 0.2)
    y = window_labels(t, 1, 5.0, 5.6)
    assert y.sum() > 0 and y[t < 5.3].sum() == 0 and y[t > 7.7].sum() == 0
    assert window_labels(t, 0, np.nan, np.nan).sum() == 0


def test_scenario_sampling_is_stratified():
    sc = sample_scenarios(200, np.random.default_rng(0))
    assert len(sc) == 200 and set(sc) == set(ALL)


def test_alert_logic_needs_consecutive_windows_and_has_refractory():
    t = np.arange(3.0, 10.0, 0.2)
    p = np.zeros(len(t))
    p[10] = 0.9
    assert alert_times(p.copy(), t, 0.5) == []
    p[11] = 0.9
    assert len(alert_times(p.copy(), t, 0.5)) == 1
    p[12:20] = 0.9
    assert len(alert_times(p.copy(), t, 0.5)) == 1


def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson(45, 50)
    assert lo < 0.9 < hi


def test_rule_detector_separates_fall_from_walk():
    split = build_split(60, NARROW, seed=11)
    rule = RuleDetector()
    p = rule.predict_proba(split.X)
    walk_ids = [i for i, m in enumerate(split.clips) if m["scenario"] == "walk"]
    walk_mask = np.isin(split.clip_id, walk_ids)
    assert p[walk_mask].sum() == 0
    assert p[split.y == 1].mean() > 0.4
