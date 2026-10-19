from types import SimpleNamespace

import numpy as np
import pytest

from falldet.config import NARROW
from falldet.models import GBMDetector
from falldet.pipeline import build_split
from falldet.simulate import JI, observe, simulate_clip
from falldet.video import (analyze_track, coco_to_joints, hold_last, person_from_result,
                           pick_person)


def test_coco_mapping_uses_midpoints_and_min_confidence():
    kp17 = np.arange(34, dtype=float).reshape(17, 2)
    conf17 = np.ones(17)
    conf17[6] = 0.2          # right shoulder is unsure
    kp, conf = coco_to_joints(kp17, conf17)
    assert kp.shape == (15, 2) and conf.shape == (15,)
    assert np.allclose(kp[JI["head"]], kp17[0])
    assert np.allclose(kp[JI["neck"]], (kp17[5] + kp17[6]) / 2)
    assert np.allclose(kp[JI["mid_hip"]], (kp17[11] + kp17[12]) / 2)
    assert np.allclose(kp[JI["l_ank"]], kp17[15])
    assert conf[JI["neck"]] == pytest.approx(0.2) and conf[JI["head"]] == 1.0


def test_coco_mapping_accepts_a_whole_track():
    kp, conf = coco_to_joints(np.zeros((40, 17, 2)), np.ones((40, 17)))
    assert kp.shape == (40, 15, 2) and conf.shape == (40, 15)


def test_hold_last_fills_missing_and_leading_gaps():
    kp = np.zeros((6, 15, 2))
    kp[:, 0, 0] = [9, 9, 3, 4, 5, 6]
    conf = np.ones((6, 15))
    conf[[0, 1, 4], 0] = 0.0
    out = hold_last(kp, conf)
    assert out[:, 0, 0].tolist() == [3, 3, 3, 4, 4, 6]


def test_pick_person_largest_first_then_nearest():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 220, 300]])
    assert pick_person(boxes) == 1
    assert pick_person(boxes, prev_center=(5, 5)) == 0
    assert pick_person(np.zeros((0, 4))) is None


def test_person_from_result_handles_missing_and_present():
    none = SimpleNamespace(keypoints=None, boxes=None)
    assert person_from_result(none) is None
    empty = SimpleNamespace(keypoints=SimpleNamespace(xy=np.zeros((0, 17, 2)), conf=np.zeros((0, 17))),
                            boxes=SimpleNamespace(xyxy=np.zeros((0, 4))))
    assert person_from_result(empty) is None
    res = SimpleNamespace(
        keypoints=SimpleNamespace(xy=np.ones((2, 17, 2)) * [[[1]], [[2]]], conf=np.ones((2, 17)) * 0.9),
        boxes=SimpleNamespace(xyxy=np.array([[0, 0, 10, 10], [0, 0, 50, 80]])))
    kp, conf, centre = person_from_result(res)
    assert kp[0, 0] == 2 and conf[0] == pytest.approx(0.9) and centre == (25.0, 40.0)


@pytest.fixture(scope="module")
def small_model():
    split = build_split(200, NARROW, seed=21)
    return GBMDetector(seed=0).fit(split.X, split.y)


def test_analyze_track_alerts_on_a_fall_and_not_on_walking(small_model):
    rng = np.random.default_rng(31)
    fall = simulate_clip("fall_forward", NARROW, rng, t0=4.5)
    walk = simulate_clip("walk", NARROW, rng)
    kp_fall, _ = observe(fall.kp2d, fall.fps, NARROW, rng)
    kp_walk, _ = observe(walk.kp2d, walk.fps, NARROW, rng)
    a_fall = analyze_track(kp_fall, fall.fps, small_model, tau=0.5)
    a_walk = analyze_track(kp_walk, walk.fps, small_model, tau=0.5)
    assert a_fall.alerts and fall.fall_t0 - 0.5 <= a_fall.alerts[0] <= fall.fall_t1 + 3.0
    assert not a_walk.alerts
    assert len(a_fall.prob) == len(a_fall.t_end) and a_fall.peak_prob > 0.5


def test_analyze_track_rejects_short_video_and_warns_on_low_detection(small_model):
    with pytest.raises(ValueError):
        analyze_track(np.zeros((25, 15, 2)), 25.0, small_model, 0.5)
    rng = np.random.default_rng(32)
    clip = simulate_clip("walk", NARROW, rng)
    an = analyze_track(clip.kp2d, clip.fps, small_model, 0.5, detect_rate=0.5)
    assert any("only 50%" in w for w in an.warnings)
