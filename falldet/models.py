"""Detectors: a transparent rule-based baseline and a gradient-boosted model."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURE_NAMES

FI = {n: i for i, n in enumerate(FEATURE_NAMES)}


class RuleDetector:
    """Classic heuristic: fast hip drop + horizontal / low final posture."""

    name = "rule_based"

    def __init__(self, v_thr=1.2, ang_thr=50.0, ratio_thr=0.55):
        self.v_thr, self.ang_thr, self.ratio_thr = v_thr, ang_thr, ratio_thr

    def predict_proba(self, X):
        fast = X[:, FI["vy_hip_min"]] < -self.v_thr
        posture = (X[:, FI["ang_last"]] > self.ang_thr) | (X[:, FI["asp_last"]] > 0.0)
        low = X[:, FI["hip_h_ratio"]] < self.ratio_thr
        return (fast & posture & low).astype(float)


class GBMDetector:
    name = "gbm"

    def __init__(self, seed=0):
        self.clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.08, max_leaf_nodes=24, l2_regularization=1.0,
            class_weight="balanced", random_state=seed)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def importances(self, X, y, seed=0):
        from sklearn.inspection import permutation_importance
        r = permutation_importance(self.clf, X, y, n_repeats=3, random_state=seed,
                                   scoring="average_precision")
        return dict(zip(FEATURE_NAMES, r.importances_mean))
