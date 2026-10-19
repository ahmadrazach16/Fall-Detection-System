"""Train detectors on narrow / randomised synthetic data and stress-test them.

Usage:  python scripts/run_experiments.py [--n-train 1500] [--n-test 400] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from falldet.config import NARROW, TEST_DOMAINS, WIDE  # noqa: E402
from falldet.evaluate import pick_threshold, summarise  # noqa: E402
from falldet.features import FEATURE_NAMES  # noqa: E402
from falldet.models import GBMDetector, RuleDetector  # noqa: E402
from falldet.pipeline import build_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    t0 = time.time()

    print("building training / validation splits ...")
    train = {d.name: build_split(a.n_train, d, a.seed + i) for i, d in enumerate([NARROW, WIDE])}
    val = {d.name: build_split(a.n_val, d, a.seed + 100 + i) for i, d in enumerate([NARROW, WIDE])}
    print("building test splits ...")
    tests = [build_split(a.n_test, d, a.seed + 1000 + i) for i, d in enumerate(TEST_DOMAINS)]
    print(f"  data ready in {time.time() - t0:.0f}s  "
          f"(train windows: narrow={len(train['narrow'].y)}, wide={len(train['wide'].y)})")

    models = {}
    models["rule_based"] = (RuleDetector(), None)
    for key, label in [("narrow", "gbm_narrow_train"), ("wide", "gbm_wide_train")]:
        m = GBMDetector(seed=a.seed).fit(train[key].X, train[key].y)
        tau = pick_threshold(val[key], m.predict_proba(val[key].X), max_fa_clip_rate=0.03)
        models[label] = (m, tau)
        print(f"  trained {label}: tau={tau:.2f}")

    results = {}
    for name, (m, tau) in models.items():
        results[name] = {}
        for sp in tests:
            p = m.predict_proba(sp.X)
            s = summarise(sp, p, 0.5 if tau is None else tau)
            s["tau"] = 0.5 if tau is None else tau
            results[name][sp.domain] = s
            print(f"  {name:18s} {sp.domain:16s} sens={s['sensitivity']:.3f} "
                  f"FA/clip={s['fa_clip_rate']:.3f} AUC={s['window_auc']:.3f} lat={s['median_latency_s']:.2f}s")

    # feature importance for the wide-trained model (what drives the decision?)
    imp = models["gbm_wide_train"][0].importances(val["wide"].X, val["wide"].y, seed=a.seed)
    imp = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

    meta = dict(seed=a.seed, n_train=a.n_train, n_val=a.n_val, n_test=a.n_test,
                n_features=len(FEATURE_NAMES), feature_names=FEATURE_NAMES,
                runtime_s=round(time.time() - t0, 1))
    (out / "metrics.json").write_text(json.dumps(
        dict(meta=meta, results=results, importance=imp), indent=2, default=float))
    joblib.dump(dict(model=models["gbm_wide_train"][0], tau=models["gbm_wide_train"][1],
                     feature_names=FEATURE_NAMES), ROOT / "models" / "gbm_wide.joblib")
    joblib.dump(dict(model=models["gbm_narrow_train"][0], tau=models["gbm_narrow_train"][1],
                     feature_names=FEATURE_NAMES), ROOT / "models" / "gbm_narrow.joblib")
    print(f"done in {time.time() - t0:.0f}s -> {out/'metrics.json'}")


if __name__ == "__main__":
    main()
