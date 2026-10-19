# Changelog

## 0.1.0

* 3D biomechanical simulator for walker users: gait, posture, tipping falls, impact, ground contact, walker tipping, hand release.
* Four fall types (forward, backward, sideways, slow slump) and seven non-fall types, including hard negatives (sitting, bending, crouching, stumble-and-recover, controlled lowering).
* Camera model with yaw, pitch and distance; pose-estimator-style noise, occlusion and dropout.
* 26 causal window features from 2D keypoints.
* Rule-based baseline and gradient-boosted detector.
* Clip-level evaluation with Wilson intervals, latency and per-scenario alert rates.
* Robustness benchmark: narrow-view vs domain-randomised training on five test conditions.
* Pixel renderer, optical-flow motion energy and a still-frame contact sheet script.
* 25 unit tests and a CI workflow.
