# Model card: synthetic-trained fall detector

## Summary

A gradient-boosted classifier that scores 3-second windows of 2D body keypoints for the presence of a fall. Trained only on simulator output. A transparent rule-based baseline is provided for comparison.

## Intended use

* Research and education on synthetic data, domain randomisation and fall detection.
* Benchmarking pose-based fall detectors under camera, noise, frame-rate and occlusion shift.

## Not intended for

* Safety-critical or clinical use, or as a substitute for human supervision.
* Deployment on real people without independent validation on real data.

## Training data

Simulated clips (10 s each) of a walker user: 4 fall types and 7 non-fall types. Two regimes:

| Regime | Camera yaw | Pitch | Distance | Frame rate | Keypoint noise | Occlusion |
|---|---|---|---|---|---|---|
| Narrow-view | 0–35° | 8–22° | 4.0–5.2 m | 25 fps | 0.8–1.5 px | none |
| Domain-randomised | 0–90°, either side | 5–30° | 3.5–6.0 m | 15 / 25 / 30 fps | 0.5–4.5 px | 0–2 blocks |

No real person, image or video was used.

## Evaluation

400 fresh simulated clips per test condition (144 falls, 256 non-falls), five conditions, thresholds frozen from validation data of each model's training domain. Full tables are in the README and `results/metrics.json`.

## Known failure modes

* Controlled lowering to the floor and deep crouches end in a fall-like posture and still trigger alerts.
* Higher keypoint noise and occlusion increase false alarms.
* Hold-last-value filling of occluded joints can hide motion.
* Per-track calibration on the first second assumes the person starts upright.

## Ethical considerations

Fall detection involves vulnerable people. A missed fall can cause harm and so can alert fatigue from false alarms. Any real deployment needs consent, privacy review, validation on representative real data, and a clinical or care-home owner for the alerts.
