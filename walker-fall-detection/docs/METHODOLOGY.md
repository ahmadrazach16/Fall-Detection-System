# Methodology notes

## Simulator

Local posture is computed from joint angles in a body frame: torso flexion, and hip and knee flexion for each leg. Walking adds sinusoidal hip swing and knee flexion damped by a gait weight, which falls to zero when the person stops.

Falls, stumbles and controlled lowering add a rigid rotation `Theta(t)` of the whole body about the ankle midpoint, with fall direction `phi`. For falls:

* loss of balance: `Theta` rises to 0.15–0.30 rad over 0.5 s,
* descent: `Theta` follows an accelerating quadratic profile to 1.3–1.6 rad over 0.45–0.95 s (slump: 1.1–1.8 s, ending 1.0–1.5 rad with knee collapse),
* impact: small damped oscillation,
* ground contact: the body is lifted so no joint is below the floor.

Hands switch from the walker handles to hanging arms as `g(t)` rises, and elbows come from two-link IK. The walker tips with a gain depending on fall direction (0.9 forward, 0.5 sideways, 0 backward).

## Labels

* Clip label: contains a fall.
* Window label: positive if the window ends between 60% of the descent and 2 s after impact (so the whole descent is inside the window). Windows that contain only the aftermath, and slow lowering, are negative. The detector therefore detects the *event*, not the lying state.

## Metrics

* **Sensitivity**: fall clips with an alert between loss of balance and 3 s after impact.
* **False-alarm rate**: non-fall clips with any alert. Alerts before loss of balance in a fall clip also count as false.
* **False alerts per hour**: distinct alert events divided by total clip time (clips are 10 s and contain scripted events, so this is only comparable between models, not to real life).
* **Latency**: first valid alert time minus descent start.
* Intervals are 95% Wilson.

## Threshold policy

For each GBM, the threshold is the smallest value giving ≤3% false-alarm clips on a validation set drawn from that model's own training domain. It is then frozen for all test domains. This mimics deployment, where you cannot retune on a distribution you have not seen.

## Known weaknesses of the design

* Normalisation uses a calibration at track start; large depth changes (walking toward the camera) distort it.
* Hold-last-value fill makes occluded joints look still, which can hide motion.
* Window features see only 3 s of context.
