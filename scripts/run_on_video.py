"""Run the trained fall detector on a real video.

Pose comes from Ultralytics YOLO pose (COCO-17), which is mapped to the 15-joint
layout the detector was trained on.

Usage (from the project folder):
    python scripts/run_on_video.py assets/Demo.mp4
    python scripts/run_on_video.py assets/Demo.mp4 --annotate

Needs models/gbm_wide.joblib, which scripts/run_experiments.py creates.
Outputs go to results/: video_test.json, video_test_signals.png and, with
--annotate, video_test_annotated.mp4.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from falldet.render import draw_skeleton  # noqa: E402
from falldet.video import (analyze_track, coco_to_joints, hold_last, load_bundle,  # noqa: E402
                           person_from_result)


def extract_keypoints(video, pose_model):
    """Run pose estimation on every frame. Returns (kp17, conf17, detected mask, fps)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed. Run:  pip install ultralytics")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Could not open the video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps <= 0:
        print("  warning: video reports no frame rate, assuming 25 fps")
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Loading pose model {pose_model} (downloads on first use) ...")
    pose = YOLO(pose_model)
    kp17, conf17, detected, prev = [], [], [], None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        found = person_from_result(pose(frame, verbose=False)[0], prev)
        if found is None:
            kp17.append(np.zeros((17, 2)))
            conf17.append(np.zeros(17))
            detected.append(False)
        else:
            kp, conf, prev = found
            kp17.append(kp)
            conf17.append(conf)
            detected.append(True)
        i += 1
        if i % 30 == 0:
            print(f"  pose estimation: {i}/{total if total > 0 else '?'} frames", end="\r")
    cap.release()
    print(" " * 60, end="\r")
    if not any(detected):
        sys.exit("The pose model did not find a person in any frame.")
    return np.array(kp17), np.array(conf17), np.array(detected), float(fps)


def plot_signals(an, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(an.t, an.hip_rel, color="#0072b2", lw=2)
    ax[0].set_ylabel("hip height\n(relative to upright)")
    ax[1].plot(an.t, an.torso_deg, color="#009e73", lw=2)
    ax[1].set_ylabel("torso angle (deg)")
    ax[2].plot(an.t_end, an.prob, color="#d55e00", lw=2)
    ax[2].axhline(an.tau, color="#6b7280", ls="--", lw=1)
    ax[2].text(an.t[-1], an.tau, " threshold", va="bottom", ha="right", color="#6b7280", fontsize=8)
    ax[2].set_ylabel("fall probability")
    ax[2].set_ylim(0, 1.02)
    ax[2].set_xlabel("time (s)")
    for a in an.alerts:
        for axis in ax:
            axis.axvline(a, color="#c0392b", lw=1.5, alpha=0.8)
    ax[0].set_title("Detector input and output on your video" + ("  (red line = alert)" if an.alerts else ""),
                    loc="left")
    for axis in ax:
        axis.grid(color="#e5e7eb")
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_annotated(video, kp15, an, out_path):
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), an.fps, (w, h))
    per_frame = np.zeros(len(kp15))
    for t_end, p in zip(an.t_end, an.prob):
        per_frame[int(round(t_end * an.fps)):] = p
    for i in range(len(kp15)):
        ok, frame = cap.read()
        if not ok:
            break
        draw_skeleton(frame, kp15[i])
        t = i / an.fps
        alerted = any(0 <= t - a <= 3.0 for a in an.alerts)
        cv2.rectangle(frame, (0, 0), (w, 34), (31, 25, 22), -1)
        msg = "FALL ALERT" if alerted else "monitoring"
        cv2.putText(frame, f"t={t:5.1f}s  p(fall)={per_frame[i]:.2f}  {msg}", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 240) if alerted else (240, 236, 232), 2, cv2.LINE_AA)
        if alerted:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (60, 60, 230), 6)
        writer.write(frame)
    writer.release()
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", default="assets/Demo.mp4")
    ap.add_argument("--model", default=str(ROOT / "models" / "gbm_wide.joblib"))
    ap.add_argument("--pose-model", default="yolo11n-pose.pt", help="any Ultralytics COCO-17 pose weights")
    ap.add_argument("--tau", type=float, default=None, help="alert threshold (default: the one saved with the model)")
    ap.add_argument("--min-conf", type=float, default=0.3, help="keypoints below this are treated as missing")
    ap.add_argument("--annotate", action="store_true", help="also write results/video_test_annotated.mp4")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()

    video = Path(a.video)
    if not video.is_absolute() and not video.exists():
        video = ROOT / video
    if not video.exists():
        sys.exit(f"Video not found: {a.video}")
    if not Path(a.model).exists():
        sys.exit(f"Model not found: {a.model}\nRun first:  python scripts/run_experiments.py")
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    kp17, conf17, detected, fps = extract_keypoints(video, a.pose_model)
    kp15, c15 = coco_to_joints(kp17, conf17)
    kp15 = hold_last(kp15, c15, a.min_conf)

    model, tau = load_bundle(a.model)
    tau = tau if a.tau is None else a.tau
    try:
        an = analyze_track(kp15, fps, model, tau, detect_rate=float(detected.mean()))
    except ValueError as e:
        sys.exit(str(e))

    print(f"\nVideo: {video.name}   {len(kp15)} frames at {fps:.1f} fps ({len(kp15) / fps:.1f} s)")
    print(f"Person found in {detected.mean():.0%} of frames")
    print(f"Alert threshold: {tau:.2f}")
    print(f"Peak fall probability: {an.peak_prob:.2f} at t = {an.peak_time:.1f} s")
    if an.alerts:
        print("FALL ALERT at: " + ", ".join(f"{t:.1f} s" for t in an.alerts))
    else:
        print("No fall alert raised.")
    for w in an.warnings:
        print("WARNING: " + w)
    print("The script does not know what really happens in the video. Compare the result with what you see.")

    (out / "video_test.json").write_text(json.dumps(dict(
        video=video.name, fps=fps, frames=len(kp15), detect_rate=float(detected.mean()), tau=tau,
        alerts=an.alerts, peak_prob=an.peak_prob, peak_time=an.peak_time, warnings=an.warnings,
        t_end=np.round(an.t_end, 3).tolist(), prob=np.round(an.prob, 4).tolist()), indent=2))
    plot_signals(an, out / "video_test_signals.png")
    print(f"Saved: {out / 'video_test.json'}")
    print(f"Saved: {out / 'video_test_signals.png'}")
    if a.annotate:
        write_annotated(video, kp15, an, out / "video_test_annotated.mp4")
        print(f"Saved: {out / 'video_test_annotated.mp4'}")


if __name__ == "__main__":
    main()
