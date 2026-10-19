"""Figures for the README: scenario signals, robustness comparison, per-scenario alerts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from falldet.config import NARROW  # noqa: E402
from falldet.features import calibrate, frame_signals  # noqa: E402
from falldet.simulate import observe, simulate_clip  # noqa: E402

INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"
C_RULE, C_NARROW, C_WIDE = "#9aa5b1", "#d55e00", "#0072b2"   # Okabe-Ito based

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 150,
})

DOMAIN_LABELS = {
    "in_distribution": "In-distribution", "frontal_view": "Frontal view",
    "noisy_occluded": "Noisy + occluded", "low_fps_12": "Low FPS (12)",
    "deployment_mix": "Deployment mix",
}


def fig_signals(out):
    rng = np.random.default_rng(3)
    scen = [("fall_forward", "Forward fall", C_WIDE), ("sit_down", "Sit down", C_RULE),
            ("bend_pick", "Bend to pick up", "#009e73"), ("controlled_lower", "Controlled lowering", "#e69f00")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    for sc, label, col in scen:
        clip = simulate_clip(sc, NARROW, rng, t0=4.0)
        kp, _ = observe(clip.kp2d, clip.fps, NARROW, rng)
        cal = calibrate(kp, clip.fps)
        sig = frame_signals(kp, cal, clip.fps)
        axes[0].plot(clip.t, sig["hip_h"] / cal.hip_h0, color=col, lw=2, label=label)
        axes[1].plot(clip.t, sig["ang"], color=col, lw=2, label=label)
        axes[0].text(clip.t[-1] + 0.1, (sig["hip_h"] / cal.hip_h0)[-1], label, color=INK, va="center", fontsize=8.5)
    axes[0].set_title("Hip height (relative to upright)", loc="left", color=INK, fontsize=11)
    axes[1].set_title("Torso angle from vertical (deg)", loc="left", color=INK, fontsize=11)
    for ax in axes:
        ax.set_xlabel("time (s)")
    axes[0].set_xlim(0, 12.8)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_robustness(res, out):
    doms = list(DOMAIN_LABELS)
    models = [("rule_based", "Rule-based baseline", C_RULE),
              ("gbm_narrow_train", "GBM, narrow-view synthetic", C_NARROW),
              ("gbm_wide_train", "GBM, domain-randomised synthetic", C_WIDE)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    x = np.arange(len(doms))
    w = 0.26
    for ax, key, title in [(axes[0], "sensitivity", "Fall sensitivity (higher is better)"),
                           (axes[1], "fa_clip_rate", "False-alarm rate on non-fall clips (lower is better)")]:
        for k, (m, lab, col) in enumerate(models):
            vals = [res[m][d][key] * 100 for d in doms]
            ax.bar(x + (k - 1) * w, vals, w * 0.92, color=col, label=lab)
            for xi, v in zip(x + (k - 1) * w, vals):
                ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([DOMAIN_LABELS[d] for d in doms], fontsize=8.5)
        ax.set_ylim(0, 112 if key == "sensitivity" else 40)
        ax.set_ylabel("%")
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.grid(axis="x", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_scenarios(res, out):
    order = ["fall_forward", "fall_backward", "fall_side", "slump",
             "walk", "stand", "sit_down", "bend_pick", "crouch_pick", "stumble_recover", "controlled_lower"]
    names = {"fall_forward": "Fall forward", "fall_backward": "Fall backward", "fall_side": "Fall sideways",
             "slump": "Slow slump", "walk": "Walk", "stand": "Stand", "sit_down": "Sit down",
             "bend_pick": "Bend / pick up", "crouch_pick": "Crouch", "stumble_recover": "Stumble + recover",
             "controlled_lower": "Controlled lowering"}
    d = "deployment_mix"
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    y = np.arange(len(order))[::-1]
    for k, (m, lab, col) in enumerate([("gbm_narrow_train", "narrow-view", C_NARROW),
                                       ("gbm_wide_train", "domain-randomised", C_WIDE)]):
        v = [res[m][d]["per_scenario"][s] * 100 for s in order]
        ax.barh(y + (0.2 if k == 0 else -0.2), v, 0.38, color=col, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([names[s] for s in order], fontsize=9)
    ax.axhline(6.5 - 0.0, color=GRID, lw=1.2)
    ax.set_xlabel("% of clips that raise an alert")
    ax.set_title("Alert rate per scenario, deployment-mix test set", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    ax.grid(axis="y", visible=False)
    ax.text(101, y[0] + 0.0, "falls: should alert", fontsize=8, color=MUTED, va="center")
    ax.text(101, y[-1] + 0.0, "ADL: should not", fontsize=8, color=MUTED, va="center")
    ax.set_xlim(0, 125)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    res = json.loads((ROOT / "results" / "metrics.json").read_text())["results"]
    fig_signals(ROOT / "assets" / "signals.png")
    fig_robustness(res, ROOT / "assets" / "robustness.png")
    fig_scenarios(res, ROOT / "assets" / "per_scenario.png")
    print("figures written to", ROOT / "assets")
