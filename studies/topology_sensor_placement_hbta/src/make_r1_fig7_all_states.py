"""Revision R1: Figure 7 extended to all damage states.

Reads outputs/R1_physical_consistency/link_level_all_states.{npz,json} (make_r1_link_level_all_states.py)
and draws, for each state, the modelled response weights q (left) and the measured link response R
(right) on the plan view of the bridge, with the same line styling as the submitted Figure 7.

Outputs (new files only):
  outputs/R1_physical_consistency/figures/fig7_link_level_consistency_all_states.{pdf,png}  (DS1-DS4, DS8)
  outputs/R1_physical_consistency/figures/figS1_bracing_states.{pdf,png}                    (DS5-DS7)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STUDY_ROOT = Path(__file__).resolve().parents[1]
DATA = STUDY_ROOT / "outputs" / "R1_physical_consistency"
FIGURES = DATA / "figures"
MAPPING_PATH = STUDY_ROOT / "configs" / "r1_damage_mapping.json"
COLORS = {"navy": "#255B78", "orange": "#E08B2C", "red": "#C53B32", "blue": "#2E6F9E",
          "grid": "#E8EDF0", "light": "#d2d9df"}
SHORT = {"DS1": "stringer--floor-beam connection", "DS2": "stringer--floor-beam connections",
         "DS3": "stringer cross beam", "DS4": "stringer cross beams",
         "DS5": "lateral-bracing connection", "DS6": "lateral-bracing connection",
         "DS7": "lateral-bracing connections", "DS8": "floor-beam--main-member connection"}


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.0, "axes.titlesize": 7.2,
                         "axes.linewidth": 0.7, "savefig.dpi": 360, "pdf.fonttype": 42,
                         "ps.fonttype": 42})


def draw(ax, pos, edges, values, kind, damage_points, title):
    for u, v in edges:
        ax.plot(pos[[u, v], 0], pos[[u, v], 1], color=COLORS["light"], lw=0.6, alpha=0.85, zorder=1)
    if kind == "q":
        floor, vmax = 1e-4, float(values.max())
        norm = np.clip((np.log10(np.maximum(values, floor)) - np.log10(floor))
                       / (np.log10(vmax) - np.log10(floor)), 0.0, 1.0)
        shown, color = values >= floor, COLORS["orange"]
    else:
        vmax = float(values.max())
        norm = np.clip(np.log10(1.0 + values) / np.log10(1.0 + vmax), 0.0, 1.0)
        shown, color = values > 0, COLORS["blue"]
    for m, (u, v) in enumerate(edges):
        if shown[m]:
            ax.plot(pos[[u, v], 0], pos[[u, v], 1], color=color, lw=0.6 + 2.6 * norm[m],
                    alpha=0.35 + 0.65 * norm[m], zorder=2, solid_capstyle="round")
    ax.scatter(pos[:, 0], pos[:, 1], s=6, color=COLORS["navy"], edgecolor="white", lw=0.3, zorder=3)
    ax.scatter(damage_points[:, 0], damage_points[:, 1], marker="X", s=34, color=COLORS["red"],
               edgecolor="white", lw=0.5, zorder=6)
    ax.set_xlim(-17, 17)
    ax.set_ylim(-2.7, 2.7)
    ax.set_xticks([-14, -7, 0, 7, 14])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=5.8, length=2)
    ax.grid(color=COLORS["grid"], lw=0.4, zorder=0)
    ax.set_title(title, loc="left")


def figure(states, data, rows, mapping, stem):
    names = data["sensor_names"].astype(str)
    pos, edges = data["positions"], data["edges"]
    state_list = list(data["states"].astype(str))
    index = {}
    for m, (u, v) in enumerate(edges):
        index[f"{names[u]}--{names[v]}"] = index[f"{names[v]}--{names[u]}"] = m
    rho = {r["state"]: r["spearman_rho"] for r in rows if r["variant"] == "primary"}
    fig, axes = plt.subplots(len(states), 2, figsize=(7.2, 1.02 * len(states) + 0.25),
                             constrained_layout=True, squeeze=False, sharex=True, sharey=True)
    for row, state in enumerate(states):
        k = state_list.index(state)
        springs = next(s for s in mapping["states"] if s["state"] == state)["springs"]
        damaged = [index[s] for s in springs]
        points = (pos[edges[damaged, 0], :2] + pos[edges[damaged, 1], :2]) / 2
        draw(axes[row, 0], pos, edges, data["q"][k], "q", points,
             f"({chr(97 + 2 * row)}) {state} model: {SHORT[state]}")
        draw(axes[row, 1], pos, edges, data["R"][k], "R", points,
             f"({chr(98 + 2 * row)}) {state} measured (Spearman $\\rho$ = {rho[state]:.2f})")
    for ax in axes[-1]:
        ax.set_xlabel("Longitudinal coordinate, x (m)", fontsize=6.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)", fontsize=6.5)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}")
    plt.close(fig)


def main() -> None:
    style()
    data = np.load(DATA / "link_level_all_states.npz", allow_pickle=False)
    rows = json.loads((DATA / "link_level_all_states.json").read_text(encoding="utf-8"))["rows"]
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    figure(["DS1", "DS2", "DS3", "DS4", "DS8"], data, rows, mapping, "fig7_link_level_consistency_all_states")
    figure(["DS5", "DS6", "DS7"], data, rows, mapping, "figS1_bracing_states")
    print("saved figures in", FIGURES)


if __name__ == "__main__":
    main()
