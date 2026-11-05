"""Draft of Figure 7 extended to all damage states (R1.5): modelled q vs measured R, per state.

Same plan view and line styling as the submitted Figure 7 (make_physical_consistency_figure.py).
Main figure: DS1, DS2, DS3, DS4, DS8.  Supplementary figure: DS5, DS6, DS7 (lateral bracing).

Run from SHM-R1:  python analysis_R1/r1_make_fig7_all_states.py
Writes only to analysis_R1/outputs/fig7_all_states/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "topology_sensor_placement_hbta"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(STUDY / "src"))
sys.path.insert(0, str(ROOT / "analysis_R1"))
from r1_multistate_link_level import measured_responses, response_weights  # noqa: E402

OUT = ROOT / "analysis_R1" / "outputs" / "fig7_all_states"
COLORS = {"navy": "#255B78", "orange": "#E08B2C", "red": "#C53B32", "blue": "#2E6F9E",
          "grid": "#E8EDF0", "light": "#d2d9df"}
SHORT = {"DS1": "stringer--floor-beam connection", "DS2": "stringer--floor-beam connections",
         "DS3": "stringer cross beam", "DS4": "stringer cross beams",
         "DS5": "lateral-bracing connection", "DS6": "lateral-bracing connection",
         "DS7": "lateral-bracing connections", "DS8": "floor-beam--main-member connection"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.0, "axes.titlesize": 7.2,
                         "axes.linewidth": 0.7, "savefig.dpi": 360, "pdf.fonttype": 42})


def draw(ax, pos, edges, values, kind, damage_pts, label):
    for u, v in edges:
        ax.plot(pos[[u, v], 0], pos[[u, v], 1], color=COLORS["light"], lw=0.6, alpha=0.85, zorder=1)
    if kind == "q":
        floor, vmax = 1e-4, float(values.max())
        norm = np.clip((np.log10(np.maximum(values, floor)) - np.log10(floor))
                       / (np.log10(vmax) - np.log10(floor)), 0, 1)
        show, color = values >= floor, COLORS["orange"]
    else:
        vmax = float(values.max())
        norm = np.clip(np.log10(1 + values) / np.log10(1 + vmax), 0, 1)
        show, color = values > 0, COLORS["blue"]
    for m, (u, v) in enumerate(edges):
        if show[m]:
            ax.plot(pos[[u, v], 0], pos[[u, v], 1], color=color, lw=0.6 + 2.6 * norm[m],
                    alpha=0.35 + 0.65 * norm[m], zorder=2, solid_capstyle="round")
    ax.scatter(pos[:, 0], pos[:, 1], s=6, color=COLORS["navy"], edgecolor="white", lw=0.3, zorder=3)
    ax.scatter(damage_pts[:, 0], damage_pts[:, 1], marker="X", s=34, color=COLORS["red"],
               edgecolor="white", lw=0.5, zorder=6)
    ax.set_xlim(-17, 17)
    ax.set_ylim(-2.7, 2.7)
    ax.set_xticks([-14, -7, 0, 7, 14])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=5.8, length=2)
    ax.grid(color=COLORS["grid"], lw=0.4, zorder=0)
    ax.set_title(label, loc="left")


def figure(states, mapping, R, pos, edges, B, w, rel, red, alph, key, fname):
    fig, axes = plt.subplots(len(states), 2, figsize=(7.2, 1.02 * len(states) + 0.25),
                             constrained_layout=True, squeeze=False, sharex=True, sharey=True)
    for row, s in enumerate(states):
        st = next(x for x in mapping["states"] if x["state"] == s)
        D = [key[e] for e in st["springs"]]
        q = response_weights(B, w, rel, [D], red, alph)[:, 0]
        r = R[s]
        rho = spearmanr(q, r).statistic
        pts = (pos[edges[D, 0], :2] + pos[edges[D, 1], :2]) / 2
        draw(axes[row, 0], pos, edges, q, "q", pts, f"({chr(97 + 2 * row)}) {s} model: {SHORT[s]}")
        draw(axes[row, 1], pos, edges, r, "R", pts,
             f"({chr(98 + 2 * row)}) {s} measured  (Spearman $\\rho$ = {rho:.2f})")
    for ax in axes[-1]:
        ax.set_xlabel("x (m)", fontsize=6.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)", fontsize=6.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{fname}.png")
    fig.savefig(OUT / f"{fname}.pdf")
    plt.close(fig)


def main() -> None:
    style()
    config = json.loads((STUDY / "configs" / "virtual_damage_pareto.json").read_text(encoding="utf-8"))
    mapping = json.loads((ROOT / "analysis_R1" / "r1_damage_mapping.json").read_text(encoding="utf-8"))
    from hbta_pipeline.spatial_topology import build_hbta_spatial_graph
    d = np.load(ROOT / "data" / "derived" / "hbta" / "edge_spectra_multicondition_10s.npz")
    names, pos = d["sensor_names"].astype(str), d["sensor_positions"].astype(float)
    bank = np.load(STUDY / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz")
    rel, edges = bank["reliability"].astype(float), bank["edges"]
    g = build_hbta_spatial_graph(names, pos)
    N, M = len(names), len(edges)
    B = np.zeros((M, N)); B[np.arange(M), edges[:, 0]] = 1; B[np.arange(M), edges[:, 1]] = -1
    w = g.weights * np.sqrt(rel); w /= w.mean()
    key = {}
    for m, (u, v) in enumerate(edges):
        key[f"{names[u]}--{names[v]}"] = key[f"{names[v]}--{names[u]}"] = m
    R = measured_responses(config)
    args = (mapping, R, pos, edges, B, w, rel, config["damage_weight_reductions"],
            config["resolvent_alpha_factors"], key)
    figure(["DS1", "DS2", "DS3", "DS4", "DS8"], *args, "fig7_all_states_main_draft")
    figure(["DS5", "DS6", "DS7"], *args, "fig7_bracing_states_supplementary_draft")
    print("saved", OUT)


if __name__ == "__main__":
    main()
