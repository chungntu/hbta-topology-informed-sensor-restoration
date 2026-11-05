"""Draft figure for revised Section 3.1: the sensor graph as a spring model (R1.1).

(a) a fragment of the HBTA elevation graph (Fig. 2a style),
(b) the same fragment as a spring model with vertical DOFs and ground springs.

Revision R1, Section 3.1 (new figure).  Run from the study folder: python src/make_r1_spring_model_figure.py
Writes only to outputs/R1_physical_consistency/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

STUDY_ROOT = Path(__file__).resolve().parents[1]
ROOT = STUDY_ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hbta_pipeline.spatial_topology import build_hbta_spatial_graph  # noqa: E402

OUT = STUDY_ROOT / "outputs" / "R1_physical_consistency" / "figures"

COLORS = {"navy": "#1F4E79", "blue": "#3B82B6", "orange": "#E69F00",
          "purple": "#8F63A8", "red": "#C74440", "grey": "#6B7280"}
REL_STYLE = {
    "truss_diagonal": (COLORS["blue"], "-", 1.8, "Truss diagonal"),
    "truss_chord": (COLORS["navy"], "-", 2.2, "Truss chord/path"),
    "deck_longitudinal": (COLORS["orange"], "-", 1.7, "Deck longitudinal path"),
    "deck_truss_interface": (COLORS["purple"], "--", 1.8, "Deck--truss interface"),
}
FRAGMENT = ["AG01", "AG02", "AG03", "AL04", "AL08", "AL12"]
DAMAGED = ("AL04", "AL08")          # spring shown as weakened in panel (b)
UNINSTRUMENTED_JOINT = (-10.5, 1.2)  # bottom-chord joint on the AG01--AG03 path (schematic z)
# schematic elevations: real x, vertical spacing between deck and truss exaggerated for legibility
SCHEMATIC_Z = {"AG02": 4.1, "AG01": 1.2, "AG03": 1.2, "AL04": -0.3, "AL08": -0.3, "AL12": -0.3}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.0, "axes.titlesize": 8.5,
        "axes.edgecolor": "#374151", "axes.linewidth": 0.8, "legend.fontsize": 7.0,
        "figure.dpi": 160, "savefig.dpi": 600, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans",
    })


def zigzag(p0, p1, coils=4, amp=0.17, lead=None, coil_len=1.1, center=0.5):
    """Spring polyline: straight leads and a compact zigzag in the middle."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    length = np.linalg.norm(v)
    t, nrm = v / length, np.array([-v[1], v[0]]) / length
    cl = min(coil_len, 0.6 * length)
    a = center * length - cl / 2 if lead is None else lead
    s = np.linspace(a, a + cl if lead is None else length - a, 2 * coils + 1)
    offs = np.zeros_like(s)
    offs[1:-1:2], offs[2:-1:2] = amp, -amp
    pts = [p0] + [p0 + si * t + oi * nrm for si, oi in zip(s, offs)] + [p1]
    return np.array(pts)


def ground_symbol(ax, x, y, w=0.36):
    ax.plot([x - w / 2, x + w / 2], [y, y], color="k", lw=1.0, zorder=2)
    for k in np.linspace(-w / 2, w / 2, 5):
        ax.plot([x + k, x + k - 0.09], [y, y - 0.12], color="k", lw=0.7, zorder=2)


def load_fragment():
    d = np.load(ROOT / "data" / "derived" / "hbta" / "edge_spectra_multicondition_10s.npz")
    names = d["sensor_names"].astype(str)
    pos = d["sensor_positions"].astype(float)
    g = build_hbta_spatial_graph(names, pos)
    idx = {n: i for i, n in enumerate(names)}
    xz = {n: np.array([pos[idx[n]][0], SCHEMATIC_Z[n]]) for n in FRAGMENT}
    links = []
    for m, (u, v) in enumerate(g.edges):
        if names[u] in xz and names[v] in xz:
            links.append((names[u], names[v], g.relation_names[g.relation_ids[m]]))
    return xz, links


def setup_axis(ax, title):
    ax.set_title(title, loc="left")
    ax.set_aspect("equal")
    ax.set_xlim(-16.9, -5.8)
    ax.set_ylim(-1.45, 4.85)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_nodes(ax, xz, labels=True, dy=0.0):
    for n, (x, z) in xz.items():
        is_ag = n.startswith("AG")
        ax.scatter(x, z + dy, s=46, marker="o" if is_ag else "s", zorder=5,
                   color="white", edgecolor=COLORS["navy"] if is_ag else COLORS["orange"],
                   linewidth=1.4)
        if labels:
            ox, oz = {"AG02": (0.0, 0.38), "AG01": (-0.62, 0.22), "AG03": (0.62, 0.22)}.get(
                n, (0.0, -0.45))
            ax.text(x + ox, z + oz, n, ha="center", va="center", fontsize=7)


LINK_LABEL = {("AG01", "AG02"): ("diagonal", -1.05, 0.05), ("AG01", "AG03"): ("chord path", 0.0, 0.0),
              ("AL04", "AL08"): ("deck path", 0.0, -0.32), ("AG01", "AL04"): ("interface", -1.0, 0.0)}


def panel_a(ax, xz, links):
    setup_axis(ax, "(a) Sensor graph (fragment)")
    for u, v, rel in links:
        c, ls, lw, _ = REL_STYLE[rel]
        ax.plot(*zip(xz[u], xz[v]), color=c, ls=ls, lw=lw, zorder=3)
        if (u, v) in LINK_LABEL:
            lab, dx, dz = LINK_LABEL[(u, v)]
            mx, mz = (xz[u] + xz[v]) / 2
            if lab == "chord path":
                mx, mz = -12.4, 1.42
            ax.text(mx + dx, mz + dz, lab, fontsize=6.5, color=c, ha="center", va="center")
    jx, jz = UNINSTRUMENTED_JOINT
    ax.scatter(jx, jz, s=26, color=COLORS["grey"], marker="x", zorder=6, linewidth=1.2)
    ax.text(jx, 0.55, "joint without\nsensor", fontsize=6.5, ha="center", va="center",
            color=COLORS["grey"])
    draw_nodes(ax, xz)


def panel_b(ax, xz, links):
    setup_axis(ax, "(b) Equivalent spring model")
    for u, v, rel in links:
        damaged = {u, v} == set(DAMAGED)
        pts = zigzag(xz[u], xz[v], center=0.72 if rel == "truss_chord" else 0.5)
        ax.plot(pts[:, 0], pts[:, 1], color=COLORS["red"] if damaged else "#374151",
                lw=1.6 if damaged else 1.1, ls="-", zorder=3)
    jx, jz = UNINSTRUMENTED_JOINT
    ax.scatter(jx, jz, s=22, color=COLORS["grey"], marker="x", zorder=6, linewidth=1.0)
    ax.text(jx, 0.55, "one spring for the\npath through the joint", fontsize=6.3,
            color=COLORS["grey"], ha="center", va="center")
    a, b = xz["AG01"], xz["AG02"]
    ax.text((a[0] + b[0]) / 2 - 0.55, (a[1] + b[1]) / 2 + 0.15, r"$w_m$", fontsize=8.5,
            ha="center")
    mid = (xz[DAMAGED[0]] + xz[DAMAGED[1]]) / 2
    ax.text(mid[0], mid[1] - 0.42, r"$(1-\eta)\,w_d$", fontsize=8, color=COLORS["red"],
            ha="center", va="center")
    for n, (x, z) in xz.items():
        top = (x, z)
        bot = (x, z - 0.85)
        pts = zigzag(top, bot, amp=0.09, coil_len=0.45)
        ax.plot(pts[:, 0], pts[:, 1], color=COLORS["grey"], lw=0.8, zorder=2)
        ground_symbol(ax, bot[0], bot[1], w=0.34)
    x3, z3 = xz["AL12"]
    ax.text(x3 + 0.3, z3 - 0.55, r"$\alpha$", fontsize=8.5, color=COLORS["grey"], va="center")
    draw_nodes(ax, xz, labels=False)


def main() -> None:
    style()
    xz, links = load_fragment()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"wspace": 0.04})
    panel_a(ax_a, xz, links)
    panel_b(ax_b, xz, links)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_spring_model.{ext}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
