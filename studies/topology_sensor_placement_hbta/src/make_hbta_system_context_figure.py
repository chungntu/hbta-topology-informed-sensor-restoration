"""Create a compact HBTA sensor-topology figure for the restoration paper.

The geometry and relation definitions are reused from the HBTA topology study,
but the elevation and plan views are arranged side by side to fit a two-column
paper without leaving a large blank area around the figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLORS = {
    "navy": "#1F4E79",
    "blue": "#3B82B6",
    "orange": "#E69F00",
    "green": "#2A9D8F",
    "purple": "#8F63A8",
    "red": "#C74440",
}

RELATION_STYLE = {
    "truss_diagonal": (COLORS["blue"], "-", 1.8),
    "truss_chord": (COLORS["navy"], "-", 2.2),
    "deck_longitudinal": (COLORS["orange"], "-", 1.7),
    "deck_transverse": (COLORS["green"], "-", 1.4),
    "deck_truss_interface": (COLORS["purple"], "--", 1.8),
    "lateral_bracing": (COLORS["red"], "-", 2.5),
}

RELATION_LABEL = {
    "truss_diagonal": "Truss diagonal",
    "truss_chord": "Truss chord/path",
    "deck_longitudinal": "Deck longitudinal/stringer path",
    "deck_transverse": "Cross-girder path",
    "deck_truss_interface": "Deck--truss interface",
    "lateral_bracing": "Lateral bracing",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_figure(source_npz: Path, repository: Path, output_stem: Path) -> None:
    sys.path.insert(0, str(repository))
    from hbta_pipeline.spatial_topology import build_hbta_spatial_graph

    with np.load(source_npz) as data:
        names = data["sensor_names"].astype(str)
        positions = data["sensor_positions"].astype(float)
    graph = build_hbta_spatial_graph(names, positions)

    fig, (ax_side, ax_plan) = plt.subplots(
        1,
        2,
        figsize=(10.6, 3.25),
        gridspec_kw={"wspace": 0.12},
    )

    for relation_id, relation in enumerate(graph.relation_names):
        color, linestyle, width = RELATION_STYLE[relation]
        selected = graph.edges[graph.relation_ids == relation_id]
        for left, right in selected:
            p0, p1 = positions[left], positions[right]
            ax_side.plot(
                [p0[0], p1[0]],
                [p0[2], p1[2]],
                color=color,
                linestyle=linestyle,
                linewidth=0.92 * width,
                alpha=0.86,
                zorder=1,
            )
            ax_plan.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=color,
                linestyle=linestyle,
                linewidth=0.92 * width,
                alpha=0.86,
                zorder=1,
            )

    truss = np.array([name.startswith("AG") for name in names])
    deck = ~truss
    for axis, coordinate in ((ax_side, 2), (ax_plan, 1)):
        axis.scatter(
            positions[deck, 0],
            positions[deck, coordinate],
            s=17,
            color="white",
            edgecolor=COLORS["orange"],
            linewidth=1.0,
            marker="s",
            zorder=3,
        )
        axis.scatter(
            positions[truss, 0],
            positions[truss, coordinate],
            s=21,
            color="white",
            edgecolor=COLORS["navy"],
            linewidth=1.05,
            marker="o",
            zorder=4,
        )
        axis.grid(True, color="#D8DEE6", linewidth=0.55, alpha=0.75)
        axis.set_axisbelow(True)
        axis.set_xlabel("Longitudinal coordinate, x (m)")

    ax_side.set_title("(a) Elevation: truss and deck paths")
    ax_side.set_ylabel("Elevation, z (m)")
    ax_side.set_ylim(-0.35, 5.55)

    ax_plan.set_title("(b) Plan: deck grid and lateral connectivity")
    ax_plan.set_ylabel("Transverse coordinate, y (m)")
    ax_plan.set_ylim(-2.8, 2.8)

    relation_handles = [
        Line2D(
            [0],
            [0],
            color=RELATION_STYLE[relation][0],
            linestyle=RELATION_STYLE[relation][1],
            linewidth=0.92 * RELATION_STYLE[relation][2],
            label=RELATION_LABEL[relation],
        )
        for relation in graph.relation_names
    ]
    sensor_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["navy"],
            markeredgewidth=1.05,
            label="Truss sensor (AG)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["orange"],
            markeredgewidth=1.0,
            label="Deck sensor (AL)",
        ),
    ]
    fig.legend(
        handles=relation_handles + sensor_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.91, bottom=0.27)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    build_figure(args.source_npz, args.repository, args.output_stem)


if __name__ == "__main__":
    main()
