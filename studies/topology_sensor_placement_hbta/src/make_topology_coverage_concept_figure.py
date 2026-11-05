"""Create the introductory topology-coverage concept figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "topology_coverage_restoration" / "figures"

POSITIONS = np.asarray(
    [
        (-2.2, 0.0),
        (-1.25, 0.85),
        (-1.25, -0.85),
        (0.0, 0.0),
        (1.25, 0.85),
        (1.25, -0.85),
        (2.2, 0.0),
    ],
    dtype=float,
)
EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (3, 4),
    (3, 5),
    (4, 6),
    (5, 6),
)
DAMAGE_EDGE = 4
RESPONSE_WEIGHT = np.asarray((0.05, 0.08, 0.18, 0.13, 0.22, 0.15, 0.12, 0.07))
DISPLAY_LINEWIDTH_MIN = 1.0
DISPLAY_LINEWIDTH_MAX = 5.0

BLUE = "#3478a8"
BLUE_DARK = "#225f8a"
GREY = "#aab4be"
GREY_LIGHT = "#d8dee5"
ORANGE = "#e78a19"
RED = "#cc4535"
GREEN = "#25875f"
INK = "#26323d"


def midpoint(edge_index: int) -> np.ndarray:
    u, v = EDGES[edge_index]
    return 0.5 * (POSITIONS[u] + POSITIONS[v])


def response_linewidth(edge_index: int) -> float:
    """Min-max scale response weight for display without changing q_md."""
    weight_min = float(RESPONSE_WEIGHT.min())
    weight_span = float(RESPONSE_WEIGHT.max() - weight_min)
    relative_weight = (float(RESPONSE_WEIGHT[edge_index]) - weight_min) / weight_span
    return DISPLAY_LINEWIDTH_MIN + relative_weight * (
        DISPLAY_LINEWIDTH_MAX - DISPLAY_LINEWIDTH_MIN
    )


def draw_damage_location(axis: plt.Axes) -> None:
    centre = midpoint(DAMAGE_EDGE)
    axis.scatter(
        centre[0],
        centre[1],
        marker="X",
        s=72,
        color=RED,
        edgecolor="white",
        linewidth=0.8,
        zorder=7,
    )


def draw_nodes(axis: plt.Axes, active: set[int], restored: int | None = None) -> None:
    active_nodes = sorted(active - ({restored} if restored is not None else set()))
    if active_nodes:
        axis.scatter(
            POSITIONS[active_nodes, 0],
            POSITIONS[active_nodes, 1],
            s=42,
            color=BLUE,
            edgecolor="white",
            linewidth=0.7,
            zorder=6,
        )
    inactive_nodes = sorted(set(range(len(POSITIONS))) - active)
    if inactive_nodes:
        axis.scatter(
            POSITIONS[inactive_nodes, 0],
            POSITIONS[inactive_nodes, 1],
            s=55,
            marker="x",
            color=GREY,
            linewidth=1.7,
            zorder=6,
        )
    if restored is not None:
        axis.scatter(
            POSITIONS[restored, 0],
            POSITIONS[restored, 1],
            s=78,
            color="#fff1d6",
            edgecolor=ORANGE,
            linewidth=2.1,
            zorder=7,
        )
        axis.text(
            POSITIONS[restored, 0],
            POSITIONS[restored, 1] - 0.02,
            r"$i$",
            color="#8c4b09",
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
        )


def draw_graph(
    axis: plt.Axes,
    active: set[int],
    *,
    show_response_weight: bool = False,
    newly_active: set[int] | None = None,
    restored: int | None = None,
) -> None:
    newly_active = newly_active or set()
    for edge_index, (u, v) in enumerate(EDGES):
        is_active = u in active and v in active
        if edge_index in newly_active:
            color = ORANGE
            linewidth = response_linewidth(edge_index)
            linestyle = "-"
            zorder = 5
        elif is_active:
            color = BLUE_DARK
            linewidth = response_linewidth(edge_index) if show_response_weight else 1.8
            linestyle = "-"
            zorder = 3
        else:
            color = GREY_LIGHT
            linewidth = 1.2
            linestyle = (0, (2, 2))
            zorder = 1
        axis.plot(
            POSITIONS[[u, v], 0],
            POSITIONS[[u, v], 1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=zorder,
        )
    draw_damage_location(axis)
    draw_nodes(axis, active, restored=restored)
    axis.set_xlim(-2.55, 2.55)
    axis.set_ylim(-1.32, 1.36)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.18), constrained_layout=True)

    all_active = set(range(len(POSITIONS)))
    draw_graph(axes[0], all_active, show_response_weight=True)
    axes[0].set_title("(a) Before sensor loss", pad=5)

    active_before = {0, 1, 2, 3, 6}
    draw_graph(axes[1], active_before, show_response_weight=True)
    axes[1].set_title("(b) After sensor loss", pad=5)

    active_after = active_before | {4}
    draw_graph(
        axes[2],
        active_after,
        show_response_weight=True,
        newly_active={4, 6},
        restored=4,
    )
    axes[2].set_title(r"(c) After sensor $i$ is restored", pad=5)

    legend = (
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=6, label="active sensor"),
        Line2D([0], [0], marker="x", color=GREY, linestyle="none", markersize=6, label="out-of-service sensor"),
        Line2D([0], [0], color=BLUE_DARK, linewidth=3.0, label="thicker link = larger response weight"),
        Line2D([0], [0], color=ORANGE, linewidth=2.8, label="newly active link"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=RED, markeredgecolor="white", markersize=6, label="candidate damage location"),
    )
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=5,
        frameon=False,
        fontsize=6.9,
        handlelength=2.0,
        columnspacing=1.2,
    )

    fig.savefig(OUTPUT / "fig1_topology_coverage_concept.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "fig1_topology_coverage_concept.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
