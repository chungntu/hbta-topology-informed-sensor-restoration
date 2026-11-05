"""Create publication figures for topology-coverage restoration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "topology_coverage_restoration_10sensor"
FIGURES = OUTPUT / "figures"

BLUE = "#3274a1"
GREEN = "#2a8f68"
ORANGE = "#e58b26"
RED = "#cf4436"
GRAY = "#aeb8c2"
DARK = "#25313c"
CASE_COLORS = {
    "deck_row_10": "#3274a1",
    "truss_wall_9": "#2a8f68",
    "central_cluster_10": "#b66a2b",
    "distributed_10_seed_1": "#7656a6",
}
CASE_LABELS = {
    "deck_row_10": "deck-row outage (10)",
    "truss_wall_9": "truss-wall outage (9)",
    "central_cluster_10": "central cluster (10)",
    "distributed_10_seed_1": "distributed outage (10)",
}
CASE_TITLES = {
    "deck_row_10": "One longitudinal deck row",
    "truss_wall_9": "One truss-wall acquisition branch",
    "central_cluster_10": "Central sensor cluster",
    "distributed_10_seed_1": "Spatially distributed loss",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.3,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e5e9ed",
            "grid.linewidth": 0.65,
        }
    )


def draw_graph(ax, positions, edges, color="#d5dce3", lw=0.75, alpha=0.8):
    for left, right in edges:
        ax.plot(
            positions[[left, right], 0],
            positions[[left, right], 1],
            color=color,
            lw=lw,
            alpha=alpha,
            zorder=1,
        )


def save(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def figure_paper_overview(protocol: dict, sensitivity: np.ndarray) -> None:
    """One-page graphical summary of the paper logic and data separation."""
    positions = np.asarray(protocol["sensor_positions"], dtype=float)
    edges = np.asarray(protocol["edges"], dtype=int)
    names = np.asarray(protocol["sensor_names"], dtype=str)
    case = next(
        row for row in protocol["cases"] if row["case_name"] == "central_cluster_10"
    )
    unavailable = set(case["initial_outage_sensor_indices"])
    active = set(range(len(names))) - unavailable
    first = int(case["restoration_steps"][0]["restored_sensor_index"])

    active_mask = np.zeros(len(names), dtype=bool)
    active_mask[list(active)] = True
    active_edges = active_mask[edges[:, 0]] & active_mask[edges[:, 1]]
    initial_coverage = sensitivity[active_edges].sum(axis=0)
    ordered_coverage = np.sort(initial_coverage)

    fig = plt.figure(figsize=(8.6, 3.0))
    fig.patch.set_facecolor("white")
    lefts = [0.018, 0.218, 0.418, 0.618, 0.805]
    widths = [0.165, 0.165, 0.165, 0.155, 0.177]
    axes = [fig.add_axes([left, 0.19, width, 0.70]) for left, width in zip(lefts, widths)]

    def card(ax, number: int, title: str, subtitle: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (-0.06, -0.15), 1.12, 1.26,
                boxstyle="round,pad=0.018,rounding_size=0.035",
                transform=ax.transAxes, facecolor="#fbfcfd", edgecolor="#cbd3da",
                lw=1.0, zorder=-10, clip_on=False,
            )
        )
        ax.text(
            0.0, 1.05, f"{number}", transform=ax.transAxes, ha="left", va="top",
            fontsize=10, weight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.28", facecolor=DARK, edgecolor="none"),
        )
        ax.text(0.17, 1.055, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=10.4, weight="bold", color=DARK)
        ax.text(0.5, -0.075, subtitle, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.2, color="#56626d")

    # 1. Observed outage.
    ax = axes[0]
    card(ax, 1, "Observe outage", "active + out-of-service sensors")
    draw_graph(ax, positions, edges, color="#d5dce2", lw=0.55, alpha=0.85)
    ax.scatter(positions[list(active), 0], positions[list(active), 1], s=12, color=BLUE, zorder=3)
    ax.scatter(positions[list(unavailable), 0], positions[list(unavailable), 1], s=15,
               marker="x", color="#9ca6af", linewidth=0.9, zorder=3)
    ax.axvline(0, color="#7e8891", ls="--", lw=0.8)
    ax.set_xlim(-17, 17); ax.set_ylim(-2.8, 2.8); ax.axis("off")
    ax.text(0.50, 0.10, "left-half outage", transform=ax.transAxes, ha="center",
            fontsize=8.2, color=DARK)

    # 2. Unknown damage ensemble.
    ax = axes[1]
    card(ax, 2, "Span damage space", "topology + healthy response only")
    draw_graph(ax, positions, edges, color="#b9c4cd", lw=0.62, alpha=0.9)
    midpoints = (positions[edges[:, 0]] + positions[edges[:, 1]]) / 2
    ax.scatter(midpoints[:, 0], midpoints[:, 1], s=6, color=RED, alpha=0.75, zorder=3)
    ax.scatter(positions[:, 0], positions[:, 1], s=8, color="#6d7a85", zorder=2)
    ax.set_xlim(-17, 17); ax.set_ylim(-2.8, 2.8); ax.axis("off")
    ax.text(0.50, 0.18, "122 candidate links", transform=ax.transAxes, ha="center",
            fontsize=8.6, weight="bold", color=DARK)
    ax.text(0.50, 0.08, r"3 severities $\times$ 3 graph scales", transform=ax.transAxes,
            ha="center", fontsize=7.8, color="#56626d")

    # 3. Lower-tail objective.
    ax = axes[2]
    card(ax, 3, "Measure coverage", "lowest-covered 10%")
    x = np.arange(1, len(ordered_coverage) + 1)
    ax.plot(x, ordered_coverage, color="#8ea0ad", lw=1.5)
    ax.fill_between(x[:13], ordered_coverage[:13], 1e-6, color=ORANGE, alpha=0.35)
    ax.scatter(x[:13], ordered_coverage[:13], s=10, color=ORANGE, zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(1, 122); ax.set_ylim(1e-5, 1.15)
    ax.set_xticks([1, 13, 122]); ax.set_xticklabels(["1", "13", "122"])
    ax.set_yticks([1e-4, 1e-2, 1]); ax.set_yticklabels(["$10^{-4}$", "$10^{-2}$", "$1$"])
    ax.set_ylabel("coverage", fontsize=7.5, labelpad=1)
    ax.tick_params(labelsize=7, length=2)
    ax.grid(True, color="#e3e8ec", lw=0.55)
    ax.text(0.07, 0.18, "$U(A)$ = mean of\nlowest 13", transform=ax.transAxes,
            fontsize=8.0, color=DARK, weight="bold")

    # 4. Outage-conditioned repair mechanism.
    ax = axes[3]
    card(ax, 4, "Restore and repeat", r"choose $i^*=\arg\max_i\Delta U$")
    al20 = first
    ag05 = int(np.flatnonzero(names == "AG05")[0])
    al24 = int(np.flatnonzero(names == "AL24")[0])
    local = {ag05: (0.50, 0.82), al20: (0.22, 0.48), al24: (0.80, 0.48)}
    for neighbour in (ag05, al24):
        ax.plot([local[al20][0], local[neighbour][0]], [local[al20][1], local[neighbour][1]],
                color=ORANGE, lw=4.0, solid_capstyle="round", zorder=2)
    ax.scatter(*local[al20], s=175, marker="*", color=ORANGE, edgecolor="white",
               linewidth=0.8, zorder=4)
    ax.scatter(*local[ag05], s=55, color=BLUE, zorder=3)
    ax.scatter(*local[al24], s=55, color=BLUE, zorder=3)
    for sensor in (al20, ag05, al24):
        x0, y0 = local[sensor]
        ax.text(x0, y0 + (0.10 if sensor != al20 else -0.15), names[sensor], ha="center",
                fontsize=8.0, weight="bold" if sensor == al20 else "normal", color=DARK)
    ax.text(0.50, 0.17, "new graph links", transform=ax.transAxes, ha="center",
            fontsize=8.1, color=DARK)
    ax.text(0.50, 0.07, r"$U(A')\geq U(A)$", transform=ax.transAxes, ha="center",
            fontsize=10.0, weight="bold", color=GREEN)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 5. Lock first, validate second.
    ax = axes[4]
    card(ax, 5, "Lock, then validate", "DS1--DS8 never select")
    ax.add_patch(FancyBboxPatch((0.04, 0.66), 0.38, 0.19, boxstyle="round,pad=0.02",
                                facecolor="#eef2f5", edgecolor=DARK, lw=1.0))
    ax.text(0.23, 0.755, "PROTOCOL", ha="center", va="center", fontsize=8.0,
            weight="bold", color=DARK)
    ax.text(0.23, 0.685, "LOCK", ha="center", va="center", fontsize=8.0,
            weight="bold", color=DARK)
    ax.add_patch(FancyArrowPatch((0.45, 0.75), (0.61, 0.75), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.2, color="#65717b"))
    ax.text(0.79, 0.755, "DS1--DS8", ha="center", va="center", fontsize=8.2,
            weight="bold", color=RED,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="#fff4f1", edgecolor="#e8a094"))
    ax.plot([0.08, 0.20, 0.33, 0.47, 0.60, 0.74, 0.91],
            [0.18, 0.25, 0.31, 0.43, 0.52, 0.67, 0.82], color=GREEN, lw=2.0,
            marker="o", ms=3.5)
    ax.text(0.08, 0.09, r"coverage: guaranteed $\nearrow$", fontsize=7.8, color=DARK)
    ax.plot([0.08, 0.24, 0.40, 0.56, 0.72, 0.91],
            [0.47, 0.42, 0.51, 0.37, 0.44, 0.40], color=BLUE, lw=1.6,
            marker="o", ms=3.2)
    ax.text(0.52, 0.30, "AUROC: may fluctuate", fontsize=7.8, color=DARK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Arrows between the five stages.
    for left, width, next_left in zip(lefts[:-1], widths[:-1], lefts[1:]):
        fig.add_artist(
            FancyArrowPatch(
                (left + width + 0.005, 0.54), (next_left - 0.007, 0.54),
                transform=fig.transFigure, arrowstyle="-|>", mutation_scale=13,
                lw=1.35, color="#76828c",
            )
        )

    fig.text(0.34, 0.055, "SELECTION: topology and healthy response",
             ha="center", va="center", fontsize=9.2, weight="bold", color=GREEN)
    fig.add_artist(FancyArrowPatch((0.075, 0.105), (0.595, 0.105), transform=fig.transFigure,
                                   arrowstyle="-[", mutation_scale=8, lw=1.2, color=GREEN))
    fig.text(0.89, 0.055, "HELD-OUT PHYSICAL CHECK",
             ha="center", va="center", fontsize=9.2, weight="bold", color=BLUE)
    fig.add_artist(FancyArrowPatch((0.82, 0.105), (0.965, 0.105), transform=fig.transFigure,
                                   arrowstyle="-[", mutation_scale=8, lw=1.2, color=BLUE))
    save(fig, "fig2_paper_overview")


def figure_paper_overview_v2(protocol: dict, sensitivity: np.ndarray) -> None:
    """Four-stage conceptual overview with dataset-specific details kept in the text."""
    positions = np.asarray(protocol["sensor_positions"], dtype=float)
    edges = np.asarray(protocol["edges"], dtype=int)
    names = np.asarray(protocol["sensor_names"], dtype=str)
    case = next(
        row for row in protocol["cases"] if row["case_name"] == "central_cluster_10"
    )
    unavailable = set(case["initial_outage_sensor_indices"])
    active = set(range(len(names))) - unavailable
    active_mask = np.zeros(len(names), dtype=bool)
    active_mask[list(active)] = True
    tail_count = int(np.ceil(0.10 * sensitivity.shape[1]))

    def lower_tail_coverage(sensor_mask: np.ndarray) -> float:
        edge_mask = sensor_mask[edges[:, 0]] & sensor_mask[edges[:, 1]]
        location_coverage = sensitivity[edge_mask].sum(axis=0)
        return float(np.partition(location_coverage, tail_count - 1)[:tail_count].mean())

    initial_tail = lower_tail_coverage(active_mask)
    candidate_sensors = np.asarray(sorted(unavailable), dtype=int)
    marginal_gains = []
    for sensor in candidate_sensors:
        trial_mask = active_mask.copy()
        trial_mask[sensor] = True
        marginal_gains.append(lower_tail_coverage(trial_mask) - initial_tail)
    marginal_gains = np.asarray(marginal_gains, dtype=float)
    selected_candidate = int(np.argmax(marginal_gains))

    original_family = plt.rcParams["font.family"]
    original_serif = list(plt.rcParams["font.serif"])
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })

    fig = plt.figure(figsize=(7.6, 5.15))
    fig.patch.set_facecolor("white")
    panel_positions = [
        [0.045, 0.555, 0.390, 0.365],
        [0.565, 0.555, 0.390, 0.365],
        [0.565, 0.075, 0.390, 0.365],
        [0.045, 0.075, 0.390, 0.365],
    ]
    axes = [fig.add_axes(position) for position in panel_positions]

    def card(ax, number: int, title: str, face_color: str, border_color: str) -> None:
        ax.add_patch(FancyBboxPatch(
            (-0.035, -0.055), 1.07, 1.15,
            boxstyle="round,pad=0.018,rounding_size=0.032",
            transform=ax.transAxes, facecolor=face_color, edgecolor=border_color,
            lw=1.7, zorder=-10, clip_on=False,
        ))
        ax.text(0.0, 1.035, f"{number}", transform=ax.transAxes, ha="left", va="top",
                fontsize=10.3, weight="bold", color="white",
                bbox=dict(boxstyle="circle,pad=0.27", facecolor=border_color,
                          edgecolor="none"))
        ax.text(0.14, 1.04, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=11.0, weight="bold", color="#111111")

    xnorm = (positions[:, 0] - positions[:, 0].min()) / np.ptp(positions[:, 0])
    ynorm = (positions[:, 1] - positions[:, 1].min()) / np.ptp(positions[:, 1])

    # Stage 1: observe which sensors and links survive the outage.
    ax = axes[0]
    card(ax, 1, "Observe the Outage", "#e3f2fd", "#1565c0")
    xy = np.column_stack((0.08 + 0.84 * xnorm, 0.10 + 0.73 * ynorm))
    for left, right in edges:
        both_active = left in active and right in active
        ax.plot(xy[[left, right], 0], xy[[left, right], 1],
                color=BLUE if both_active else "#d7dee4",
                lw=0.68 if both_active else 0.45,
                alpha=0.90 if both_active else 0.70, zorder=1)
    ax.scatter(xy[list(active), 0], xy[list(active), 1], s=11, color=BLUE, zorder=3)
    ax.scatter(xy[list(unavailable), 0], xy[list(unavailable), 1], s=14,
               marker="x", color="#9ca6af", linewidth=0.9, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Stage 2: span unknown damage with a topology-wide candidate ensemble.
    ax = axes[1]
    card(ax, 2, "Represent Damage Uncertainty", "#f3e5f5", "#7b1fa2")
    xy = np.column_stack((0.08 + 0.84 * xnorm, 0.10 + 0.73 * ynorm))
    for left, right in edges:
        ax.plot(xy[[left, right], 0], xy[[left, right], 1],
                color="#cbd4db", lw=0.48, zorder=1)
    mids = (xy[edges[:, 0]] + xy[edges[:, 1]]) / 2
    ax.scatter(mids[:, 0], mids[:, 1], s=7.0, color=RED, alpha=0.82, zorder=3)
    ax.scatter(xy[:, 0], xy[:, 1], s=7.0, color="#697681", zorder=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Stage 3: compare one marginal gain per unavailable sensor and select the maximum.
    ax = axes[2]
    card(ax, 3, "Maximize Coverage Gain", "#e8f5e9", "#2e7d32")
    gain_ax = ax.inset_axes([0.09, 0.17, 0.83, 0.66])
    candidate_index = np.arange(len(candidate_sensors))
    bar_colors = np.full(len(candidate_sensors), "#9aa8b3", dtype=object)
    bar_colors[selected_candidate] = ORANGE
    gain_ax.bar(candidate_index, marginal_gains, color=bar_colors, width=0.68)
    best_gain = float(marginal_gains[selected_candidate])
    gain_ax.scatter(
        [selected_candidate], [best_gain * 1.08], marker="*", s=62,
        color=ORANGE, edgecolor="white", linewidth=0.6, zorder=4,
    )
    gain_ax.text(
        selected_candidate, best_gain * 1.18, r"$i^*$",
        ha="center", va="bottom", fontsize=7.5, weight="bold", color=DARK,
    )
    gain_ax.set_xticks(candidate_index, [str(i) for i in range(1, len(candidate_sensors) + 1)])
    gain_ax.set_xlabel("Candidate sensors in S", fontsize=7.0)
    gain_ax.set_ylabel(r"$\Delta_U(i\mid A)$", fontsize=8.0)
    gain_ax.set_ylim(0, best_gain * 1.34)
    gain_ax.tick_params(axis="x", labelsize=6.2, length=2)
    gain_ax.tick_params(axis="y", labelsize=6.2)
    gain_ax.grid(axis="y", color="#dfe6e1", lw=0.55)
    gain_ax.spines["bottom"].set_color("#9ba8a0")
    gain_ax.spines["left"].set_color("#9ba8a0")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Stage 4: freeze selection before the held-out physical check.
    ax = axes[3]
    card(ax, 4, "Lock, Then Validate", "#fff3e0", "#e65100")
    ax.add_patch(FancyBboxPatch((0.08, 0.61), 0.38, 0.18, boxstyle="round,pad=0.02",
                                facecolor="#eef2f5", edgecolor=DARK, lw=1.0))
    ax.text(0.27, 0.70, "LOCK", ha="center", va="center", fontsize=8.0,
            weight="bold", color=DARK)
    ax.add_patch(FancyArrowPatch((0.49, 0.70), (0.65, 0.70), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.2, color="#65717b"))
    ax.scatter([0.79], [0.70], marker="D", s=60, color=RED, edgecolor="white",
               linewidth=0.7, zorder=4)
    ax.plot([0.08, 0.20, 0.33, 0.47, 0.60, 0.74, 0.91],
            [0.13, 0.20, 0.26, 0.38, 0.47, 0.61, 0.76], color=GREEN, lw=2.0,
            marker="o", ms=3.3)
    ax.plot([0.08, 0.24, 0.40, 0.56, 0.72, 0.91],
            [0.39, 0.34, 0.43, 0.29, 0.36, 0.32], color=BLUE, lw=1.6,
            marker="o", ms=3.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    connectors = (
        ((0.445, 0.738), (0.555, 0.738)),
        ((0.760, 0.545), (0.760, 0.450)),
        ((0.555, 0.258), (0.445, 0.258)),
    )
    for start, end in connectors:
        fig.add_artist(FancyArrowPatch(
            start, end, transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=13, lw=1.8, color="#444444",
        ))
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
                   markeredgecolor=BLUE, label="active sensor"),
            Line2D([0], [0], marker="x", color="#9ca6af", linestyle="none",
                   label="out-of-service sensor"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=RED,
                   markeredgecolor=RED, markersize=4, label="candidate location"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor=ORANGE,
                   markeredgecolor=ORANGE, markersize=8,
                   label=r"selected repair $i^*$"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.5,
        fontsize=7.6,
    )
    save(fig, "fig2_paper_overview")
    plt.rcParams["font.family"] = original_family
    plt.rcParams["font.serif"] = original_serif


def figure_outage_situations(protocol: dict) -> None:
    """Show the four bounded outage states before any restoration decision."""
    positions = np.asarray(protocol["sensor_positions"], dtype=float)
    edges = np.asarray(protocol["edges"], dtype=int)
    sensor_count = len(protocol["sensor_names"])

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 5.75), constrained_layout=True)
    for panel, (ax, case) in enumerate(zip(axes.ravel(), protocol["cases"]), start=1):
        unavailable = set(case["initial_outage_sensor_indices"])
        active = sorted(set(range(sensor_count)) - unavailable)
        unavailable_ordered = sorted(unavailable)
        active_mask = np.zeros(sensor_count, dtype=bool)
        active_mask[active] = True

        for left, right in edges:
            survives = active_mask[left] and active_mask[right]
            ax.plot(
                positions[[left, right], 0],
                positions[[left, right], 1],
                color=BLUE if survives else "#d6dde3",
                lw=0.85 if survives else 0.55,
                alpha=0.82 if survives else 0.72,
                zorder=1,
            )
        ax.scatter(
            positions[active, 0], positions[active, 1], s=16, color=BLUE,
            edgecolor="white", linewidth=0.35, zorder=3,
        )
        ax.scatter(
            positions[unavailable_ordered, 0], positions[unavailable_ordered, 1],
            s=43, marker="x", color=RED, linewidth=1.6, zorder=4,
        )
        ax.set_title(
            f"({chr(96 + panel)}) {CASE_TITLES[case['case_name']]} "
            f"({len(unavailable)} out of service)",
            loc="left",
        )
        ax.set_xlim(-17, 17)
        ax.set_ylim(-2.8, 2.8)
        ax.set_xlabel("Longitudinal coordinate, x (m)")
        ax.set_ylabel("Transverse coordinate, y (m)")

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
                   markeredgecolor=BLUE, label="active sensor"),
            Line2D([0], [0], marker="x", color=RED, linestyle="none",
                   markeredgewidth=1.6, label="out-of-service sensor"),
            Line2D([0], [0], color=BLUE, lw=1.2, label="surviving measurement link"),
            Line2D([0], [0], color="#d6dde3", lw=1.0,
                   label="link lost after sensor loss"),
        ],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    save(fig, "fig3_observed_outage_situations")


def figure_all_coverage(protocol: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.35), constrained_layout=True)
    for case in protocol["cases"]:
        name = case["case_name"]
        values = [case["initial_coverage"]["lower_tail_mean"]] + [
            row["coverage"]["lower_tail_mean"] for row in case["restoration_steps"]
        ]
        x = np.arange(len(values))
        ax.plot(
            x, values, color=CASE_COLORS[name], marker="o", ms=3.2, lw=1.7,
            label=CASE_LABELS[name],
        )
    ax.set_title("Locked damage-location coverage paths", loc="left")
    ax.set_xlabel("Number of restored sensors")
    ax.set_ylabel("Lower-tail damage-location coverage")
    ax.set_ylim(-0.025, 1.025)
    ax.legend(loc="upper left", frameon=False)
    fig.text(
        0.5, -0.02,
        "Every marker is one restored sensor; each sensor-loss state produces a different coverage-recovery path.",
        ha="center", color=DARK,
    )
    save(fig, "fig3_monotone_coverage_paths")


def figure_combined_restoration_cases(protocol: dict) -> None:
    positions = np.asarray(protocol["sensor_positions"], dtype=float)
    edges = np.asarray(protocol["edges"], dtype=int)
    names = np.asarray(protocol["sensor_names"], dtype=str)
    case_order = [
        ("deck_row_10", "Deck-row loss"),
        ("truss_wall_9", "Truss-wall loss"),
        ("central_cluster_10", "Central-cluster loss"),
        ("distributed_10_seed_1", "Distributed loss"),
    ]
    cases = {row["case_name"]: row for row in protocol["cases"]}

    fig, axes = plt.subplots(4, 2, figsize=(9.2, 11.5))
    for row_index, (case_name, case_title) in enumerate(case_order):
        case = cases[case_name]
        unavailable = set(case["initial_outage_sensor_indices"])
        available = set(range(len(names))) - unavailable
        steps = case["restoration_steps"]
        selected = [int(row["restored_sensor_index"]) for row in steps]
        color = CASE_COLORS[case_name]

        ax = axes[row_index, 0]
        draw_graph(ax, positions, edges, color="#d2d9df", lw=0.65, alpha=0.85)
        ax.scatter(
            positions[list(available), 0], positions[list(available), 1],
            s=13, color=BLUE, zorder=3,
        )
        for rank, sensor in enumerate(selected, start=1):
            ax.scatter(
                positions[sensor, 0], positions[sensor, 1], s=82, color=ORANGE,
                edgecolor="white", linewidth=0.7, zorder=5,
            )
            ax.text(
                positions[sensor, 0], positions[sensor, 1], str(rank),
                ha="center", va="center", fontsize=6.2, weight="bold",
                color="white", zorder=6,
            )
        left_letter = chr(ord("a") + 2 * row_index)
        ax.set_title(f"({left_letter}) {case_title}: priority path", loc="left")
        ax.set_ylabel("y (m)")
        ax.set_xlim(-17, 17)
        ax.set_ylim(-2.8, 2.8)
        if row_index == len(case_order) - 1:
            ax.set_xlabel("Longitudinal coordinate, x (m)")
        else:
            ax.set_xticklabels([])

        ax = axes[row_index, 1]
        values = [case["initial_coverage"]["lower_tail_mean"]] + [
            step["coverage"]["lower_tail_mean"] for step in steps
        ]
        x = np.arange(len(values))
        ax.plot(x, values, color=color, marker="o", ms=4.2, lw=1.9)
        right_letter = chr(ord("a") + 2 * row_index + 1)
        ax.set_title(f"({right_letter}) {case_title}: coverage path", loc="left")
        ax.set_yscale("log")
        ax.set_ylim(0.035, 1.08)
        ax.set_yticks([0.05, 0.1, 0.5, 1.0])
        ax.set_yticklabels(["0.05", "0.1", "0.5", "1.0"])
        ax.set_xticks(x)
        ax.set_ylabel("Lower-tail coverage")
        if row_index == len(case_order) - 1:
            ax.set_xlabel("Number of restored sensors")
        else:
            ax.set_xticklabels([])

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
                   markeredgecolor=BLUE, label="initially active sensor"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE,
                   markeredgecolor=ORANGE,
                   label="out-of-service sensor numbered by priority path"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.075, right=0.99, top=0.975, bottom=0.075,
        hspace=0.43, wspace=0.24,
    )
    save(fig, "fig6_combined_restoration_sequences")


def figure_validation(result: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0), constrained_layout=True)
    for case in result["validation"]:
        name = case["case_name"]
        x = [row["restored_sensor_count"] for row in case["rows"]]
        coverage = [row["coverage"]["lower_tail_mean"] for row in case["rows"]]
        auroc = [row["physical_validation"]["block_auroc"] for row in case["rows"]]
        axes[0].plot(
            x, coverage, color=CASE_COLORS[name], marker="o", lw=1.8,
            label=CASE_LABELS[name],
        )
        axes[1].plot(
            x, auroc, color=CASE_COLORS[name], marker="o", lw=1.8,
            label=CASE_LABELS[name],
        )
    axes[0].set_title("(a) Coverage along the locked sequences", loc="left")
    axes[0].set_xlabel("Number of restored sensors")
    axes[0].set_ylabel("Lower-tail damage-location coverage")
    axes[0].set_yscale("log")
    axes[1].set_title("(b) Held-out DS1--DS8 detector check", loc="left")
    axes[1].set_xlabel("Number of restored sensors")
    axes[1].set_ylabel("Aggregate AUROC")
    axes[1].set_ylim(0.84, 0.945)
    fig.legend(
        handles=[
            Line2D([0], [0], color=CASE_COLORS[name], marker="o", lw=1.8,
                   label=CASE_LABELS[name])
            for name in CASE_LABELS
        ],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    save(fig, "fig5_locked_validation")


def main() -> None:
    style()
    protocol = json.loads((OUTPUT / "protocol.lock.json").read_text(encoding="utf-8"))
    figure_outage_situations(protocol)
    figure_combined_restoration_cases(protocol)
    print(f"saved figures to {FIGURES.resolve()}")


if __name__ == "__main__":
    main()
