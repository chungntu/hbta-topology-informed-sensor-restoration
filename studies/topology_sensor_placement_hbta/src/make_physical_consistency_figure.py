"""Generate standalone figures for Section 5:
- Figure 6: Full-bridge sensor topology & DS1 damage location.
- Figure 7: Link-level response consistency (Test 1):
    (a) Theoretical response weights q_{m,d_DS1} across all 122 links (log-scaled).
    (b) Measured vibration response R_{m,DS1} across all 122 links (log-scaled).
    (c) Predicted weight vs. measured response across all 122 links.
- Figure 8: Sensor-restoration consistency under companion outages (Test 2).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


STUDY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = STUDY_ROOT / "outputs" / "physical_consistency"
PREPARED_PATH = (
    STUDY_ROOT.parents[1]
    / "data"
    / "derived"
    / "hbta"
    / "edge_spectra_multicondition_10s.npz"
)
SENSITIVITY_PATH = (
    STUDY_ROOT / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz"
)

COLORS = {
    "navy": "#255B78",
    "green": "#2F8B68",
    "orange": "#E08B2C",
    "red": "#C53B32",
    "gray": "#9AA6AF",
    "light": "#D7DEE3",
    "text": "#25313A",
    "grid": "#E8EDF0",
    "blue": "#2E6F9E",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    results = json.loads(
        (OUTPUT_ROOT / "physical_consistency_results.json").read_text(
            encoding="utf-8"
        )
    )
    links = read_csv(OUTPUT_ROOT / "link_level_ds1.csv")
    restorations = read_csv(OUTPUT_ROOT / "restoration_consistency_ds1.csv")

    with np.load(PREPARED_PATH) as data:
        names = data["sensor_names"].astype(str)
        positions = data["sensor_positions"].astype(float)
    with np.load(SENSITIVITY_PATH) as data:
        sensitivity = data["sensitivity"].astype(float)
        edges = data["edges"].astype(int)

    lookup = {name: index for index, name in enumerate(names)}
    ds1_location = next(
        index
        for index, edge in enumerate(edges)
        if set(map(int, edge)) == {lookup["AL14"], lookup["AL18"]}
    )
    q = sensitivity[:, ds1_location]
    measured = np.asarray([float(row["R_m_ds1"]) for row in links])
    top = np.asarray([row["in_top_10_percent_predicted"] == "True" for row in links])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure_dir = OUTPUT_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Figure 6: Full-Bridge Sensor Topology & DS1 Damage Location
    # =========================================================================
    fig6, ax6 = plt.subplots(figsize=(7.6, 2.3), constrained_layout=True)

    # Underlying measurement links
    for left, right in edges:
        ax6.plot(
            positions[[left, right], 0],
            positions[[left, right], 1],
            color="#d2d9df",
            lw=0.65,
            alpha=0.85,
            zorder=1,
        )

    # Base bridge truss grid (all 122 links as clean structural wireframe)
    for left, right in edges:
        ax6.plot(
            positions[[left, right], 0],
            positions[[left, right], 1],
            color="#d2d9df",
            lw=0.85,
            zorder=1,
        )

    # Highlight damaged member AL14--AL18
    idx_al14, idx_al18 = lookup["AL14"], lookup["AL18"]
    ax6.plot(
        positions[[idx_al14, idx_al18], 0],
        positions[[idx_al14, idx_al18], 1],
        color=COLORS["red"],
        lw=2.2,
        ls="--",
        alpha=0.9,
        zorder=2,
    )

    # Active healthy sensors (navy dots)
    ax6.scatter(
        positions[:, 0],
        positions[:, 1],
        s=16,
        color=COLORS["navy"],
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )

    # Companion outage sensors: AG05, AG07 (red warning rings)
    for name, offset, ha, va in [
        ("AG05", (0, 7), "center", "bottom"),
        ("AG07", (0, 7), "center", "bottom"),
    ]:
        idx = lookup[name]
        ax6.scatter(
            positions[idx, 0],
            positions[idx, 1],
            s=55,
            facecolor="#fee2e2",
            edgecolor=COLORS["red"],
            linewidth=1.4,
            zorder=4,
        )
        ax6.annotate(
            name,
            positions[idx, :2],
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=6.8,
            fontweight="bold",
            color=COLORS["red"],
            zorder=5,
        )

    # Restored sensor candidate: AL14 (amber orange ring)
    idx14 = lookup["AL14"]
    ax6.scatter(
        positions[idx14, 0],
        positions[idx14, 1],
        s=55,
        facecolor="#ffedd5",
        edgecolor="#c25e00",
        linewidth=1.4,
        zorder=4,
    )
    ax6.annotate(
        "AL14",
        positions[idx14, :2],
        xytext=(-8, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
        color="#c25e00",
        zorder=5,
    )

    # Restored sensor candidate: AL15 (amber orange ring)
    idx15 = lookup["AL15"]
    ax6.scatter(
        positions[idx15, 0],
        positions[idx15, 1],
        s=55,
        facecolor="#ffedd5",
        edgecolor="#c25e00",
        linewidth=1.4,
        zorder=4,
    )
    ax6.annotate(
        "AL15",
        positions[idx15, :2],
        xytext=(-8, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
        color="#c25e00",
        zorder=5,
    )

    # Restored sensor candidate: AL18 (amber orange ring)
    idx18 = lookup["AL18"]
    ax6.scatter(
        positions[idx18, 0],
        positions[idx18, 1],
        s=55,
        facecolor="#ffedd5",
        edgecolor="#c25e00",
        linewidth=1.4,
        zorder=4,
    )
    ax6.annotate(
        "AL18",
        positions[idx18, :2],
        xytext=(8, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
        color="#c25e00",
        zorder=5,
    )

    # DS1 Damage Location marker
    damage_xy = (-3.5, -0.55)
    ax6.scatter(
        [damage_xy[0]],
        [damage_xy[1]],
        marker="X",
        s=75,
        color=COLORS["red"],
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    ax6.annotate(
        "DS1",
        damage_xy,
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=COLORS["red"],
        fontweight="bold",
        fontsize=7.5,
    )

    ax6.set_xlim(-17, 17)
    ax6.set_ylim(-2.8, 3.0)
    ax6.set_xlabel("Longitudinal coordinate, x (m)")
    ax6.set_ylabel("y (m)")
    ax6.grid(color=COLORS["grid"], linewidth=0.45, zorder=0)

    fig6.savefig(figure_dir / "fig6_ds1_damage_location.pdf", bbox_inches="tight")
    fig6.savefig(
        figure_dir / "fig6_ds1_damage_location.png", dpi=360, bbox_inches="tight"
    )
    plt.close(fig6)

    # =========================================================================
    # Figure 7: Link-Level Response Consistency (Test 1) - 2 Panels (a), (b)
    # =========================================================================
    fig7, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(7.2, 4.4), constrained_layout=True, sharex=True
    )

    # (a) Theoretical response weights q across all 122 links (log-scaled)
    q_floor = 1e-4
    q_max = float(np.max(q))
    for left, right in edges:
        ax_a.plot(
            positions[[left, right], 0],
            positions[[left, right], 1],
            color="#d2d9df",
            lw=0.75,
            alpha=0.85,
            zorder=1,
        )
    for index, (left, right) in enumerate(edges):
        val = q[index]
        if val >= q_floor:
            norm = (np.log10(val) - np.log10(q_floor)) / (
                np.log10(q_max) - np.log10(q_floor)
            )
            norm = float(np.clip(norm, 0.0, 1.0))
            ax_a.plot(
                positions[[left, right], 0],
                positions[[left, right], 1],
                color=COLORS["orange"],
                lw=0.8 + 3.8 * norm,
                alpha=0.35 + 0.65 * norm,
                zorder=2,
                solid_capstyle="round",
            )
    ax_a.scatter(
        positions[:, 0],
        positions[:, 1],
        s=16,
        color=COLORS["navy"],
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    damage_xy = (-3.5, -0.55)
    ax_a.scatter(
        [damage_xy[0]],
        [damage_xy[1]],
        marker="X",
        s=60,
        color=COLORS["red"],
        edgecolor="white",
        linewidth=0.7,
        zorder=6,
    )
    ax_a.annotate(
        "DS1",
        damage_xy,
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        color=COLORS["red"],
        fontweight="bold",
        fontsize=7.0,
    )
    ax_a.set_title(
        r"(a) Modeled response weights $q_{m,\mathrm{DS1}}$ across all 122 links (theory)",
        loc="left",
    )
    ax_a.set_xlim(-17, 17)
    ax_a.set_ylim(-2.7, 3.5)
    ax_a.set_ylabel("y (m)")
    ax_a.grid(color=COLORS["grid"], linewidth=0.45, zorder=0)

    # Legend for (a): sample line thicknesses
    legend_q = [
        (None, "#d2d9df", 0.75, 0.85, r"$q \leq 10^{-4}$"),
        (
            1e-2,
            COLORS["orange"],
            0.8
            + 3.8
            * (
                (np.log10(1e-2) - np.log10(q_floor))
                / (np.log10(q_max) - np.log10(q_floor))
            ),
            0.35
            + 0.65
            * (
                (np.log10(1e-2) - np.log10(q_floor))
                / (np.log10(q_max) - np.log10(q_floor))
            ),
            r"$q = 10^{-2}$",
        ),
        (
            1e-1,
            COLORS["orange"],
            0.8
            + 3.8
            * (
                (np.log10(1e-1) - np.log10(q_floor))
                / (np.log10(q_max) - np.log10(q_floor))
            ),
            0.35
            + 0.65
            * (
                (np.log10(1e-1) - np.log10(q_floor))
                / (np.log10(q_max) - np.log10(q_floor))
            ),
            r"$q = 10^{-1}$",
        ),
    ]
    lines_a, labels_a = [], []
    for _, col, lw, alpha, lbl in legend_q:
        lines_a.append(Line2D([0], [0], color=col, lw=lw, alpha=alpha))
        labels_a.append(lbl)
    leg_a = ax_a.legend(
        lines_a,
        labels_a,
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor=COLORS["light"],
        ncol=3,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.35,
        borderpad=0.3,
    )
    leg_a.set_zorder(10)

    # (b) Measured Physical Response Rm across all 122 links (log-scaled)
    for left, right in edges:
        ax_b.plot(
            positions[[left, right], 0],
            positions[[left, right], 1],
            color="#d2d9df",
            lw=0.75,
            alpha=0.85,
            zorder=1,
        )
    max_r = float(np.max(measured))
    for index, (left, right) in enumerate(edges):
        r_val = measured[index]
        if r_val > 0:
            norm = np.log10(1.0 + r_val) / np.log10(1.0 + max_r)
            norm = float(np.clip(norm, 0.0, 1.0))
            ax_b.plot(
                positions[[left, right], 0],
                positions[[left, right], 1],
                color=COLORS["blue"],
                lw=0.8 + 3.8 * norm,
                alpha=0.35 + 0.65 * norm,
                zorder=2,
                solid_capstyle="round",
            )
    ax_b.scatter(
        positions[:, 0],
        positions[:, 1],
        s=16,
        color=COLORS["navy"],
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    ax_b.scatter(
        [damage_xy[0]],
        [damage_xy[1]],
        marker="X",
        s=60,
        color=COLORS["red"],
        edgecolor="white",
        linewidth=0.7,
        zorder=6,
    )
    ax_b.annotate(
        "DS1",
        damage_xy,
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        color=COLORS["red"],
        fontweight="bold",
        fontsize=7.0,
    )
    ax_b.set_title(
        r"(b) Measured vibration response $R_{m,\mathrm{DS1}}$ across all 122 links (experiment)",
        loc="left",
    )
    ax_b.set_xlim(-17, 17)
    ax_b.set_ylim(-2.7, 3.5)
    ax_b.set_xlabel("Longitudinal coordinate, x (m)")
    ax_b.set_ylabel("y (m)")
    ax_b.grid(color=COLORS["grid"], linewidth=0.45, zorder=0)

    # Legend for (b): sample line thicknesses including R=0 baseline
    norm_05 = np.log10(1.0 + 0.5) / np.log10(1.0 + max_r)
    norm_20 = np.log10(1.0 + 2.0) / np.log10(1.0 + max_r)
    legend_r = [
        (None, "#d2d9df", 0.75, 0.85, r"$R = 0$"),
        (
            0.5,
            COLORS["blue"],
            0.8 + 3.8 * norm_05,
            0.35 + 0.65 * norm_05,
            r"$R = 0.5$",
        ),
        (
            2.0,
            COLORS["blue"],
            0.8 + 3.8 * norm_20,
            0.35 + 0.65 * norm_20,
            r"$R = 2.0$",
        ),
    ]
    lines_b, labels_b = [], []
    for _, col, lw, alpha, lbl in legend_r:
        lines_b.append(Line2D([0], [0], color=col, lw=lw, alpha=alpha))
        labels_b.append(lbl)
    leg_b = ax_b.legend(
        lines_b,
        labels_b,
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor=COLORS["light"],
        ncol=3,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.35,
        borderpad=0.3,
    )
    leg_b.set_zorder(10)

    fig7.savefig(figure_dir / "fig7_link_level_consistency.pdf", bbox_inches="tight")
    fig7.savefig(
        figure_dir / "fig7_link_level_consistency.png", dpi=360, bbox_inches="tight"
    )
    plt.close(fig7)

    # =========================================================================
    # Figure 8: Sensor-Restoration Consistency (Test 2) - Two Panels (a) & (b)
    # =========================================================================
    fig8, (ax8_a, ax8_b) = plt.subplots(
        1, 2, figsize=(6.6, 2.7), sharey=True, constrained_layout=True
    )

    panels = [
        ("AG05", ax8_a, COLORS["navy"], "o", "(a) Initial outage AG05"),
        ("AG07", ax8_b, COLORS["green"], "^", "(b) Initial outage AG07"),
    ]

    for companion, ax, color, marker, title in panels:
        rows = [row for row in restorations if row["companion_outage"] == companion]
        x = np.asarray([float(row["predicted_delta_c"]) for row in rows])
        y = 100.0 * np.asarray([float(row["measured_delta_tpr"]) for row in rows])

        # Background candidate repair points
        ax.scatter(
            x,
            y,
            s=18,
            marker=marker,
            facecolor="none",
            edgecolor=color,
            linewidth=0.75,
            alpha=0.72,
            zorder=2,
        )

        # Highlighted top restored sensors
        top_sensors = [
            row for row in rows if row["restored_sensor"] in ("AL14", "AL15", "AL18")
        ]
        tx = np.asarray([float(row["predicted_delta_c"]) for row in top_sensors])
        ty = 100.0 * np.asarray([float(row["measured_delta_tpr"]) for row in top_sensors])
        ax.scatter(
            tx,
            ty,
            s=52,
            marker=marker,
            facecolor="#c25e00",
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )

        # Annotate top sensors
        for row, xx, yy in zip(top_sensors, tx, ty):
            ax.annotate(
                row["restored_sensor"],
                (xx, yy),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="#c25e00",
            )

        ax.axhline(0.0, color=COLORS["gray"], linewidth=0.7, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(8e-7, 1.5e0)
        ax.set_ylim(-3.5, 37.0)
        ax.set_xlabel(r"Predicted recovery $\Delta c_{\mathrm{DS1}}$")
        ax.set_title(title, fontsize=8.0, fontweight="bold", pad=5)
        ax.grid(color=COLORS["grid"], linewidth=0.45, which="both", zorder=0)

    ax8_a.set_ylabel(r"Measured recovery $\Delta\mathrm{TPR}$ (%)")

    fig8.savefig(figure_dir / "fig8_restoration_consistency.pdf", bbox_inches="tight")
    fig8.savefig(
        figure_dir / "fig8_restoration_consistency.png", dpi=360, bbox_inches="tight"
    )
    plt.close(fig8)

    print(f"saved figures in {figure_dir.resolve()}")


if __name__ == "__main__":
    main()
