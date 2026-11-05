"""Check figure for the draft DS1-DS8 damage mapping (plan view, one panel per state).

Run from SHM-R1:  python analysis_R1/r1_damage_mapping_check_figure.py
Writes only to analysis_R1/outputs/damage_mapping/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hbta_pipeline.spatial_topology import build_hbta_spatial_graph  # noqa: E402

MAPPING = ROOT / "analysis_R1" / "r1_damage_mapping.json"
OUT = ROOT / "analysis_R1" / "outputs" / "damage_mapping"
SHORT = {"DS1": "stringer conn., FB x=-3.5, south", "DS2": "stringer conns., FB x=-3.5, both",
         "DS3": "cross beam, x=-8.75", "DS4": "cross beams, x=-8.75 and -5.25",
         "DS5": "bracing conn., south chord x=-10.5", "DS6": "bracing conn., north chord x=0",
         "DS7": "DS5 + DS6 connections", "DS8": "floor beam-chord conn., x=+3.5, south"}


def main() -> None:
    d = np.load(ROOT / "data" / "derived" / "hbta" / "edge_spectra_multicondition_10s.npz")
    names = d["sensor_names"].astype(str)
    pos = d["sensor_positions"].astype(float)
    g = build_hbta_spatial_graph(names, pos)
    label = {f"{names[u]}--{names[v]}": m for m, (u, v) in enumerate(g.edges)}
    label.update({f"{names[v]}--{names[u]}": m for m, (u, v) in enumerate(g.edges)})
    states = json.loads(MAPPING.read_text(encoding="utf-8"))["states"]

    fig, axes = plt.subplots(4, 2, figsize=(7.2, 5.6))
    for ax, st in zip(axes.ravel(), states):
        for m, (u, v) in enumerate(g.edges):
            ax.plot(pos[[u, v], 0], pos[[u, v], 1], color="#D1D5DB", lw=0.7, zorder=1)
        alt = {e for links in st.get("alternatives", {}).values() for e in links} - set(st["springs"])
        for e in alt:
            u, v = g.edges[label[e]]
            ax.plot(pos[[u, v], 0], pos[[u, v], 1], color="#F59E0B", lw=1.6, ls="--", zorder=2)
        for e in st["springs"]:
            u, v = g.edges[label[e]]
            ax.plot(pos[[u, v], 0], pos[[u, v], 1], color="#C74440", lw=2.6, zorder=3)
        ag = np.char.startswith(names, "AG")
        ax.scatter(pos[ag, 0], pos[ag, 1], s=9, color="white", edgecolor="#1F4E79", lw=0.8, zorder=4)
        ax.scatter(pos[~ag, 0], pos[~ag, 1], s=7, marker="s", color="white", edgecolor="#E69F00",
                   lw=0.8, zorder=4)
        for e in st["springs"]:
            for s in e.split("--"):
                i = int(np.flatnonzero(names == s)[0])
                ax.text(pos[i, 0], pos[i, 1] + 0.42, s, fontsize=5.2, ha="center", zorder=5)
        flag = "" if st.get("usable_for_detection_validation", True) else "  (not detectable)"
        ax.set_title(f"{st['state']}: {SHORT[st['state']]}{flag}", fontsize=7, loc="left")
        ax.set_xlim(-17, 17)
        ax.set_ylim(-2.9, 2.9)
        ax.set_xticks([-14, -7, 0, 7, 14])
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal")
    fig.text(0.5, 0.005, "x (m), west to east;  red: proposed springs;  amber dashed: alternatives",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "damage_mapping_check.png", dpi=220)
    print("saved", OUT)


if __name__ == "__main__":
    main()
