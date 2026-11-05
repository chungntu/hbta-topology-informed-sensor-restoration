"""Regenerate the four LaTeX table bodies used by the final HBTA paper."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
STUDY = ROOT / "studies" / "topology_sensor_placement_hbta"
PAPER = ROOT / "paper" / "outage_conditioned_prioritization"
GENERATED = PAPER / "generated_tables"
PREPARED = ROOT / "data" / "derived" / "hbta" / "edge_spectra_multicondition_10s.npz"


def write(name: str, lines: list[str]) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def structural_relations() -> None:
    sys.path.insert(0, str(SOURCE))
    from hbta_pipeline.spatial_topology import build_hbta_spatial_graph, graph_audit

    with np.load(PREPARED) as data:
        names = data["sensor_names"].astype(str)
        positions = data["sensor_positions"].astype(float)
    audit = graph_audit(build_hbta_spatial_graph(names, positions), names)
    definitions = [
        ("truss_diagonal", "Truss diagonal", "AG--AG", "Direct diagonal member on one truss wall"),
        ("truss_chord", "Truss chord/path", "AG--AG", "Chord path, sometimes through an uninstrumented joint"),
        ("deck_longitudinal", "Deck longitudinal", "AL--AL", "Adjacent stringer-panel midpoint sensors"),
        ("deck_transverse", "Deck transverse", "AL--AL", "Cross-girder path through a panel-end joint"),
        ("deck_truss_interface", "Deck--truss interface", "AG--AL", "Outer deck panel to a deck-level truss joint"),
        ("lateral_bracing", "Lateral bracing", "AG--AG", "Under-deck diagonal path between opposite truss walls"),
    ]
    lines = [f"{label} & {audit['relation_counts'][key]} & {pair} & {meaning} \\\\" for key, label, pair, meaning in definitions]
    family = audit["sensor_family_counts"]
    lines += ["\\midrule", f"Total & {audit['edge_count']} & -- & {family['AG-AG']} AG--AG, {family['AG-AL']} AG--AL, and {family['AL-AL']} AL--AL links \\\\"]
    write("table_structural_relations.tex", lines + ["\\bottomrule"])


def restoration_cases() -> None:
    protocol = json.loads((STUDY / "outputs" / "topology_coverage_restoration_10sensor" / "protocol.lock.json").read_text(encoding="utf-8"))
    labels = {
        "deck_row_10": "Longitudinal deck row",
        "truss_wall_9": "One truss wall",
        "central_cluster_10": "Central cluster",
        "distributed_10_seed_1": "Spatially distributed",
    }
    lines = []
    for case in protocol["cases"]:
        steps = case["restoration_steps"]
        lines.append(
            f"{labels[case['case_name']]} & {len(case['initial_outage_sensor_names'])} & "
            f"{case['initial_coverage']['lower_tail_mean']:.6f} & {steps[4]['coverage']['lower_tail_mean']:.6f} & "
            f"{steps[0]['restored_sensor_name']} \\\\"
        )
    write("table_restoration_cases.tex", lines + ["\\bottomrule"])


def damage_mapping() -> None:
    rows = load_csv(STUDY / "outputs" / "physical_consistency" / "damage_mapping.csv")
    by_state = {row["state"]: row for row in rows}
    expected = {
        "DS1": ("One stringer--floor-beam connection", "Use AL14--AL18; the path traverses the damaged joint."),
        "DS2": ("Multiple stringer--floor-beam connections", "Exclude; complete damaged-link set is not named."),
        "DS3": ("Main part of one stringer cross beam removed", "Exclude; the member spans several sensor-projected paths."),
        "DS4": ("Main parts of multiple cross beams removed", "Exclude; intervention spans multiple candidate locations."),
        "DS5": ("One lateral-bracing connection", "Exclude; damaged uninstrumented joint is not a unique graph link."),
        "DS6": ("One lateral-bracing connection", "Exclude for the same graph-representation ambiguity."),
        "DS7": ("Multiple lateral-bracing connections", "Exclude; intervention spans multiple locations."),
        "DS8": ("One floor-beam--main-member connection", "Exclude; several represented paths meet the uninstrumented joint."),
    }
    if set(by_state) != set(expected):
        raise ValueError("Damage-mapping states differ from the accepted DS1--DS8 definition")
    write("table_damage_mapping.tex", [f"{state} & {intervention} & {decision} \\\\" for state, (intervention, decision) in expected.items()] + ["\\bottomrule"])


def restoration_chain() -> None:
    root = STUDY / "outputs" / "physical_consistency"
    selected = {(row["companion_outage"], row["restored_sensor"]): row for row in load_csv(root / "restoration_selected_examples_ds1.csv")}
    activated = load_csv(root / "restoration_activated_links_ds1.csv")

    def row(companion: str, restored: str, description: str | None = None) -> str:
        item = selected[(companion, restored)]
        if description is None:
            links = [entry for entry in activated if entry["companion_outage"] == companion and entry["restored_sensor"] == restored]
            description = "; ".join(f"{entry['activated_measurement_link']}: {float(entry['q_m_ds1']):.6f}" for entry in links)
        return f"{companion} & {restored} & {description} & {float(item['predicted_delta_c']):.6f} & {100.0 * float(item['measured_delta_tpr']):.2f} \\\\"

    lines = [
        row("AG05", "AL14"),
        row("AG05", "AG12", "Median predicted control; four activated links"),
        row("AG05", "AL37", "Lowest predicted control; two activated links"),
        row("AG07", "AL14", "Highest predicted action; four activated links"),
        row("AG07", "AL15"),
        row("AG07", "AG04", "Median predicted control; four activated links"),
        row("AG07", "AL37", "Lowest predicted control; two activated links"),
    ]
    write("table_ds1_restoration_chain.tex", lines + ["\\bottomrule"])


def main() -> None:
    structural_relations()
    restoration_cases()
    damage_mapping()
    restoration_chain()
    print(f"generated four LaTeX table bodies in {GENERATED}")


if __name__ == "__main__":
    main()
