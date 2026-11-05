"""Verify regenerated HBTA paper results against the accepted final baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
STUDY = ROOT / "studies" / "topology_sensor_placement_hbta"
DATA = ROOT / "data" / "derived" / "hbta"
BASELINE = json.loads((ROOT / "verification" / "accepted_results.json").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise AssertionError(f"{actual} != {expected} within atol={tolerance}")


def verify() -> None:
    for name, expected in BASELINE["prepared_inputs"].items():
        actual = sha256_file(DATA / name)
        if actual.lower() != expected.lower():
            raise AssertionError(f"Prepared input hash mismatch: {name}")

    sys.path.insert(0, str(SOURCE))
    from hbta_pipeline.spatial_topology import build_hbta_spatial_graph, graph_audit

    with np.load(DATA / "edge_spectra_multicondition_10s.npz") as prepared:
        names = prepared["sensor_names"].astype(str)
        positions = prepared["sensor_positions"].astype(float)
    graph = build_hbta_spatial_graph(names, positions)
    audit = graph_audit(graph, names)
    accepted_graph = BASELINE["graph"]
    assert len(names) == accepted_graph["sensor_count"]
    assert audit["edge_count"] == accepted_graph["edge_count"]
    assert audit["relation_counts"] == accepted_graph["relation_counts"]
    assert audit["sensor_family_counts"] == accepted_graph["sensor_family_counts"]

    with np.load(STUDY / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz") as bank:
        sensitivity = bank["sensitivity"]
        reliability = bank["reliability"]
        edges = bank["edges"]
    assert list(sensitivity.shape) == BASELINE["q_md"]["shape"]
    assert np.array_equal(edges, graph.edges)
    if sha256_array(sensitivity) != BASELINE["q_md"]["sensitivity_array_sha256"]:
        raise AssertionError("q_md sensitivity array changed")
    if sha256_array(reliability) != BASELINE["q_md"]["reliability_array_sha256"]:
        raise AssertionError("healthy reliability array changed")
    # The stored bank is float32, so column sums can differ from one by a few
    # units in the seventh decimal place even though generation normalizes them.
    if not np.allclose(sensitivity.sum(axis=0), 1.0, rtol=0.0, atol=1e-6):
        raise AssertionError("q_md columns are not normalized")

    protocol = json.loads((STUDY / "outputs" / "topology_coverage_restoration_10sensor" / "protocol.lock.json").read_text(encoding="utf-8"))
    assert protocol["utility"]["tail_count"] == accepted_graph["tail_count"]
    assert len(protocol["cases"]) == 4
    for case in protocol["cases"]:
        expected = BASELINE["restoration_cases"][case["case_name"]]
        assert len(case["initial_outage_sensor_names"]) == expected["outage_count"]
        close(case["initial_coverage"]["lower_tail_mean"], expected["initial_lower_tail_coverage"])
        close(case["restoration_steps"][4]["coverage"]["lower_tail_mean"], expected["five_action_lower_tail_coverage"])
        assert [step["restored_sensor_name"] for step in case["restoration_steps"]] == expected["priority_path"]

    physical = json.loads((STUDY / "outputs" / "physical_consistency" / "physical_consistency_results.json").read_text(encoding="utf-8"))
    expected_pc = BASELINE["physical_consistency"]
    link = physical["link_level"]
    assert link["state"] == expected_pc["damage_state"]
    assert link["candidate_location"] == expected_pc["candidate_location"]
    close(link["spearman_rho"], expected_pc["link_spearman_rho"])
    close(link["kendall_tau_b"], expected_pc["link_kendall_tau_b"])
    assert link["top_link_check"]["top_count"] == expected_pc["top_10_percent_count"]
    assert link["top_link_check"]["top_predicted_and_top_measured_overlap"] == expected_pc["top_10_percent_overlap"]
    summaries = {row["companion_outage"]: row for row in physical["restoration_consistency"]["all_candidate_rank_results"]}
    for companion in ("AG05", "AG07"):
        close(summaries[companion]["spearman_rho"], expected_pc[f"{companion}_spearman_rho"])
        close(summaries[companion]["kendall_tau_b"], expected_pc[f"{companion}_kendall_tau_b"])
    selected = {(row["companion_outage"], row["restored_sensor"]): row for row in physical["restoration_consistency"]["preselected_examples"]}
    for key, (delta_c, delta_tpr) in expected_pc["selected_actions"].items():
        companion, restored = key.split("/")
        close(selected[(companion, restored)]["predicted_delta_c"], delta_c)
        close(selected[(companion, restored)]["measured_delta_tpr"], delta_tpr)

    figure_paths = {
        "fig_hbta_bridge_photo.jpg": STUDY / "outputs" / "topology_coverage_restoration" / "figures" / "fig_hbta_bridge_photo.jpg",
        "fig_hbta_sensor_topology_wide.png": STUDY / "outputs" / "topology_coverage_restoration" / "figures" / "fig_hbta_sensor_topology_wide.png",
        "fig1_topology_coverage_concept.png": STUDY / "outputs" / "topology_coverage_restoration" / "figures" / "fig1_topology_coverage_concept.png",
        "fig3_observed_outage_situations.png": STUDY / "outputs" / "topology_coverage_restoration_10sensor" / "figures" / "fig3_observed_outage_situations.png",
        "fig6_combined_restoration_sequences.png": STUDY / "outputs" / "topology_coverage_restoration_10sensor" / "figures" / "fig6_combined_restoration_sequences.png",
        "fig6_ds1_damage_location.png": STUDY / "outputs" / "physical_consistency" / "figures" / "fig6_ds1_damage_location.png",
        "fig7_link_level_consistency.png": STUDY / "outputs" / "physical_consistency" / "figures" / "fig7_link_level_consistency.png",
        "fig8_restoration_consistency.png": STUDY / "outputs" / "physical_consistency" / "figures" / "fig8_restoration_consistency.png",
    }
    for name, expected in BASELINE["figure_png_sha256"].items():
        if sha256_file(figure_paths[name]).lower() != expected.lower():
            raise AssertionError(f"Final figure changed: {name}")

    generated = ROOT / "paper" / "outage_conditioned_prioritization" / "generated_tables"
    expected_tables = {
        "table_structural_relations.tex",
        "table_restoration_cases.tex",
        "table_damage_mapping.tex",
        "table_ds1_restoration_chain.tex",
    }
    if {path.name for path in generated.glob("*.tex")} != expected_tables:
        raise AssertionError("Generated table set is incomplete or contains unexpected files")
    print("SCIENTIFIC_RESULTS_VERIFIED")


def verify_pdf(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 100_000:
        raise AssertionError(f"Final PDF is missing or unexpectedly small: {path}")
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        target = Path(tempfile.mkdtemp()) / "paper.txt"
        subprocess.run([pdftotext, str(path), str(target)], check=True)
        text = target.read_text(encoding="utf-8", errors="replace")
        if BASELINE["paper_title"] not in " ".join(text.split()):
            raise AssertionError("Final PDF title does not match the accepted manuscript")
    print(f"FINAL_PDF_VERIFIED={path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    verify()
    if args.pdf:
        verify_pdf(args.pdf.resolve())


if __name__ == "__main__":
    main()
