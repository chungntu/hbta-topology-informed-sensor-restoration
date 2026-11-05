"""Physical consistency checks for topology-informed response weights.

This analysis intentionally does not compare graph-wide lower-tail coverage
with aggregate detector AUROC.  It asks a narrower question: for the one HBTA
intervention that maps unambiguously to a represented candidate link (DS1), do
the virtual-damage weights for that link agree with measured edge-local change?

The mapping, response metric, top-link fraction, statistical tests, and
restoration controls below are declared in code before results are evaluated.
Physical damage data never modify the sensitivity matrix or restoration paths.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import (
    hypergeom,
    kendalltau,
    mannwhitneyu,
    rankdata,
    spearmanr,
)


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
CONFIG_PATH = STUDY_ROOT / "configs" / "virtual_damage_pareto.json"
SENSITIVITY_PATH = (
    STUDY_ROOT / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz"
)
OUTPUT = STUDY_ROOT / "outputs" / "physical_consistency"

USABLE_STATE = "DS1"
DS1_CANDIDATE_LINK = ("AL14", "AL18")
TOP_FRACTION = 0.10
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260824
CONTROL_QUANTILES = ("highest", "median", "lowest")
FOCAL_RESTORATIONS = (("AG05", "AL14"), ("AG07", "AL15"))
EXPECTED_FOCAL_TPR_GAINS = {("AG05", "AL14"): 0.32530120481927716,
                            ("AG07", "AL15"): 0.2168674698795181}

DAMAGE_MAPPING = [
    {
        "state": "DS1",
        "documented_intervention": "one stringer-to-floor-beam connection damaged",
        "status": "used",
        "candidate_links": ["AL14--AL18"],
        "basis": (
            "The damaged uninstrumented connection lies on the drawing-derived "
            "longitudinal AL14--AL18 member path at x=-3.5 m, y=-0.55 m."
        ),
    },
    {
        "state": "DS2",
        "documented_intervention": "multiple stringer-to-floor-beam connections damaged",
        "status": "excluded",
        "candidate_links": [],
        "basis": "The benchmark description does not identify the complete damaged connection set in graph-link names.",
    },
    {
        "state": "DS3",
        "documented_intervention": "main part of one stringer cross beam removed",
        "status": "excluded",
        "candidate_links": [],
        "basis": (
            "A removed cross beam spans several stringer paths, while each sensor-projected "
            "transverse link can run through either adjacent panel-end joint; no unique link mapping exists."
        ),
    },
    {
        "state": "DS4",
        "documented_intervention": "main parts of multiple stringer cross beams removed",
        "status": "excluded",
        "candidate_links": [],
        "basis": "The intervention spans multiple cross beams and is not a single represented candidate location.",
    },
    {
        "state": "DS5",
        "documented_intervention": "one lateral-bracing connection damaged",
        "status": "excluded",
        "candidate_links": [],
        "basis": (
            "The damaged connection is an uninstrumented brace joint, whereas graph links represent "
            "complete sensor-to-sensor bracing paths; the documentation does not name a unique graph link."
        ),
    },
    {
        "state": "DS6",
        "documented_intervention": "one lateral-bracing connection damaged",
        "status": "excluded",
        "candidate_links": [],
        "basis": (
            "The damaged connection is an uninstrumented brace joint, whereas graph links represent "
            "complete sensor-to-sensor bracing paths; the documentation does not name a unique graph link."
        ),
    },
    {
        "state": "DS7",
        "documented_intervention": "multiple lateral-bracing connections damaged",
        "status": "excluded",
        "candidate_links": [],
        "basis": "The intervention spans multiple connections and is not a single represented candidate location.",
    },
    {
        "state": "DS8",
        "documented_intervention": "one floor-beam-to-main-member connection damaged",
        "status": "excluded",
        "candidate_links": [],
        "basis": (
            "The connection is not instrumented directly; several deck-to-truss and deck-transverse "
            "sensor paths meet the region, with no documented one-to-one graph-link correspondence."
        ),
    },
]


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def edge_labels(edges: np.ndarray, names: np.ndarray) -> list[str]:
    return [f"{names[left]}--{names[right]}" for left, right in edges]


def find_edge(edges: np.ndarray, names: np.ndarray, pair: tuple[str, str]) -> int:
    indices = {name: index for index, name in enumerate(names)}
    target = {indices[pair[0]], indices[pair[1]]}
    matches = [index for index, edge in enumerate(edges) if set(map(int, edge)) == target]
    if len(matches) != 1:
        raise ValueError(f"Expected one graph edge for {pair}, found {matches}")
    return matches[0]


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def bootstrap_spearman(
    predicted: np.ndarray, measured: np.ndarray, rng: np.random.Generator
) -> list[float]:
    estimates = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, predicted.size, predicted.size)
        coefficient = spearmanr(predicted[selected], measured[selected]).statistic
        if np.isfinite(coefficient):
            estimates.append(float(coefficient))
    return percentile_interval(np.asarray(estimates))


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    differences = left[:, None] - right[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sys.path.insert(0, str(SOURCE_ROOT))
    sys.path.insert(0, str(STUDY_ROOT / "src"))

    from hbta_pipeline.edge_spectral_topology_experiment import (
        edge_spectral_features,
        rolling_recording_ids,
    )
    from hbta_pipeline.spatial_topology import build_hbta_spatial_graph
    from run_placement import raw_edge_summary, robust_center_scale

    prepared_path = (STUDY_ROOT / config["prepared_spectra"]).resolve()
    teacher_path = (STUDY_ROOT / config["teacher_state"]).resolve()
    with np.load(prepared_path) as data:
        coefficients = data["fourier_coefficients"].astype(np.complex64)
        source_ids = data["recording_ids"].astype(np.int16)
        recording_names = data["recording_names"].astype(str)
        sensor_names = data["sensor_names"].astype(str)
        positions = data["sensor_positions"].astype(np.float64)
    with np.load(teacher_path) as teacher:
        coordinate_center = teacher["coord_center"].astype(np.float32)
        coordinate_scale = teacher["coord_scale"].astype(np.float32)
    with np.load(SENSITIVITY_PATH) as data:
        sensitivity = data["sensitivity"].astype(np.float64)
        locked_edges = data["edges"].astype(int)
        locked_names = data["sensor_names"].astype(str)

    graph = build_hbta_spatial_graph(sensor_names, positions)
    if not np.array_equal(sensor_names, locked_names) or not np.array_equal(graph.edges, locked_edges):
        raise ValueError("Physical graph and sensitivity-bank graph do not match")
    edges = graph.edges
    labels = edge_labels(edges, sensor_names)
    ds1_edge = find_edge(edges, sensor_names, DS1_CANDIDATE_LINK)
    q_ds1 = sensitivity[:, ds1_edge]

    recording_ids = rolling_recording_ids(source_ids, int(config["rolling_blocks"]))
    fit_indices = np.flatnonzero(np.isin(recording_ids, config["fit_recording_ids"]))
    ds1_recording_id = int(
        np.flatnonzero(np.char.find(recording_names, "_DS1_") >= 0)[0]
    )
    ds1_indices = np.flatnonzero(recording_ids == ds1_recording_id)

    features = edge_spectral_features(
        coefficients, edges, source_ids, int(config["rolling_blocks"])
    )
    standardized = ((features - coordinate_center[None]) / coordinate_scale[None]).astype(np.float32)
    del features, coefficients

    # Predeclared empirical response: the detector's existing edge-local branch.
    local_raw = raw_edge_summary(
        standardized,
        int(config["top_frequency_count"]),
        remove_spatial_median=False,
    )
    local_center, local_scale = robust_center_scale(local_raw[fit_indices], 1e-4)
    local_edges = np.maximum((local_raw - local_center[None]) / local_scale[None], 0.0)
    measured_ds1 = np.median(local_edges[ds1_indices], axis=0)

    spearman = spearmanr(q_ds1, measured_ds1)
    kendall = kendalltau(q_ds1, measured_ds1, variant="b")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    spearman_ci = bootstrap_spearman(q_ds1, measured_ds1, rng)

    top_count = int(math.ceil(TOP_FRACTION * len(edges)))
    predicted_order = np.argsort(-q_ds1, kind="stable")
    measured_order = np.argsort(-measured_ds1, kind="stable")
    top_predicted = predicted_order[:top_count]
    remainder = predicted_order[top_count:]
    top_measured = set(map(int, measured_order[:top_count]))
    overlap = len(set(map(int, top_predicted)) & top_measured)
    rank_sum_test = mannwhitneyu(
        measured_ds1[top_predicted], measured_ds1[remainder], alternative="greater"
    )
    enrichment_p = float(
        hypergeom.sf(overlap - 1, len(edges), top_count, top_count)
    )
    top_summary = {
        "fraction": TOP_FRACTION,
        "top_count": top_count,
        "median_measured_response_top": float(np.median(measured_ds1[top_predicted])),
        "median_measured_response_remainder": float(np.median(measured_ds1[remainder])),
        "mean_rank_measured_response_top": float(np.mean(rankdata(-measured_ds1)[top_predicted])),
        "cliffs_delta": cliffs_delta(measured_ds1[top_predicted], measured_ds1[remainder]),
        "mann_whitney_u": float(rank_sum_test.statistic),
        "mann_whitney_one_sided_p": float(rank_sum_test.pvalue),
        "top_predicted_and_top_measured_overlap": overlap,
        "hypergeometric_enrichment_p": enrichment_p,
    }

    name_to_index = {name: index for index, name in enumerate(sensor_names)}
    all_sensor_indices = set(range(len(sensor_names)))

    def activated_links(companion: int, restored: int) -> np.ndarray:
        active_after = all_sensor_indices - {companion}
        return np.asarray(
            [
                edge_index
                for edge_index, edge in enumerate(edges)
                if restored in set(map(int, edge))
                and set(map(int, edge)).issubset(active_after)
            ],
            dtype=int,
        )

    def predicted_recovery(companion: int, restored: int) -> float:
        return float(np.sum(q_ds1[activated_links(companion, restored)]))

    single_rows = {
        row["sensor"]: row
        for row in csv.DictReader(
            (OUTPUT.parent / "expanded_criticality" / "single_quarantine_state_metrics.csv")
            .open(encoding="utf-8-sig", newline="")
        )
    }
    pair_rows: dict[frozenset[str], dict[str, Any]] = {}
    pair_root = OUTPUT.parent / "expanded_criticality" / "pairwise_parts"
    for path in sorted(pair_root.glob("pairwise_part*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["pairs"]:
            key = frozenset((row["first_sensor_name"], row["second_sensor_name"]))
            pair_rows[key] = row
    if len(pair_rows) != math.comb(len(sensor_names), 2):
        raise ValueError(f"Expected exhaustive pair audit, found {len(pair_rows)} pairs")

    selected_rules: dict[tuple[int, int], list[str]] = {}
    for companion_name, focal_name in FOCAL_RESTORATIONS:
        companion = name_to_index[companion_name]
        focal = name_to_index[focal_name]
        selected_rules.setdefault((companion, focal), []).append("focal")
        candidates = [index for index in range(len(sensor_names)) if index != companion]
        ordered = sorted(
            candidates,
            key=lambda index: (-predicted_recovery(companion, index), index),
        )
        selected_controls = {
            "highest": ordered[0],
            "median": ordered[(len(ordered) - 1) // 2],
            "lowest": ordered[-1],
        }
        for label, restored in selected_controls.items():
            selected_rules.setdefault((companion, restored), []).append(label)

    all_restoration_rows = []
    restoration_summaries = []
    activated_rows = []
    for companion_name, _ in FOCAL_RESTORATIONS:
        companion = name_to_index[companion_name]
        tpr_after = float(single_rows[companion_name]["DS1_tpr"])
        companion_rows = []
        for restored in range(len(sensor_names)):
            if restored == companion:
                continue
            restored_name = str(sensor_names[restored])
            links = activated_links(companion, restored)
            pair = pair_rows[frozenset((companion_name, restored_name))]
            tpr_before = float(
                pair["metrics_after_quarantine"]["per_state"]
                ["MVS_P2_DS1_NM_Z_01"]["detection_rate"]
            )
            rules = selected_rules.get((companion, restored), [])
            row = {
                "damage_state": USABLE_STATE,
                "companion_outage": companion_name,
                "restored_sensor": restored_name,
                "selection_rule": "; ".join(rules) if rules else "not selected",
                "activated_link_count": int(links.size),
                "predicted_delta_c": float(np.sum(q_ds1[links])),
                "measured_tpr_before": tpr_before,
                "measured_tpr_after": tpr_after,
                "measured_delta_tpr": tpr_after - tpr_before,
            }
            companion_rows.append(row)
            all_restoration_rows.append(row)
            if rules:
                for edge_index in links:
                    activated_rows.append(
                        {
                            "damage_state": USABLE_STATE,
                            "companion_outage": companion_name,
                            "restored_sensor": restored_name,
                            "selection_rule": row["selection_rule"],
                            "activated_measurement_link": labels[int(edge_index)],
                            "q_m_ds1": float(q_ds1[int(edge_index)]),
                        }
                    )
        predicted = np.asarray([row["predicted_delta_c"] for row in companion_rows])
        measured = np.asarray([row["measured_delta_tpr"] for row in companion_rows])
        rho = spearmanr(predicted, measured)
        tau = kendalltau(predicted, measured, variant="b")
        restoration_summaries.append(
            {
                "companion_outage": companion_name,
                "candidate_sensor_count": len(companion_rows),
                "spearman_rho": float(rho.statistic),
                "spearman_p": float(rho.pvalue),
                "spearman_bootstrap_95_ci": bootstrap_spearman(predicted, measured, rng),
                "kendall_tau_b": float(tau.statistic),
                "kendall_p": float(tau.pvalue),
                "negative_measured_recovery_count": int(np.sum(measured < 0)),
                "zero_measured_recovery_count": int(np.sum(measured == 0)),
            }
        )

    restoration_rows = [
        row for row in all_restoration_rows if row["selection_rule"] != "not selected"
    ]
    for (companion, restored), expected in EXPECTED_FOCAL_TPR_GAINS.items():
        actual = next(
            row["measured_delta_tpr"]
            for row in restoration_rows
            if row["companion_outage"] == companion
            and row["restored_sensor"] == restored
        )
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Focal restoration mismatch for {companion}/{restored}: {actual} != {expected}"
            )

    link_rows = [
        {
            "measurement_link": label,
            "q_m_ds1": float(q_ds1[index]),
            "R_m_ds1": float(measured_ds1[index]),
            "predicted_rank": int(rankdata(-q_ds1, method="min")[index]),
            "measured_rank": int(rankdata(-measured_ds1, method="min")[index]),
            "in_top_10_percent_predicted": bool(index in set(map(int, top_predicted))),
        }
        for index, label in enumerate(labels)
    ]

    protocol = {
        "purpose": "physical consistency check of q_md, not validation of the restoration trajectory",
        "damage_mapping": DAMAGE_MAPPING,
        "usable_state": USABLE_STATE,
        "candidate_location": "AL14--AL18",
        "empirical_link_response": {
            "symbol": "R_m,s",
            "healthy_reference_recording_ids": config["fit_recording_ids"],
            "healthy_reference_description": (
                "MVS_P1_UDS_SM_Z_01, MVS_P1_UDS_SM_Z_02, and MVS_P2_UDS_SM_Z_01"
            ),
            "damage_recording": "MVS_P2_DS1_NM_Z_01",
            "window": "six rolling 10-s blocks (60 s), 10-s step",
            "edge_descriptors": (
                "forward and reverse log-transfer magnitude, coherence-weighted phase cosine "
                "and sine, and magnitude-squared coherence"
            ),
            "aggregation": (
                "healthy-standardize every descriptor; average log1p(z^2) across the five "
                "descriptors; average the eight largest frequency-bin values; robustly "
                "standardize each edge on healthy fit windows; truncate below zero; take the "
                "median over DS1 windows"
            ),
            "unavailable_links": "not applicable in the full-array link-level check",
        },
        "primary_comparison": "Spearman rank correlation across all 122 measurement links",
        "robustness_comparison": "Kendall tau-b",
        "uncertainty": f"percentile bootstrap over links, {BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}",
        "top_link_check": "top 10% by q (ceil(0.10*122)=13) versus remaining 109 links",
        "restoration_controls": (
            "for each fixed companion outage, choose highest, median, and lowest predicted "
            "DS1-specific coverage recovery; sensor index breaks ties"
        ),
        "restoration_comparison": (
            "Spearman and Kendall rank association across all 57 possible restored sensors "
            "for each fixed companion outage; measured recovery is DS1 TPR with only the "
            "companion absent minus DS1 TPR with the companion and candidate both absent"
        ),
    }
    results = {
        "protocol": protocol,
        "link_level": {
            "state": USABLE_STATE,
            "candidate_edge_index": ds1_edge,
            "candidate_location": "AL14--AL18",
            "link_count": int(len(edges)),
            "damage_window_count": int(ds1_indices.size),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "spearman_bootstrap_95_ci": spearman_ci,
            "kendall_tau_b": float(kendall.statistic),
            "kendall_p": float(kendall.pvalue),
            "top_link_check": top_summary,
        },
        "restoration_consistency": {
            "all_candidate_rank_results": restoration_summaries,
            "preselected_examples": restoration_rows,
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "analysis_protocol.json").write_text(
        json.dumps(protocol, indent=2, default=json_default), encoding="utf-8"
    )
    (OUTPUT / "physical_consistency_results.json").write_text(
        json.dumps(results, indent=2, default=json_default), encoding="utf-8"
    )
    write_csv(OUTPUT / "link_level_ds1.csv", link_rows)
    write_csv(OUTPUT / "restoration_consistency_ds1.csv", all_restoration_rows)
    write_csv(OUTPUT / "restoration_selected_examples_ds1.csv", restoration_rows)
    write_csv(OUTPUT / "restoration_activated_links_ds1.csv", activated_rows)
    write_csv(OUTPUT / "damage_mapping.csv", DAMAGE_MAPPING)
    print(json.dumps(results["link_level"], indent=2))
    print(json.dumps(restoration_summaries, indent=2))
    print(json.dumps(restoration_rows, indent=2))
    print(f"saved={OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
