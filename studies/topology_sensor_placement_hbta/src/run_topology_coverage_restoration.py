"""Lock topology-coverage restoration sequences for bounded sensor-loss states.

The selection phase is blind to the physical damage recordings. It uses only
sensor metadata and the healthy-graph virtual-damage response-weight bank.
For each outage, sensors are restored greedily to maximize the mean coverage of
the lowest-covered 10% of graph-wide candidate damage locations.  The complete
selection protocol is written before any response coefficient is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from run_placement import active_edge_indices, compact_metric, exact_score_bank, json_default


CASE_NAMES = (
    "deck_row_10",
    "truss_wall_9",
    "central_cluster_10",
    "distributed_10_seed_1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(path: Path) -> str:
    """Hash JSON content independently of indentation and line endings."""
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def case_indices(
    case_name: str, sensor_names: np.ndarray, positions: np.ndarray
) -> list[int]:
    is_al = np.char.startswith(sensor_names.astype(str), "AL")
    is_ag = np.char.startswith(sensor_names.astype(str), "AG")
    if case_name == "deck_row_10":
        return np.flatnonzero(is_al & np.isclose(positions[:, 1], -0.95)).astype(int).tolist()
    if case_name == "truss_wall_9":
        return np.flatnonzero(is_ag & (positions[:, 1] > 0.0)).astype(int).tolist()
    if case_name == "central_cluster_10":
        return np.flatnonzero(np.abs(positions[:, 0]) <= 1.75 + 1e-6).astype(int).tolist()
    if case_name == "distributed_10_seed_1":
        return sorted(
            np.random.default_rng(1)
            .choice(len(sensor_names), size=10, replace=False)
            .astype(int)
            .tolist()
        )
    raise ValueError(f"Unknown case: {case_name}")


def coverage_vector(
    active_sensors: set[int], edges: np.ndarray, sensitivity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    active = np.zeros(int(edges.max()) + 1, dtype=bool)
    active[list(active_sensors)] = True
    active_edges = np.flatnonzero(active[edges[:, 0]] & active[edges[:, 1]])
    if active_edges.size:
        coverage = sensitivity[active_edges].sum(axis=0, dtype=np.float64)
    else:
        coverage = np.zeros(sensitivity.shape[1], dtype=np.float64)
    return coverage, active_edges


def coverage_summary(coverage: np.ndarray, active_edge_count: int, tail_count: int) -> dict:
    lower_tail = np.partition(coverage, tail_count - 1)[:tail_count]
    return {
        "lower_tail_mean": float(np.mean(lower_tail)),
        "mean": float(np.mean(coverage)),
        "minimum": float(np.min(coverage)),
        "median": float(np.median(coverage)),
        "active_edge_count": int(active_edge_count),
    }


def lock_case(
    case_name: str,
    initial_outage: list[int],
    sensor_names: np.ndarray,
    edges: np.ndarray,
    sensitivity: np.ndarray,
    tail_count: int,
) -> dict:
    all_sensors = set(range(len(sensor_names)))
    active = all_sensors - set(initial_outage)
    coverage, active_edges = coverage_vector(active, edges, sensitivity)
    current = coverage_summary(coverage, active_edges.size, tail_count)
    initial = dict(current)
    remaining = set(initial_outage)
    steps = []

    while remaining:
        candidates = []
        for sensor in sorted(remaining):
            candidate_coverage, candidate_edges = coverage_vector(
                active | {sensor}, edges, sensitivity
            )
            summary = coverage_summary(
                candidate_coverage, candidate_edges.size, tail_count
            )
            candidates.append(
                {
                    "sensor_index": sensor,
                    "sensor_name": str(sensor_names[sensor]),
                    "summary": summary,
                    "lower_tail_gain": summary["lower_tail_mean"]
                    - current["lower_tail_mean"],
                    "mean_gain": summary["mean"] - current["mean"],
                    "active_edges_added": summary["active_edge_count"]
                    - current["active_edge_count"],
                }
            )
        chosen = max(
            candidates,
            key=lambda row: (
                round(row["lower_tail_gain"], 15),
                round(row["mean_gain"], 15),
                -row["sensor_index"],
            ),
        )
        active.add(int(chosen["sensor_index"]))
        remaining.remove(int(chosen["sensor_index"]))
        current = chosen["summary"]
        steps.append(
            {
                "step": len(steps) + 1,
                "restored_sensor_index": chosen["sensor_index"],
                "restored_sensor_name": chosen["sensor_name"],
                "lower_tail_gain": chosen["lower_tail_gain"],
                "mean_gain": chosen["mean_gain"],
                "active_edges_added": chosen["active_edges_added"],
                "coverage": chosen["summary"],
            }
        )

    return {
        "case_name": case_name,
        "initial_outage_sensor_indices": initial_outage,
        "initial_outage_sensor_names": [str(sensor_names[i]) for i in initial_outage],
        "initial_active_sensor_count": len(sensor_names) - len(initial_outage),
        "initial_coverage": initial,
        "restoration_steps": steps,
    }


def mean_tpr(metric: dict) -> float:
    return float(np.mean([row["detection_rate"] for row in metric["per_state"].values()]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/placement.json"))
    parser.add_argument(
        "--virtual-config",
        type=Path,
        default=Path("configs/virtual_damage_pareto.json"),
    )
    parser.add_argument(
        "--sensitivity",
        type=Path,
        default=Path("outputs/virtual_damage_pareto/virtual_damage_sensitivity.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/topology_coverage_restoration")
    )
    parser.add_argument("--tail-fraction", type=float, default=0.10)
    parser.add_argument(
        "--validation-steps", type=int, nargs="+", default=[0, 1, 2, 3, 5, 10]
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Write the state-dependent restoration protocol without legacy detector validation.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    config_path = args.config.resolve()
    virtual_config_path = args.virtual_config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    virtual_config = json.loads(virtual_config_path.read_text(encoding="utf-8"))

    # Selection phase: only topology metadata and healthy-graph sensitivity are read.
    with np.load(args.sensitivity) as data:
        sensitivity = data["sensitivity"].astype(np.float64)
        edges = data["edges"].astype(np.int64)
        sensor_names = data["sensor_names"].astype(str)
    with np.load(config["prepared_spectra"]) as metadata:
        metadata_names = metadata["sensor_names"].astype(str)
        positions = metadata["sensor_positions"].astype(np.float64)
    if not np.array_equal(sensor_names, metadata_names):
        raise ValueError("Sensor ordering differs between topology metadata and sensitivity bank")
    if not np.allclose(sensitivity.sum(axis=0), 1.0, atol=1e-5):
        raise ValueError("Each candidate damage-location sensitivity must sum to one")

    tail_count = max(1, int(math.ceil(args.tail_fraction * sensitivity.shape[1])))
    cases = []
    for case_name in CASE_NAMES:
        initial_outage = case_indices(case_name, sensor_names, positions)
        cases.append(
            lock_case(
                case_name,
                initial_outage,
                sensor_names,
                edges,
                sensitivity,
                tail_count,
            )
        )

    protocol = {
        "experiment": "outage-conditioned topology-coverage restoration",
        "selection_is_blind_to_physical_damage_recordings": True,
        "selection_inputs": {
            "sensor_metadata": "sensor names and coordinates only",
            "sensitivity_bank": str(args.sensitivity),
            "sensitivity_bank_sha256": sha256(args.sensitivity.resolve()),
            "virtual_damage_config_sha256": canonical_json_sha256(virtual_config_path),
            "candidate_damage_location_count": int(sensitivity.shape[1]),
            "measurement_edge_count": int(sensitivity.shape[0]),
            "damage_weight_reductions": virtual_config["damage_weight_reductions"],
            "resolvent_alpha_factors": virtual_config["resolvent_alpha_factors"],
        },
        "utility": {
            "name": "lower-tail topology coverage",
            "definition": (
                "mean of the lowest-covered ceil(qD) candidate damage locations; "
                "coverage at a location is the sum of normalized sensitivities over "
                "measurement edges whose two endpoint sensors are active"
            ),
            "tail_fraction": args.tail_fraction,
            "tail_count": tail_count,
            "greedy_tie_breaks": ["mean coverage gain", "lowest sensor index"],
            "monotonicity": (
                "adding a sensor cannot remove an active measurement edge, so every "
                "location coverage and the lower-tail mean are nondecreasing"
            ),
        },
        "outage_cases": {
            "deck_row_10": (
                "ten AL sensors on one longitudinal deck row unavailable"
            ),
            "truss_wall_9": (
                "nine AG sensors on one truss wall unavailable"
            ),
            "central_cluster_10": (
                "ten sensors in the two central deck columns and adjacent truss locations unavailable"
            ),
            "distributed_10_seed_1": (
                "ten unavailable sensors sampled without replacement, seed 1"
            ),
        },
        "sensor_names": sensor_names.tolist(),
        "sensor_positions": positions.tolist(),
        "edges": edges.tolist(),
        "cases": cases,
    }
    protocol_path = args.output / "protocol.lock.json"
    serialized_protocol = json.dumps(protocol, indent=2, default=json_default)
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError(
                "Existing protocol lock differs; use a new output directory for a new protocol"
            )
    protocol_path.write_text(serialized_protocol, encoding="utf-8", newline="\n")
    protocol_hash = sha256(protocol_path)
    print(f"locked={protocol_path.resolve()} sha256={protocol_hash}", flush=True)

    if args.selection_only:
        return

    # Physical validation phase starts only after the topology-only protocol is locked.
    sys.path.insert(0, config["source_repo"])
    from hbta_pipeline.edge_spectral_topology_experiment import (
        calibration_normalized_sum,
        candidate_edge_kernel,
        edge_spectral_features,
        rolling_recording_ids,
    )
    from hbta_pipeline.spatial_topology import build_hbta_spatial_graph
    from hbta_pipeline.topology_location_scan_experiment import (
        graph_distance_matrix,
        scan_score,
        standardized_location_scores,
    )
    from hbta_pipeline.vqpiae_experiment import metric_result

    with np.load(config["prepared_spectra"]) as data:
        coefficients = data["fourier_coefficients"].astype(np.complex64)
        source_ids = data["recording_ids"].astype(np.int16)
        recording_names = data["recording_names"].astype(str)
    with np.load(config["teacher_state"]) as teacher:
        coordinate_center = teacher["coord_center"].astype(np.float32)
        coordinate_scale = teacher["coord_scale"].astype(np.float32)

    graph = build_hbta_spatial_graph(sensor_names, positions)
    if not np.array_equal(graph.edges, edges):
        raise ValueError("Rebuilt physical topology does not match locked sensitivity topology")
    distances = graph_distance_matrix(graph, len(sensor_names))
    recording_ids = rolling_recording_ids(source_ids, config["rolling_blocks"])
    fit_indices = np.flatnonzero(np.isin(recording_ids, config["fit_recording_ids"]))
    calibration_indices = np.flatnonzero(recording_ids == config["calibration_recording_id"])
    healthy_test_indices = np.flatnonzero(recording_ids == config["healthy_test_recording_id"])
    features = edge_spectral_features(
        coefficients, graph.edges, source_ids, config["rolling_blocks"]
    )
    standardized = ((features - coordinate_center[None]) / coordinate_scale[None]).astype(
        np.float32
    )
    del features, coefficients

    metric_args = dict(
        recording_ids=recording_ids,
        recording_names=recording_names,
        calibration_indices=calibration_indices,
        external_healthy_id=config["healthy_test_recording_id"],
        damage_start_id=config["damage_start_recording_id"],
        external_healthy_indices=healthy_test_indices,
    )
    score_args = dict(
        standardized_features=standardized,
        graph_edges=graph.edges,
        graph_distances=distances,
        fit_indices=fit_indices,
        calibration_indices=calibration_indices,
        top_frequency_count=config["top_frequency_count"],
        kernel_length_m=config["kernel_length_m"],
        topology_weight=config["topology_weight"],
        candidate_edge_kernel=candidate_edge_kernel,
        standardized_location_scores=standardized_location_scores,
        scan_score=scan_score,
        calibration_normalized_sum=calibration_normalized_sum,
    )
    all_sensors = set(range(len(sensor_names)))

    def evaluate(active_sensors: set[int]) -> dict[str, Any]:
        active = active_edge_indices(graph.edges, sorted(active_sensors))
        if active.size < 2:
            raise ValueError("Physical validation requires at least two active edges")
        scores, _ = exact_score_bank(active_edges=active, **score_args)
        metric = compact_metric(metric_result(scores, **metric_args))
        metric["mean_tpr"] = mean_tpr(metric)
        metric["active_sensor_count"] = len(active_sensors)
        metric["active_edge_count"] = int(active.size)
        return metric

    validation = []
    for case in cases:
        initial_outage = set(case["initial_outage_sensor_indices"])
        restored_order = [
            int(row["restored_sensor_index"]) for row in case["restoration_steps"]
        ]
        available_steps = sorted(
            {step for step in args.validation_steps if 0 <= step <= len(restored_order)}
        )
        rows = []
        for step in available_steps:
            active_sensors = (all_sensors - initial_outage) | set(restored_order[:step])
            metric = evaluate(active_sensors)
            coverage = (
                case["initial_coverage"]
                if step == 0
                else case["restoration_steps"][step - 1]["coverage"]
            )
            rows.append(
                {
                    "restored_sensor_count": step,
                    "last_restored_sensor_name": (
                        None if step == 0 else case["restoration_steps"][step - 1]["restored_sensor_name"]
                    ),
                    "coverage": coverage,
                    "physical_validation": metric,
                }
            )
            print(
                f"case={case['case_name']} restored={step:02d} "
                f"coverage={coverage['lower_tail_mean']:.6f} "
                f"auroc={metric['block_auroc']:.6f}",
                flush=True,
            )
        validation.append({"case_name": case["case_name"], "rows": rows})

    result = {
        "experiment": "locked topology-coverage restoration with held-out physical validation",
        "protocol_lock": str(protocol_path),
        "protocol_lock_sha256": protocol_hash,
        "physical_validation_data_are_not_used_for_selection": True,
        "validation_steps": args.validation_steps,
        "validation": validation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = args.output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, default=json_default), encoding="utf-8"
    )
    print(f"saved={result_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
