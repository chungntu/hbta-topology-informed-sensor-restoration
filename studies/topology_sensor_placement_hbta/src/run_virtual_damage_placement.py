"""Healthy graph-perturbation sensor placement with one-sensor-fault guarantees."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from run_placement import (
    active_edge_indices,
    compact_metric,
    exact_score_bank,
    json_default,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_sha256(path: Path) -> str:
    """Hash JSON content independently of indentation and line endings."""
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def edge_incidence(edges: np.ndarray, node_count: int) -> np.ndarray:
    matrix = np.zeros((len(edges), node_count), dtype=np.float64)
    matrix[np.arange(len(edges)), edges[:, 0]] = 1.0
    matrix[np.arange(len(edges)), edges[:, 1]] = -1.0
    return matrix


def healthy_edge_reliability(
    features: np.ndarray, healthy_indices: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reliability from healthy coherence and transfer-magnitude stability only."""
    healthy = features[healthy_indices]
    coherence = np.median(healthy[..., 4], axis=(0, 2))
    forward_reverse = healthy[..., :2]
    frequency_medians = np.median(forward_reverse, axis=2)
    variability = np.median(np.std(frequency_medians, axis=0), axis=1)
    raw = coherence / np.maximum(variability, 1e-4)
    low, high = np.quantile(raw, [0.05, 0.95])
    clipped = np.clip(raw, low, high)
    reliability = clipped / np.mean(clipped)
    return reliability.astype(np.float64), {
        "coherence_median": float(np.median(coherence)),
        "variability_median": float(np.median(variability)),
        "raw_reliability_quantiles": np.quantile(raw, [0, 0.05, 0.5, 0.95, 1]).tolist(),
        "normalized_reliability_quantiles": np.quantile(
            reliability, [0, 0.05, 0.5, 0.95, 1]
        ).tolist(),
    }


def virtual_damage_sensitivity(
    edges: np.ndarray,
    physical_weights: np.ndarray,
    reliability: np.ndarray,
    damage_reductions: list[float],
    alpha_factors: list[float],
    node_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Edge-measurement sensitivity to every virtual physical-edge weakening."""
    incidence = edge_incidence(edges, node_count)
    effective = physical_weights.astype(np.float64) * np.sqrt(reliability)
    effective /= np.mean(effective)
    laplacian = incidence.T @ (effective[:, None] * incidence)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    positive = eigenvalues[eigenvalues > 1e-8]
    reference = float(np.median(positive))
    alphas = [reference * float(factor) for factor in alpha_factors]
    base_resolvents = [
        np.linalg.inv(laplacian + alpha * np.eye(node_count)) for alpha in alphas
    ]
    sensitivity = np.zeros((len(edges), len(edges)), dtype=np.float64)
    for damage_edge, vector in enumerate(incidence):
        edge_outer = np.outer(vector, vector)
        for reduction in damage_reductions:
            damaged_laplacian = (
                laplacian - float(reduction) * effective[damage_edge] * edge_outer
            )
            for alpha, baseline in zip(alphas, base_resolvents):
                damaged = np.linalg.inv(
                    damaged_laplacian + alpha * np.eye(node_count)
                )
                response_change = incidence @ (damaged - baseline)
                sensitivity[:, damage_edge] += (
                    np.sum(response_change**2, axis=1) * reliability
                )
    column_sums = sensitivity.sum(axis=0)
    if np.any(column_sums <= 0) or not np.isfinite(sensitivity).all():
        raise ValueError("Virtual-damage sensitivity is degenerate")
    normalized = sensitivity / column_sums[None, :]
    return normalized, {
        "surrogate_warning": (
            "Graph edge weights are coordinate/drawing-informed transfer proxies, "
            "not verified member stiffnesses."
        ),
        "laplacian_positive_eigenvalue_median": reference,
        "resolvent_alphas": alphas,
        "damage_weight_reductions": damage_reductions,
        "sensitivity_min": float(normalized.min()),
        "sensitivity_max": float(normalized.max()),
        "effective_weight_quantiles": np.quantile(
            effective, [0, 0.05, 0.5, 0.95, 1]
        ).tolist(),
    }


def robust_observability(
    sensitivity: np.ndarray,
    edges: np.ndarray,
    active_edges: np.ndarray,
    node_count: int,
) -> dict[str, Any]:
    values = []
    ratios = []
    active_mask = np.zeros(len(edges), dtype=bool)
    active_mask[active_edges] = True
    for damage_edge in range(len(edges)):
        for failed_sensor in range(node_count):
            available = active_mask & ~np.any(edges == failed_sensor, axis=1)
            full_available = ~np.any(edges == failed_sensor, axis=1)
            observed = float(sensitivity[available, damage_edge].sum())
            reference = float(sensitivity[full_available, damage_edge].sum())
            values.append(observed)
            ratios.append(observed / max(reference, 1e-12))
    return {
        "worst_absolute_observability": float(np.min(values)),
        "worst_fraction_of_fault_matched_full": float(np.min(ratios)),
        "p05_fraction_of_fault_matched_full": float(np.quantile(ratios, 0.05)),
        "median_fraction_of_fault_matched_full": float(np.median(ratios)),
    }


def solve_layout(
    sensitivity: np.ndarray,
    edges: np.ndarray,
    fraction: float,
    node_count: int,
    two_fold_coverage: bool,
    degree_two: bool,
    time_limit: float,
) -> tuple[list[int], np.ndarray, dict[str, Any]]:
    edge_count = len(edges)
    variable_count = node_count + edge_count
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[:node_count] = 1.0
    objective[node_count:] = -1e-6
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    # y_uv is exactly x_u AND x_v.
    for edge_index, (left, right) in enumerate(edges):
        y = node_count + edge_index
        row = np.zeros(variable_count); row[y] = 1; row[left] = -1
        rows.append(row); lower.append(-np.inf); upper.append(0)
        row = np.zeros(variable_count); row[y] = 1; row[right] = -1
        rows.append(row); lower.append(-np.inf); upper.append(0)
        row = np.zeros(variable_count); row[y] = 1; row[left] = -1; row[right] = -1
        rows.append(row); lower.append(-1); upper.append(np.inf)

    adjacency = np.zeros((node_count, node_count), dtype=np.float64)
    for left, right in edges:
        adjacency[left, right] = adjacency[right, left] = 1.0
    if two_fold_coverage:
        closed = adjacency + np.eye(node_count)
        for sensor in range(node_count):
            row = np.zeros(variable_count); row[:node_count] = closed[sensor]
            rows.append(row); lower.append(2); upper.append(np.inf)
    if degree_two:
        for sensor in range(node_count):
            row = np.zeros(variable_count)
            row[:node_count] = adjacency[sensor]
            row[sensor] -= 2
            rows.append(row); lower.append(0); upper.append(np.inf)

    # Preserve a fraction of full-graph sensitivity for every virtual damage
    # edge after every possible one-node sensor failure.
    for damage_edge in range(edge_count):
        for failed_sensor in range(node_count):
            available = ~np.any(edges == failed_sensor, axis=1)
            reference = float(sensitivity[available, damage_edge].sum())
            row = np.zeros(variable_count)
            row[node_count:][available] = sensitivity[available, damage_edge]
            rows.append(row)
            lower.append(float(fraction) * reference)
            upper.append(np.inf)

    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(
            np.asarray(rows), np.asarray(lower), np.asarray(upper)
        ),
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
    )
    if result.x is None:
        raise RuntimeError(f"MILP failed for fraction={fraction}: {result.message}")
    sensors = np.flatnonzero(result.x[:node_count] > 0.5).tolist()
    active = np.flatnonzero(result.x[node_count:] > 0.5)
    exact_active = active_edge_indices(edges, sensors)
    if not np.array_equal(active, exact_active):
        raise AssertionError("MILP edge variables do not match induced topology")
    return sensors, active, {
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": result.message,
        "objective": float(result.fun),
        "elapsed_seconds": time.perf_counter() - started,
    }


def state_robust_metric(metric: dict) -> dict[str, Any]:
    states = metric["per_state"]
    worst_auc_name, worst_auc = min(
        states.items(), key=lambda row: row[1]["auroc"]
    )
    worst_detection_name, worst_detection = min(
        states.items(), key=lambda row: row[1]["detection_rate"]
    )
    return {
        "worst_state_auroc": worst_auc["auroc"],
        "worst_auroc_state": worst_auc_name,
        "worst_state_detection_rate": worst_detection["detection_rate"],
        "worst_detection_state": worst_detection_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--sensitivity-only",
        action="store_true",
        help="Generate the locked q_md sensitivity bank without running legacy layout evaluation.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, config["source_repo"])
    from hbta_pipeline.edge_spectral_topology_experiment import (
        calibration_normalized_sum,
        candidate_edge_kernel,
        edge_spectral_features,
        rolling_recording_ids,
    )
    from hbta_pipeline.spatial_topology import build_hbta_spatial_graph, graph_audit
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
        sensor_names = data["sensor_names"].astype(str)
        positions = data["sensor_positions"].astype(np.float64)
    with np.load(config["teacher_state"]) as teacher:
        coordinate_center = teacher["coord_center"].astype(np.float32)
        coordinate_scale = teacher["coord_scale"].astype(np.float32)
    graph = build_hbta_spatial_graph(sensor_names, positions)
    node_count = len(sensor_names)
    recording_ids = rolling_recording_ids(source_ids, config["rolling_blocks"])
    features = edge_spectral_features(
        coefficients, graph.edges, source_ids, config["rolling_blocks"]
    )
    healthy_selection = np.flatnonzero(
        np.isin(recording_ids, config["healthy_selection_recording_ids"])
    )
    reliability, reliability_audit = healthy_edge_reliability(
        features, healthy_selection
    )
    sensitivity, sensitivity_audit = virtual_damage_sensitivity(
        graph.edges, graph.weights, reliability,
        config["damage_weight_reductions"], config["resolvent_alpha_factors"],
        node_count,
    )
    np.savez_compressed(
        output / "virtual_damage_sensitivity.npz",
        sensitivity=sensitivity.astype(np.float32),
        reliability=reliability.astype(np.float32),
        edges=graph.edges,
        sensor_names=sensor_names,
    )
    if args.sensitivity_only:
        metadata = {
            "experiment": "healthy_graph_virtual_damage_response_weights",
            "config": args.config.as_posix(),
            "config_sha256": canonical_json_sha256(config_path),
            "prepared_spectra_sha256": sha256(Path(config["prepared_spectra"])),
            "selection_recording_ids": config["healthy_selection_recording_ids"],
            "physical_damage_data_used": False,
            "candidate_damage_location_count": int(sensitivity.shape[1]),
            "measurement_link_count": int(sensitivity.shape[0]),
            "graph": graph_audit(graph, sensor_names),
            "healthy_reliability_audit": reliability_audit,
            "sensitivity_audit": sensitivity_audit,
        }
        (output / "sensitivity_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=json_default),
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps(metadata, indent=2, default=json_default))
        return

    layouts = {}
    for fraction in config["observability_fractions"]:
        sensors, active, solver = solve_layout(
            sensitivity, graph.edges, float(fraction), node_count,
            config["require_two_fold_node_coverage"],
            config["require_selected_degree_two"],
            config["solver_time_limit_seconds"],
        )
        key = f"virtual_eta_{float(fraction):.2f}".replace(".", "p")
        layouts[key] = {
            "method": "healthy_graph_virtual_damage_robust_milp",
            "observability_fraction": float(fraction),
            "budget": len(sensors),
            "sensors": sensors,
            "sensor_names": sensor_names[sensors].tolist(),
            "active_edge_indices": active.tolist(),
            "active_edge_count": int(active.size),
            "virtual_observability": robust_observability(
                sensitivity, graph.edges, active, node_count
            ),
            "solver": solver,
        }
        print(
            f"fraction={fraction:.2f} sensors={len(sensors)} edges={active.size} "
            f"worst={layouts[key]['virtual_observability']['worst_fraction_of_fault_matched_full']:.4f}",
            flush=True,
        )

    primary_fraction = float(config["primary_observability_fraction"])
    primary_key = f"virtual_eta_{primary_fraction:.2f}".replace(".", "p")
    if primary_key not in layouts:
        raise ValueError("Primary observability fraction is not in the locked sweep")
    lock = {
        "experiment": config["experiment"],
        "status": "healthy/topology-only selection locked before real damage evaluation",
        "config": args.config.as_posix(),
        "config_sha256": canonical_json_sha256(config_path),
        "prepared_spectra_sha256": sha256(Path(config["prepared_spectra"])),
        "selection_recording_ids": config["healthy_selection_recording_ids"],
        "damage_recordings_or_locations_used_for_selection": False,
        "virtual_damage_domain": {
            "candidate_damage_edges": int(len(graph.edges)),
            "weight_reductions": config["damage_weight_reductions"],
            "possible_single_sensor_failures": node_count,
        },
        "primary_layout_key_selected_before_damage_evaluation": primary_key,
        "healthy_reliability_audit": reliability_audit,
        "sensitivity_audit": sensitivity_audit,
        "layouts": layouts,
    }
    lock_path = output / "protocol.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, default=json_default), encoding="utf-8")

    # Real DS1-DS8 evaluation begins only after the lock above exists.
    standardized = ((features - coordinate_center[None]) / coordinate_scale[None]).astype(np.float32)
    distances = graph_distance_matrix(graph, node_count)
    fit_indices = np.flatnonzero(np.isin(recording_ids, config["fit_recording_ids"]))
    calibration_indices = np.flatnonzero(recording_ids == config["calibration_recording_id"])
    healthy_test_indices = np.flatnonzero(recording_ids == config["healthy_test_recording_id"])
    results = {}
    for key, layout in layouts.items():
        active = np.asarray(layout["active_edge_indices"], dtype=int)
        scores, _ = exact_score_bank(
            standardized, graph.edges, distances, active, fit_indices,
            calibration_indices, config["top_frequency_count"],
            config["kernel_length_m"], config["topology_weight"],
            candidate_edge_kernel, standardized_location_scores, scan_score,
            calibration_normalized_sum,
        )
        metric = compact_metric(metric_result(
            scores, recording_ids, recording_names, calibration_indices,
            external_healthy_id=config["healthy_test_recording_id"],
            damage_start_id=config["damage_start_recording_id"],
            external_healthy_indices=healthy_test_indices,
        ))
        results[key] = {
            **layout,
            "status": "evaluated_after_protocol_lock",
            "metrics": metric,
            "state_robust_metrics": state_robust_metric(metric),
        }
    payload = {
        "experiment": config["experiment"],
        "interpretation": (
            "The method is damage-location-agnostic at selection time; real DS1-DS8 "
            "remain a retrospective single-structure validation set."
        ),
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": sha256(lock_path),
        "primary_layout_key": primary_key,
        "graph": graph_audit(graph, sensor_names),
        "results": results,
    }
    result_path = output / "placement_results.json"
    result_path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps({
        key: {
            "budget": row["budget"],
            "nominal_auroc": row["metrics"]["block_auroc"],
            "fpr": row["metrics"]["external_healthy_fpr"],
            **row["state_robust_metrics"],
        }
        for key, row in results.items()
    }, indent=2, default=json_default))


if __name__ == "__main__":
    main()
