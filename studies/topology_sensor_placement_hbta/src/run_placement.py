"""Minimum sensor placement using the original HBTA topology detector."""

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


def json_default(value: Any):
    """Convert NumPy containers/scalars without changing experiment values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_center_scale(values: np.ndarray, floor: float = 1e-6):
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center[None]), axis=0)
    standard_deviation = np.std(values, axis=0)
    scale = np.maximum.reduce(
        [1.4826 * mad, 0.1 * standard_deviation, np.full_like(mad, floor)]
    )
    return center.astype(np.float32), scale.astype(np.float32)


def top_k_mean(values: np.ndarray, count: int) -> np.ndarray:
    selected = np.partition(values, values.shape[-1] - count, axis=-1)[..., -count:]
    return np.mean(selected, axis=-1)


def raw_edge_summary(
    standardized: np.ndarray,
    top_frequency_count: int | None,
    remove_spatial_median: bool,
) -> np.ndarray:
    values = standardized
    if remove_spatial_median:
        values = values - np.median(values, axis=1, keepdims=True)
    frequency_scores = np.mean(np.log1p(values**2), axis=3)
    if top_frequency_count is None:
        return np.mean(frequency_scores, axis=2)
    return top_k_mean(frequency_scores, top_frequency_count)


def active_edge_indices(edges: np.ndarray, sensors: list[int]) -> np.ndarray:
    selected = np.zeros(int(edges.max()) + 1, dtype=bool)
    selected[sensors] = True
    return np.flatnonzero(selected[edges[:, 0]] & selected[edges[:, 1]])


def normalized_scalar(scores: np.ndarray, fit_indices: np.ndarray) -> np.ndarray:
    center, scale = robust_center_scale(scores[fit_indices], 1e-6)
    return (scores - center) / scale


def approximate_components(
    robust_edges: np.ndarray,
    local_edges: np.ndarray,
    kernel: np.ndarray,
    active_edges: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stable = np.mean(robust_edges[:, active_edges], axis=1)
    selected_kernel = kernel[:, active_edges].copy()
    selected_kernel /= np.maximum(selected_kernel.sum(axis=1, keepdims=True), 1e-12)
    location = local_edges[:, active_edges] @ selected_kernel.T
    center, scale = robust_center_scale(location[fit_indices], 1e-6)
    topology = np.max((location - center[None]) / scale[None], axis=1)
    return stable, topology


def graph_penalty(
    sensors: list[int],
    edges: np.ndarray,
    relation_ids: np.ndarray,
    relation_count: int,
    node_count: int,
) -> float:
    selected = np.zeros(node_count, dtype=bool)
    selected[sensors] = True
    active = selected[edges[:, 0]] & selected[edges[:, 1]]
    if active.sum() < 2:
        return 100.0
    covered = selected.copy()
    for left, right in edges:
        if selected[left] or selected[right]:
            covered[left] = True
            covered[right] = True
    coverage_loss = 1.0 - float(np.mean(covered))
    counts = np.bincount(relation_ids[active], minlength=relation_count)
    relation_loss = float(np.mean(np.exp(-counts / 2.0)))
    incident = np.zeros(node_count, dtype=int)
    for left, right in edges[active]:
        incident[left] += 1
        incident[right] += 1
    isolated_loss = float(np.mean(incident[sensors] == 0))
    return coverage_loss + relation_loss + isolated_loss


def subset_objective(
    sensors: list[int],
    edges: np.ndarray,
    relation_ids: np.ndarray,
    robust_edges: np.ndarray,
    local_edges: np.ndarray,
    kernel: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    teacher_targets: tuple[np.ndarray, np.ndarray],
    teacher_weight: float,
) -> float:
    active = active_edge_indices(edges, sensors)
    if active.size < 2:
        return 1e6
    stable, topology = approximate_components(
        robust_edges, local_edges, kernel, active, fit_indices
    )
    stable = normalized_scalar(stable, fit_indices)
    topology = normalized_scalar(topology, fit_indices)
    teacher_stable, teacher_topology = teacher_targets
    difference = np.mean((stable[validation_indices] - teacher_stable[validation_indices]) ** 2)
    difference += 0.25 * np.mean(
        (topology[validation_indices] - teacher_topology[validation_indices]) ** 2
    )
    correlations = []
    for candidate, target in ((stable, teacher_stable), (topology, teacher_topology)):
        left = candidate[validation_indices]
        right = target[validation_indices]
        if np.std(left) > 1e-8 and np.std(right) > 1e-8:
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
        else:
            correlations.append(0.0)
    representation_loss = difference + 0.2 * sum(1.0 - value for value in correlations)
    topology_loss = graph_penalty(
        sensors, edges, relation_ids, int(relation_ids.max()) + 1, int(edges.max()) + 1
    )
    return teacher_weight * representation_loss + 0.2 * topology_loss


def backward_path(
    edges: np.ndarray,
    relation_ids: np.ndarray,
    robust_edges: np.ndarray,
    local_edges: np.ndarray,
    kernel: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    budgets: list[int],
    teacher_targets: tuple[np.ndarray, np.ndarray],
    teacher_weight: float,
) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
    current = list(range(int(edges.max()) + 1))
    requested = set(budgets)
    snapshots = {len(current): current.copy()} if len(current) in requested else {}
    trace = []
    while len(current) > min(budgets):
        candidates = []
        for removed in current:
            subset = [sensor for sensor in current if sensor != removed]
            loss = subset_objective(
                subset, edges, relation_ids, robust_edges, local_edges, kernel,
                fit_indices, validation_indices, teacher_targets, teacher_weight
            )
            candidates.append((loss, removed, subset))
        loss, removed, current = min(candidates, key=lambda row: (row[0], row[1]))
        trace.append({"sensor_count": len(current), "removed_sensor_index": removed, "objective": loss})
        if len(current) in requested:
            snapshots[len(current)] = current.copy()
    return snapshots, trace


def exact_score_bank(
    standardized_features: np.ndarray,
    graph_edges: np.ndarray,
    graph_distances: np.ndarray,
    active_edges: np.ndarray,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    top_frequency_count: int,
    kernel_length_m: float,
    topology_weight: float,
    candidate_edge_kernel,
    standardized_location_scores,
    scan_score,
    calibration_normalized_sum,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = standardized_features[:, active_edges]
    return exact_score_bank_from_selected(
        selected, graph_edges, graph_distances, active_edges, fit_indices,
        calibration_indices, top_frequency_count, kernel_length_m,
        topology_weight, candidate_edge_kernel, standardized_location_scores,
        scan_score, calibration_normalized_sum,
    )


def exact_score_bank_from_selected(
    selected: np.ndarray,
    graph_edges: np.ndarray,
    graph_distances: np.ndarray,
    active_edges: np.ndarray,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    top_frequency_count: int,
    kernel_length_m: float,
    topology_weight: float,
    candidate_edge_kernel,
    standardized_location_scores,
    scan_score,
    calibration_normalized_sum,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate an already sliced edge bank (also supports injected faults)."""
    robust_raw = raw_edge_summary(selected, None, True)
    robust_center, robust_scale = robust_center_scale(robust_raw[fit_indices], 1e-4)
    robust_edges = np.maximum(
        (robust_raw - robust_center[None]) / robust_scale[None], 0.0
    )
    stable_global = np.mean(robust_edges, axis=1)
    local_raw = raw_edge_summary(selected, top_frequency_count, False)
    local_center, local_scale = robust_center_scale(local_raw[fit_indices], 1e-4)
    local_edges = np.maximum((local_raw - local_center[None]) / local_scale[None], 0.0)
    kernel = candidate_edge_kernel(
        graph_distances, graph_edges[active_edges], kernel_length_m
    )
    locations = standardized_location_scores(local_edges, kernel, fit_indices)
    topology = scan_score(locations)
    combined = calibration_normalized_sum(
        stable_global, topology, calibration_indices, topology_weight
    )
    return combined, {
        "active_edge_count": int(active_edges.size),
        "stable_global": stable_global.astype(np.float32),
        "topology": topology.astype(np.float32),
    }


def compact_metric(result: dict) -> dict:
    per_state = {
        name: {
            "auroc": values["auroc_vs_external_healthy"],
            "detection_rate": values["detection_rate"],
        }
        for name, values in result["by_recording"].items()
        if values["auroc_vs_external_healthy"] is not None
    }
    return {
        "block_auroc": result["block_auroc"],
        "block_auprc": result["block_auprc"],
        "external_healthy_fpr": result["external_healthy_false_alarm_rate"],
        "threshold": result["threshold"],
        "macro_state_auroc": float(np.mean([row["auroc"] for row in per_state.values()])),
        "per_state": per_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
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
        teacher_robust_center = teacher["robust_center"].astype(np.float32)
        teacher_robust_scale = teacher["robust_scale"].astype(np.float32)
        teacher_local_center = teacher["local_center"].astype(np.float32)
        teacher_local_scale = teacher["local_scale"].astype(np.float32)
    graph = build_hbta_spatial_graph(sensor_names, positions)
    distances = graph_distance_matrix(graph, len(sensor_names))
    recording_ids = rolling_recording_ids(source_ids, config["rolling_blocks"])
    fit_indices = np.flatnonzero(np.isin(recording_ids, config["fit_recording_ids"]))
    validation_indices = np.flatnonzero(recording_ids == config["validation_recording_id"])
    calibration_indices = np.flatnonzero(recording_ids == config["calibration_recording_id"])
    healthy_test_indices = np.flatnonzero(recording_ids == config["healthy_test_recording_id"])
    features = edge_spectral_features(
        coefficients, graph.edges, source_ids, config["rolling_blocks"]
    )
    standardized = ((features - coordinate_center[None]) / coordinate_scale[None]).astype(np.float32)

    # Approximate full edge banks used only for healthy subset selection.
    full_robust_raw = raw_edge_summary(standardized, None, True)
    full_robust_edges = np.maximum(
        (full_robust_raw - teacher_robust_center[None]) / teacher_robust_scale[None], 0.0
    ).astype(np.float32)
    full_local_raw = raw_edge_summary(standardized, config["top_frequency_count"], False)
    full_local_edges = np.maximum(
        (full_local_raw - teacher_local_center[None]) / teacher_local_scale[None], 0.0
    ).astype(np.float32)
    full_kernel = candidate_edge_kernel(
        distances, graph.edges, config["kernel_length_m"]
    ).astype(np.float32)
    selection_limit = config["validation_recording_id"]
    selection_indices = np.flatnonzero(recording_ids <= selection_limit)
    selection_fit = np.flatnonzero(np.isin(recording_ids[selection_indices], config["fit_recording_ids"]))
    selection_validation = np.flatnonzero(
        recording_ids[selection_indices] == config["validation_recording_id"]
    )
    selection_robust = full_robust_edges[selection_indices]
    selection_local = full_local_edges[selection_indices]
    teacher_stable, teacher_topology = approximate_components(
        selection_robust, selection_local, full_kernel,
        np.arange(len(graph.edges)), selection_fit
    )
    teacher_targets = (
        normalized_scalar(teacher_stable, selection_fit),
        normalized_scalar(teacher_topology, selection_fit),
    )

    teacher_subsets, teacher_trace = backward_path(
        graph.edges, graph.relation_ids, selection_robust, selection_local,
        full_kernel, selection_fit, selection_validation, config["budgets"],
        teacher_targets, teacher_weight=1.0
    )
    topology_subsets, topology_trace = backward_path(
        graph.edges, graph.relation_ids, selection_robust, selection_local,
        full_kernel, selection_fit, selection_validation, config["budgets"],
        teacher_targets, teacher_weight=0.0
    )
    degree = np.zeros(len(sensor_names), dtype=np.float64)
    for (left, right), weight in zip(graph.edges, graph.weights):
        degree[left] += weight
        degree[right] += weight
    locked_subsets: dict[str, dict[str, Any]] = {
        "full_58": {"method": "full", "budget": 58, "sensors": list(range(58))}
    }
    for budget in config["budgets"]:
        locked_subsets[f"teacher_{budget}"] = {
            "method": "healthy_teacher_backward", "budget": budget,
            "sensors": teacher_subsets[budget],
        }
        locked_subsets[f"topology_{budget}"] = {
            "method": "topology_backward", "budget": budget,
            "sensors": topology_subsets[budget],
        }
        locked_subsets[f"degree_{budget}"] = {
            "method": "weighted_degree", "budget": budget,
            "sensors": np.argsort(-degree)[:budget].tolist(),
        }
        for seed in config["random_seeds"]:
            rng = np.random.default_rng(seed + budget)
            locked_subsets[f"random_{budget}_seed{seed}"] = {
                "method": "random", "budget": budget, "seed": seed,
                "sensors": rng.choice(len(sensor_names), budget, replace=False).tolist(),
            }
    for row in locked_subsets.values():
        active = active_edge_indices(graph.edges, row["sensors"])
        row["active_edge_indices"] = active.tolist()
        row["active_edge_count"] = int(active.size)
        row["sensor_names"] = sensor_names[row["sensors"]].tolist()
        row["relation_edge_counts"] = {
            str(name): int(np.sum(graph.relation_ids[active] == relation_id))
            for relation_id, name in enumerate(graph.relation_names)
        }
    lock = {
        "experiment": config["experiment"],
        "config": str(config_path), "config_sha256": sha256(config_path),
        "prepared_spectra": config["prepared_spectra"],
        "prepared_spectra_sha256": sha256(Path(config["prepared_spectra"])),
        "teacher_state": config["teacher_state"],
        "teacher_state_sha256": sha256(Path(config["teacher_state"])),
        "selection_recording_ids": [0, 1, 2, 3],
        "calibration_recording_id": config["calibration_recording_id"],
        "healthy_test_recording_id": config["healthy_test_recording_id"],
        "damage_start_recording_id": config["damage_start_recording_id"],
        "damage_or_external_healthy_used_for_selection": False,
        "subsets": locked_subsets,
    }
    lock_path = output / "protocol.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, default=json_default), encoding="utf-8")

    results = {}
    score_payload = {"recording_ids": recording_ids, "recording_names": recording_names}
    for index, (name, row) in enumerate(locked_subsets.items(), start=1):
        active = np.asarray(row["active_edge_indices"], dtype=int)
        if active.size < 2:
            results[name] = {**row, "status": "insufficient induced edges"}
            continue
        scores, diagnostics = exact_score_bank(
            standardized, graph.edges, distances, active, fit_indices,
            calibration_indices, config["top_frequency_count"],
            config["kernel_length_m"], config["topology_weight"],
            candidate_edge_kernel, standardized_location_scores, scan_score,
            calibration_normalized_sum,
        )
        metric = metric_result(
            scores, recording_ids, recording_names, calibration_indices,
            external_healthy_id=config["healthy_test_recording_id"],
            damage_start_id=config["damage_start_recording_id"],
            external_healthy_indices=healthy_test_indices,
        )
        results[name] = {**row, "status": "evaluated", "metrics": compact_metric(metric)}
        score_payload[f"score_{name}"] = scores.astype(np.float32)
        if index % 10 == 0:
            print(f"evaluated={index}/{len(locked_subsets)}", flush=True)
    full = results["full_58"]["metrics"]
    success_floor = full["block_auroc"] - config["success_auroc_drop"]
    for row in results.values():
        if row.get("status") == "evaluated":
            row["passes_primary_gate"] = bool(
                row["metrics"]["block_auroc"] >= success_floor
                and row["metrics"]["external_healthy_fpr"] <= config["success_fpr_max"]
            )
    summaries = {}
    for budget in config["budgets"]:
        random_rows = [
            row for row in results.values()
            if row.get("method") == "random" and row["budget"] == budget and row.get("status") == "evaluated"
        ]
        summaries[str(budget)] = {
            "teacher": results[f"teacher_{budget}"],
            "topology": results[f"topology_{budget}"],
            "degree": results[f"degree_{budget}"],
            "random": {
                "count": len(random_rows),
                "median_auroc": float(np.median([row["metrics"]["block_auroc"] for row in random_rows])),
                "auroc_range": [
                    float(np.min([row["metrics"]["block_auroc"] for row in random_rows])),
                    float(np.max([row["metrics"]["block_auroc"] for row in random_rows])),
                ],
                "median_fpr": float(np.median([row["metrics"]["external_healthy_fpr"] for row in random_rows])),
                "pass_count": int(sum(row["passes_primary_gate"] for row in random_rows)),
            },
        }
    payload = {
        "experiment": config["experiment"],
        "status": "retrospective placement feasibility study",
        "protocol_lock": str(lock_path), "protocol_lock_sha256": sha256(lock_path),
        "graph": graph_audit(graph, sensor_names),
        "full_reference": full,
        "success_gate": {"auroc_min": success_floor, "external_healthy_fpr_max": config["success_fpr_max"]},
        "budget_summaries": summaries,
        "results": results,
        "selection_traces": {"teacher": teacher_trace, "topology": topology_trace},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "placement_results.json").write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )
    np.savez_compressed(output / "placement_scores.npz", **score_payload)
    print(json.dumps({
        budget: {
            "teacher": {
                "auroc": row["teacher"]["metrics"]["block_auroc"],
                "fpr": row["teacher"]["metrics"]["external_healthy_fpr"],
                "edges": row["teacher"]["active_edge_count"],
                "pass": row["teacher"]["passes_primary_gate"],
            },
            "topology": {
                "auroc": row["topology"]["metrics"]["block_auroc"],
                "fpr": row["topology"]["metrics"]["external_healthy_fpr"],
                "edges": row["topology"]["active_edge_count"],
                "pass": row["topology"]["passes_primary_gate"],
            },
            "random": row["random"],
        }
        for budget, row in summaries.items()
    }, indent=2, default=json_default))


if __name__ == "__main__":
    main()
