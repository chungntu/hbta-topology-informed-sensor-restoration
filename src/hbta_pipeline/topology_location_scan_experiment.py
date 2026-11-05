"""Scan every HBTA graph location with distance-decayed sensor weights."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from hbta_pipeline.edge_conditional_experiment import normalized_sum, summarize
from hbta_pipeline.localized_topology_experiment import (
    healthy_normalized_unit_scores,
)
from hbta_pipeline.piae_topology_experiment import (
    evaluate_scores,
    train_graph_model,
)
from hbta_pipeline.spatial_topology import (
    SpatialGraph,
    build_hbta_spatial_graph,
    graph_audit,
    permuted_spatial_control,
    random_spatial_control,
)
from hbta_pipeline.vqpiae_experiment import Autoencoder, residuals


KERNEL_LENGTHS_M = (3.5, 7.0)
COMBINATION_WEIGHTS = (0.1, 0.25, 0.5)


def graph_distance_matrix(graph: SpatialGraph, node_count: int) -> np.ndarray:
    """All-pairs member-following distance on an undirected sensor graph."""
    rows = np.concatenate([graph.edges[:, 0], graph.edges[:, 1]])
    columns = np.concatenate([graph.edges[:, 1], graph.edges[:, 0]])
    lengths = np.concatenate(
        [graph.structural_path_lengths, graph.structural_path_lengths]
    )
    adjacency = csr_matrix(
        (lengths, (rows, columns)), shape=(node_count, node_count)
    )
    return shortest_path(adjacency, directed=False, unweighted=False)


def euclidean_distance_matrix(positions: np.ndarray) -> np.ndarray:
    differences = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(differences, axis=2)


def distance_kernel(
    distances: np.ndarray,
    length_scale_m: float,
) -> np.ndarray:
    """Rows are candidate locations and columns are weighted sensors."""
    if length_scale_m <= 0.0:
        raise ValueError("length_scale_m must be positive")
    weights = np.exp(-distances / length_scale_m)
    weights[~np.isfinite(distances)] = 0.0
    row_sums = np.sum(weights, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Every candidate location must weight at least itself")
    return weights / row_sums


def standardized_location_scores(
    node_scores: np.ndarray,
    kernel: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    """Healthy-standardized score for every sample and candidate location."""
    local_scores = node_scores @ kernel.T
    healthy = local_scores[train_indices]
    center = np.median(healthy, axis=0)
    mad = np.median(np.abs(healthy - center[None, :]), axis=0)
    robust_scale = np.maximum(1.4826 * mad, 1e-6)
    return (local_scores - center[None, :]) / robust_scale[None, :]


def scan_score(location_scores: np.ndarray) -> np.ndarray:
    """Unknown-location detector: use the strongest candidate in each block."""
    return np.max(location_scores, axis=1)


def oracle_location_diagnostic(
    location_scores: np.ndarray,
    recording_ids: np.ndarray,
    recording_names: np.ndarray,
    sensor_names: np.ndarray,
    positions: np.ndarray,
    healthy_indices: np.ndarray,
) -> dict[str, list[dict]]:
    """Label-leaking upper bound used only to audit localization."""
    output = {}
    for recording_id in range(2, int(recording_ids.max()) + 1):
        damage_indices = np.flatnonzero(recording_ids == recording_id)
        selected = np.concatenate([healthy_indices, damage_indices])
        labels = np.concatenate(
            [
                np.zeros(healthy_indices.size, dtype=np.int64),
                np.ones(damage_indices.size, dtype=np.int64),
            ]
        )
        aurocs = np.array(
            [
                roc_auc_score(labels, location_scores[selected, candidate])
                for candidate in range(location_scores.shape[1])
            ]
        )
        ranking = np.argsort(-aurocs)[:6]
        output[str(recording_names[recording_id])] = [
            {
                "candidate_sensor": str(sensor_names[index]),
                "position_xyz_m": positions[index].tolist(),
                "oracle_auroc": float(aurocs[index]),
            }
            for index in ranking
        ]
    return output


def run(
    prepared_path: Path,
    seeds: list[int],
    max_epochs: int,
    device: torch.device,
) -> dict:
    with np.load(prepared_path) as data:
        raw = data["features"].astype(np.float32)
        recording_ids = data["recording_ids"]
        recording_names = data["recording_names"].astype(str)
        sensor_names = data["sensor_names"].astype(str)
        positions = data["sensor_positions"].astype(np.float64)
        descriptor_groups = data["descriptor_groups"].astype(str)
        feature_names = data["feature_names"].astype(str)

    selected = np.isin(
        descriptor_groups, ["response", "transmissibility"]
    )
    raw = raw[:, :, selected]
    descriptor_groups = descriptor_groups[selected]
    feature_names = feature_names[selected]

    uds01 = np.flatnonzero(recording_ids == 0)
    split = int(np.floor(0.8 * uds01.size))
    train_indices = uds01[:split]
    validation_indices = uds01[split:]
    uds02 = np.flatnonzero(recording_ids == 1)

    flat_raw = raw.reshape(raw.shape[0], -1)
    scaler = StandardScaler().fit(flat_raw[train_indices])
    flat = scaler.transform(flat_raw).astype(np.float32)

    physical = build_hbta_spatial_graph(sensor_names, positions)
    physical_distances = graph_distance_matrix(physical, sensor_names.size)
    euclidean_distances = euclidean_distance_matrix(positions)
    physical_kernels = {
        scale: distance_kernel(physical_distances, scale)
        for scale in KERNEL_LENGTHS_M
    }
    euclidean_kernels = {
        scale: distance_kernel(euclidean_distances, scale)
        for scale in KERNEL_LENGTHS_M
    }

    methods: dict[str, list[dict]] = {}
    physical_locations_accumulator = {
        scale: np.zeros((raw.shape[0], sensor_names.size), dtype=np.float64)
        for scale in KERNEL_LENGTHS_M
    }

    def add(name: str, scores: np.ndarray, seed: int) -> None:
        row = evaluate_scores(
            scores,
            recording_ids,
            recording_names,
            validation_indices,
            healthy_test_indices=uds02,
        )
        row["seed"] = seed
        methods.setdefault(name, []).append(row)

    for seed in seeds:
        torch.manual_seed(seed)
        initial = Autoencoder(flat.shape[1], 32).to(device)
        model, best_epoch = train_graph_model(
            copy.deepcopy(initial.state_dict()),
            flat[train_indices],
            flat[validation_indices],
            raw.shape[1],
            raw.shape[2],
            None,
            None,
            0.0,
            device,
            max_epochs,
        )
        squared_grid = residuals(model, flat, device).reshape(raw.shape) ** 2
        raw_node_scores = np.mean(squared_grid, axis=2)
        node_scores = healthy_normalized_unit_scores(
            raw_node_scores, train_indices
        )
        reconstruction = np.mean(raw_node_scores, axis=1)
        baseline = evaluate_scores(
            reconstruction,
            recording_ids,
            recording_names,
            validation_indices,
            healthy_test_indices=uds02,
        )
        baseline.update({"seed": seed, "best_epoch": best_epoch})
        methods.setdefault("plain_global_mean", []).append(baseline)

        random_graph = random_spatial_control(
            physical, sensor_names, positions, 7000 + seed
        )
        permuted_graph = permuted_spatial_control(
            physical, sensor_names, positions, 8000 + seed
        )
        graph_distances = {
            "physical": physical_distances,
            "euclidean": euclidean_distances,
            "random": graph_distance_matrix(random_graph, sensor_names.size),
            "permuted": graph_distance_matrix(
                permuted_graph, sensor_names.size
            ),
        }

        for scale in KERNEL_LENGTHS_M:
            graph_kernels = {
                "physical": physical_kernels[scale],
                "euclidean": euclidean_kernels[scale],
                "random": distance_kernel(graph_distances["random"], scale),
                "permuted": distance_kernel(
                    graph_distances["permuted"], scale
                ),
            }
            for graph_name, kernel in graph_kernels.items():
                location_scores = standardized_location_scores(
                    node_scores, kernel, train_indices
                )
                score = scan_score(location_scores)
                method = f"{graph_name}_scan_scale_{scale:g}m"
                add(method, score, seed)
                for weight in COMBINATION_WEIGHTS:
                    add(
                        f"combined_plain_{method}_weight_{weight}",
                        normalized_sum(
                            reconstruction, score, train_indices, weight
                        ),
                        seed,
                    )
                if graph_name == "physical":
                    physical_locations_accumulator[scale] += location_scores
        print(
            f"seed={seed} plain={baseline['block_auroc']:.4f} "
            f"physical3.5={methods['physical_scan_scale_3.5m'][-1]['block_auroc']:.4f}",
            flush=True,
        )

    oracle = {}
    for scale in KERNEL_LENGTHS_M:
        mean_locations = physical_locations_accumulator[scale] / len(seeds)
        oracle[f"physical_scale_{scale:g}m"] = (
            oracle_location_diagnostic(
                mean_locations,
                recording_ids,
                recording_names,
                sensor_names,
                positions,
                uds02,
            )
        )

    return {
        "protocol": {
            "model_train": "First 80% UDS_01",
            "model_validation_and_threshold": "Last 20% UDS_01",
            "healthy_test": "All UDS_02 only",
            "damage_test": "DS1-DS8; never used for fitting or selection",
            "unknown_location_rule": (
                "Every sensor node is scanned as a candidate damage center; "
                "the maximum healthy-standardized location score is used"
            ),
        },
        "input": {
            "features_per_sensor": int(raw.shape[2]),
            "feature_names": feature_names.tolist(),
            "descriptor_groups": descriptor_groups.tolist(),
        },
        "scan": {
            "kernel": "exp(-shortest_path_distance / length_scale)",
            "length_scales_m": KERNEL_LENGTHS_M,
            "combination_weights": COMBINATION_WEIGHTS,
            "location_standardization": (
                "Per candidate median and median absolute deviation from "
                "healthy training only"
            ),
        },
        "physical_graph": graph_audit(physical, sensor_names),
        "seeds": seeds,
        "summary": summarize(methods),
        "runs": methods,
        "oracle_location_diagnostic": {
            "warning": (
                "This ranking uses each damage label and is only an upper-bound "
                "localization audit. It is not a valid deployable detector."
            ),
            "physical": oracle,
        },
        "warning": (
            "Both length scales and all combination weights are exploratory; "
            "matched controls are required for any topology interpretation."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("HBTA/results/topology_location_scan_external.json"),
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(10))
    )
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"
    )
    result = run(args.prepared, args.seeds, args.max_epochs, device)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
