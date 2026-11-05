"""Test whether local aggregation preserves damage near affected HBTA sensors."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from hbta_pipeline.edge_conditional_experiment import (
    conditional_edge_scores,
    normalized_sum,
    select_ridge,
    summarize,
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


SENSOR_TOP_K = (1, 2, 3, 4, 6, 8, 12)
NEIGHBORHOOD_TOP_K = (1, 2, 3, 4, 6)
EDGE_TOP_K = (1, 2, 4, 8, 12, 20)
COMBINATION_WEIGHTS = (0.05, 0.1, 0.25, 0.5)


def healthy_normalized_unit_scores(
    squared_scores: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    """Normalize each sensor or edge without using damage observations."""
    healthy_median = np.median(squared_scores[train_indices], axis=0)
    return np.log1p(
        squared_scores / np.maximum(healthy_median[None, :], 1e-8)
    )


def top_k_mean(unit_scores: np.ndarray, count: int) -> np.ndarray:
    """Dynamically average the largest ``count`` units in every sample."""
    if unit_scores.ndim != 2:
        raise ValueError("unit_scores must have shape (samples, units)")
    if not 1 <= count <= unit_scores.shape[1]:
        raise ValueError("count must be between one and the number of units")
    selected = np.partition(
        unit_scores, unit_scores.shape[1] - count, axis=1
    )[:, -count:]
    return np.mean(selected, axis=1)


def graph_neighborhood_scores(
    node_scores: np.ndarray,
    graph: SpatialGraph,
) -> np.ndarray:
    """Average each sensor with its one-hop physical neighbors."""
    neighborhoods = []
    for center in range(node_scores.shape[1]):
        incident = graph.edges[
            np.any(graph.edges == center, axis=1)
        ]
        neighbors = set(incident.reshape(-1).tolist())
        neighbors.add(center)
        neighborhoods.append(sorted(neighbors))
    return np.column_stack(
        [np.mean(node_scores[:, nodes], axis=1) for nodes in neighborhoods]
    )


def graph_edge_cluster_score(
    node_scores: np.ndarray,
    graph: SpatialGraph,
    reducer: str,
) -> np.ndarray:
    """Scan for a physical edge whose two endpoints are jointly anomalous."""
    endpoints = node_scores[:, graph.edges]
    if reducer == "mean":
        edge_scores = np.mean(endpoints, axis=2)
    elif reducer == "minimum":
        edge_scores = np.min(endpoints, axis=2)
    else:
        raise ValueError("reducer must be 'mean' or 'minimum'")
    return np.max(edge_scores, axis=1)


def graph_neighborhood_best_pair_score(
    node_scores: np.ndarray,
    graph: SpatialGraph,
) -> np.ndarray:
    """Scan for the best two-sensor cluster within a one-hop neighborhood."""
    cluster_scores = []
    for center in range(node_scores.shape[1]):
        incident = graph.edges[np.any(graph.edges == center, axis=1)]
        nodes = sorted(set(incident.reshape(-1).tolist()) | {center})
        count = min(2, len(nodes))
        cluster_scores.append(top_k_mean(node_scores[:, nodes], count))
    return np.max(np.column_stack(cluster_scores), axis=1)


def localization_diagnostic(
    node_scores: np.ndarray,
    recording_ids: np.ndarray,
    recording_names: np.ndarray,
    sensor_names: np.ndarray,
    positions: np.ndarray,
    healthy_indices: np.ndarray,
) -> dict[str, list[dict]]:
    """Rank sensors after evaluation; this diagnostic never selects a detector."""
    healthy_mean = np.mean(node_scores[healthy_indices], axis=0)
    output = {}
    for recording_id in range(2, int(recording_ids.max()) + 1):
        indices = np.flatnonzero(recording_ids == recording_id)
        if indices.size == 0:
            continue
        delta = np.mean(node_scores[indices], axis=0) - healthy_mean
        ranking = np.argsort(-delta)[:6]
        damage_name = str(recording_names[recording_id])
        output[damage_name] = [
            {
                "sensor": str(sensor_names[index]),
                "position_xyz_m": positions[index].tolist(),
                "mean_score_increase_vs_UDS02": float(delta[index]),
            }
            for index in ranking
        ]
    return output


def edge_localization_diagnostic(
    edge_scores: np.ndarray,
    graph: SpatialGraph,
    recording_ids: np.ndarray,
    recording_names: np.ndarray,
    sensor_names: np.ndarray,
    healthy_indices: np.ndarray,
) -> dict[str, list[dict]]:
    healthy_mean = np.mean(edge_scores[healthy_indices], axis=0)
    output = {}
    for recording_id in range(2, int(recording_ids.max()) + 1):
        indices = np.flatnonzero(recording_ids == recording_id)
        if indices.size == 0:
            continue
        delta = np.mean(edge_scores[indices], axis=0) - healthy_mean
        ranking = np.argsort(-delta)[:6]
        damage_name = str(recording_names[recording_id])
        output[damage_name] = [
            {
                "edge": [
                    str(sensor_names[graph.edges[index, 0]]),
                    str(sensor_names[graph.edges[index, 1]]),
                ],
                "relation": graph.relation_names[
                    int(graph.relation_ids[index])
                ],
                "mean_score_increase_vs_UDS02": float(delta[index]),
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
    grid = flat.reshape(raw.shape)

    physical = build_hbta_spatial_graph(sensor_names, positions)
    ridge, ridge_validation = select_ridge(
        grid, train_indices, validation_indices, physical
    )
    physical_edge_scores = conditional_edge_scores(
        grid, train_indices, physical, ridge
    )

    methods: dict[str, list[dict]] = {}

    def add(name: str, scores: np.ndarray, seed: int | None = None) -> None:
        row = evaluate_scores(
            scores,
            recording_ids,
            recording_names,
            validation_indices,
            healthy_test_indices=uds02,
        )
        if seed is not None:
            row["seed"] = seed
        methods.setdefault(name, []).append(row)

    for count in EDGE_TOP_K:
        add(
            f"conditional_physical_dynamic_top_{count}_edges",
            top_k_mean(physical_edge_scores, count),
        )

    accumulated_node_scores = np.zeros(
        (raw.shape[0], raw.shape[1]), dtype=np.float64
    )
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
        accumulated_node_scores += node_scores
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

        for count in SENSOR_TOP_K:
            sensor_localized = top_k_mean(node_scores, count)
            add(
                f"sensor_dynamic_top_{count}",
                sensor_localized,
                seed,
            )
            if count == 6:
                for weight in COMBINATION_WEIGHTS:
                    add(
                        f"combined_plain_sensor_top_6_weight_{weight}",
                        normalized_sum(
                            reconstruction,
                            sensor_localized,
                            train_indices,
                            weight,
                        ),
                        seed,
                    )

        physical_neighborhoods = graph_neighborhood_scores(
            node_scores, physical
        )
        for count in NEIGHBORHOOD_TOP_K:
            localized = top_k_mean(physical_neighborhoods, count)
            add(
                f"physical_neighborhood_dynamic_top_{count}",
                localized,
                seed,
            )
            if count == 1:
                for weight in COMBINATION_WEIGHTS:
                    add(
                        f"combined_plain_physical_local_weight_{weight}",
                        normalized_sum(
                            reconstruction,
                            localized,
                            train_indices,
                            weight,
                        ),
                        seed,
                    )

        random_graph = random_spatial_control(
            physical, sensor_names, positions, 5000 + seed
        )
        permuted_graph = permuted_spatial_control(
            physical, sensor_names, positions, 6000 + seed
        )
        random_localized = top_k_mean(
            graph_neighborhood_scores(node_scores, random_graph), 1
        )
        permuted_localized = top_k_mean(
            graph_neighborhood_scores(node_scores, permuted_graph), 1
        )
        add("random_neighborhood_dynamic_top_1", random_localized, seed)
        add("permuted_neighborhood_dynamic_top_1", permuted_localized, seed)
        for weight in COMBINATION_WEIGHTS:
            add(
                f"combined_plain_random_local_weight_{weight}",
                normalized_sum(
                    reconstruction,
                    random_localized,
                    train_indices,
                    weight,
                ),
                seed,
            )
            add(
                f"combined_plain_permuted_local_weight_{weight}",
                normalized_sum(
                    reconstruction,
                    permuted_localized,
                    train_indices,
                    weight,
                ),
                seed,
            )

        graph_variants = {
            "physical": physical,
            "random": random_graph,
            "permuted": permuted_graph,
        }
        for graph_name, graph in graph_variants.items():
            cluster_methods = {
                "edge_mean_scan": graph_edge_cluster_score(
                    node_scores, graph, "mean"
                ),
                "edge_minimum_scan": graph_edge_cluster_score(
                    node_scores, graph, "minimum"
                ),
                "neighborhood_best_pair_scan":
                    graph_neighborhood_best_pair_score(node_scores, graph),
            }
            for scan_name, scan_scores in cluster_methods.items():
                add(f"{graph_name}_{scan_name}", scan_scores, seed)
                for weight in COMBINATION_WEIGHTS:
                    add(
                        f"combined_plain_{graph_name}_{scan_name}_weight_{weight}",
                        normalized_sum(
                            reconstruction,
                            scan_scores,
                            train_indices,
                            weight,
                        ),
                        seed,
                    )
        print(
            f"seed={seed} plain={baseline['block_auroc']:.4f} "
            f"physical_local={methods['physical_neighborhood_dynamic_top_1'][-1]['block_auroc']:.4f}",
            flush=True,
        )

    mean_node_scores = accumulated_node_scores / len(seeds)
    return {
        "protocol": {
            "model_train": "First 80% UDS_01",
            "model_validation_and_threshold": "Last 20% UDS_01",
            "healthy_test": "All UDS_02 only",
            "damage_test": "DS1-DS8; never used for fitting or selection",
            "local_rule": (
                "Units are ranked independently for every test block; no "
                "fixed sensor subset is selected from damage labels"
            ),
        },
        "input": {
            "features_per_sensor": int(raw.shape[2]),
            "feature_names": feature_names.tolist(),
            "descriptor_groups": descriptor_groups.tolist(),
        },
        "candidate_counts": {
            "sensor_top_k": SENSOR_TOP_K,
            "neighborhood_top_k": NEIGHBORHOOD_TOP_K,
            "edge_top_k": EDGE_TOP_K,
            "combination_weights": COMBINATION_WEIGHTS,
        },
        "conditional_ridge": {
            "selected": ridge,
            "validation_mse": ridge_validation,
        },
        "physical_graph": graph_audit(physical, sensor_names),
        "seeds": seeds,
        "summary": summarize(methods),
        "runs": methods,
        "posthoc_localization_diagnostic": {
            "warning": (
                "Damage labels are used only for this diagnostic ranking, "
                "never for detector fitting or hyperparameter selection."
            ),
            "top_sensors": localization_diagnostic(
                mean_node_scores,
                recording_ids,
                recording_names,
                sensor_names,
                positions,
                uds02,
            ),
            "top_conditional_edges": edge_localization_diagnostic(
                physical_edge_scores,
                physical,
                recording_ids,
                recording_names,
                sensor_names,
                uds02,
            ),
        },
        "warning": (
            "The top-k grid is exploratory. A future physical experiment is "
            "required before one count can be reported as confirmatory."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("HBTA/results/localized_topology_external.json"),
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
