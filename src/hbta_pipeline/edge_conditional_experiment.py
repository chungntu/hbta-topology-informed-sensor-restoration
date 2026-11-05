"""Healthy-only conditional prediction along drawing-verified HBTA edges."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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


RIDGE_CANDIDATES = (0.01, 0.1, 1.0, 10.0, 100.0)
COMBINATION_WEIGHTS = (0.1, 0.25, 0.5, 1.0)


def directed_edges(graph: SpatialGraph) -> tuple[np.ndarray, np.ndarray]:
    edges = np.concatenate([graph.edges, graph.edges[:, ::-1]], axis=0)
    relations = np.concatenate([graph.relation_ids, graph.relation_ids])
    return edges, relations


def validation_error(
    grid: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    graph: SpatialGraph,
    ridge: float,
) -> float:
    edges, _ = directed_edges(graph)
    errors = []
    for source, target in edges:
        model = Ridge(alpha=ridge)
        model.fit(grid[train_indices, source], grid[train_indices, target])
        prediction = model.predict(grid[validation_indices, source])
        errors.append(np.mean((prediction - grid[validation_indices, target]) ** 2))
    return float(np.mean(errors))


def select_ridge(
    grid: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    graph: SpatialGraph,
) -> tuple[float, dict[str, float]]:
    errors = {
        str(ridge): validation_error(
            grid, train_indices, validation_indices, graph, ridge
        )
        for ridge in RIDGE_CANDIDATES
    }
    selected = min(RIDGE_CANDIDATES, key=lambda ridge: errors[str(ridge)])
    return float(selected), errors


def conditional_scores(
    grid: np.ndarray,
    train_indices: np.ndarray,
    graph: SpatialGraph,
    ridge: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit both edge directions and return robust healthy-normalized errors."""
    edge_scores = conditional_edge_scores(
        grid, train_indices, graph, ridge
    )
    all_edges = np.mean(edge_scores, axis=1)

    by_relation = {}
    relation_scores = []
    for relation_id, relation in enumerate(graph.relation_names):
        mask = graph.relation_ids == relation_id
        if not np.any(mask):
            continue
        score = np.mean(edge_scores[:, mask], axis=1)
        by_relation[relation] = score
        relation_scores.append(score)
    relation_balanced = np.mean(np.column_stack(relation_scores), axis=1)
    return all_edges, {
        "relation_balanced": relation_balanced,
        **by_relation,
    }


def conditional_edge_scores(
    grid: np.ndarray,
    train_indices: np.ndarray,
    graph: SpatialGraph,
    ridge: float,
) -> np.ndarray:
    """Return one robust prediction-error score per sample and undirected edge."""
    edges, _ = directed_edges(graph)
    squared_errors = np.empty(
        (grid.shape[0], edges.shape[0], grid.shape[2]), dtype=np.float64
    )
    for edge_index, (source, target) in enumerate(edges):
        model = Ridge(alpha=ridge)
        model.fit(grid[train_indices, source], grid[train_indices, target])
        prediction = model.predict(grid[:, source])
        squared_errors[:, edge_index] = (
            prediction - grid[:, target]
        ) ** 2

    healthy_median = np.median(squared_errors[train_indices], axis=0)
    normalized = squared_errors / np.maximum(
        healthy_median[None, :, :], 1e-8
    )
    robust = np.log1p(normalized)
    undirected_count = graph.edges.shape[0]
    forward = np.mean(robust[:, :undirected_count], axis=2)
    reverse = np.mean(robust[:, undirected_count:], axis=2)
    return 0.5 * (forward + reverse)


def normalized_sum(
    baseline: np.ndarray,
    conditional: np.ndarray,
    train_indices: np.ndarray,
    weight: float,
) -> np.ndarray:
    baseline_scale = max(float(np.median(baseline[train_indices])), 1e-12)
    conditional_scale = max(
        float(np.median(conditional[train_indices])), 1e-12
    )
    return baseline / baseline_scale + weight * conditional / conditional_scale


def summarize(rows: dict[str, list[dict]]) -> dict:
    result = {}
    for method, method_rows in rows.items():
        result[method] = {}
        for metric in (
            "block_auroc",
            "block_auprc",
            "external_healthy_false_alarm_rate",
        ):
            values = np.asarray([row[metric] for row in method_rows])
            result[method][f"mean_{metric}"] = float(values.mean())
            result[method][f"std_{metric}"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0
            )
    return result


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
    selected_ridge, ridge_validation = select_ridge(
        grid, train_indices, validation_indices, physical
    )
    physical_all, physical_relations = conditional_scores(
        grid, train_indices, physical, selected_ridge
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

    # Physical conditional scores do not depend on neural-network initialization.
    add("conditional_physical_all_edges", physical_all)
    add(
        "conditional_physical_relation_balanced",
        physical_relations["relation_balanced"],
    )
    for relation, scores in physical_relations.items():
        if relation != "relation_balanced":
            add(f"conditional_relation_{relation}", scores)

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
        reconstruction = np.mean(
            residuals(model, flat, device).reshape(raw.shape) ** 2,
            axis=(1, 2),
        )
        baseline = evaluate_scores(
            reconstruction,
            recording_ids,
            recording_names,
            validation_indices,
            healthy_test_indices=uds02,
        )
        baseline.update({"seed": seed, "best_epoch": best_epoch})
        methods.setdefault("plain_reconstruction", []).append(baseline)

        random_graph = random_spatial_control(
            physical, sensor_names, positions, 3000 + seed
        )
        permuted_graph = permuted_spatial_control(
            physical, sensor_names, positions, 4000 + seed
        )
        random_all, _ = conditional_scores(
            grid, train_indices, random_graph, selected_ridge
        )
        permuted_all, _ = conditional_scores(
            grid, train_indices, permuted_graph, selected_ridge
        )
        add("conditional_random_all_edges", random_all, seed)
        add("conditional_permuted_all_edges", permuted_all, seed)

        for weight in COMBINATION_WEIGHTS:
            add(
                f"combined_physical_weight_{weight}",
                normalized_sum(
                    reconstruction, physical_all, train_indices, weight
                ),
                seed,
            )
            add(
                f"combined_random_weight_{weight}",
                normalized_sum(
                    reconstruction, random_all, train_indices, weight
                ),
                seed,
            )
            add(
                f"combined_permuted_weight_{weight}",
                normalized_sum(
                    reconstruction, permuted_all, train_indices, weight
                ),
                seed,
            )
        print(f"seed={seed} plain={baseline['block_auroc']:.3f}", flush=True)

    return {
        "protocol": {
            "model_train": "First 80% UDS_01",
            "model_validation": "Last 20% UDS_01",
            "healthy_test": "All UDS_02; never used for fitting or selection",
            "damage_test": "DS1-DS8; never used for fitting or selection",
        },
        "input": {
            "features_per_sensor": int(raw.shape[2]),
            "feature_names": feature_names.tolist(),
            "descriptor_groups": descriptor_groups.tolist(),
        },
        "conditional_model": {
            "definition": (
                "Separate bidirectional multi-output ridge predictor for each "
                "drawing-verified edge; log1p squared error normalized by the "
                "healthy-training median for every edge-feature coordinate"
            ),
            "ridge_candidates": RIDGE_CANDIDATES,
            "selected_ridge": selected_ridge,
            "ridge_validation_mse": ridge_validation,
            "combination_weights": COMBINATION_WEIGHTS,
            "primary_combination_weight": 1.0,
        },
        "physical_graph": graph_audit(physical, sensor_names),
        "seeds": seeds,
        "summary": summarize(methods),
        "runs": methods,
        "warning": (
            "Combination-weight comparisons are exploratory; damage labels "
            "must not be used to select the reported primary coefficient."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("HBTA/results/edge_conditional_external.json"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
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
