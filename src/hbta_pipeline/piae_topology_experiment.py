"""Test HBTA topology in PIAE reconstruction scoring and training."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from hbta_pipeline.graph_experiment import geometry_edges
from hbta_pipeline.spatial_topology import (
    build_hbta_spatial_graph,
    graph_audit,
    permuted_spatial_control,
    random_spatial_control,
)
from hbta_pipeline.vqpiae_experiment import (
    Autoencoder,
    metric_result,
    random_edges,
    residuals,
    sensor_topology,
)


def correlation_edges(train_grid: np.ndarray, neighbors: int = 3) -> np.ndarray:
    sensor_signatures = train_grid.transpose(1, 0, 2).reshape(train_grid.shape[1], -1)
    correlation = np.nan_to_num(np.corrcoef(sensor_signatures), nan=0.0)
    np.fill_diagonal(correlation, 0.0)
    nearest = np.argsort(-np.abs(correlation), axis=1)[:, :neighbors]
    edges = {
        tuple(sorted((source, int(target))))
        for source, targets in enumerate(nearest)
        for target in targets
    }
    return np.asarray(sorted(edges), dtype=np.int64)


def edge_weights(train_grid: np.ndarray, edges: np.ndarray) -> np.ndarray:
    signatures = train_grid.transpose(1, 0, 2).reshape(train_grid.shape[1], -1)
    correlation = np.nan_to_num(np.corrcoef(signatures), nan=0.0)
    weights = np.abs(correlation[edges[:, 0], edges[:, 1]])
    return weights / max(weights.mean(), 1e-12)


def edge_residual_score(
    residual_grid: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    differences = (
        residual_grid[:, edges[:, 0], :] - residual_grid[:, edges[:, 1], :]
    ) ** 2
    return np.mean(differences * weights[None, :, None], axis=(1, 2))


def graph_loss(
    residual_grid: torch.Tensor,
    edges: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    differences = (
        residual_grid[:, edges[:, 0], :] - residual_grid[:, edges[:, 1], :]
    ) ** 2
    return torch.mean(differences * weights[None, :, None])


def train_graph_model(
    initial_state: dict[str, torch.Tensor],
    train: np.ndarray,
    validation: np.ndarray,
    sensor_count: int,
    feature_count: int,
    edges: np.ndarray | None,
    weights: np.ndarray | None,
    graph_weight: float,
    device: torch.device,
    max_epochs: int,
) -> tuple[Autoencoder, int]:
    model = Autoencoder(train.shape[1], 32).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_tensor = torch.from_numpy(train).to(device)
    validation_tensor = torch.from_numpy(validation).to(device)
    edge_tensor = None if edges is None else torch.from_numpy(edges).to(device)
    weight_tensor = None if weights is None else torch.from_numpy(weights.astype(np.float32)).to(device)
    best_loss = np.inf
    best_epoch = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_tensor)
        reconstruction_loss = criterion(output, train_tensor)
        loss = reconstruction_loss
        if edge_tensor is not None and weight_tensor is not None:
            residual_grid = (output - train_tensor).reshape(
                -1, sensor_count, feature_count
            )
            loss = loss + graph_weight * graph_loss(
                residual_grid, edge_tensor, weight_tensor
            )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_tensor), validation_tensor))
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        elif epoch + 1 - best_epoch >= 60:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch


def evaluate_scores(
    scores: np.ndarray,
    recording_ids: np.ndarray,
    recording_names: np.ndarray,
    calibration_indices: np.ndarray,
    healthy_test_indices: np.ndarray | None = None,
) -> dict:
    return metric_result(
        scores,
        recording_ids,
        recording_names,
        calibration_indices,
        external_healthy_id=1,
        damage_start_id=2,
        external_healthy_indices=healthy_test_indices,
    )


def run(
    prepared_path: Path,
    seeds: list[int],
    graph_weight: float,
    max_epochs: int,
    device: torch.device,
    both_healthy: bool,
    selected_groups: list[str] | None = None,
    requested_graphs: list[str] | None = None,
) -> dict:
    with np.load(prepared_path) as data:
        raw_grid = data["features"].astype(np.float32)
        recording_ids = data["recording_ids"]
        recording_names = data["recording_names"].astype(str)
        sensor_names = data["sensor_names"].astype(str)
        positions = data["sensor_positions"].astype(np.float64)
        feature_names = data["feature_names"].astype(str)
        descriptor_groups = data["descriptor_groups"].astype(str)

    if selected_groups:
        unknown = sorted(set(selected_groups) - set(descriptor_groups))
        if unknown:
            raise ValueError(
                f"Unknown descriptor groups {unknown}; available groups are "
                f"{sorted(set(descriptor_groups))}"
            )
        selected = np.isin(descriptor_groups, selected_groups)
        raw_grid = raw_grid[:, :, selected]
        feature_names = feature_names[selected]
        descriptor_groups = descriptor_groups[selected]

    if both_healthy:
        first = np.flatnonzero(recording_ids == 0)
        second = np.flatnonzero(recording_ids == 1)
        first_split = int(np.floor(0.80 * first.size))
        second_split = int(np.floor(0.80 * second.size))
        train_indices = np.concatenate([first[:first_split], second[:second_split]])
        healthy_test_indices = np.concatenate([first[first_split:], second[second_split:]])
        validation_indices = healthy_test_indices
        calibration_indices = healthy_test_indices
    else:
        development = np.flatnonzero(recording_ids == 0)
        train_end = int(np.floor(0.80 * development.size))
        train_indices = development[:train_end]
        validation_indices = development[train_end:]
        calibration_indices = validation_indices
        healthy_test_indices = None
    sensor_count, feature_count = raw_grid.shape[1:]
    flat_raw = raw_grid.reshape(raw_grid.shape[0], -1)
    scaler = StandardScaler().fit(flat_raw[train_indices])
    flat = scaler.transform(flat_raw).astype(np.float32)
    grid = flat.reshape(-1, sensor_count, feature_count)

    structural = sensor_topology(sensor_names)
    geometric = geometry_edges(positions, neighbors=3)
    spatial3d = build_hbta_spatial_graph(sensor_names, positions)
    spatial3d_no_lateral = spatial3d.subset(
        set(spatial3d.relation_names) - {"lateral_bracing"}
    )
    correlated = correlation_edges(grid[train_indices], neighbors=3)
    structural_weights = np.ones(structural.shape[0], dtype=np.float64)
    graph_specs_base = {
        "structural": (structural, structural_weights),
        "geometry": (geometric, np.ones(geometric.shape[0], dtype=np.float64)),
        "healthy_correlation": (
            correlated,
            np.ones(correlated.shape[0], dtype=np.float64),
        ),
        "hybrid": (structural, edge_weights(grid[train_indices], structural)),
        "spatial3d": (spatial3d.edges, spatial3d.weights),
        "spatial3d_uniform": (
            spatial3d.edges,
            np.ones(spatial3d.edges.shape[0], dtype=np.float64),
        ),
        "spatial3d_no_lateral": (
            spatial3d_no_lateral.edges,
            spatial3d_no_lateral.weights,
        ),
    }
    methods: dict[str, list[dict]] = {}

    for seed in seeds:
        torch.manual_seed(seed)
        initial_model = Autoencoder(flat.shape[1], 32).to(device)
        initial_state = copy.deepcopy(initial_model.state_dict())
        random_graph = random_edges(sensor_count, structural.shape[0], 1000 + seed)
        permutation = np.random.default_rng(2000 + seed).permutation(sensor_count)
        permuted = np.sort(permutation[structural], axis=1)
        spatial3d_random = random_spatial_control(
            spatial3d, sensor_names, positions, 3000 + seed
        )
        spatial3d_permuted = permuted_spatial_control(
            spatial3d, sensor_names, positions, 4000 + seed
        )
        graph_specs = {
            **graph_specs_base,
            "random": (random_graph, np.ones(random_graph.shape[0], dtype=np.float64)),
            "permuted": (permuted, structural_weights),
            "spatial3d_random": (
                spatial3d_random.edges,
                spatial3d_random.weights,
            ),
            "spatial3d_random_uniform": (
                spatial3d_random.edges,
                np.ones(spatial3d_random.edges.shape[0], dtype=np.float64),
            ),
            "spatial3d_permuted": (
                spatial3d_permuted.edges,
                spatial3d_permuted.weights,
            ),
            "spatial3d_permuted_uniform": (
                spatial3d_permuted.edges,
                np.ones(spatial3d_permuted.edges.shape[0], dtype=np.float64),
            ),
        }
        if requested_graphs:
            unknown_graphs = sorted(set(requested_graphs) - set(graph_specs))
            if unknown_graphs:
                raise ValueError(
                    f"Unknown graphs {unknown_graphs}; available graphs are "
                    f"{sorted(graph_specs)}"
                )
            graph_specs = {
                name: graph_specs[name] for name in requested_graphs
            }

        plain_model, plain_epoch = train_graph_model(
            initial_state,
            flat[train_indices],
            flat[validation_indices],
            sensor_count,
            feature_count,
            None,
            None,
            0.0,
            device,
            max_epochs,
        )
        plain_residual = residuals(plain_model, flat, device).reshape(
            -1, sensor_count, feature_count
        )
        total_reconstruction = np.mean(plain_residual**2, axis=(1, 2))
        result = evaluate_scores(
            total_reconstruction, recording_ids, recording_names, calibration_indices,
            healthy_test_indices,
        )
        result.update({"seed": seed, "best_epoch": plain_epoch})
        methods.setdefault("plain_reconstruction", []).append(result)

        for graph_name, (edges, weights) in graph_specs.items():
            edge_score = edge_residual_score(plain_residual, edges, weights)
            normalized_total = total_reconstruction / max(
                np.median(total_reconstruction[train_indices]), 1e-12
            )
            normalized_edge = edge_score / max(
                np.median(edge_score[train_indices]), 1e-12
            )
            combined_score = normalized_total + normalized_edge
            for score_name, scores in [
                (f"score_{graph_name}_edge", edge_score),
                (f"score_{graph_name}_combined", combined_score),
            ]:
                score_result = evaluate_scores(
                    scores, recording_ids, recording_names, calibration_indices,
                    healthy_test_indices,
                )
                score_result.update({"seed": seed, "best_epoch": plain_epoch})
                methods.setdefault(score_name, []).append(score_result)

            graph_model, best_epoch = train_graph_model(
                initial_state,
                flat[train_indices],
                flat[validation_indices],
                sensor_count,
                feature_count,
                edges,
                weights,
                graph_weight,
                device,
                max_epochs,
            )
            graph_residual = residuals(graph_model, flat, device)
            graph_reconstruction = np.mean(graph_residual**2, axis=1)
            train_result = evaluate_scores(
                graph_reconstruction,
                recording_ids,
                recording_names,
                calibration_indices,
                healthy_test_indices,
            )
            train_result.update({"seed": seed, "best_epoch": best_epoch})
            methods.setdefault(f"train_{graph_name}", []).append(train_result)

        progress_graph = next(iter(graph_specs))
        print(
            f"seed={seed} "
            f"plain={methods['plain_reconstruction'][-1]['block_auroc']:.3f} "
            f"score-{progress_graph}="
            f"{methods[f'score_{progress_graph}_combined'][-1]['block_auroc']:.3f} "
            f"train-{progress_graph}="
            f"{methods[f'train_{progress_graph}'][-1]['block_auroc']:.3f}",
            flush=True,
        )

    summary = {}
    for method_name, rows in methods.items():
        summary[method_name] = {}
        for metric in ["block_auroc", "block_auprc", "external_healthy_false_alarm_rate"]:
            values = np.asarray([row[metric] for row in rows])
            summary[method_name][f"mean_{metric}"] = float(values.mean())
            summary[method_name][f"std_{metric}"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0
            )

    return {
        "protocol": (
            "P2/NM/Z, both UDS recordings split chronological 80/20"
            if both_healthy
            else "P2/NM/Z, UDS_01 chronological 80/20, UDS_02 external healthy"
        ),
        "model": "PIAE continuous autoencoder with 32-dimensional bottleneck",
        "input": {
            "sensors": int(sensor_count),
            "features_per_sensor": int(feature_count),
            "dimension": int(flat.shape[1]),
            "feature_names": feature_names.tolist(),
            "descriptor_groups": descriptor_groups.tolist(),
        },
        "topology_scoring": (
            "Equal healthy-median-normalized sum of total reconstruction and edge residual contrast"
        ),
        "topology_training": (
            f"MSE plus {graph_weight} times weighted edge-consistency residual loss"
        ),
        "graphs": {
            "structural_edges": int(structural.shape[0]),
            "geometry_edges": int(geometric.shape[0]),
            "healthy_correlation_edges": int(correlated.shape[0]),
            "hybrid": "structural edges weighted by absolute healthy sensor correlation",
            "spatial3d": {
                "definition": (
                    "Sensor-projected joint/member graph checked against "
                    "Sensor layout.pdf"
                ),
                "weighting": (
                    "relation-balanced inverse structural-path length; AL "
                    "mid-panel sensors route through cross-girder joints"
                ),
                "audit": graph_audit(spatial3d, sensor_names),
            },
        },
        "seeds": seeds,
        "summary": summary,
        "runs": methods,
        "warning": "Block and seed variation do not provide physical-test replication.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("HBTA/results/piae_topology_reconstruction.json")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--graph-weight", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--both-healthy", action="store_true")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--graphs", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"
    )
    results = run(
        args.prepared,
        args.seeds,
        args.graph_weight,
        args.max_epochs,
        device,
        args.both_healthy,
        args.groups,
        args.graphs,
    )
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
