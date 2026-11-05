"""Matched healthy-only AE and geometry-graph controls for HBTA."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from hbta_pipeline.distance_baseline import robust_scale_fit


class Autoencoder(nn.Module):
    def __init__(self, sensor_count: int, feature_count: int) -> None:
        super().__init__()
        size = sensor_count * feature_count
        self.sensor_count = sensor_count
        self.feature_count = feature_count
        self.network = nn.Sequential(
            nn.Linear(size, 128),
            nn.GELU(),
            nn.Linear(128, 16),
            nn.GELU(),
            nn.Linear(16, 128),
            nn.GELU(),
            nn.Linear(128, size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        reconstructed = self.network(values.flatten(1))
        return reconstructed.reshape(-1, self.sensor_count, self.feature_count)


def geometry_edges(positions: np.ndarray, neighbors: int = 3) -> np.ndarray:
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    directed = np.argsort(distances, axis=1)[:, :neighbors]
    edges = {
        tuple(sorted((source, int(target))))
        for source, targets in enumerate(directed)
        for target in targets
    }
    return np.asarray(sorted(edges), dtype=np.int64)


def random_edges(sensor_count: int, edge_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    candidates = np.asarray(
        [(left, right) for left in range(sensor_count) for right in range(left + 1, sensor_count)],
        dtype=np.int64,
    )
    selected = rng.choice(candidates.shape[0], size=edge_count, replace=False)
    return candidates[selected]


def permute_edges(edges: np.ndarray, sensor_count: int, seed: int) -> np.ndarray:
    permutation = np.random.default_rng(seed).permutation(sensor_count)
    return np.sort(permutation[edges], axis=1)


def loss_value(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    edges: torch.Tensor | None,
    graph_weight: float,
) -> torch.Tensor:
    reconstruction = torch.mean((outputs - inputs) ** 2)
    if edges is None:
        return reconstruction
    energy_change = torch.sum(outputs**2, dim=-1) - torch.sum(inputs**2, dim=-1)
    graph = torch.mean(
        (energy_change[:, edges[:, 0]] - energy_change[:, edges[:, 1]]) ** 2
    )
    return reconstruction + graph_weight * graph


def train_model(
    train: np.ndarray,
    validation: np.ndarray,
    edges: np.ndarray | None,
    graph_weight: float,
    seed: int,
    epochs: int,
) -> tuple[Autoencoder, int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = Autoencoder(train.shape[1], train.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_tensor = torch.from_numpy(train.astype(np.float32))
    validation_tensor = torch.from_numpy(validation.astype(np.float32))
    edge_tensor = None if edges is None else torch.from_numpy(edges)
    best_state = None
    best_loss = np.inf
    best_epoch = 0
    patience = 40

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_tensor)
        loss = loss_value(train_tensor, output, edge_tensor, graph_weight)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_output = model(validation_tensor)
            validation_loss = float(
                loss_value(validation_tensor, validation_output, edge_tensor, graph_weight)
            )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        elif epoch + 1 - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch


def anomaly_scores(model: Autoencoder, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(features.astype(np.float32))
        outputs = model(inputs).numpy()
    sensor_scores = np.mean((outputs - features) ** 2, axis=-1)
    top_count = max(1, int(np.ceil(sensor_scores.shape[1] * 0.1)))
    return np.mean(np.partition(sensor_scores, -top_count, axis=1)[:, -top_count:], axis=1)


def evaluate(
    scores: np.ndarray,
    recording_ids: np.ndarray,
    recording_names: np.ndarray,
    calibration_indices: np.ndarray,
) -> dict:
    threshold = float(np.quantile(scores[calibration_indices], 0.95))
    external_healthy = recording_ids == 1
    damaged = recording_ids >= 2
    selected = external_healthy | damaged
    labels = damaged[selected].astype(np.int8)
    by_recording = {}
    for recording_id, name in enumerate(recording_names):
        values = scores[recording_ids == recording_id]
        by_recording[str(name)] = {
            "median_score": float(np.median(values)),
            "positive_rate": float(np.mean(values > threshold)),
        }
    return {
        "threshold": threshold,
        "block_level_auroc": float(roc_auc_score(labels, scores[selected])),
        "block_level_auprc": float(average_precision_score(labels, scores[selected])),
        "external_healthy_false_positive_rate": float(
            np.mean(scores[external_healthy] > threshold)
        ),
        "by_recording": by_recording,
    }


def run(path: Path, seeds: list[int], epochs: int, graph_weight: float) -> dict:
    with np.load(path) as data:
        raw_features = data["features"].astype(np.float64)
        recording_ids = data["recording_ids"]
        recording_names = data["recording_names"].astype(str)
        positions = data["sensor_positions"].astype(np.float64)

    development = np.flatnonzero(recording_ids == 0)
    split = int(np.floor(0.6 * development.size))
    train_indices = development[:split]
    calibration_indices = development[split:]
    center, scale = robust_scale_fit(raw_features[train_indices])
    features = np.clip((raw_features - center) / scale, -20.0, 20.0).astype(np.float32)

    geometric = geometry_edges(positions)
    methods = {}
    for seed in seeds:
        controls = {
            "ae": (None, 0.0),
            "geometry_knn": (geometric, graph_weight),
            "random_graph": (
                random_edges(positions.shape[0], geometric.shape[0], 1000 + seed),
                graph_weight,
            ),
            "permuted_geometry": (
                permute_edges(geometric, positions.shape[0], 2000 + seed),
                graph_weight,
            ),
        }
        for method, (edges, weight) in controls.items():
            model, best_epoch = train_model(
                features[train_indices],
                features[calibration_indices],
                edges,
                weight,
                seed,
                epochs,
            )
            scores = anomaly_scores(model, features)
            result = evaluate(scores, recording_ids, recording_names, calibration_indices)
            result["seed"] = seed
            result["best_epoch"] = best_epoch
            methods.setdefault(method, []).append(result)
            print(
                f"{method} seed={seed}: AUROC={result['block_level_auroc']:.3f}, "
                f"external FPR={result['external_healthy_false_positive_rate']:.3f}"
            )

    summary = {}
    for method, runs in methods.items():
        summary[method] = {
            metric: float(np.mean([run[metric] for run in runs]))
            for metric in [
                "block_level_auroc",
                "block_level_auprc",
                "external_healthy_false_positive_rate",
            ]
        }
    return {
        "protocol": "P2/NM/Z healthy-only; UDS_01 chronological 60/40 train/calibration",
        "graph_definition": "symmetric 3-nearest-neighbor graph from HDF5 sensor coordinates",
        "graph_caveat": "Geometry kNN is not verified member connectivity and is not called a structural graph.",
        "graph_weight": graph_weight,
        "seeds": seeds,
        "summary": summary,
        "runs": methods,
        "inference_warning": (
            "Blocks are correlated within each recording; only one damaged recording exists per state."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--output", type=Path, default=Path("HBTA/results/graph_controls.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--graph-weight", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = run(args.prepared, args.seeds, args.epochs, args.graph_weight)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
