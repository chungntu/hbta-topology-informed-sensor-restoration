"""Apply VQ-PIAE-A and topology-aware reliability attention to HBTA."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from hbta_pipeline.spatial_topology import (
    build_hbta_spatial_graph,
    graph_audit,
    permuted_spatial_control,
    random_spatial_control,
)


class Autoencoder(nn.Module):
    """PIAE-A architecture used in the Z24 VQ-PIAE-A experiment."""

    def __init__(self, dimension: int, bottleneck: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dimension, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))

    def quantization_loss(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)


class ProductVectorQuantizer(nn.Module):
    def __init__(
        self,
        dimension: int,
        codewords: int = 32,
        groups: int = 4,
        commitment: float = 0.25,
    ) -> None:
        super().__init__()
        if dimension % groups:
            raise ValueError("Bottleneck dimension must be divisible by groups")
        self.codewords = codewords
        self.groups = groups
        self.subdimension = dimension // groups
        self.commitment = commitment
        self.embeddings = nn.Parameter(
            torch.randn(groups, codewords, self.subdimension) * 0.1
        )
        self.last_loss: torch.Tensor | None = None
        self.last_indices: torch.Tensor | None = None

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        grouped = latent.reshape(batch_size, self.groups, self.subdimension)
        distances = (
            (grouped.unsqueeze(2) - self.embeddings.unsqueeze(0)) ** 2
        ).sum(-1)
        indices = distances.argmin(-1)
        group_indices = torch.arange(self.groups, device=latent.device).unsqueeze(0)
        group_indices = group_indices.expand(batch_size, self.groups)
        quantized = self.embeddings[group_indices, indices]
        self.last_loss = ((quantized - grouped.detach()) ** 2).mean() + self.commitment * (
            (grouped - quantized.detach()) ** 2
        ).mean()
        self.last_indices = indices.detach()
        straight_through = grouped + (quantized - grouped).detach()
        return straight_through.reshape(batch_size, -1)


class VQAutoencoder(nn.Module):
    def __init__(
        self,
        dimension: int,
        bottleneck: int = 32,
        codewords: int = 32,
        groups: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dimension, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck),
            nn.ReLU(),
        )
        self.quantizer = ProductVectorQuantizer(bottleneck, codewords, groups)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.quantizer(self.encoder(values)))

    def quantization_loss(self) -> torch.Tensor:
        if self.quantizer.last_loss is None:
            raise RuntimeError("Forward must be called before quantization_loss")
        return self.quantizer.last_loss

    def code_indices(self, values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.quantizer(self.encoder(values))
        if self.quantizer.last_indices is None:
            raise RuntimeError("Quantizer did not produce indices")
        return self.quantizer.last_indices


def train_model(
    model: nn.Module,
    train: np.ndarray,
    validation: np.ndarray,
    device: torch.device,
    seed: int,
    max_epochs: int,
) -> tuple[nn.Module, int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_tensor = torch.from_numpy(train).to(device)
    validation_tensor = torch.from_numpy(validation).to(device)
    best_loss = np.inf
    best_epoch = 0
    best_state = None
    patience = 60

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        reconstruction = model(train_tensor)
        loss = criterion(reconstruction, train_tensor) + model.quantization_loss()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_tensor), validation_tensor))
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        elif epoch + 1 - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch


def sensor_topology(sensor_names: np.ndarray) -> np.ndarray:
    """HBTA response-sensor topology derived from the truss and deck grids."""
    index = {str(name): position for position, name in enumerate(sensor_names)}
    edges: set[tuple[int, int]] = set()

    north = [f"AG{i:02d}" for i in range(1, 10)]
    south = [f"AG{i:02d}" for i in range(10, 19)]
    for wall in (north, south):
        edges.update(
            tuple(sorted((index[left], index[right])))
            for left, right in zip(wall[:-1], wall[1:])
        )
    edges.update(
        tuple(sorted((index[north_sensor], index[south_sensor])))
        for north_sensor, south_sensor in zip(north, south)
    )

    deck = np.asarray(
        [[index[f"AL{4 * column + row + 1:02d}"] for row in range(4)] for column in range(10)]
    )
    for column in range(deck.shape[0]):
        edges.update(
            tuple(sorted((int(deck[column, row]), int(deck[column, row + 1]))))
            for row in range(deck.shape[1] - 1)
        )
    for row in range(deck.shape[1]):
        edges.update(
            tuple(sorted((int(deck[column, row]), int(deck[column + 1, row]))))
            for column in range(deck.shape[0] - 1)
        )
    return np.asarray(sorted(edges), dtype=np.int64)


def adjacency(
    sensor_count: int,
    edges: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    matrix = np.zeros((sensor_count, sensor_count), dtype=np.float64)
    if weights is None:
        weights = np.ones(edges.shape[0], dtype=np.float64)
    if weights.shape != (edges.shape[0],):
        raise ValueError("weights must have one value per edge")
    matrix[edges[:, 0], edges[:, 1]] = weights
    matrix[edges[:, 1], edges[:, 0]] = weights
    return matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)


def random_edges(sensor_count: int, edge_count: int, seed: int) -> np.ndarray:
    candidates = np.asarray(
        [(left, right) for left in range(sensor_count) for right in range(left + 1, sensor_count)],
        dtype=np.int64,
    )
    selected = np.random.default_rng(seed).choice(
        candidates.shape[0], size=edge_count, replace=False
    )
    return candidates[selected]


def attention_from_log_reliability(log_reliability: np.ndarray) -> tuple[np.ndarray, float]:
    reliability = 10.0**log_reliability
    betas = np.geomspace(1e-3, 10.0, 2000)
    log_betas = np.log10(betas)
    participation = np.empty(betas.size)
    for position, beta in enumerate(betas):
        logits = -reliability / beta
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        participation[position] = 1.0 / np.sum(weights**2)
    normalized_x = (log_betas - log_betas.min()) / np.ptp(log_betas)
    normalized_y = (participation - participation.min()) / max(
        np.ptp(participation), 1e-12
    )
    beta_low = betas[int(np.argmin(normalized_y - normalized_x))]
    beta_high = betas[int(np.argmax(np.gradient(participation, log_betas)))]
    beta = float(np.sqrt(beta_low * beta_high))
    logits = -reliability / beta
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum(), beta


def residuals(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(values).to(device)
        return (model(tensor) - tensor).cpu().numpy()


def residualize_against_eov(
    values: np.ndarray,
    eov_features: np.ndarray,
    train_indices: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, dict]:
    """Remove the healthy-train linear component explained by shaker EOVs."""
    if ridge < 0:
        raise ValueError("EOV ridge penalty must be non-negative")
    eov_scaler = StandardScaler().fit(eov_features[train_indices])
    eov = eov_scaler.transform(eov_features).astype(np.float64)
    target_mean = values[train_indices].mean(axis=0, keepdims=True)
    centered_train = values[train_indices] - target_mean
    design = eov[train_indices]
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    coefficients = np.linalg.solve(gram, design.T @ centered_train)
    adjusted = values - eov @ coefficients
    fitted_train = design @ coefficients
    denominator = float(np.sum(centered_train**2))
    explained_fraction = 1.0 - float(
        np.sum((centered_train - fitted_train) ** 2)
    ) / max(denominator, 1e-12)
    return adjusted, {
        "method": "Healthy-train ridge residualization against shaker descriptors",
        "ridge": ridge,
        "eov_dimension": int(eov.shape[1]),
        "healthy_train_variance_explained": explained_fraction,
    }


def metric_result(
    scores: np.ndarray,
    recording_ids: np.ndarray,
    recording_names: np.ndarray,
    calibration_indices: np.ndarray,
    external_healthy_id: int,
    damage_start_id: int,
    external_healthy_indices: np.ndarray | None = None,
) -> dict:
    threshold = float(np.quantile(scores[calibration_indices], 0.95))
    if external_healthy_indices is None:
        external_healthy = recording_ids == external_healthy_id
    else:
        external_healthy = np.zeros(recording_ids.size, dtype=bool)
        external_healthy[external_healthy_indices] = True
    damaged = recording_ids >= damage_start_id
    selected = external_healthy | damaged
    labels = damaged[selected].astype(np.int8)
    by_recording = {}
    for recording_id, name in enumerate(recording_names):
        recording_mask = recording_ids == recording_id
        values = scores[recording_mask]
        auroc_vs_external_healthy = None
        if recording_id >= damage_start_id:
            comparison = external_healthy | recording_mask
            comparison_labels = recording_mask[comparison].astype(np.int8)
            auroc_vs_external_healthy = float(
                roc_auc_score(comparison_labels, scores[comparison])
            )
        by_recording[str(name)] = {
            "blocks": int(values.size),
            "median_score": float(np.median(values)),
            "detection_rate": float(np.mean(values > threshold)),
            "auroc_vs_external_healthy": auroc_vs_external_healthy,
        }
    return {
        "block_auroc": float(roc_auc_score(labels, scores[selected])),
        "block_auprc": float(average_precision_score(labels, scores[selected])),
        "external_healthy_false_alarm_rate": float(
            np.mean(scores[external_healthy] > threshold)
        ),
        "threshold": threshold,
        "by_recording": by_recording,
    }


def run(
    prepared_path: Path,
    seeds: list[int],
    max_epochs: int,
    smoothing: float,
    device: torch.device,
    split_mode: str,
    selected_groups: list[str] | None = None,
    eov_adjustment: str = "none",
    eov_ridge: float = 1.0,
) -> dict:
    with np.load(prepared_path) as data:
        feature_tensor = data["features"].astype(np.float32)
        recording_ids = data["recording_ids"]
        recording_names = data["recording_names"].astype(str)
        sensor_names = data["sensor_names"].astype(str)
        positions = data["sensor_positions"].astype(np.float64)
        feature_names = data["feature_names"].astype(str)
        descriptor_groups = data["descriptor_groups"].astype(str)
        shaker_eov = (
            data["shaker_eov_features"].astype(np.float64)
            if "shaker_eov_features" in data.files
            else None
        )
        shaker_eov_names = (
            data["shaker_eov_feature_names"].astype(str)
            if "shaker_eov_feature_names" in data.files
            else None
        )

    if selected_groups:
        unknown = sorted(set(selected_groups) - set(descriptor_groups))
        if unknown:
            raise ValueError(
                f"Unknown descriptor groups {unknown}; available groups are "
                f"{sorted(set(descriptor_groups))}"
            )
        selected = np.isin(descriptor_groups, selected_groups)
        feature_tensor = feature_tensor[:, :, selected]
        feature_names = feature_names[selected]
        descriptor_groups = descriptor_groups[selected]

    healthy_test_indices = None
    if split_mode in {"controlled", "controlled80"}:
        development = np.flatnonzero(recording_ids == 0)
        if split_mode == "controlled80":
            train_end = int(np.floor(0.80 * development.size))
            train_indices = development[:train_end]
            validation_indices = development[train_end:]
            calibration_indices = validation_indices
        else:
            train_end = int(np.floor(0.50 * development.size))
            validation_end = int(np.floor(0.75 * development.size))
            train_indices = development[:train_end]
            validation_indices = development[train_end:validation_end]
            calibration_indices = development[validation_end:]
        external_healthy_id = 1
        damage_start_id = 2
    elif split_mode == "single_recording":
        development = np.flatnonzero(recording_ids == 0)
        train_end = int(np.floor(0.60 * development.size))
        validation_end = int(np.floor(0.80 * development.size))
        train_indices = development[:train_end]
        validation_indices = development[train_end:validation_end]
        calibration_indices = validation_indices
        healthy_test_indices = development[validation_end:]
        external_healthy_id = 0
        damage_start_id = 2
    elif split_mode == "both_healthy80":
        first = np.flatnonzero(recording_ids == 0)
        second = np.flatnonzero(recording_ids == 1)
        first_split = int(np.floor(0.80 * first.size))
        second_split = int(np.floor(0.80 * second.size))
        train_indices = np.concatenate([first[:first_split], second[:second_split]])
        healthy_test_indices = np.concatenate([first[first_split:], second[second_split:]])
        validation_indices = healthy_test_indices
        calibration_indices = healthy_test_indices
        external_healthy_id = 1
        damage_start_id = 2
    elif split_mode == "multicondition":
        expected = [
            "MVS_P1_UDS_SM_Z_01",
            "MVS_P1_UDS_SM_Z_02",
            "MVS_P2_UDS_SM_Z_01",
            "MVS_P2_UDS_SM_Z_02",
            "MVS_P2_UDS_NM_Z_01",
            "MVS_P2_UDS_NM_Z_02",
        ]
        if recording_names[:6].tolist() != expected:
            raise ValueError("Prepared recording order does not match multicondition split")
        train_indices = np.flatnonzero(np.isin(recording_ids, [0, 1, 2]))
        validation_indices = np.flatnonzero(recording_ids == 3)
        calibration_indices = np.flatnonzero(recording_ids == 4)
        external_healthy_id = 5
        damage_start_id = 6
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

    flat_raw = feature_tensor.reshape(feature_tensor.shape[0], -1).astype(np.float64)
    eov_metadata = {"method": "None"}
    if eov_adjustment == "linear":
        if shaker_eov is None or shaker_eov_names is None:
            raise ValueError(
                "Prepared file has no shaker EOV descriptors; regenerate it "
                "with hbta_pipeline.prepare_vqpiae"
            )
        flat_raw, eov_metadata = residualize_against_eov(
            flat_raw, shaker_eov, train_indices, eov_ridge
        )
        eov_metadata["feature_names"] = shaker_eov_names.tolist()
    elif eov_adjustment != "none":
        raise ValueError(f"Unknown EOV adjustment: {eov_adjustment}")
    scaler = StandardScaler().fit(flat_raw[train_indices])
    flat = scaler.transform(flat_raw).astype(np.float32)
    sensor_count, feature_count = feature_tensor.shape[1:]
    topology_edges = sensor_topology(sensor_names)
    topology_matrix = adjacency(sensor_count, topology_edges)
    spatial3d = build_hbta_spatial_graph(sensor_names, positions)
    spatial3d_matrix = adjacency(
        sensor_count, spatial3d.edges, spatial3d.weights
    )
    methods: dict[str, list[dict]] = {}

    for seed in seeds:
        torch.manual_seed(seed)
        models = {
            "piae": Autoencoder(flat.shape[1], 32),
            "vq_piae": VQAutoencoder(flat.shape[1], 32, 32, 4),
        }
        trained = {}
        for model_name, model in models.items():
            trained[model_name] = train_model(
                model,
                flat[train_indices],
                flat[validation_indices],
                device,
                seed,
                max_epochs,
            )

        random_matrix = adjacency(
            sensor_count,
            random_edges(sensor_count, topology_edges.shape[0], 1000 + seed),
        )
        permutation = np.random.default_rng(2000 + seed).permutation(sensor_count)
        permuted_matrix = topology_matrix[permutation][:, permutation]
        spatial3d_random = random_spatial_control(
            spatial3d, sensor_names, positions, 3000 + seed
        )
        spatial3d_permuted = permuted_spatial_control(
            spatial3d, sensor_names, positions, 4000 + seed
        )
        spatial3d_random_matrix = adjacency(
            sensor_count, spatial3d_random.edges, spatial3d_random.weights
        )
        spatial3d_permuted_matrix = adjacency(
            sensor_count, spatial3d_permuted.edges, spatial3d_permuted.weights
        )

        for model_name, (model, best_epoch) in trained.items():
            all_residuals = residuals(model, flat, device)
            train_reliability = np.mean(all_residuals[train_indices] ** 2, axis=0)
            reliability_grid = np.log10(train_reliability + 1e-12).reshape(
                sensor_count, feature_count
            )
            attention_variants = {
                "attention": reliability_grid,
                "topology_attention": (1.0 - smoothing) * reliability_grid
                + smoothing * (topology_matrix @ reliability_grid),
                "random_graph_attention": (1.0 - smoothing) * reliability_grid
                + smoothing * (random_matrix @ reliability_grid),
                "permuted_topology_attention": (1.0 - smoothing) * reliability_grid
                + smoothing * (permuted_matrix @ reliability_grid),
                "spatial3d_attention": (1.0 - smoothing) * reliability_grid
                + smoothing * (spatial3d_matrix @ reliability_grid),
                "spatial3d_random_attention": (1.0 - smoothing) * reliability_grid
                + smoothing * (spatial3d_random_matrix @ reliability_grid),
                "spatial3d_permuted_attention": (1.0 - smoothing) * reliability_grid
                + smoothing * (spatial3d_permuted_matrix @ reliability_grid),
                "global_shrink_attention": (1.0 - smoothing) * reliability_grid
                + smoothing * reliability_grid.mean(),
            }
            squared = all_residuals**2
            plain_name = f"{model_name}_plain"
            plain_result = metric_result(
                squared.sum(axis=1), recording_ids, recording_names, calibration_indices,
                external_healthy_id, damage_start_id, healthy_test_indices,
            )
            plain_result.update({"seed": seed, "best_epoch": best_epoch})
            methods.setdefault(plain_name, []).append(plain_result)

            for variant_name, log_reliability in attention_variants.items():
                weights, beta = attention_from_log_reliability(log_reliability.reshape(-1))
                scores = np.sum(squared * weights, axis=1)
                method_name = f"{model_name}_{variant_name}"
                result = metric_result(
                    scores, recording_ids, recording_names, calibration_indices,
                    external_healthy_id, damage_start_id, healthy_test_indices,
                )
                result.update(
                    {
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "beta": beta,
                        "effective_features": float(1.0 / np.sum(weights**2)),
                    }
                )
                methods.setdefault(method_name, []).append(result)

        vq_model = trained["vq_piae"][0]
        code_indices = vq_model.code_indices(
            torch.from_numpy(flat[train_indices]).to(device)
        ).cpu().numpy()
        code_usage = int(np.unique(code_indices, axis=0).shape[0])
        for method_name in methods:
            if method_name.startswith("vq_piae") and methods[method_name][-1]["seed"] == seed:
                methods[method_name][-1]["joint_codes_used"] = code_usage

        print(
            f"seed={seed} "
            f"VQ-attn={methods['vq_piae_attention'][-1]['block_auroc']:.3f} "
            f"VQ-topology={methods['vq_piae_topology_attention'][-1]['block_auroc']:.3f} "
            f"healthy-FAR={methods['vq_piae_topology_attention'][-1]['external_healthy_false_alarm_rate']:.3f}",
            flush=True,
        )

    summary = {}
    for method_name, rows in methods.items():
        summary[method_name] = {}
        for metric in ["block_auroc", "block_auprc", "external_healthy_false_alarm_rate"]:
            values = np.asarray([row[metric] for row in rows])
            summary[method_name][f"mean_{metric}"] = float(values.mean())
            summary[method_name][f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    return {
        "model_source": "VQ-PIAE-A architecture and reliability attention from vq_piae_z24.py",
        "input": {
            "blocks": int(feature_tensor.shape[0]),
            "sensors": sensor_count,
            "features_per_sensor": feature_count,
            "dimension": int(flat.shape[1]),
            "feature_names": feature_names.tolist(),
            "descriptor_groups": descriptor_groups.tolist(),
            "preprocessing": (
                "StandardScaler fitted on the selected healthy training blocks only"
            ),
        },
        "eov_adjustment": eov_metadata,
        "split": {
            "mode": split_mode,
            "train": int(train_indices.size),
            "validation": int(validation_indices.size),
            "calibration": int(calibration_indices.size),
            "external_healthy": (
                recording_names[:2].tolist()
                if split_mode == "both_healthy80"
                else (
                    f"Final 20% of {recording_names[0]}"
                    if split_mode == "single_recording"
                    else recording_names[external_healthy_id]
                )
            ),
            "damage": recording_names[damage_start_id:].tolist(),
        },
        "topology": {
            "definition": "AG two-wall truss grid plus AL 10x4 deck grid",
            "sensor_edges": int(topology_edges.shape[0]),
            "smoothing": smoothing,
            "affects": "reliability attention only; trained model and parameter count are unchanged",
            "spatial3d": {
                "definition": (
                    "Coordinate-constrained truss, deck, and deck-truss interface; "
                    "not member-verified"
                ),
                "weighting": "relation-balanced inverse edge length",
                "audit": graph_audit(spatial3d, sensor_names),
            },
        },
        "seeds": seeds,
        "device": str(device),
        "summary": summary,
        "runs": methods,
        "inference_warning": (
            "Blocks within a recording are correlated and each damaged state has one recording. "
            "Seed variation measures initialization, not physical-test uncertainty."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("HBTA/results/vqpiae_topology.json")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--split-mode",
        choices=[
            "controlled",
            "controlled80",
            "single_recording",
            "both_healthy80",
            "multicondition",
        ],
        default="controlled80",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        help=(
            "Optional predefined descriptor groups to retain, for example "
            "--groups response transmissibility"
        ),
    )
    parser.add_argument(
        "--eov-adjustment",
        choices=["none", "linear"],
        default="none",
        help="Healthy-only shaker EOV adjustment applied before standardization",
    )
    parser.add_argument("--eov-ridge", type=float, default=1.0)
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
        args.max_epochs,
        args.smoothing,
        device,
        args.split_mode,
        args.groups,
        args.eov_adjustment,
        args.eov_ridge,
    )
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
