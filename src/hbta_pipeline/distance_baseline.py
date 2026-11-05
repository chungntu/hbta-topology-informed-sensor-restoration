"""Healthy-only robust-distance baseline for prepared HBTA features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def robust_scale_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(features, axis=0)
    mad = np.median(np.abs(features - center), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(features, axis=0, ddof=1)
    scale = np.where(scale > 1e-6, scale, np.maximum(fallback, 1e-6))
    return center, scale


def score(features: np.ndarray, center: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sensor_scores = np.mean(((features - center) / scale) ** 2, axis=-1)
    top_count = max(1, int(np.ceil(sensor_scores.shape[1] * 0.1)))
    global_scores = np.mean(
        np.partition(sensor_scores, -top_count, axis=1)[:, -top_count:], axis=1
    )
    return global_scores, sensor_scores


def run(prepared_path: Path) -> dict:
    with np.load(prepared_path) as data:
        features = data["features"].astype(np.float64)
        recording_ids = data["recording_ids"]
        recording_names = data["recording_names"].astype(str)

    development_indices = np.flatnonzero(recording_ids == 0)
    split = int(np.floor(0.6 * development_indices.size))
    train_indices = development_indices[:split]
    calibration_indices = development_indices[split:]
    center, scale = robust_scale_fit(features[train_indices])
    scores, _ = score(features, center, scale)
    threshold = float(np.quantile(scores[calibration_indices], 0.95))

    external_healthy = recording_ids == 1
    damaged = recording_ids >= 2
    labels = damaged[external_healthy | damaged].astype(np.int8)
    evaluation_scores = scores[external_healthy | damaged]
    by_recording = {}
    for recording_id, name in enumerate(recording_names):
        selected = recording_ids == recording_id
        values = scores[selected]
        by_recording[name] = {
            "blocks": int(selected.sum()),
            "median_score": float(np.median(values)),
            "mean_score": float(np.mean(values)),
            "positive_rate_at_calibration_q95": float(np.mean(values > threshold)),
        }

    return {
        "method": "robust_feature_distance",
        "split": {
            "development_recording": recording_names[0],
            "train_blocks": int(train_indices.size),
            "calibration_blocks": int(calibration_indices.size),
            "external_healthy_recording": recording_names[1],
            "damaged_recordings": recording_names[2:].tolist(),
        },
        "threshold_calibration_q95": threshold,
        "block_level_auroc": float(roc_auc_score(labels, evaluation_scores)),
        "block_level_auprc": float(average_precision_score(labels, evaluation_scores)),
        "external_healthy_false_positive_rate": float(
            np.mean(scores[external_healthy] > threshold)
        ),
        "by_recording": by_recording,
        "inference_warning": (
            "Blocks within a recording are correlated. The controlled data provide "
            "only one damaged recording per state, so these are descriptive results."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--output", type=Path, default=Path("HBTA/results/distance.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = run(args.prepared)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
