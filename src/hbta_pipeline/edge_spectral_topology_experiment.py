"""Healthy-only HBTA detector using full complex spectra along graph edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from hbta_pipeline.localized_topology_experiment import top_k_mean
from hbta_pipeline.spatial_topology import (
    SpatialGraph,
    build_hbta_spatial_graph,
    graph_audit,
    permuted_spatial_control,
    random_spatial_control,
)
from hbta_pipeline.topology_location_scan_experiment import (
    euclidean_distance_matrix,
    graph_distance_matrix,
    oracle_location_diagnostic,
    scan_score,
    standardized_location_scores,
)
from hbta_pipeline.vqpiae_experiment import metric_result


EDGE_FEATURE_NAMES = (
    "forward_transfer_log_magnitude",
    "reverse_transfer_log_magnitude",
    "coherence_weighted_phase_cosine",
    "coherence_weighted_phase_sine",
    "magnitude_squared_coherence",
)
FREQUENCY_TOP_COUNT = 8
EDGE_TOP_COUNT = 12
KERNEL_LENGTHS_M = (3.5, 7.0)
ROBUST_COMBINATION_WEIGHTS = (0.01, 0.025, 0.05, 0.1)


def calibration_normalized_sum(
    baseline: np.ndarray,
    topology: np.ndarray,
    calibration_indices: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Combine two scores on robust calibration scales, allowing signed scans."""
    components = []
    for scores in (baseline, topology):
        calibration = scores[calibration_indices]
        center = float(np.median(calibration))
        mad = float(np.median(np.abs(calibration - center)))
        scale = max(1.4826 * mad, float(np.std(calibration)), 1e-6)
        components.append((scores - center) / scale)
    return components[0] + weight * components[1]


def edge_spectral_features(
    coefficients: np.ndarray,
    edges: np.ndarray,
    recording_ids: np.ndarray | None = None,
    rolling_blocks: int = 1,
    chunk_size: int = 16,
) -> np.ndarray:
    """Return five full-frequency transfer descriptors for every graph edge."""
    if rolling_blocks < 1:
        raise ValueError("rolling_blocks must be positive")
    if rolling_blocks > 1 and recording_ids is None:
        raise ValueError("recording_ids are required for rolling spectra")
    if rolling_blocks == 1:
        sample_count = coefficients.shape[0]
    else:
        sample_count = rolling_recording_ids(
            np.asarray(recording_ids), rolling_blocks
        ).size
    frequency_count = coefficients.shape[-1]
    output = np.empty(
        (sample_count, edges.shape[0], frequency_count, 5),
        dtype=np.float32,
    )
    epsilon = 1e-12
    for start in range(0, edges.shape[0], chunk_size):
        stop = min(start + chunk_size, edges.shape[0])
        selected = edges[start:stop]
        source = coefficients[:, selected[:, 0]]
        target = coefficients[:, selected[:, 1]]
        source_power = np.mean(np.abs(source) ** 2, axis=2)
        target_power = np.mean(np.abs(target) ** 2, axis=2)
        cross_power = np.mean(np.conj(source) * target, axis=2)
        if rolling_blocks > 1:
            source_power = rolling_mean_by_recording(
                source_power, recording_ids, rolling_blocks
            )
            target_power = rolling_mean_by_recording(
                target_power, recording_ids, rolling_blocks
            )
            cross_power = rolling_mean_by_recording(
                cross_power, recording_ids, rolling_blocks
            )
        forward = cross_power / np.maximum(source_power, epsilon)
        reverse = np.conj(cross_power) / np.maximum(
            target_power, epsilon
        )
        coherence = np.abs(cross_power) ** 2 / np.maximum(
            source_power * target_power, epsilon
        )
        coherence = np.clip(coherence, 0.0, 1.0)
        phase = np.angle(cross_power)
        phase_reliability = np.sqrt(coherence)
        output[:, start:stop] = np.stack(
            [
                np.log(np.maximum(np.abs(forward), epsilon)),
                np.log(np.maximum(np.abs(reverse), epsilon)),
                phase_reliability * np.cos(phase),
                phase_reliability * np.sin(phase),
                coherence,
            ],
            axis=-1,
        ).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Edge spectral features contain non-finite values")
    return output


def rolling_mean_by_recording(
    values: np.ndarray,
    recording_ids: np.ndarray,
    window: int,
) -> np.ndarray:
    """Valid rolling mean that never crosses a recording boundary."""
    output = []
    for recording_id in np.unique(recording_ids):
        selected = values[recording_ids == recording_id]
        if selected.shape[0] < window:
            raise ValueError(
                f"Recording {recording_id} has fewer than {window} blocks"
            )
        cumulative = np.concatenate(
            [
                np.zeros((1, *selected.shape[1:]), dtype=selected.dtype),
                np.cumsum(selected, axis=0),
            ],
            axis=0,
        )
        output.append(
            (cumulative[window:] - cumulative[:-window]) / window
        )
    return np.concatenate(output, axis=0)


def rolling_recording_ids(
    recording_ids: np.ndarray,
    window: int,
) -> np.ndarray:
    output = []
    for recording_id in np.unique(recording_ids):
        count = int(np.sum(recording_ids == recording_id))
        if count < window:
            raise ValueError(
                f"Recording {recording_id} has fewer than {window} blocks"
            )
        output.append(
            np.full(count - window + 1, recording_id, dtype=recording_ids.dtype)
        )
    return np.concatenate(output)


def healthy_standardized_coordinates(
    features: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    healthy = features[train_indices]
    center = np.median(healthy, axis=0)
    mad = np.median(np.abs(healthy - center[None, :]), axis=0)
    standard_deviation = np.std(healthy, axis=0)
    scale = np.maximum.reduce(
        [1.4826 * mad, 0.1 * standard_deviation, np.full_like(mad, 1e-4)]
    )
    return ((features - center[None, :]) / scale[None, :]).astype(
        np.float32
    )


def edge_excess_scores(
    standardized: np.ndarray,
    train_indices: np.ndarray,
    top_frequency_count: int | None,
    remove_spatial_median: bool,
    studentize_edges_within_block: bool = False,
) -> np.ndarray:
    """Aggregate full curves, retaining only excess over healthy edge behavior."""
    values = standardized
    if remove_spatial_median:
        values = values - np.median(
            values, axis=1, keepdims=True
        )
    frequency_scores = np.mean(np.log1p(values**2), axis=3)
    if top_frequency_count is None:
        raw_edge_scores = np.mean(frequency_scores, axis=2)
    else:
        raw_edge_scores = top_k_mean(
            frequency_scores.reshape(
                -1, frequency_scores.shape[2]
            ),
            top_frequency_count,
        ).reshape(frequency_scores.shape[:2])

    if studentize_edges_within_block:
        spatial_center = np.median(
            raw_edge_scores, axis=1, keepdims=True
        )
        spatial_mad = np.median(
            np.abs(raw_edge_scores - spatial_center),
            axis=1,
            keepdims=True,
        )
        raw_edge_scores = (
            raw_edge_scores - spatial_center
        ) / np.maximum(1.4826 * spatial_mad, 1e-4)

    healthy = raw_edge_scores[train_indices]
    center = np.median(healthy, axis=0)
    mad = np.median(np.abs(healthy - center[None, :]), axis=0)
    standard_deviation = np.std(healthy, axis=0)
    scale = np.maximum.reduce(
        [1.4826 * mad, 0.1 * standard_deviation, np.full_like(mad, 1e-4)]
    )
    return np.maximum(
        (raw_edge_scores - center[None, :]) / scale[None, :],
        0.0,
    )


def candidate_edge_kernel(
    node_distances: np.ndarray,
    edges: np.ndarray,
    length_scale_m: float,
) -> np.ndarray:
    distance_to_edge = np.minimum(
        node_distances[:, edges[:, 0]],
        node_distances[:, edges[:, 1]],
    )
    weights = np.exp(-distance_to_edge / length_scale_m)
    weights[~np.isfinite(distance_to_edge)] = 0.0
    weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
    return weights


def graph_edge_scores(
    coefficients: np.ndarray,
    graph: SpatialGraph,
    train_indices: np.ndarray,
    source_recording_ids: np.ndarray,
    rolling_blocks: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features = edge_spectral_features(
        coefficients,
        graph.edges,
        source_recording_ids,
        rolling_blocks,
    )
    standardized = healthy_standardized_coordinates(
        features, train_indices
    )
    diagnostics = {
        "absolute_all_frequencies": edge_excess_scores(
            standardized, train_indices, None, False
        ),
        "absolute_top_frequencies": edge_excess_scores(
            standardized, train_indices, FREQUENCY_TOP_COUNT, False
        ),
        "contrast_all_frequencies": edge_excess_scores(
            standardized, train_indices, None, True
        ),
        "studentized_top_frequencies": edge_excess_scores(
            standardized,
            train_indices,
            FREQUENCY_TOP_COUNT,
            False,
            studentize_edges_within_block=True,
        ),
    }
    primary = edge_excess_scores(
        standardized, train_indices, FREQUENCY_TOP_COUNT, True
    )
    return primary, diagnostics


def evaluate_graph_scores(
    edge_scores: np.ndarray,
    graph: SpatialGraph,
    node_distances: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[float, np.ndarray]]:
    global_mean = np.mean(edge_scores, axis=1)
    dynamic_top = top_k_mean(edge_scores, EDGE_TOP_COUNT)
    scores = {
        "global_edge_mean": global_mean,
        f"dynamic_top_{EDGE_TOP_COUNT}_edges": dynamic_top,
        f"top_{EDGE_TOP_COUNT}_concentration_ratio": (
            dynamic_top / np.maximum(global_mean, 1e-8)
        ),
        "edge_coefficient_of_variation": (
            np.std(edge_scores, axis=1) / np.maximum(global_mean, 1e-8)
        ),
    }
    location_matrices = {}
    for length_scale in KERNEL_LENGTHS_M:
        kernel = candidate_edge_kernel(
            node_distances, graph.edges, length_scale
        )
        locations = standardized_location_scores(
            edge_scores, kernel, train_indices
        )
        location_matrices[length_scale] = locations
        scores[f"location_scan_{length_scale:g}m"] = scan_score(locations)
    return scores, location_matrices


def summarize(rows: dict[str, list[dict]]) -> dict:
    output = {}
    for method, method_rows in rows.items():
        output[method] = {}
        for metric in (
            "block_auroc",
            "block_auprc",
            "external_healthy_false_alarm_rate",
        ):
            values = np.asarray([row[metric] for row in method_rows])
            output[method][f"mean_{metric}"] = float(np.mean(values))
            output[method][f"std_{metric}"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
    return output


def run(
    prepared_path: Path,
    control_seeds: list[int],
    rolling_blocks: int,
) -> dict:
    with np.load(prepared_path) as data:
        coefficients = data["fourier_coefficients"]
        frequencies = data["frequencies_hz"].astype(np.float64)
        source_recording_ids = data["recording_ids"]
        recording_names = data["recording_names"].astype(str)
        sensor_names = data["sensor_names"].astype(str)
        positions = data["sensor_positions"].astype(np.float64)
        source_block_seconds = float(data["block_seconds"])

    recording_ids = rolling_recording_ids(
        source_recording_ids, rolling_blocks
    )
    if recording_names[0].startswith("MVS_P1_UDS"):
        train_indices = np.flatnonzero(
            np.isin(recording_ids, [0, 1, 2])
        )
        validation_indices = np.flatnonzero(recording_ids == 3)
        calibration_indices = np.flatnonzero(recording_ids == 4)
        uds02 = np.flatnonzero(recording_ids == 5)
        external_healthy_id = 5
        damage_start_id = 6
        protocol = {
            "fit_and_coordinate_normalization": (
                "P1/SM/Z/01, P1/SM/Z/02, and P2/SM/Z/01"
            ),
            "healthy_validation": "P2/SM/Z/02",
            "threshold_calibration": "P2/NM/Z/01",
            "healthy_test": "P2/NM/Z/02 only",
            "damage_test": "DS1-DS8; never used for fitting or selection",
        }
    else:
        uds01 = np.flatnonzero(recording_ids == 0)
        split = int(np.floor(0.8 * uds01.size))
        train_indices = uds01[:split]
        validation_indices = uds01[split:]
        calibration_indices = validation_indices
        uds02 = np.flatnonzero(recording_ids == 1)
        external_healthy_id = 1
        damage_start_id = 2
        protocol = {
            "fit_and_coordinate_normalization": "First 80% UDS_01",
            "healthy_validation": "Last 20% UDS_01",
            "threshold_calibration": "Last 20% UDS_01",
            "healthy_test": "All UDS_02 only",
            "damage_test": "DS1-DS8; never used for fitting or selection",
        }

    physical = build_hbta_spatial_graph(sensor_names, positions)
    physical_distances = graph_distance_matrix(
        physical, sensor_names.size
    )
    euclidean_distances = euclidean_distance_matrix(positions)
    rows: dict[str, list[dict]] = {}

    def add(name: str, scores: np.ndarray, seed: int | None = None) -> None:
        result = metric_result(
            scores,
            recording_ids,
            recording_names,
            calibration_indices,
            external_healthy_id=external_healthy_id,
            damage_start_id=damage_start_id,
            external_healthy_indices=uds02,
        )
        if seed is not None:
            result["control_seed"] = seed
        rows.setdefault(name, []).append(result)

    physical_edges, diagnostics = graph_edge_scores(
        coefficients,
        physical,
        train_indices,
        source_recording_ids,
        rolling_blocks,
    )
    physical_scores, physical_locations = evaluate_graph_scores(
        physical_edges, physical, physical_distances, train_indices
    )
    for name, scores in physical_scores.items():
        add(f"physical_{name}", scores)

    physical_diagnostic_scores = {}
    for diagnostic_name, edge_scores in diagnostics.items():
        diagnostic_scores, _ = evaluate_graph_scores(
            edge_scores, physical, physical_distances, train_indices
        )
        physical_diagnostic_scores[diagnostic_name] = diagnostic_scores
        for name, scores in diagnostic_scores.items():
            add(f"physical_{diagnostic_name}_{name}", scores)
        for length_scale in KERNEL_LENGTHS_M:
            diagnostic_euclidean_kernel = candidate_edge_kernel(
                euclidean_distances, physical.edges, length_scale
            )
            diagnostic_euclidean_locations = (
                standardized_location_scores(
                    edge_scores,
                    diagnostic_euclidean_kernel,
                    train_indices,
                )
            )
            add(
                f"euclidean_{diagnostic_name}_location_scan_"
                f"{length_scale:g}m",
                scan_score(diagnostic_euclidean_locations),
            )

    robust_physical = physical_diagnostic_scores[
        "contrast_all_frequencies"
    ]["global_edge_mean"]
    for length_scale in KERNEL_LENGTHS_M:
        topology_physical = physical_diagnostic_scores[
            "absolute_top_frequencies"
        ][f"location_scan_{length_scale:g}m"]
        for weight in ROBUST_COMBINATION_WEIGHTS:
            add(
                f"physical_robust_plus_absolute_topology_"
                f"{length_scale:g}m_weight_{weight}",
                calibration_normalized_sum(
                    robust_physical,
                    topology_physical,
                    calibration_indices,
                    weight,
                ),
            )

    for length_scale in KERNEL_LENGTHS_M:
        euclidean_kernel = candidate_edge_kernel(
            euclidean_distances, physical.edges, length_scale
        )
        euclidean_locations = standardized_location_scores(
            physical_edges, euclidean_kernel, train_indices
        )
        add(
            f"euclidean_location_scan_{length_scale:g}m",
            scan_score(euclidean_locations),
        )

    for seed in control_seeds:
        graph_controls = {
            "random": random_spatial_control(
                physical, sensor_names, positions, 9000 + seed
            ),
            "permuted": permuted_spatial_control(
                physical, sensor_names, positions, 10000 + seed
            ),
        }
        for control_name, control in graph_controls.items():
            control_edges, control_diagnostics = graph_edge_scores(
                coefficients,
                control,
                train_indices,
                source_recording_ids,
                rolling_blocks,
            )
            control_distances = graph_distance_matrix(
                control, sensor_names.size
            )
            control_scores, _ = evaluate_graph_scores(
                control_edges,
                control,
                control_distances,
                train_indices,
            )
            for name, scores in control_scores.items():
                add(f"{control_name}_{name}", scores, seed)
            control_diagnostic_scores = {}
            for diagnostic_name, diagnostic_edges in (
                control_diagnostics.items()
            ):
                diagnostic_scores, _ = evaluate_graph_scores(
                    diagnostic_edges,
                    control,
                    control_distances,
                    train_indices,
                )
                control_diagnostic_scores[
                    diagnostic_name
                ] = diagnostic_scores
                for name, scores in diagnostic_scores.items():
                    add(
                        f"{control_name}_{diagnostic_name}_{name}",
                        scores,
                        seed,
                    )
            robust_control = control_diagnostic_scores[
                "contrast_all_frequencies"
            ]["global_edge_mean"]
            for length_scale in KERNEL_LENGTHS_M:
                topology_control = control_diagnostic_scores[
                    "absolute_top_frequencies"
                ][f"location_scan_{length_scale:g}m"]
                for weight in ROBUST_COMBINATION_WEIGHTS:
                    add(
                        f"{control_name}_robust_plus_absolute_topology_"
                        f"{length_scale:g}m_weight_{weight}",
                        calibration_normalized_sum(
                            robust_control,
                            topology_control,
                            calibration_indices,
                            weight,
                        ),
                        seed,
                    )
        print(f"control_seed={seed}", flush=True)

    oracle = {}
    for length_scale, locations in physical_locations.items():
        oracle[f"physical_scale_{length_scale:g}m"] = (
            oracle_location_diagnostic(
                locations,
                recording_ids,
                recording_names,
                sensor_names,
                positions,
                uds02,
            )
        )

    return {
        "protocol": protocol,
        "input": {
            "frequency_min_hz": float(frequencies.min()),
            "frequency_max_hz": float(frequencies.max()),
            "frequency_count": int(frequencies.size),
            "frequency_resolution_hz": float(np.median(np.diff(frequencies))),
            "edge_feature_names": EDGE_FEATURE_NAMES,
            "features_per_edge": int(
                frequencies.size * len(EDGE_FEATURE_NAMES)
            ),
            "frequency_top_count": FREQUENCY_TOP_COUNT,
            "edge_top_count": EDGE_TOP_COUNT,
            "robust_combination_weights": ROBUST_COMBINATION_WEIGHTS,
            "source_block_seconds": source_block_seconds,
            "rolling_blocks": rolling_blocks,
            "effective_spectral_window_seconds": (
                source_block_seconds * rolling_blocks
            ),
            "effective_step_seconds": source_block_seconds,
        },
        "primary_score": (
            "Healthy-standardized full edge spectrum; median spatial shift "
            "removed per frequency/feature; top eight frequencies retained"
        ),
        "physical_graph": graph_audit(physical, sensor_names),
        "control_seeds": control_seeds,
        "summary": summarize(rows),
        "runs": rows,
        "oracle_location_diagnostic": {
            "warning": (
                "Uses damage labels only after evaluation; not a deployable "
                "detector or valid selection mechanism."
            ),
            "physical": oracle,
        },
        "warning": (
            "Diagnostic aggregation variants are reported together and require "
            "independent physical confirmation before selecting one."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("HBTA/results/edge_spectral_topology_external.json"),
    )
    parser.add_argument(
        "--control-seeds", type=int, nargs="+", default=list(range(10))
    )
    parser.add_argument("--rolling-blocks", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        args.prepared, args.control_seeds, args.rolling_blocks
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
