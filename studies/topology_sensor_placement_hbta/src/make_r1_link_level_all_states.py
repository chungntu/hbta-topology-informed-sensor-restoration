"""Revision R1, Section 5.1: link-level physical consistency for all damage states DS1-DS8.

Each damage state s is represented by the pre-registered set of damaged springs D_s
(configs/r1_damage_mapping.json).  For every state:
  q_s : modelled response weights when all springs of D_s are weakened together (same spring model,
        severities and ground-spring values as the response-weight bank of Section 3.1)
  R_s : measured link response, computed exactly as for DS1 in the submitted Section 5.1
and the Spearman rank correlation between q_s and R_s is reported with a 95% bootstrap interval.
The rank of the true damage set among all 122 single-spring locations of the bank is reported as
a location check.

Outputs (new folder, nothing submitted is overwritten):
  outputs/R1_physical_consistency/link_level_all_states.json
  outputs/R1_physical_consistency/link_level_all_states.npz   (q and R for the primary sets)
  ../../paper/outage_conditioned_prioritization/generated_tables_R1/table_r1_*.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(STUDY_ROOT / "src"))

from make_physical_consistency_analysis import bootstrap_spearman  # noqa: E402

CONFIG_PATH = STUDY_ROOT / "configs" / "virtual_damage_pareto.json"
MAPPING_PATH = STUDY_ROOT / "configs" / "r1_damage_mapping.json"
BANK_PATH = STUDY_ROOT / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz"
OUTPUT = STUDY_ROOT / "outputs" / "R1_physical_consistency"
TABLES = REPO_ROOT / "paper" / "outage_conditioned_prioritization" / "generated_tables_R1"
STATES = [f"DS{k}" for k in range(1, 9)]
SEED = 20261105
COMPONENT = {
    "DS1": "Stringer--floor-beam connection", "DS2": "Stringer--floor-beam connections",
    "DS3": "Stringer cross beam", "DS4": "Stringer cross beams",
    "DS5": "Lateral-bracing connection", "DS6": "Lateral-bracing connection",
    "DS7": "Lateral-bracing connections", "DS8": "Floor-beam--main-member connection",
}


def response_weights(incidence, weights, reliability, damage_sets, reductions, alpha_factors):
    """Spring-model response weights for weakening every spring of each set together."""
    laplacian = incidence.T @ (weights[:, None] * incidence)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    reference = float(np.median(eigenvalues[eigenvalues > 1e-8]))
    size = laplacian.shape[0]
    base = [np.linalg.inv(laplacian + reference * f * np.eye(size)) for f in alpha_factors]
    q = np.zeros((incidence.shape[0], len(damage_sets)))
    for column, damaged in enumerate(damage_sets):
        delta = sum(weights[d] * np.outer(incidence[d], incidence[d]) for d in damaged)
        for rho in reductions:
            for factor, flexibility in zip(alpha_factors, base):
                damaged_flex = np.linalg.inv(laplacian - rho * delta + reference * factor * np.eye(size))
                q[:, column] += np.sum((incidence @ (damaged_flex - flexibility)) ** 2, axis=1) * reliability
    return q / q.sum(axis=0, keepdims=True)


def measured_responses(config):
    """Measured link response R for every damage state (identical to the submitted DS1 analysis)."""
    from hbta_pipeline.edge_spectral_topology_experiment import (
        edge_spectral_features, rolling_recording_ids)
    from run_placement import raw_edge_summary, robust_center_scale
    with np.load((STUDY_ROOT / config["prepared_spectra"]).resolve()) as data:
        coefficients = data["fourier_coefficients"].astype(np.complex64)
        source_ids = data["recording_ids"].astype(np.int16)
        recording_names = data["recording_names"].astype(str)
    with np.load((STUDY_ROOT / config["teacher_state"]).resolve()) as teacher:
        center = teacher["coord_center"].astype(np.float32)
        scale = teacher["coord_scale"].astype(np.float32)
    edges = np.load(BANK_PATH)["edges"]
    recording_ids = rolling_recording_ids(source_ids, int(config["rolling_blocks"]))
    fit = np.flatnonzero(np.isin(recording_ids, config["fit_recording_ids"]))
    features = edge_spectral_features(coefficients, edges, source_ids, int(config["rolling_blocks"]))
    standardized = ((features - center[None]) / scale[None]).astype(np.float32)
    raw = raw_edge_summary(standardized, int(config["top_frequency_count"]), remove_spatial_median=False)
    local_center, local_scale = robust_center_scale(raw[fit], 1e-4)
    local = np.maximum((raw - local_center[None]) / local_scale[None], 0.0)
    out = {}
    for state in STATES:
        recording = int(np.flatnonzero(np.char.find(recording_names, f"_{state}_") >= 0)[0])
        out[state] = np.median(local[recording_ids == recording], axis=0)
    return out


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    from hbta_pipeline.spatial_topology import build_hbta_spatial_graph
    with np.load((STUDY_ROOT / config["prepared_spectra"]).resolve()) as data:
        names = data["sensor_names"].astype(str)
        positions = data["sensor_positions"].astype(np.float64)
    with np.load(BANK_PATH) as bank:
        q_bank = bank["sensitivity"].astype(np.float64)
        reliability = bank["reliability"].astype(np.float64)
        edges = bank["edges"].astype(int)
    graph = build_hbta_spatial_graph(names, positions)
    if not np.array_equal(graph.edges, edges):
        raise ValueError("Graph and response-weight bank do not match")
    node_count, edge_count = len(names), len(edges)
    incidence = np.zeros((edge_count, node_count))
    incidence[np.arange(edge_count), edges[:, 0]] = 1.0
    incidence[np.arange(edge_count), edges[:, 1]] = -1.0
    weights = graph.weights * np.sqrt(reliability)
    weights /= weights.mean()
    reductions, alphas = config["damage_weight_reductions"], config["resolvent_alpha_factors"]
    index = {}
    for m, (u, v) in enumerate(edges):
        index[f"{names[u]}--{names[v]}"] = index[f"{names[v]}--{names[u]}"] = m

    check = response_weights(incidence, weights, reliability, [[0], [edge_count - 1]], reductions, alphas)
    if not np.allclose(check, q_bank[:, [0, edge_count - 1]], atol=1e-6):
        raise ValueError("Spring model does not reproduce the stored response-weight bank")

    measured = measured_responses(config)
    rng = np.random.default_rng(SEED)
    rows, q_primary = [], np.zeros((len(STATES), edge_count))
    for state in STATES:
        entry = next(item for item in mapping["states"] if item["state"] == state)
        variants = {"primary": entry["springs"], **entry.get("alternatives", {})}
        for variant, springs in variants.items():
            damaged = [index[s] for s in springs]
            q = response_weights(incidence, weights, reliability, [damaged], reductions, alphas)[:, 0]
            r = measured[state]
            rho = float(spearmanr(q, r).statistic)
            rho_bank = np.array([spearmanr(q_bank[:, j], r).statistic for j in range(edge_count)])
            row = {
                "state": state, "variant": variant, "springs": springs,
                "component": COMPONENT[state],
                "detectable": bool(entry.get("usable_for_detection_validation", True)),
                "spearman_rho": rho,
                "location_rank_of_122": int(1 + np.sum(rho_bank > rho)),
            }
            if variant == "primary":
                row["spearman_ci95"] = bootstrap_spearman(q, r, rng)
                q_primary[STATES.index(state)] = q
            rows.append(row)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "link_level_all_states.json").write_text(json.dumps(
        {"mapping": str(MAPPING_PATH.name), "seed": SEED, "rows": rows}, indent=2), encoding="utf-8")
    np.savez_compressed(OUTPUT / "link_level_all_states.npz", states=np.array(STATES), q=q_primary,
                        R=np.array([measured[s] for s in STATES]), edges=edges,
                        positions=positions, sensor_names=names)

    TABLES.mkdir(parents=True, exist_ok=True)
    primary = [r for r in rows if r["variant"] == "primary"]
    with (TABLES / "table_r1_damage_sets.tex").open("w", encoding="utf-8", newline="\n") as f:
        for r in primary:
            springs = ", ".join(s.replace("--", "--") for s in r["springs"])
            f.write(f"{r['state']} & {r['component']} & {springs} \\\\\n")
        f.write("\\bottomrule\n")  # inside the file, as for the submitted generated tables
    with (TABLES / "table_r1_link_level.tex").open("w", encoding="utf-8", newline="\n") as f:
        for r in primary:
            lo, hi = r["spearman_ci95"]
            fmt = lambda v: f"${v:.2f}$" if v < 0 else f"{v:.2f}"
            f.write(f"{r['state']} & {r['component']} & {fmt(r['spearman_rho'])} & "
                    f"[{fmt(lo)}, {fmt(hi)}] & {r['location_rank_of_122']} \\\\\n")
        f.write("\\bottomrule\n")
    for r in rows:
        ci = r.get("spearman_ci95")
        ci = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
        print(f"{r['state']} {r['variant']:26} rho={r['spearman_rho']:.3f} {ci:14} rank={r['location_rank_of_122']}")


if __name__ == "__main__":
    main()
