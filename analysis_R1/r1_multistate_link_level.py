"""R1.5 / R2.2 / R2.3: link-level consistency for all damage states, with baselines.

For each damage state s with pre-registered spring set D_s (analysis_R1/r1_damage_mapping.json):
  q_s  : response weights from weakening all springs in D_s together (same model as the paper)
  R_s  : measured link response (same empirical scalar as the submitted DS1 analysis)
and Spearman(q_s, R_s) is compared with
  - graph-distance baseline   : -(hop distance from the link to D_s)
  - Euclidean baseline        : -(distance from link midpoint to the nearest midpoint of D_s)
  - topology-only model       : same spring model with uniform stiffness and no reliability
  - location test             : rank of the true D_s among all 122 single-spring locations

Run from SHM-R1:  python analysis_R1/r1_multistate_link_level.py
Writes only to analysis_R1/outputs/multistate_link_level/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "topology_sensor_placement_hbta"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(STUDY / "src"))

from make_physical_consistency_analysis import bootstrap_spearman  # noqa: E402

OUT = ROOT / "analysis_R1" / "outputs" / "multistate_link_level"
MAPPING = ROOT / "analysis_R1" / "r1_damage_mapping.json"
SEED = 20261105


def response_weights(B, w, rel, sets, reductions, alpha_factors):
    """q for weakening every spring of each set together (columns = sets)."""
    L = B.T @ (w[:, None] * B)
    lam = np.linalg.eigvalsh(L)
    ref = float(np.median(lam[lam > 1e-8]))
    N = L.shape[0]
    base = [np.linalg.inv(L + ref * f * np.eye(N)) for f in alpha_factors]
    Q = np.zeros((B.shape[0], len(sets)))
    for j, idx in enumerate(sets):
        dL = sum(w[d] * np.outer(B[d], B[d]) for d in idx)
        for rho in reductions:
            for f, F0 in zip(alpha_factors, base):
                F = np.linalg.inv(L - rho * dL + ref * f * np.eye(N))
                Q[:, j] += np.sum((B @ (F - F0)) ** 2, axis=1) * rel
    return Q / Q.sum(axis=0, keepdims=True)


def measured_responses(config):
    """R_{m,s} for every recording, exactly as in the submitted DS1 analysis."""
    from hbta_pipeline.edge_spectral_topology_experiment import (
        edge_spectral_features, rolling_recording_ids)
    from run_placement import raw_edge_summary, robust_center_scale
    with np.load((STUDY / config["prepared_spectra"]).resolve()) as d:
        coeff = d["fourier_coefficients"].astype(np.complex64)
        src = d["recording_ids"].astype(np.int16)
        rec_names = d["recording_names"].astype(str)
    with np.load((STUDY / config["teacher_state"]).resolve()) as t:
        cc, cs = t["coord_center"].astype(np.float32), t["coord_scale"].astype(np.float32)
    edges = np.load(STUDY / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz")["edges"]
    rid = rolling_recording_ids(src, int(config["rolling_blocks"]))
    fit = np.flatnonzero(np.isin(rid, config["fit_recording_ids"]))
    feats = edge_spectral_features(coeff, edges, src, int(config["rolling_blocks"]))
    std = ((feats - cc[None]) / cs[None]).astype(np.float32)
    raw = raw_edge_summary(std, int(config["top_frequency_count"]), remove_spatial_median=False)
    c, s = robust_center_scale(raw[fit], 1e-4)
    local = np.maximum((raw - c[None]) / s[None], 0.0)
    out = {}
    for k in range(1, 9):
        r = int(np.flatnonzero(np.char.find(rec_names, f"_DS{k}_") >= 0)[0])
        out[f"DS{k}"] = np.median(local[rid == r], axis=0)
    return out


def main() -> None:
    config = json.loads((STUDY / "configs" / "virtual_damage_pareto.json").read_text(encoding="utf-8"))
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    from hbta_pipeline.spatial_topology import build_hbta_spatial_graph
    d = np.load(ROOT / "data" / "derived" / "hbta" / "edge_spectra_multicondition_10s.npz")
    names, pos = d["sensor_names"].astype(str), d["sensor_positions"].astype(float)
    bank = np.load(STUDY / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz")
    Qbank, rel, edges = bank["sensitivity"].astype(float), bank["reliability"].astype(float), bank["edges"]
    g = build_hbta_spatial_graph(names, pos)
    assert np.array_equal(g.edges, edges)
    N, M = len(names), len(edges)
    B = np.zeros((M, N)); B[np.arange(M), edges[:, 0]] = 1; B[np.arange(M), edges[:, 1]] = -1
    w = g.weights * np.sqrt(rel); w /= w.mean()
    red, alph = config["damage_weight_reductions"], config["resolvent_alpha_factors"]
    key = {}
    for m, (u, v) in enumerate(edges):
        key[f"{names[u]}--{names[v]}"] = key[f"{names[v]}--{names[u]}"] = m

    # sanity: singleton sets reproduce the stored bank
    check = response_weights(B, w, rel, [[0], [57]], red, alph)
    assert np.allclose(check, Qbank[:, [0, 57]], atol=1e-6), "model does not reproduce the bank"

    # geometry for baselines
    A = np.zeros((N, N)); A[edges[:, 0], edges[:, 1]] = A[edges[:, 1], edges[:, 0]] = 1
    hop = shortest_path(A, unweighted=True)
    link_hop = np.min([hop[np.ix_(edges[:, i], edges[:, j])] for i in (0, 1) for j in (0, 1)], axis=0)
    mid = (pos[edges[:, 0]] + pos[edges[:, 1]]) / 2
    link_dist = np.linalg.norm(mid[:, None] - mid[None], axis=2)
    w_uniform = np.ones(M)

    R = measured_responses(config)
    rng = np.random.default_rng(SEED)
    rows = []
    for st in mapping["states"]:
        s = st["state"]
        variants = {"primary": st["springs"], **st.get("alternatives", {})}
        for vname, springs in variants.items():
            D = [key[e] for e in springs]
            q = response_weights(B, w, rel, [D], red, alph)[:, 0]
            q_topo = response_weights(B, w_uniform, np.ones(M), [D], red, alph)[:, 0]
            graph_base = -np.min(link_hop[:, D], axis=1)
            eucl_base = -np.min(link_dist[:, D], axis=1)
            r = R[s]
            rho = spearmanr(q, r).statistic
            # location test: true set against every single-spring location of the bank
            rho_all = np.array([spearmanr(Qbank[:, j], r).statistic for j in range(M)])
            rank = int(1 + np.sum(rho_all > rho))
            rows.append({
                "state": s, "variant": vname, "springs": springs,
                "detectable": st.get("usable_for_detection_validation", True),
                "rho_model": float(rho),
                "rho_model_ci95": bootstrap_spearman(q, r, rng) if vname == "primary" else None,
                "rho_topology_only": float(spearmanr(q_topo, r).statistic),
                "rho_graph_distance": float(spearmanr(graph_base, r).statistic),
                "rho_euclidean": float(spearmanr(eucl_base, r).statistic),
                "location_rank_among_122": rank,
                "median_R": float(np.median(r)), "max_R": float(np.max(r)),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "multistate_link_level.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{'state':5} {'variant':26} {'model':>6} {'CI95':>14} {'topo':>6} {'graph':>6} {'eucl':>6} {'rank':>5}")
    for x in rows:
        ci = x["rho_model_ci95"]
        ci = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else ""
        flag = "" if x["detectable"] else " *"
        print(f"{x['state']:5} {x['variant'][:26]:26} {x['rho_model']:6.3f} {ci:>14} "
              f"{x['rho_topology_only']:6.3f} {x['rho_graph_distance']:6.3f} {x['rho_euclidean']:6.3f} "
              f"{x['location_rank_among_122']:5d}{flag}")


if __name__ == "__main__":
    main()
