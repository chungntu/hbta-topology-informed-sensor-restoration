"""R1.5 / R1.8 / R1.9 / R2.5 / R2.6 / R2.7: restoration decisions without knowing the damage location.

Scenario (all 58 x 57 two-sensor outages stored in outputs/expanded_criticality):
  companion sensor c and candidate i are both out; one of them, i, is restored.
  Measured gain on state s:  TPR_s(all but c) - TPR_s(all but c and i)   (same detector, same threshold)
Each rule ranks the 57 candidates i for every companion c using only the layout and outage state:
  lower_tail            : the paper's rule (Q from the spring model, gamma = 0.10)
  mean_coverage         : same Q, mean instead of lower tail
  lower_tail_topology   : lower tail with Q from uniform springs (topology only)
  link_count            : number of links reactivated by restoring i
  degree                : number of links at i in the full graph
  random                : expected value over candidates
Per-state rules that need the damage location (reported separately, not location-agnostic):
  nearest_to_damage     : candidate closest (hops) to the damaged spring set of that state
  damage_coverage       : Delta c_s for the known damaged set (as in submitted Section 5.2)

Run from SHM-R1:  python analysis_R1/r1_restoration_location_agnostic.py
Writes only to analysis_R1/outputs/restoration_location_agnostic/.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "topology_sensor_placement_hbta"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis_R1"))
from r1_multistate_link_level import response_weights  # noqa: E402

OUT = ROOT / "analysis_R1" / "outputs" / "restoration_location_agnostic"
DETECTABLE = ["DS1", "DS2", "DS3", "DS4", "DS7", "DS8"]
GAMMA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10


def load_tpr(names):
    idx = {n: i for i, n in enumerate(names)}
    N = len(names)
    single = {s: np.zeros(N) for s in [f"DS{k}" for k in range(1, 9)]}
    with open(STUDY / "outputs" / "expanded_criticality" / "single_quarantine_state_metrics.csv",
              encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            for s in single:
                single[s][idx[row["sensor"]]] = float(row[f"{s}_tpr"])
    pair = {s: np.full((N, N), np.nan) for s in single}
    for path in sorted(glob.glob(str(STUDY / "outputs" / "expanded_criticality" / "pairwise_parts" / "*.json"))):
        for row in json.load(open(path, encoding="utf-8"))["pairs"]:
            a, b = idx[row["first_sensor_name"]], idx[row["second_sensor_name"]]
            for s in single:
                v = row["metrics_after_quarantine"]["per_state"][f"MVS_P2_{s}_NM_Z_01"]["detection_rate"]
                pair[s][a, b] = pair[s][b, a] = v
    return single, pair


def lower_tail(cov, k):
    return np.sort(cov)[:k].mean()


def main() -> None:
    config = json.loads((STUDY / "configs" / "virtual_damage_pareto.json").read_text(encoding="utf-8"))
    mapping = json.loads((ROOT / "analysis_R1" / "r1_damage_mapping.json").read_text(encoding="utf-8"))
    from hbta_pipeline.spatial_topology import build_hbta_spatial_graph
    d = np.load(ROOT / "data" / "derived" / "hbta" / "edge_spectra_multicondition_10s.npz")
    names, pos = d["sensor_names"].astype(str), d["sensor_positions"].astype(float)
    bank = np.load(STUDY / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz")
    Q, rel, edges = bank["sensitivity"].astype(float), bank["reliability"].astype(float), bank["edges"]
    g = build_hbta_spatial_graph(names, pos)
    N, M = len(names), len(edges)
    B = np.zeros((M, N)); B[np.arange(M), edges[:, 0]] = 1; B[np.arange(M), edges[:, 1]] = -1
    red, alph = config["damage_weight_reductions"], config["resolvent_alpha_factors"]
    Q_topo = response_weights(B, np.ones(M), np.ones(M), [[m] for m in range(M)], red, alph)
    k = int(np.ceil(GAMMA * M))
    key = {}
    for m, (u, v) in enumerate(edges):
        key[f"{names[u]}--{names[v]}"] = key[f"{names[v]}--{names[u]}"] = m
    degree = np.abs(B).sum(axis=0)
    A = np.zeros((N, N)); A[edges[:, 0], edges[:, 1]] = A[edges[:, 1], edges[:, 0]] = 1
    hop = shortest_path(A, unweighted=True)

    single, pair = load_tpr(names)
    gain = {s: np.full((N, N), np.nan) for s in single}   # gain[s][c, i]
    for s in single:
        for c in range(N):
            for i in range(N):
                if i != c:
                    gain[s][c, i] = single[s][c] - pair[s][c, i]
    mean_gain = np.nanmean([gain[s] for s in DETECTABLE], axis=0)
    min_gain = np.nanmin([gain[s] for s in DETECTABLE], axis=0)

    def active(out):
        return ~np.isin(edges, list(out)).any(axis=1)

    rules = ["lower_tail", "mean_coverage", "lower_tail_topology", "link_count", "degree"]
    scores = {r: np.full((N, N), -np.inf) for r in rules}
    for c in range(N):
        chi_c = active({c})
        U_c, Ut_c, mean_c = lower_tail(Q.T @ chi_c, k), lower_tail(Q_topo.T @ chi_c, k), (Q.T @ chi_c).mean()
        for i in range(N):
            if i == c:
                continue
            chi = active({c, i})
            scores["lower_tail"][c, i] = U_c - lower_tail(Q.T @ chi, k)
            scores["mean_coverage"][c, i] = mean_c - (Q.T @ chi).mean()
            scores["lower_tail_topology"][c, i] = Ut_c - lower_tail(Q_topo.T @ chi, k)
            scores["link_count"][c, i] = float(np.sum(chi_c & ~chi))
            scores["degree"][c, i] = degree[i]

    def pick(score_row, tie_row):
        # highest score; ties by tie_row (mean-coverage gain), then sensor index
        cand = np.flatnonzero(np.isfinite(score_row))
        order = sorted(cand, key=lambda i: (-round(score_row[i], 12), -round(tie_row[i], 12), i))
        return order[0]

    summary = {}
    for r in rules:
        chosen = [pick(scores[r][c], scores["mean_coverage"][c]) for c in range(N)]
        mg = np.array([mean_gain[c, i] for c, i in enumerate(chosen)])
        ng = np.array([min_gain[c, i] for c, i in enumerate(chosen)])
        best = np.array([np.nanmax(mean_gain[c]) for c in range(N)])
        top3 = np.mean([mean_gain[c, i] >= np.sort(mean_gain[c][np.isfinite(mean_gain[c])])[-3]
                        for c, i in enumerate(chosen)])
        rho = np.nanmean([spearmanr(scores[r][c][np.arange(N) != c],
                                    mean_gain[c][np.arange(N) != c]).statistic for c in range(N)])
        summary[r] = {"mean_gain_of_choice": float(mg.mean()), "worst_state_gain_of_choice": float(ng.mean()),
                      "share_of_oracle": float(mg.mean() / best.mean()), "top3_hit_rate": float(top3),
                      "mean_spearman_over_candidates": float(rho)}
    rand = np.nanmean(mean_gain, axis=1)
    summary["random"] = {"mean_gain_of_choice": float(np.mean(rand)),
                         "worst_state_gain_of_choice": float(np.mean(np.nanmean(min_gain, axis=1))),
                         "share_of_oracle": float(np.mean(rand) / np.mean([np.nanmax(mean_gain[c]) for c in range(N)])),
                         "top3_hit_rate": 3 / 57, "mean_spearman_over_candidates": 0.0}
    summary["oracle"] = {"mean_gain_of_choice": float(np.mean([np.nanmax(mean_gain[c]) for c in range(N)]))}

    # per-state view, including rules that know the damage location
    per_state = {}
    for st in mapping["states"]:
        s = st["state"]
        if s not in DETECTABLE:
            continue
        D = [key[e] for e in st["springs"]]
        q_s = response_weights(B, g.weights * np.sqrt(rel) / np.mean(g.weights * np.sqrt(rel)), rel, [D], red, alph)[:, 0]
        dmg_nodes = np.unique(edges[D])
        res = {}
        for r in rules + ["nearest_to_damage", "damage_coverage"]:
            vals = []
            for c in range(N):
                if r == "nearest_to_damage":
                    sc = -np.min(hop[:, dmg_nodes], axis=1).astype(float); sc[c] = -np.inf
                elif r == "damage_coverage":
                    sc = np.array([q_s @ (active({c}) & ~active({c, i})) if i != c else -np.inf for i in range(N)])
                else:
                    sc = scores[r][c]
                i = pick(sc, scores["mean_coverage"][c])
                vals.append(gain[s][c, i])
            res[r] = float(np.mean(vals))
        res["random"] = float(np.nanmean(gain[s]))
        res["oracle"] = float(np.mean([np.nanmax(gain[s][c]) for c in range(N)]))
        per_state[s] = res

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"restoration_location_agnostic_gamma{GAMMA:.2f}.json").write_text(
        json.dumps({"location_agnostic": summary, "per_state": per_state,
                    "detectable_states": DETECTABLE, "tail_count": k}, indent=2), encoding="utf-8")
    print("LOCATION-AGNOSTIC (58 companions x 57 candidates; gains averaged over", DETECTABLE, ")")
    print(f"{'rule':22} {'meanGain':>9} {'worstDS':>8} {'%oracle':>8} {'top3':>6} {'rho':>6}")
    for r, v in summary.items():
        if r == "oracle":
            print(f"{r:22} {v['mean_gain_of_choice']:9.4f}")
            continue
        print(f"{r:22} {v['mean_gain_of_choice']:9.4f} {v['worst_state_gain_of_choice']:8.4f} "
              f"{100*v['share_of_oracle']:7.1f}% {v['top3_hit_rate']:6.2f} {v['mean_spearman_over_candidates']:6.3f}")
    print("\nPER STATE: mean measured TPR gain of the chosen repair (averaged over 58 companions)")
    cols = rules + ["nearest_to_damage", "damage_coverage", "random", "oracle"]
    print("state " + " ".join(f"{c[:10]:>10}" for c in cols))
    for s, v in per_state.items():
        print(f"{s:5} " + " ".join(f"{v[c]:10.4f}" for c in cols))


if __name__ == "__main__":
    main()
