"""R2.1 / R2.7: how stable are the Section 4 restoration choices under changes of the spring weights?

For the four outage cases of Section 4, compares the first five lower-tail choices obtained with the
submitted bank Q and with perturbed banks (log-normal noise on w_m, sigma 0.2 and 0.4, 20 draws each;
reliability removed; uniform springs).  Reported: same sensor at the same step, and overlap of the
unordered sets of the first five restored sensors.  Also reports how close the competing candidates
are at each step of the submitted sequence (relative gap between the best and second-best gain).

Run from SHM-R1:  python analysis_R1/r1_sequence_set_overlap.py
Writes only to analysis_R1/outputs/sensitivity/sequence_set_overlap.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "topology_sensor_placement_hbta"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(STUDY / "src"))
from make_r1_link_level_all_states import response_weights  # noqa: E402

OUT = ROOT / "analysis_R1" / "outputs" / "sensitivity"


def main() -> None:
    from hbta_pipeline.spatial_topology import build_hbta_spatial_graph
    cfg = json.loads((STUDY / "configs" / "virtual_damage_pareto.json").read_text(encoding="utf-8"))
    lock = json.loads((STUDY / "outputs" / "topology_coverage_restoration_10sensor" / "protocol.lock.json")
                      .read_text(encoding="utf-8"))
    d = np.load(ROOT / "data" / "derived" / "hbta" / "edge_spectra_multicondition_10s.npz")
    names, pos = d["sensor_names"].astype(str), d["sensor_positions"].astype(float)
    bank = np.load(STUDY / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz")
    Q0, rel, edges = bank["sensitivity"].astype(float), bank["reliability"].astype(float), bank["edges"]
    g = build_hbta_spatial_graph(names, pos)
    N, M = len(names), len(edges)
    B = np.zeros((M, N)); B[np.arange(M), edges[:, 0]] = 1; B[np.arange(M), edges[:, 1]] = -1
    red, alph = cfg["damage_weight_reductions"], cfg["resolvent_alpha_factors"]
    idx = {n: i for i, n in enumerate(names)}
    cases = {c["case_name"]: [idx[n] for n in c["initial_outage_sensor_names"]] for c in lock["cases"]}

    def chi(active):
        z = np.zeros(N, bool); z[list(active)] = True
        return z[edges[:, 0]] & z[edges[:, 1]]

    def tail(cov):
        return np.sort(cov)[:13].mean()

    def sequence(Q, outage, steps=5, gaps=None):
        act, out, seq = set(range(N)) - set(outage), set(outage), []
        for _ in range(steps):
            base = chi(act)
            gain = {i: tail(Q.T @ chi(act | {i})) - tail(Q.T @ base) for i in out}
            mean = {i: (Q.T @ chi(act | {i})).mean() - (Q.T @ base).mean() for i in out}
            best = min(out, key=lambda i: (-round(gain[i], 12), -round(mean[i], 12), i))
            if gaps is not None:
                vals = sorted(gain.values(), reverse=True)
                gaps.append(float((vals[0] - vals[1]) / vals[0]) if len(vals) > 1 and vals[0] > 0 else float("nan"))
            seq.append(best); act.add(best); out.discard(best)
        return seq

    gaps = {}
    seq0 = {}
    for name, out in cases.items():
        gaps[name] = []
        seq0[name] = sequence(Q0, out, gaps=gaps[name])

    def compare(Q):
        res = {}
        for name, out in cases.items():
            s = sequence(Q, out)
            res[name] = {"same_position": int(sum(a == b for a, b in zip(s, seq0[name]))),
                         "set_overlap": len(set(s) & set(seq0[name]))}
        return res

    w0 = g.weights * np.sqrt(rel); w0 /= w0.mean()
    def bank_for(w, r):
        return response_weights(B, w / w.mean(), r, [[m] for m in range(M)], red, alph)

    result = {"submitted_first5": {n: [str(names[i]) for i in s] for n, s in seq0.items()},
              "relative_gap_best_vs_second_submitted": gaps}
    result["reliability removed"] = compare(bank_for(g.weights.astype(float), np.ones(M)))
    result["uniform springs"] = compare(bank_for(np.ones(M), np.ones(M)))
    rng = np.random.default_rng(20261105)
    for sigma in (0.2, 0.4):
        draws = [compare(bank_for(w0 * np.exp(rng.normal(0, sigma, M)), rel)) for _ in range(20)]
        result[f"lognormal sigma={sigma}"] = {
            n: {"same_position_mean": float(np.mean([dd[n]["same_position"] for dd in draws])),
                "set_overlap_mean": float(np.mean([dd[n]["set_overlap"] for dd in draws]))} for n in cases}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sequence_set_overlap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
