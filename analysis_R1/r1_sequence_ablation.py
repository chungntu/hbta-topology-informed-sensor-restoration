"""R1.9 / R2.7: Section 4 restoration sequences with the spring model, uniform springs and link count.

For the four outage cases, the full lower-tail sequence is recomputed with
  - the submitted spring-model weights Q,
  - uniform springs (topology only, same lower-tail rule),
  - link-count coverage (every active link counts equally, i.e. coverage = number of active links of a
    location's response support is replaced by simply counting reactivated links),
and the number of steps whose decision is an exact tie or a near tie (best and second-best lower-tail
gain within 5%) is reported, together with how often the tie-break (mean coverage, then index) decides.

Run from SHM-R1:  python analysis_R1/r1_sequence_ablation.py
Writes only to analysis_R1/outputs/sensitivity/sequence_ablation.json.
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
    cfg = json.loads((STUDY / "configs" / "virtual_damage_pareto.json").read_text(encoding="utf-8"))
    lock = json.loads((STUDY / "outputs" / "topology_coverage_restoration_10sensor" / "protocol.lock.json")
                      .read_text(encoding="utf-8"))
    d = np.load(ROOT / "data" / "derived" / "hbta" / "edge_spectra_multicondition_10s.npz")
    names = d["sensor_names"].astype(str)
    bank = np.load(STUDY / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz")
    Q0, edges = bank["sensitivity"].astype(float), bank["edges"]
    N, M = len(names), len(edges)
    B = np.zeros((M, N)); B[np.arange(M), edges[:, 0]] = 1; B[np.arange(M), edges[:, 1]] = -1
    Qu = response_weights(B, np.ones(M), np.ones(M), [[m] for m in range(M)],
                          cfg["damage_weight_reductions"], cfg["resolvent_alpha_factors"])
    idx = {n: i for i, n in enumerate(names)}

    def chi(active):
        z = np.zeros(N, bool); z[list(active)] = True
        return z[edges[:, 0]] & z[edges[:, 1]]

    def tail(cov):
        return np.sort(cov)[:13].mean()

    def run(outage, kind):
        act, out, seq, ties = set(range(N)) - set(outage), set(outage), [], {"exact": 0, "near": 0, "tie_break_used": 0}
        while out:
            base = chi(act)
            if kind == "link_count":
                gain = {i: float(np.sum(chi(act | {i}) & ~base)) for i in out}
                mean = gain
            else:
                Q = Q0 if kind == "spring_model" else Qu
                gain = {i: tail(Q.T @ chi(act | {i})) - tail(Q.T @ base) for i in out}
                mean = {i: (Q.T @ chi(act | {i})).mean() - (Q.T @ base).mean() for i in out}
            order = sorted(out, key=lambda i: (-round(gain[i], 12), -round(mean[i], 12), i))
            if len(order) > 1:
                g1, g2 = gain[order[0]], gain[order[1]]
                if abs(g1 - g2) <= 1e-12:
                    ties["exact"] += 1
                    ties["tie_break_used"] += 1
                elif g1 > 0 and (g1 - g2) / g1 < 0.05:
                    ties["near"] += 1
            seq.append(str(names[order[0]])); act.add(order[0]); out.discard(order[0])
        return seq, ties

    result = {}
    for case in lock["cases"]:
        outage = [idx[n] for n in case["initial_outage_sensor_names"]]
        entry = {}
        for kind in ("spring_model", "uniform_springs", "link_count"):
            seq, ties = run(outage, kind)
            entry[kind] = {"sequence": seq, "steps": len(seq), **ties}
        paper = [s["restored_sensor_name"] for s in case["restoration_steps"]]
        assert entry["spring_model"]["sequence"] == paper, case["case_name"]
        for kind in ("uniform_springs", "link_count"):
            s = entry[kind]["sequence"]
            entry[kind]["first3_set_overlap_with_model"] = len(set(s[:3]) & set(paper[:3]))
            entry[kind]["first5_set_overlap_with_model"] = len(set(s[:5]) & set(paper[:5]))
            entry[kind]["same_position_all_steps"] = int(sum(a == b for a, b in zip(s, paper)))
        result[case["case_name"]] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sequence_ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, e in result.items():
        print(name)
        for kind, v in e.items():
            extra = {k: v[k] for k in v if k.startswith(("first", "same"))}
            print(f"  {kind:16} {' '.join(v['sequence'][:5])} | exact ties {v['exact']} near ties {v['near']} of {v['steps']} {extra}")


if __name__ == "__main__":
    main()
