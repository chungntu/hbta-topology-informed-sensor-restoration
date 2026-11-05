"""R2.5 / R2.6 / R1.8 / R1.9 / R2.7: sequential restoration after large outages, measured with the detector.

Scenarios: the four outage cases of Section 4 (9-10 sensors out) and random 10-sensor outages.
At each step every rule restores one sensor, using only the layout and the current outage state;
the same detector and the same calibrated threshold (configs/placement.json) then give the TPR of
every damage state for the restored network.

Rules: lower_tail (paper, gamma 0.10), mean_coverage, lower_tail_topology (uniform springs),
link_count, degree, random (average of seeds).

Run from SHM-R1:  python analysis_R1/r1_restoration_large_outages.py [n_random_scenarios] [steps] [outage_size]
Writes only to analysis_R1/outputs/restoration_large_outages/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "topology_sensor_placement_hbta"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(STUDY / "src"))
sys.path.insert(0, str(ROOT / "analysis_R1"))
from r1_multistate_link_level import response_weights  # noqa: E402

OUT = ROOT / "analysis_R1" / "outputs" / "restoration_large_outages"
STATES = [f"DS{k}" for k in range(1, 9)]
DETECTABLE = ["DS1", "DS2", "DS3", "DS4", "DS7", "DS8"]
RULES = ["lower_tail", "mean_coverage", "lower_tail_topology", "link_count", "degree"]
RANDOM_SEEDS = [1, 2, 3, 4, 5]


class Detector:
    """The placement-study detector evaluated for any active sensor set (cached)."""

    def __init__(self):
        from hbta_pipeline.edge_spectral_topology_experiment import (
            calibration_normalized_sum, candidate_edge_kernel, edge_spectral_features,
            rolling_recording_ids)
        from hbta_pipeline.spatial_topology import build_hbta_spatial_graph
        from hbta_pipeline.topology_location_scan_experiment import (
            graph_distance_matrix, scan_score, standardized_location_scores)
        from hbta_pipeline.vqpiae_experiment import metric_result
        from run_placement import active_edge_indices, exact_score_bank
        cfg = json.loads((STUDY / "configs" / "placement.json").read_text(encoding="utf-8"))
        self.cfg = cfg
        with np.load((STUDY / cfg["prepared_spectra"]).resolve()) as d:
            coeff = d["fourier_coefficients"].astype(np.complex64)
            src = d["recording_ids"].astype(np.int16)
            self.rec_names = d["recording_names"].astype(str)
            self.names = d["sensor_names"].astype(str)
            self.pos = d["sensor_positions"].astype(np.float64)
        with np.load((STUDY / cfg["teacher_state"]).resolve()) as t:
            cc, cs = t["coord_center"].astype(np.float32), t["coord_scale"].astype(np.float32)
        self.graph = build_hbta_spatial_graph(self.names, self.pos)
        self.dist = graph_distance_matrix(self.graph, len(self.names))
        self.rid = rolling_recording_ids(src, cfg["rolling_blocks"])
        self.fit = np.flatnonzero(np.isin(self.rid, cfg["fit_recording_ids"]))
        self.cal = np.flatnonzero(self.rid == cfg["calibration_recording_id"])
        self.healthy = np.flatnonzero(self.rid == cfg["healthy_test_recording_id"])
        feats = edge_spectral_features(coeff, self.graph.edges, src, cfg["rolling_blocks"])
        self.std = ((feats - cc[None]) / cs[None]).astype(np.float32)
        self.f = dict(calibration_normalized_sum=calibration_normalized_sum,
                      candidate_edge_kernel=candidate_edge_kernel, scan_score=scan_score,
                      standardized_location_scores=standardized_location_scores,
                      metric_result=metric_result, active_edge_indices=active_edge_indices,
                      exact_score_bank=exact_score_bank)
        self.cache: dict[frozenset, dict] = {}

    def tpr(self, active_sensors) -> dict:
        key = frozenset(int(s) for s in active_sensors)
        if key in self.cache:
            return self.cache[key]
        f, cfg = self.f, self.cfg
        active = f["active_edge_indices"](self.graph.edges, sorted(key))
        scores, _ = f["exact_score_bank"](
            self.std, self.graph.edges, self.dist, active, self.fit, self.cal,
            cfg["top_frequency_count"], cfg["kernel_length_m"], cfg["topology_weight"],
            f["candidate_edge_kernel"], f["standardized_location_scores"], f["scan_score"],
            f["calibration_normalized_sum"])
        m = f["metric_result"](scores, self.rid, self.rec_names, self.cal,
                               external_healthy_id=cfg["healthy_test_recording_id"],
                               damage_start_id=cfg["damage_start_recording_id"],
                               external_healthy_indices=self.healthy)
        out = {s: float(m["by_recording"][f"MVS_P2_{s}_NM_Z_01"]["detection_rate"]) for s in STATES}
        out["fpr"] = float(m["external_healthy_false_alarm_rate"])
        self.cache[key] = out
        return out


def main() -> None:
    n_random = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    random_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    outage_size = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    include_paper = outage_size == 10
    t0 = time.perf_counter()
    det = Detector()
    names, edges = det.names, det.graph.edges
    N, M = len(names), len(edges)
    idx = {n: i for i, n in enumerate(names)}

    # reproduce stored results before anything else
    nominal = det.tpr(range(N))
    assert abs(nominal["DS1"] - 0.9397590361445783) < 1e-12, nominal
    pair = det.tpr([i for i in range(N) if names[i] not in ("AG05", "AL14")])
    assert abs(pair["DS1"] - (0.9397590361445783 - 0.32530120481927716)) < 1e-9, pair
    print(f"detector reproduces stored results; one evaluation ~{(time.perf_counter()-t0)/2:.1f}s incl. setup",
          flush=True)

    vcfg = json.loads((STUDY / "configs" / "virtual_damage_pareto.json").read_text(encoding="utf-8"))
    bank = np.load(STUDY / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz")
    Q = bank["sensitivity"].astype(float)
    B = np.zeros((M, N)); B[np.arange(M), edges[:, 0]] = 1; B[np.arange(M), edges[:, 1]] = -1
    Q_topo = response_weights(B, np.ones(M), np.ones(M), [[m] for m in range(M)],
                              vcfg["damage_weight_reductions"], vcfg["resolvent_alpha_factors"])
    k_tail = int(np.ceil(0.10 * M))
    degree = np.abs(B).sum(axis=0)

    def chi(active):
        a = np.zeros(N, bool); a[list(active)] = True
        return a[edges[:, 0]] & a[edges[:, 1]]

    def tail(cov):
        return np.sort(cov)[:k_tail].mean()

    def choose(rule, active, out, rng=None):
        out = sorted(out)
        if rule == "random":
            return int(rng.choice(out))
        base = chi(active)
        def gains(i):
            c = chi(active | {i})
            return {"lower_tail": tail(Q.T @ c) - tail(Q.T @ base),
                    "mean_coverage": (Q.T @ c).mean() - (Q.T @ base).mean(),
                    "lower_tail_topology": tail(Q_topo.T @ c) - tail(Q_topo.T @ base),
                    "link_count": float(np.sum(c & ~base)),
                    "degree": float(degree[i])}
        g = {i: gains(i) for i in out}
        return min(out, key=lambda i: (-round(g[i][rule], 12), -round(g[i]["mean_coverage"], 12), i))

    def run(initial_out, steps):
        out0 = set(initial_out)
        act0 = set(range(N)) - out0
        curves = {}
        for rule in RULES + [f"random_{s}" for s in RANDOM_SEEDS]:
            rng = np.random.default_rng(int(rule.split("_")[1])) if rule.startswith("random") else None
            act, out = set(act0), set(out0)
            seq, curve = [], [det.tpr(act)]
            for _ in range(min(steps, len(out0))):
                i = choose("random" if rng is not None else rule, act, out, rng)
                act.add(i); out.discard(i); seq.append(str(names[i]))
                curve.append(det.tpr(act))
            curves[rule] = {"sequence": seq, "tpr": curve}
        return curves

    def summarize(curves):
        res = {}
        for rule, c in curves.items():
            steps = c["tpr"][1:]
            res[rule] = {
                "mean_tpr_over_steps": float(np.mean([np.mean([t[s] for s in DETECTABLE]) for t in steps])),
                "worst_state_tpr_over_steps": float(np.mean([np.min([t[s] for s in DETECTABLE]) for t in steps])),
                "first_step_mean_tpr": float(np.mean([steps[0][s] for s in DETECTABLE])),
            }
        rnd = [res.pop(f"random_{s}") for s in RANDOM_SEEDS]
        res["random"] = {k: float(np.mean([r[k] for r in rnd])) for k in rnd[0]}
        return res

    lock = json.loads((STUDY / "outputs" / "topology_coverage_restoration_10sensor" / "protocol.lock.json")
                      .read_text(encoding="utf-8"))
    results = {"paper_cases": {}, "random_cases": {}}
    for case in (lock["cases"] if include_paper else []):
        out = [idx[n] for n in case["initial_outage_sensor_names"]]
        curves = run(out, len(out))
        paper_seq = [s["restored_sensor_name"] for s in case["restoration_steps"]]
        assert curves["lower_tail"]["sequence"] == paper_seq, (case["case_name"], curves["lower_tail"]["sequence"])
        results["paper_cases"][case["case_name"]] = {"initial_tpr": curves["lower_tail"]["tpr"][0],
                                                     "curves": curves, "summary": summarize(curves)}
        print(f"{case['case_name']} done ({len(det.cache)} evaluations, {time.perf_counter()-t0:.0f}s)", flush=True)
    rng = np.random.default_rng(20261105 + (0 if outage_size == 10 else outage_size))
    for r in range(n_random):
        out = rng.choice(N, outage_size, replace=False).tolist()
        curves = run(out, random_steps)
        results["random_cases"][f"random{outage_size}_{r}"] = {"outage": [str(names[i]) for i in out],
                                                    "initial_tpr": curves["lower_tail"]["tpr"][0],
                                                    "curves": curves, "summary": summarize(curves)}
        print(f"random scenario {r} done ({len(det.cache)} evaluations, {time.perf_counter()-t0:.0f}s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("restoration_large_outages.json" if outage_size == 10 else f"restoration_outage{outage_size}.json")).write_text(json.dumps(results, indent=2), encoding="utf-8")
    cols = RULES + ["random"]
    for group in ("paper_cases", "random_cases"):
        if not results[group]:
            continue
        print(f"\n{group}: mean TPR over restoration steps (detectable states)")
        print(f"{'case':24} " + " ".join(f"{c[:10]:>10}" for c in cols))
        agg = {c: [] for c in cols}
        for name, v in results[group].items():
            print(f"{name[:24]:24} " + " ".join(f"{v['summary'][c]['mean_tpr_over_steps']:10.3f}" for c in cols))
            for c in cols:
                agg[c].append(v["summary"][c]["mean_tpr_over_steps"])
        print(f"{'MEAN':24} " + " ".join(f"{np.mean(agg[c]):10.3f}" for c in cols))


if __name__ == "__main__":
    main()
