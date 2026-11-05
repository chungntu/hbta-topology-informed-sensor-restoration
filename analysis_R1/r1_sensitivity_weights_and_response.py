"""R2.1 and R1.7: sensitivity of the results to the spring weights and to the construction of R.

R2.1 (spring weights w_m and reliability r_m):
  variants of the response-weight bank Q
    - reliability removed (r_m = 1)
    - relation balancing removed (w_m = 1 / path length, not balanced across relation types)
    - random perturbation of w_m (log-normal, sigma = 0.2 and 0.4; 20 draws each)
  reported: median Spearman correlation between perturbed and original columns of Q,
            link-level rho for DS1-DS4 and DS8 (primary damaged sets),
            agreement of the first five lower-tail restoration choices in the four Section 4 cases.

R1.7 (measured response R): link-level rho for DS1-DS4 and DS8 when R is built from
    - each descriptor group alone (transfer magnitudes, coherence-weighted phase, coherence)
    - K = 4, 8 (submitted), 16 and all 74 frequency bins

Run from SHM-R1:  python analysis_R1/r1_sensitivity_weights_and_response.py
Writes only to analysis_R1/outputs/sensitivity/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "topology_sensor_placement_hbta"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(STUDY / "src"))
from make_r1_link_level_all_states import response_weights  # noqa: E402

OUT = ROOT / "analysis_R1" / "outputs" / "sensitivity"
STATES = ["DS1", "DS2", "DS3", "DS4", "DS8"]


def main() -> None:
    from hbta_pipeline.edge_spectral_topology_experiment import edge_spectral_features, rolling_recording_ids
    from hbta_pipeline.spatial_topology import build_hbta_spatial_graph
    from run_placement import robust_center_scale, top_k_mean

    cfg = json.loads((STUDY / "configs" / "virtual_damage_pareto.json").read_text(encoding="utf-8"))
    mapping = json.loads((STUDY / "configs" / "r1_damage_mapping.json").read_text(encoding="utf-8"))
    lock = json.loads((STUDY / "outputs" / "topology_coverage_restoration_10sensor" / "protocol.lock.json")
                      .read_text(encoding="utf-8"))
    with np.load((STUDY / cfg["prepared_spectra"]).resolve()) as d:
        coeff = d["fourier_coefficients"].astype(np.complex64)
        src = d["recording_ids"].astype(np.int16)
        rec_names = d["recording_names"].astype(str)
        names = d["sensor_names"].astype(str)
        pos = d["sensor_positions"].astype(np.float64)
    with np.load((STUDY / cfg["teacher_state"]).resolve()) as t:
        cc, cs = t["coord_center"].astype(np.float32), t["coord_scale"].astype(np.float32)
    bank = np.load(STUDY / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz")
    Q0, rel, edges = bank["sensitivity"].astype(float), bank["reliability"].astype(float), bank["edges"]
    g = build_hbta_spatial_graph(names, pos)
    N, M = len(names), len(edges)
    B = np.zeros((M, N)); B[np.arange(M), edges[:, 0]] = 1; B[np.arange(M), edges[:, 1]] = -1
    red, alph = cfg["damage_weight_reductions"], cfg["resolvent_alpha_factors"]
    key = {}
    for m, (u, v) in enumerate(edges):
        key[f"{names[u]}--{names[v]}"] = key[f"{names[v]}--{names[u]}"] = m
    idx = {n: i for i, n in enumerate(names)}
    sets = {s["state"]: [key[e] for e in s["springs"]] for s in mapping["states"] if s["state"] in STATES}

    # ---------- measured responses for several constructions of R ----------
    rid = rolling_recording_ids(src, int(cfg["rolling_blocks"]))
    fit = np.flatnonzero(np.isin(rid, cfg["fit_recording_ids"]))
    feats = edge_spectral_features(coeff, edges, src, int(cfg["rolling_blocks"]))
    std = ((feats - cc[None]) / cs[None]).astype(np.float32)
    del feats, coeff
    state_rows = {s: np.flatnonzero(rid == int(np.flatnonzero(np.char.find(rec_names, f"_{s}_") >= 0)[0]))
                  for s in STATES}

    def responses(descriptors, k):
        freq = np.mean(np.log1p(std[..., descriptors] ** 2), axis=3)
        raw = np.mean(freq, axis=2) if k is None else top_k_mean(freq, k)
        c, s = robust_center_scale(raw[fit], 1e-4)
        local = np.maximum((raw - c[None]) / s[None], 0.0)
        return {st: np.median(local[rows], axis=0) for st, rows in state_rows.items()}

    w0 = g.weights * np.sqrt(rel); w0 /= w0.mean()
    q0 = {s: response_weights(B, w0, rel, [D], red, alph)[:, 0] for s, D in sets.items()}
    variants_R = {"submitted (all five, K=8)": ([0, 1, 2, 3, 4], 8),
                  "transfer magnitudes only": ([0, 1], 8), "coherence-weighted phase only": ([2, 3], 8),
                  "coherence only": ([4], 8), "K=4": ([0, 1, 2, 3, 4], 4), "K=16": ([0, 1, 2, 3, 4], 16),
                  "all 74 bins": ([0, 1, 2, 3, 4], None)}
    r17 = {}
    R_sub = None
    for name, (desc, k) in variants_R.items():
        R = responses(desc, k)
        if R_sub is None:
            R_sub = R
        r17[name] = {s: float(spearmanr(q0[s], R[s]).statistic) for s in STATES}
        print("R1.7", name, {s: round(v, 2) for s, v in r17[name].items()}, flush=True)

    # ---------- spring-weight variants ----------
    lengths = g.structural_path_lengths.astype(float)
    def bank_for(w, r):
        w = w / w.mean()
        return response_weights(B, w, r, [[m] for m in range(M)], red, alph)

    def tail(cov, k=13):
        return np.sort(cov)[:k].mean()

    def lower_tail_sequence(Q, outage, steps=5):
        act, out, seq = set(range(N)) - set(outage), set(outage), []
        def chi(a):
            z = np.zeros(N, bool); z[list(a)] = True
            return z[edges[:, 0]] & z[edges[:, 1]]
        for _ in range(steps):
            base = chi(act)
            def score(i):
                c = chi(act | {i})
                return (-round(tail(Q.T @ c) - tail(Q.T @ base), 12),
                        -round((Q.T @ c).mean() - (Q.T @ base).mean(), 12), i)
            best = min(out, key=score)
            seq.append(best); act.add(best); out.discard(best)
        return seq

    cases = {c["case_name"]: [idx[n] for n in c["initial_outage_sensor_names"]] for c in lock["cases"]}
    seq0 = {name: lower_tail_sequence(Q0, out) for name, out in cases.items()}
    for name, c in zip(cases, lock["cases"]):
        assert [names[i] for i in seq0[name]] == [s["restored_sensor_name"] for s in c["restoration_steps"][:5]]

    def evaluate(Q, w, r):
        col = np.median([spearmanr(Q[:, j], Q0[:, j]).statistic for j in range(M)])
        rho = {s: float(spearmanr(response_weights(B, w / w.mean(), r, [D], red, alph)[:, 0], R_sub[s]).statistic)
               for s, D in sets.items()}
        same = {}
        for name, out in cases.items():
            seq = lower_tail_sequence(Q, out)
            same[name] = int(sum(a == b for a, b in zip(seq, seq0[name])))
        return {"median_column_spearman": float(col), "link_rho": rho, "first5_same_position": same}

    r21 = {}
    variants_W = {"reliability removed (r=1)": (g.weights.astype(float), np.ones(M)),
                  "relation balancing removed": ((1.0 / lengths) * np.sqrt(rel), rel)}
    for name, (w, r) in variants_W.items():
        r21[name] = evaluate(bank_for(w, r), w, r)
        print("R2.1", name, r21[name], flush=True)
    rng = np.random.default_rng(20261105)
    for sigma in (0.2, 0.4):
        draws = []
        for _ in range(20):
            w = w0 * np.exp(rng.normal(0.0, sigma, M))
            draws.append(evaluate(bank_for(w, rel), w, rel))
        r21[f"log-normal perturbation sigma={sigma} (20 draws)"] = {
            "median_column_spearman": float(np.median([d["median_column_spearman"] for d in draws])),
            "link_rho_median": {s: float(np.median([d["link_rho"][s] for d in draws])) for s in STATES},
            "link_rho_range": {s: [float(min(d["link_rho"][s] for d in draws)),
                                   float(max(d["link_rho"][s] for d in draws))] for s in STATES},
            "first5_same_position_mean": {n: float(np.mean([d["first5_same_position"][n] for d in draws]))
                                          for n in cases},
        }
        print("R2.1 sigma", sigma, r21[f"log-normal perturbation sigma={sigma} (20 draws)"], flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sensitivity_weights_and_response.json").write_text(
        json.dumps({"R1.7_link_rho_by_R_construction": r17, "R2.1_weight_variants": r21,
                    "submitted_link_rho": r17["submitted (all five, K=8)"],
                    "submitted_first5": {n: [str(names[i]) for i in s] for n, s in seq0.items()}}, indent=2),
        encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
