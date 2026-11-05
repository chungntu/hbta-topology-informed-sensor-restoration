import sys, json, numpy as np
R = sys.argv[1]; sys.path.insert(0, R + "/src")
S = R + "/studies/topology_sensor_placement_hbta"
bank = np.load(S + "/outputs/virtual_damage_pareto/virtual_damage_sensitivity.npz")
Q, rel, edges = bank["sensitivity"].astype(float), bank["reliability"].astype(float), bank["edges"]
cfg = json.load(open(S + "/configs/virtual_damage_pareto.json"))
from hbta_pipeline.spatial_topology import build_hbta_spatial_graph
d = np.load(R + "/data/derived/hbta/edge_spectra_multicondition_10s.npz")
g = build_hbta_spatial_graph(d["sensor_names"].astype(str), d["sensor_positions"].astype(float))
N = len(d["sensor_names"]); M = len(edges)
B = np.zeros((M, N)); B[np.arange(M), edges[:, 0]] = 1; B[np.arange(M), edges[:, 1]] = -1
w = g.weights * np.sqrt(rel); w /= w.mean(); L = B.T @ (w[:, None] * B)
lam = np.linalg.eigvalsh(L); ref = np.median(lam[lam > 1e-8])
out = np.zeros((M, M))
for a in [ref*f for f in cfg["resolvent_alpha_factors"]]:
    F = np.linalg.inv(L + a*np.eye(N)); BF = B @ F; Fmd = BF @ B.T        # f_md
    nrm = np.sum((F @ B.T)**2, axis=0)                                     # ||F b_d||^2
    for rho in cfg["damage_weight_reductions"]:
        kap = (rho*w/(1 - rho*w*np.diag(Fmd)))**2 * nrm                    # kappa_d
        out += rel[:, None] * Fmd**2 * kap[None, :]
out /= out.sum(0, keepdims=True)
print("rho/alpha sets:", cfg["damage_weight_reductions"], cfg["resolvent_alpha_factors"])
print("max |closed form - stored q|:", np.abs(out - Q).max(), " (stored float32)")
