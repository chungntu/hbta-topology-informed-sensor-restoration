import sys, json, numpy as np
from scipy.sparse.csgraph import shortest_path
R = sys.argv[1]; sys.path.insert(0, R + "/src")
S = R + "/studies/topology_sensor_placement_hbta"
bank = np.load(S + "/outputs/virtual_damage_pareto/virtual_damage_sensitivity.npz")
rel, edges = bank["reliability"].astype(float), bank["edges"]
cfg = json.load(open(S + "/configs/virtual_damage_pareto.json"))
from hbta_pipeline.spatial_topology import build_hbta_spatial_graph
d = np.load(R + "/data/derived/hbta/edge_spectra_multicondition_10s.npz")
g = build_hbta_spatial_graph(d["sensor_names"].astype(str), d["sensor_positions"].astype(float))
N = 58; M = len(edges); B = np.zeros((M, N)); B[np.arange(M), edges[:,0]] = 1; B[np.arange(M), edges[:,1]] = -1
w = g.weights*np.sqrt(rel); w /= w.mean(); L = B.T @ (w[:,None]*B)
lam = np.linalg.eigvalsh(L); ref = np.median(lam[lam > 1e-8])
A = np.zeros((N,N)); A[edges[:,0], edges[:,1]] = A[edges[:,1], edges[:,0]] = 1
hop = shortest_path(A, unweighted=True)
# hop distance between links: min over endpoints
lh = np.minimum.reduce([hop[np.ix_(edges[:,i], edges[:,j])] for i in (0,1) for j in (0,1)])
Q = bank["sensitivity"].astype(float)
print("lambda_med =", round(ref,3), " mean link stiffness = 1")
for f in cfg["resolvent_alpha_factors"]:
    a = ref*f; F = np.linalg.inv(L + a*np.eye(N)); Fmd2 = (B @ F @ B.T)**2
    share = [np.mean([Fmd2[lh[:,dd]<=h, dd].sum()/Fmd2[:,dd].sum() for dd in range(M)]) for h in range(5)]
    print(f"alpha={f}*lam_med={a:.3f}: share of f_md^2 within <=h link-hops (h=0..4):", np.round(share,2))
share = [np.mean([Q[lh[:,dd]<=h, dd].sum() for dd in range(M)]) for h in range(5)]
print("final q_md (all alphas): share within <=h hops:", np.round(share,2))
from scipy.stats import spearmanr
print("self share q_dd: mean", round(np.mean(np.diag(Q)),3), " median", round(np.median(np.diag(Q)),3))
cols = []
for f in cfg["resolvent_alpha_factors"]:
    a = ref*f; F = np.linalg.inv(L + a*np.eye(N)); X = (B@F@B.T)**2*rel[:,None]; cols.append(X/X.sum(0))
for i,j in [(0,1),(0,2),(1,2)]:
    rr = [spearmanr(cols[i][:,dd], cols[j][:,dd])[0] for dd in range(M)]
    print(f"alpha {i} vs {j}: median within-column Spearman {np.median(rr):.3f}")
# off-self share by hop
offs = [np.mean([Q[(lh[:,dd]<=h) & (np.arange(M)!=dd), dd].sum() for dd in range(M)]) for h in range(3)]
print("share excluding the damaged link itself, within <=h hops:", np.round(offs,2))
