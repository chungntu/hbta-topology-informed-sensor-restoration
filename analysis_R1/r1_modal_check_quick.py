import sys, numpy as np
from scipy.signal import find_peaks
R = sys.argv[1]
sys.path.insert(0, R + "/src")
from hbta_pipeline.spatial_topology import build_hbta_spatial_graph
d = np.load(R + "/data/derived/hbta/edge_spectra_multicondition_10s.npz")
X = d["fourier_coefficients"]; f = d["frequencies_hz"]; st = d["states"]
names = d["sensor_names"].astype(str); pos = d["sensor_positions"].astype(float)
H = X[st == "UDS"].astype(np.complex128)          # blocks, sensors, seg, freq
G = np.einsum("bisf,bjsf->fij", H, H.conj()) / (H.shape[0]*H.shape[2])
U, S, _ = np.linalg.svd(G)
s1 = S[:, 0]
pk, _ = find_peaks(np.log(s1), prominence=0.3)
print("FDD peaks (Hz):", np.round(f[pk], 2))
def realify(v):
    v = v * np.exp(-1j*np.angle(v[np.argmax(np.abs(v))]))
    return v.real
modes = np.array([realify(U[k, :, 0]) for k in pk]).T
g = build_hbta_spatial_graph(names, pos)
N = len(names); B = np.zeros((len(g.edges), N))
for m, (u, v) in enumerate(g.edges): B[m, u] = 1; B[m, v] = -1
L = B.T @ np.diag(g.weights) @ B
lam, phi = np.linalg.eigh(L)
def mac(a, b): return (a@b)**2 / ((a@a)*(b@b))
# best-matching Laplacian eigvec for each FDD mode (among first 15)
print("mode_Hz  bestMAC  lap_idx   MPC-ish(realness)")
for j, k in enumerate(pk):
    c = U[k, :, 0]; realness = np.abs(np.sum(c**2))/np.sum(np.abs(c)**2)
    macs = [mac(modes[:, j], phi[:, i]) for i in range(15)]
    i = int(np.argmax(macs)); print(f"{f[k]:6.2f}  {macs[i]:.2f}   {i:2d}   {realness:.2f}  top3={np.round(sorted(macs)[-3:],2)}")
# subspace check: fraction of each FDD mode energy captured by first n Laplacian eigvecs
for n in (5, 10, 15, 20):
    P = phi[:, :n]; cap = [np.linalg.norm(P.T@modes[:, j])**2/np.linalg.norm(modes[:, j])**2 for j in range(len(pk))]
    print(f"energy captured by first {n} Laplacian eigvecs:", np.round(cap, 2))
# random baseline for subspace capture
rng = np.random.default_rng(0); Q, _ = np.linalg.qr(rng.standard_normal((N, 10)))
print("random 10-dim subspace capture:", np.round([np.linalg.norm(Q.T@modes[:, j])**2/np.linalg.norm(modes[:, j])**2 for j in range(len(pk))], 2))
print("\n--- controls (10-dim subspaces) ---")
def capture(P, n):
    Qb, _ = np.linalg.qr(P[:, :n]); return np.round([np.linalg.norm(Qb.T@modes[:, j])**2/np.linalg.norm(modes[:, j])**2 for j in range(len(pk))], 2)
# Euclidean kNN graph Laplacian (no structural knowledge)
from scipy.spatial.distance import cdist
D = cdist(pos, pos); k = 4; Wk = np.zeros((N, N))
for i in range(N):
    for j in np.argsort(D[i])[1:k+1]: Wk[i, j] = Wk[j, i] = 1/D[i, j]
Lk = np.diag(Wk.sum(1)) - Wk; _, phik = np.linalg.eigh(Lk)
print("Euclid kNN Laplacian       :", capture(phik, 10))
# polynomial basis in coordinates (degree<=2, 10 terms)
x, y, z = (pos - pos.mean(0)).T / np.ptp(pos, 0).max()
P = np.stack([np.ones(N), x, y, z, x*x, y*y, z*z, x*y, x*z, y*z], 1)
print("quadratic polynomial basis :", capture(P, 10))
# grounded Laplacian: springs to ground at nodes closest to both abutments
xa = pos[:, 0]; ends = np.where((xa <= xa.min()+0.5) | (xa >= xa.max()-0.5))[0]
print("grounded nodes:", names[ends])
Lg = L.copy(); Lg[ends, ends] += 10*np.median(lam[1:])
lg, phig = np.linalg.eigh(Lg)
print("support-grounded Laplacian :", capture(phig, 10))
for j, kk in enumerate(pk):
    macs = [mac(modes[:, j], phig[:, i]) for i in range(15)]; i = int(np.argmax(macs))
    print(f"  grounded: {f[kk]:6.2f} Hz best MAC {macs[i]:.2f} with eigvec {i}")
print("x-range", xa.min(), xa.max())
