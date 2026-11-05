"""Revision R1, Section 5.2 (R2.5): restoration choice for DS1 compared with simple baselines.

Setting of the submitted Section 5.2: a companion sensor c and a candidate i are both out, and i
is restored.  Measured gain = TPR_DS1(all sensors but c) - TPR_DS1(all sensors but c and i), from
the stored detector results (outputs/expanded_criticality).  Every rule picks one candidate per
companion; candidates within 1% of the best score count as tied and the expected gain over the
tied candidates is used.

Rules: spring-model coverage recovery Delta c_DS1 (Eq. damage-specific-recovery), nearest to the
damage (fewest graph hops to AL14 or AL18), node degree, network-mean coverage gain, a fixed
sensitivity ranking (modeled response share of each sensor's links summed over all locations), a
modal-information ranking (Effective Independence for the vertical modes identified by FDD from
healthy data), random choice, and the best possible choice.  Results for companions AG05 and AG07 (the two cases of the submitted
Section 5.2) and the mean over all 58 companions.  Used for the response to Reviewer 2, Comment 2.5;
the figure is not used in the manuscript.

Outputs (new files only): outputs/R1_physical_consistency/restoration_baselines_ds1.json and
figures/fig_r1_restoration_baselines.{pdf,png}.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import shortest_path

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
CRIT = STUDY_ROOT / "outputs" / "expanded_criticality"
BANK = STUDY_ROOT / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz"
PREPARED = REPO_ROOT / "data" / "derived" / "hbta" / "edge_spectra_multicondition_10s.npz"
OUTPUT = STUDY_ROOT / "outputs" / "R1_physical_consistency"
TIE_TOLERANCE = 0.01
RULES = ["spring_model", "nearest_to_damage", "degree", "mean_coverage", "fixed_sensitivity", "modal_efi",
         "random", "best_possible"]
LABELS = {"spring_model": "Spring model\n($\\Delta c_{\\mathrm{DS1}}$)", "nearest_to_damage": "Nearest to\ndamage",
          "degree": "Node\ndegree", "mean_coverage": "Mean\ncoverage", "fixed_sensitivity": "Fixed\nsensitivity",
          "modal_efi": "Modal\n(EfI)", "random": "Random", "best_possible": "Best\npossible"}
COLORS = {"spring_model": "#E08B2C", "nearest_to_damage": "#2E6F9E", "degree": "#9AA6AF",
          "mean_coverage": "#7C8B96", "fixed_sensitivity": "#B5BFC6", "modal_efi": "#6B8E7F", "random": "#C9D1D7", "best_possible": "#255B78"}


def load_gain(names):
    index = {n: i for i, n in enumerate(names)}
    n = len(names)
    single = np.zeros(n)
    with (CRIT / "single_quarantine_state_metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            single[index[row["sensor"]]] = float(row["DS1_tpr"])
    pair = np.full((n, n), np.nan)
    for path in sorted(glob.glob(str(CRIT / "pairwise_parts" / "*.json"))):
        for row in json.loads(Path(path).read_text(encoding="utf-8"))["pairs"]:
            a, b = index[row["first_sensor_name"]], index[row["second_sensor_name"]]
            pair[a, b] = pair[b, a] = row["metrics_after_quarantine"]["per_state"]["MVS_P2_DS1_NM_Z_01"]["detection_rate"]
    gain = single[:, None] - pair
    np.fill_diagonal(gain, np.nan)
    return gain


def modal_efi_scores() -> np.ndarray:
    """Effective Independence of each sensor for the vertical modes identified from healthy data (FDD)."""
    from scipy.signal import find_peaks
    with np.load(PREPARED) as data:
        coeff = data["fourier_coefficients"]
        states = data["states"].astype(str)
    healthy = coeff[states == "UDS"].astype(np.complex128)
    csd = np.einsum("bisf,bjsf->fij", healthy, healthy.conj()) / (healthy.shape[0] * healthy.shape[2])
    u, s, _ = np.linalg.svd(csd)
    peaks, _ = find_peaks(np.log(s[:, 0]), prominence=0.3)
    modes = []
    for k in peaks:
        v = u[k, :, 0]
        v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
        modes.append(v.real)
    phi = np.array(modes).T
    return np.diag(phi @ np.linalg.pinv(phi.T @ phi) @ phi.T)


def main() -> None:
    with np.load(PREPARED) as data:
        names = data["sensor_names"].astype(str)
    with np.load(BANK) as bank:
        q_all = bank["sensitivity"].astype(np.float64)
        edges = bank["edges"].astype(int)
    n = len(names)
    index = {s: i for i, s in enumerate(names)}
    damaged = next(m for m, (u, v) in enumerate(edges) if {names[u], names[v]} == {"AL14", "AL18"})
    q = q_all[:, damaged]
    gain = load_gain(names)
    adjacency = np.zeros((n, n))
    adjacency[edges[:, 0], edges[:, 1]] = adjacency[edges[:, 1], edges[:, 0]] = 1
    hops = shortest_path(adjacency, unweighted=True)
    degree = adjacency.sum(axis=1)
    efi = modal_efi_scores()
    # fixed sensitivity ranking: modeled response share carried by the links of each sensor, summed over all
    # candidate damage locations (independent of the outage state and of the damage location)
    fixed = np.array([q_all[(edges == i).any(axis=1)].sum() for i in range(n)])

    def active(out):
        return ~np.isin(edges, list(out)).any(axis=1)

    def expected_gain(scores, c):
        candidates = np.array([i for i in range(n) if i != c])
        values = scores[candidates]
        # candidates within 1% of the best score are treated as tied (e.g. AL14 and AL18 have
        # Delta c = 0.6279 and 0.6275 and cannot be separated meaningfully by the model)
        tied = candidates[values >= values.max() - TIE_TOLERANCE * abs(values.max()) - 1e-12]
        return float(np.mean(gain[c, tied])), int(tied.size)

    per_companion = {r: np.zeros(n) for r in RULES}
    tie_counts = {"spring_model": [], "nearest_to_damage": []}
    for c in range(n):
        chi_c = active({c})
        mean_c = (q_all.T @ chi_c).mean()
        scores = {
            "spring_model": np.array([q @ (chi_c & ~active({c, i})) if i != c else -np.inf for i in range(n)]),
            "nearest_to_damage": -np.min(hops[:, [index["AL14"], index["AL18"]]], axis=1).astype(float),
            "degree": degree.astype(float),
            "mean_coverage": np.array([mean_c - (q_all.T @ active({c, i})).mean() if i != c else -np.inf
                                       for i in range(n)]),
            "fixed_sensitivity": fixed.astype(float),
            "modal_efi": efi.astype(float),
        }
        for rule, sc in scores.items():
            value, ties = expected_gain(sc, c)
            per_companion[rule][c] = value
            if rule in tie_counts:
                tie_counts[rule].append(ties)
        per_companion["random"][c] = float(np.nanmean(gain[c]))
        per_companion["best_possible"][c] = float(np.nanmax(gain[c]))

    result = {
        "state": "DS1",
        "setting": "companion c and candidate i out; i restored; expected gain over tied candidates",
        "mean_over_58_companions": {r: float(per_companion[r].mean()) for r in RULES},
        "AG05": {r: float(per_companion[r][index["AG05"]]) for r in RULES},
        "AG07": {r: float(per_companion[r][index["AG07"]]) for r in RULES},
        "mean_tied_candidates": {r: float(np.mean(v)) for r, v in tie_counts.items()},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "restoration_baselines_ds1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.7,
                         "savefig.dpi": 360, "pdf.fonttype": 42, "ps.fonttype": 42})
    groups = [("AG05 out", result["AG05"]), ("AG07 out", result["AG07"]),
              ("Mean over all 58\ncompanion outages", result["mean_over_58_companions"])]
    fig, ax = plt.subplots(figsize=(7.2, 2.7), constrained_layout=True)
    width = 0.105
    for k, rule in enumerate(RULES):
        xs = np.arange(len(groups)) + (k - (len(RULES) - 1) / 2) * width
        vals = [100 * g[rule] for _, g in groups]
        ax.bar(xs, vals, width=width * 0.92, color=COLORS[rule], label=LABELS[rule].replace("\n", " "),
               edgecolor="white", linewidth=0.4, zorder=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=5.6)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([g for g, _ in groups])
    ax.set_ylabel("Measured TPR gain of the\nchosen repair (percentage points)")
    ax.grid(axis="y", color="#E8EDF0", linewidth=0.5, zorder=0)
    ax.set_ylim(0, 38)
    ax.legend(ncol=8, fontsize=6.2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT / "figures" / f"fig_r1_restoration_baselines.{ext}")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
