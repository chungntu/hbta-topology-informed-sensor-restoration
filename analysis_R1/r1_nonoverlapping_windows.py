"""R2.4: TPR and Delta TPR of Section 5.2 recomputed on non-overlapping 60-s windows.

The submitted analysis uses 60-s windows advanced by 10 s, so neighbouring windows share five of six
10-s blocks.  Here every sixth window is kept (offsets 0-5), which gives six disjoint sets of
non-overlapping windows.  The detector, the score standardization and the threshold are unchanged
(threshold = 95th percentile on the full healthy calibration recording); only the windows used to
count detections change.  Reported: TPR(DS1) and Delta TPR for the Section 5.2 cases, as the value
on all windows and as mean [min, max] over the six non-overlapping sets, plus the false-alarm rate.

Run from SHM-R1:  python analysis_R1/r1_nonoverlapping_windows.py
Writes only to analysis_R1/outputs/nonoverlapping_windows/.
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
OUT = ROOT / "analysis_R1" / "outputs" / "nonoverlapping_windows"
CONFIGS = {"full": [], "AG05": ["AG05"], "AG05+AL14": ["AG05", "AL14"], "AG05+AL18": ["AG05", "AL18"],
           "AG07": ["AG07"], "AG07+AL14": ["AG07", "AL14"], "AG07+AL15": ["AG07", "AL15"],
           "AG07+AL18": ["AG07", "AL18"]}
GAINS = {"AG05: restore AL14": ("AG05", "AG05+AL14"), "AG05: restore AL18": ("AG05", "AG05+AL18"),
         "AG07: restore AL14": ("AG07", "AG07+AL14"), "AG07: restore AL15": ("AG07", "AG07+AL15"),
         "AG07: restore AL18": ("AG07", "AG07+AL18")}


def main() -> None:
    from hbta_pipeline.edge_spectral_topology_experiment import (
        calibration_normalized_sum, candidate_edge_kernel, edge_spectral_features, rolling_recording_ids)
    from hbta_pipeline.spatial_topology import build_hbta_spatial_graph
    from hbta_pipeline.topology_location_scan_experiment import (
        graph_distance_matrix, scan_score, standardized_location_scores)
    from run_placement import active_edge_indices, exact_score_bank

    cfg = json.loads((STUDY / "configs" / "placement.json").read_text(encoding="utf-8"))
    with np.load((STUDY / cfg["prepared_spectra"]).resolve()) as d:
        coeff = d["fourier_coefficients"].astype(np.complex64)
        src = d["recording_ids"].astype(np.int16)
        names = d["sensor_names"].astype(str)
        pos = d["sensor_positions"].astype(np.float64)
        rec_names = d["recording_names"].astype(str)
    with np.load((STUDY / cfg["teacher_state"]).resolve()) as t:
        cc, cs = t["coord_center"].astype(np.float32), t["coord_scale"].astype(np.float32)
    g = build_hbta_spatial_graph(names, pos)
    dist = graph_distance_matrix(g, len(names))
    rid = rolling_recording_ids(src, cfg["rolling_blocks"])
    fit = np.flatnonzero(np.isin(rid, cfg["fit_recording_ids"]))
    cal = np.flatnonzero(rid == cfg["calibration_recording_id"])
    healthy = np.flatnonzero(rid == cfg["healthy_test_recording_id"])
    ds1 = np.flatnonzero(rid == int(np.flatnonzero(np.char.find(rec_names, "_DS1_") >= 0)[0]))
    std = ((edge_spectral_features(coeff, g.edges, src, cfg["rolling_blocks"]) - cc[None]) / cs[None]).astype(np.float32)
    idx = {n: i for i, n in enumerate(names)}

    tpr_all, tpr_sets, fpr = {}, {}, {}
    for name, out in CONFIGS.items():
        active = active_edge_indices(g.edges, [i for i in range(len(names)) if names[i] not in out])
        scores, _ = exact_score_bank(std, g.edges, dist, active, fit, cal, cfg["top_frequency_count"],
                                     cfg["kernel_length_m"], cfg["topology_weight"], candidate_edge_kernel,
                                     standardized_location_scores, scan_score, calibration_normalized_sum)
        threshold = float(np.quantile(scores[cal], cfg["threshold_quantile"]))
        tpr_all[name] = float(np.mean(scores[ds1] > threshold))
        tpr_sets[name] = [float(np.mean(scores[ds1[k::6]] > threshold)) for k in range(6)]
        fpr[name] = float(np.mean(scores[healthy] > threshold))
        print(name, round(tpr_all[name], 3), np.round(tpr_sets[name], 3), "fpr", round(fpr[name], 3), flush=True)

    gains = {}
    for label, (before, after) in GAINS.items():
        per_set = [a - b for a, b in zip(tpr_sets[before], tpr_sets[after])]
        gains[label] = {"all_windows": tpr_all[before] - tpr_all[after],
                        "non_overlapping_mean": float(np.mean(per_set)),
                        "non_overlapping_range": [float(min(per_set)), float(max(per_set))]}
    result = {"ds1_windows_all": int(ds1.size), "ds1_windows_per_non_overlapping_set": [int(ds1[k::6].size) for k in range(6)],
              "tpr_all_windows": tpr_all, "tpr_non_overlapping_sets": tpr_sets, "false_alarm_rate": fpr,
              "delta_tpr": gains}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "nonoverlapping_windows.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(gains, indent=2))


if __name__ == "__main__":
    main()
