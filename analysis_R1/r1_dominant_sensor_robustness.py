"""R1.10 / R2.6: robustness of the Section 5.2 rank agreement to the dominant local sensors.

Uses the submitted Section 5.2 data (DS1, companion outages AG05 and AG07, 57 candidates each) and
reports, per companion:
  - Spearman rho with all candidates (as submitted)
  - Spearman rho after excluding AL14, AL15 and AL18
  - number of distinct measured values (ties) among the remaining candidates
  - top-k agreement: overlap of the k highest predicted and k highest measured candidates
  - mean measured gain of the k highest predicted candidates versus random choice

Run from SHM-R1:  python analysis_R1/r1_dominant_sensor_robustness.py
Writes only to analysis_R1/outputs/dominant_sensor_robustness/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = (ROOT / "studies" / "topology_sensor_placement_hbta" / "outputs" / "physical_consistency"
        / "restoration_consistency_ds1.csv")
OUT = ROOT / "analysis_R1" / "outputs" / "dominant_sensor_robustness"
DOMINANT = {"AL14", "AL15", "AL18"}


def main() -> None:
    rows = list(csv.DictReader(DATA.open(encoding="utf-8-sig", newline="")))
    result = {}
    for comp in sorted({r["companion_outage"] for r in rows}):
        sub = [r for r in rows if r["companion_outage"] == comp]
        names = np.array([r["restored_sensor"] for r in sub])
        pred = np.array([float(r["predicted_delta_c"]) for r in sub])
        meas = np.array([float(r["measured_delta_tpr"]) for r in sub])
        keep = ~np.isin(names, list(DOMINANT))
        entry = {
            "n_all": int(len(sub)),
            "rho_all": float(spearmanr(pred, meas).statistic),
            "n_without_dominant": int(keep.sum()),
            "rho_without_dominant": float(spearmanr(pred[keep], meas[keep]).statistic),
            "distinct_measured_values_without_dominant": int(len(np.unique(np.round(meas[keep], 6)))),
            "max_measured_gain_without_dominant": float(meas[keep].max()),
            "share_zero_gain_without_dominant": float(np.mean(np.isclose(meas[keep], 0.0))),
        }
        order_p, order_m = np.argsort(-pred, kind="stable"), np.argsort(-meas, kind="stable")
        for k in (1, 3, 5):
            entry[f"top{k}_overlap"] = int(len(set(order_p[:k]) & set(order_m[:k])))
            entry[f"top{k}_predicted_mean_gain"] = float(meas[order_p[:k]].mean())
        entry["random_mean_gain"] = float(meas.mean())
        entry["top3_predicted"] = names[order_p[:3]].tolist()
        entry["top3_measured"] = names[order_m[:3]].tolist()
        result[comp] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dominant_sensor_robustness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
