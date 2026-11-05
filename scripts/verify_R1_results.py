"""Verify the scientific results of the revised manuscript (R1).

Checks, without comparing figure files byte by byte (PNG/PDF metadata differ between machines):
  1. prepared inputs and the response-weight bank are unchanged (accepted_results.json),
  2. the submitted DS1 numbers are reproduced (link level and restoration consistency),
  3. the new link-level results for DS1-DS8 match accepted_results_R1.json,
  4. optionally, the revised PDF exists and contains the paper title (--pdf).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "topology_sensor_placement_hbta"
DATA = ROOT / "data" / "derived" / "hbta"
BASE = json.loads((ROOT / "verification" / "accepted_results.json").read_text(encoding="utf-8"))
R1 = json.loads((ROOT / "verification" / "accepted_results_R1.json").read_text(encoding="utf-8"))
TOL = 1e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, what: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=TOL):
        raise AssertionError(f"{what}: {actual} != {expected}")


def verify() -> None:
    for name, expected in BASE["prepared_inputs"].items():
        if sha256_file(DATA / name).lower() != expected.lower():
            raise AssertionError(f"Prepared input hash mismatch: {name}")
    with np.load(STUDY / "outputs" / "virtual_damage_pareto" / "virtual_damage_sensitivity.npz") as bank:
        sensitivity = np.ascontiguousarray(bank["sensitivity"])
    if hashlib.sha256(sensitivity.tobytes()).hexdigest() != BASE["q_md"]["sensitivity_array_sha256"]:
        raise AssertionError("Response-weight bank changed")

    physical = json.loads((STUDY / "outputs" / "physical_consistency" / "physical_consistency_results.json")
                          .read_text(encoding="utf-8"))
    expected = BASE["physical_consistency"]
    close(physical["link_level"]["spearman_rho"], expected["link_spearman_rho"], "DS1 link rho (submitted)")
    summaries = {row["companion_outage"]: row for row in physical["restoration_consistency"]["all_candidate_rank_results"]}
    for companion in ("AG05", "AG07"):
        close(summaries[companion]["spearman_rho"], expected[f"{companion}_spearman_rho"], f"{companion} rho")

    rows = json.loads((STUDY / "outputs" / "R1_physical_consistency" / "link_level_all_states.json")
                      .read_text(encoding="utf-8"))["rows"]
    got = {r["state"] + ("" if r["variant"] == "primary" else "/" + r["variant"]): r for r in rows}
    if set(got) != set(R1["states"]):
        raise AssertionError("R1 link-level result set differs from the accepted set")
    for key, acc in R1["states"].items():
        close(got[key]["spearman_rho"], acc["spearman_rho"], f"{key} rho")
        if got[key]["location_rank_of_122"] != acc["location_rank_of_122"]:
            raise AssertionError(f"{key} location rank changed")
        if "spearman_ci95" in acc:
            for a, b in zip(got[key]["spearman_ci95"], acc["spearman_ci95"]):
                close(a, b, f"{key} CI")
    close(got["DS1"]["spearman_rho"], expected["link_spearman_rho"], "DS1 R1 vs submitted")
    print("R1_SCIENTIFIC_RESULTS_VERIFIED")


def verify_pdf(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 100_000:
        raise AssertionError(f"Revised PDF missing or too small: {path}")
    tool = shutil.which("pdftotext")
    if tool:
        out = Path(tempfile.mkdtemp()) / "paper.txt"
        subprocess.run([tool, str(path), str(out)], check=True)
        text = " ".join(out.read_text(encoding="utf-8", errors="replace").split())
        if BASE["paper_title"] not in text:
            raise AssertionError("Revised PDF title does not match")
    print(f"R1_PDF_VERIFIED={path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    verify()
    if args.pdf:
        verify_pdf(args.pdf.resolve())


if __name__ == "__main__":
    main()
