# Topology-Informed Sensor Restoration after Sensor Loss in Structural Health Monitoring

Code, prepared data and results for the topology-informed sensor-restoration study on the
Hell Bridge Test Arena (HBTA). The sensor graph is interpreted as a network of equivalent
springs between the sensor locations; damage at a candidate location is a reduction of one
spring stiffness, and the exact change in flexibility gives damage-response weights for every
sensor pair. These weights define the damage-location coverage of a degraded sensor network and
a greedy lower-tail rule for prioritizing sensor repairs. The modeled weights are compared with
measured vibration responses for the eight controlled damage states of the benchmark.

The manuscript itself is not part of this repository.

## Repository structure

| Path | Content |
|---|---|
| `src/hbta_pipeline/` | HBTA sensor graph and detector-support Python package |
| `studies/topology_sensor_placement_hbta/` | Response weights, restoration sequences, physical-consistency analyses, figures and locked intermediate results |
| `analysis_R1/` | Additional analyses: all damage states, baselines, ablations and sensitivity checks |
| `data/derived/hbta/` | Prepared HBTA arrays (Git LFS); derived inputs, not raw measurements |
| `scripts/` | Table generation and verification against accepted results |
| `verification/` | Accepted numerical and figure baselines |

## Raw data

The original HBTA measurements are not included. Copy `config_local.example.json` to
`config_local.json` and set `raw_data_dir` to the directory that contains `data_100Hz.h5`.
The routine reproduction uses the prepared arrays in `data/derived/hbta/`, so the raw HDF5 file
is only needed to check its checksum.

## Environment

- Python 3.12 with the packages pinned in `requirements.txt`
- Git LFS for the prepared `.npz` arrays

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
git lfs pull
```

## Reproducing the results

From the repository root:

```powershell
.\reproduce_all.ps1   # response weights, restoration sequences, DS1 analysis, figures and tables
.\reproduce_R1.ps1    # additionally: link-level comparison for all damage states and the spring-model figure
```

Both scripts regenerate the outputs and verify them against the accepted baselines in
`verification/`. The additional analyses in `analysis_R1/` are run individually, for example

```powershell
$env:PYTHONPATH = "$PWD\src"
python analysis_R1\r1_sequence_ablation.py
```

and write only to `analysis_R1/outputs/`.
