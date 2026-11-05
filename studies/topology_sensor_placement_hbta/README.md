# HBTA topology-informed sensor restoration

This directory contains the analysis used by the final paper. The retained
pipeline constructs the 122-link HBTA sensor-projected graph, regenerates the
normalized response-weight bank `q_md`, locks restoration sequences for four
bounded sensor-loss states, and performs the limited DS1 physical-consistency
check.

Run the complete workflow from the repository root:

```powershell
.\reproduce_all.ps1
```

The master script calls the active programs in `src/` and writes results under:

- `outputs/virtual_damage_pareto/`: response-weight bank and metadata;
- `outputs/topology_coverage_restoration_10sensor/`: locked priorities and the
  two restoration figures;
- `outputs/topology_coverage_restoration/`: HBTA context and coverage-concept
  figures;
- `outputs/physical_consistency/`: DS1 response analysis and figure.

The original 2.5 GB HDF5 file remains outside the repository. Its location is
set only in the root `config_local.json`; prepared arrays required by the final
analysis are retained in `../../data/derived/hbta/`.
