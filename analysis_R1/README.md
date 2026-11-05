# analysis_R1: additional analyses

Scripts are named `r1_<topic>.py` and write only to `analysis_R1/outputs/<topic>/`.
They import the pipeline from `../src` and `../studies/topology_sensor_placement_hbta/src`
and never modify the baseline outputs of the main study.

Run from the repository root with `PYTHONPATH=src`.
