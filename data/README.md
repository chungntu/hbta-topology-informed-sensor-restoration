# HBTA source data

Raw benchmark data are intentionally kept outside this standalone code
workspace at:

`D:/BIG DATA/Hell Bridge Test Arena HBTA/`

The active studies use these external inputs when raw-data preparation or
audit is requested:

- `data_100Hz.h5` - original 100 Hz HBTA measurements (2,516,574,688 bytes)
- `Sensor layout.pdf` - sensor-coordinate/layout reference
- `Drawings.pdf` - structural drawings used to verify graph relations
- `svendsen-et-al-2022-a-hybrid-structural-health-monitoring-approach-for-damage-detection-in-steel-bridges-under.pdf`

The HDF5 file has the locked checksums:

- MD5: `3a3d941e96d50d1fd2d299790150a9a7`
- SHA-256: `467e9ecd87b367a225290f58e382657548ca81b235b08c70b7f1661174219650`

No raw file is copied into this project. Prepared arrays used by the studies
are kept under `data/derived/hbta/`.
