# Data Directory Rules

This tree is for local working data only. Keep raw PaySim files, manifests containing local paths,
sampled datasets, intermediate features, processed outputs, and external downloads out of Git.

- `raw/`: immutable downloaded source files
- `external/`: other approved source material
- `interim/`: deterministic intermediate data
- `processed/`: reproducible model-ready data

Phase 3 writes the unchanged PaySim CSV to `raw/` and generated Parquet plus manifest files to
`processed/`. All remain ignored by Git. Use the `fraud-ingest` commands in the project README for
acquisition, source validation, deterministic sampling, manifest validation, loading, and
reconciliation. Never place Kaggle credentials or personal/real customer data here.
