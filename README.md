# Spacecraft Anomaly Detection

An educational AI/ML project for detecting anomalous behavior in spacecraft telemetry using the NASA SMAP/MSL telemetry benchmarks.

> **Status:** Initial scaffold. No benchmark results are claimed yet.

## Goal
Build a reproducible pipeline that explores telemetry, trains a baseline anomaly detector, evaluates anomaly windows carefully, and later compares it with sequence models (e.g. LSTM).

## Roadmap
- [x] Project structure and baseline entry point
- [ ] Acquire NASA Telemanom-compatible data and document its license/source
- [ ] Exploratory data analysis and channel selection
- [ ] Train-only preprocessing and baseline evaluation
- [ ] Event-level scoring with documented window conventions
- [ ] Sequence model comparison
- [ ] Streamlit dashboard and reproducible report

## Quick start
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Prepare a CSV with numeric telemetry columns and optionally an `is_anomaly` column (0/1 labels). Then run:

```bash
python -m src.baseline --data data/processed/telemetry.csv --label-column is_anomaly
```

For unlabeled data, omit `--label-column`; the script reports flagged rows without supervised metrics.

## Methodology notes
- Split chronologically; do not randomly mix future samples into training.
- Fit all preprocessing on training data only.
- Isolation Forest is a **baseline**, not a validated spacecraft-grade detector.
- Point metrics can misrepresent event detection. Report event/window scoring protocol, threshold, and offsets.
- Tune on validation data, never on the held-out test anomalies.
- Do not commit proprietary or restricted telemetry. Check dataset terms before redistribution.

## Layout
- `src/baseline.py` — runnable Isolation Forest baseline
- `data/raw/` — downloaded source data (ignored by Git)
- `data/processed/` — prepared data (ignored by Git)
- `notebooks/` — exploration and experiments
- `tests/` — unit tests

## Dataset
Start with the NASA SMAP/MSL telemetry collection and the [Telemanom project](https://github.com/khundman/telemanom). Record the exact source, version, channel, split, and preprocessing steps in experiment notes.

## License
Project code licensing and third-party dataset terms need to be reviewed before public release.
