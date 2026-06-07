# NanoOracle — Streamlit web app

Predict Lipid Nanoparticle (LNP) physicochemistry (size, PDI, encapsulation, zeta potential) from a formulation, before bench experiments. Three sections in the UI:

- **My Data** — record formulations and bench measurements (manual entry or Excel import). Data is saved locally and uploaded to us.
- **Shared Models** — XGBoost models common to all researchers on the platform. A baseline is shipped pre-trained; as more partner-lab data flows in via My Data and inter-lab noise is isolated, these models become the most performant for everyone.
- **My Models** — private space where a researcher can train their own XGBoost on their own dataset, in local. The model never leaves their machine.

## Run with Docker (recommended)

```bash
docker build -t nanooracle .
docker run -p 80:80 nanooracle
```

Then open <http://localhost:80>.

The image embeds:

- The 4 pre-trained XGBoost models in `pkl/` (registered on first boot as `shared_v1`)
- The seed dataset `data/seed/example_lab.csv` (copied to `data/datasets/` on first boot)
- `libgomp1` (runtime dependency of XGBoost on Linux)

## Run locally with uv

```bash
uv sync
uv run streamlit run src/app.py
```

Requires Python 3.14 and `libomp` for XGBoost (macOS: `brew install libomp`; Linux: already covered by Docker image).

## Project layout

```
src/
  app.py                  # Streamlit entrypoint + top navbar + bootstrap
  lib/
    schema.py             # FEATURES, OUTPUTS, OUTPUT_TARGETS (green/red ranges)
    storage.py            # filesystem layout, JSON helpers
    datasets.py           # CRUD over data/datasets/*.csv
    registry.py           # list/load/save models on disk (joblib)
    training.py           # XGBRegressor per output, 5-fold CV metrics
    inference.py          # dispatch: sklearn-style or Benjamin payload
    benjamin.py           # bridge to 4 external XGBoost pkl payloads
  views/
    predict.py            # reusable: model picker + sliders + outputs
    train.py              # train a new private model
    data.py               # manage datasets in-browser + Excel I/O
pkl/                      # Benjamin's 4 pre-trained models (one per output)
data/
  seed/                   # seed CSVs copied into datasets/ on first run
  datasets/<id>.csv       # researcher's datasets (one CSV per dataset)
  models/
    shared/<id>/          # shared models (joblib per output + meta.json)
    mine/<id>/            # private models
scripts/
  generate_seed.py        # regenerate data/seed/example_lab.csv (dev-only)
Dockerfile, .dockerignore, pyproject.toml, uv.lock
```

## Regenerate the seed dataset

```bash
uv run python scripts/generate_seed.py
```

Writes 60 deterministic synthetic rows to `data/seed/example_lab.csv` (seeded with `RNG = default_rng(7)`).
