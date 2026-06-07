# d4gen — NanoOracle

LNP physicochemistry predictor built for the D4GEN hackathon. The repo splits the work between the ML experimentation phase and the mvp web app.

## Folders

### `livrables-ml/`

The ML deliverables: notebooks that explore datasets, train the XGBoost models per output (size, PDI, encapsulation, zeta), and export them as pickle payloads. Contents:

- `ml_pipeline_ben_dataset.ipynb` — pipeline on Benjamin's curated dataset
- `ml_pipeline_public_research_dataset.ipynb` — pipeline on the public-research dataset
- `nanooracle_analysis.ipynb` — exploratory analysis
- `NanoOracle_Dataset_*.xlsx` — raw and cleaned datasets
- `nanooracle/` — packaged outputs (data/, models/, notebooks/, outputs/)

The 4 `.pkl` payloads produced here are consumed by the web app under `web-mpv/pkl/`.

### `web-mpv/`

The Streamlit web app the user interacts with. Loads the pre-trained models from `pkl/`, lets researchers manage datasets, run inference, and train private models on their own data.

See [`web-mpv/README.md`](web-mpv/README.md) for build / run instructions.
