# NanoOracle

Machine learning prediction of physicochemical properties of lipid nanoparticles (LNPs).

## Context

LNPs are key delivery vectors for nucleic acids (mRNA, siRNA), used notably in mRNA vaccines. Their formulation requires optimizing 4 physicochemical properties measured experimentally, a process that is time-consuming and resource-intensive. NanoOracle provides predictive models trained on experimental data to accelerate this optimization.

## Predicted outputs

| Output | Unit | Expert threshold |
|--------|------|-----------------|
| Size (Z-avg) | nm | ± 5 nm |
| PDI | — | ± 0.05 |
| Zeta Potential | mV | ± 2 mV |
| Encapsulation Efficiency (EE%) | % | ± 5 % |

## Two complementary approaches

| | Lab model | Literature model |
|---|---|---|
| Data | ~150 formulations, 1 lab | ~224 formulations, 26 articles |
| Lipids | MC3, SM-102 | 15+ ionizable lipids |
| Split strategy | 80/20 random | Source-aware hybrid |
| Size accuracy | 65% | 30% |
| PDI accuracy | 71% | 61% |

The lab-specialized model consistently outperforms the generalist model, demonstrating that training on context-specific experimental data significantly improves predictive precision.

## Technical stack

- **XGBoost** — gradient boosting regressor
- **Optuna** — Bayesian hyperparameter optimization
- **RDKit** — molecular descriptors (MolWt, LogP, TPSA, etc.)
- **Scikit-learn** — cross-validation, metrics
- **Python 3.10+**

## Installation

```bash
git clone https://github.com/your_username/nanooracle.git
cd nanooracle
pip install -r requirements.txt
```

## Data

| File | Description |
|------|-------------|
| `NanoOracle_Dataset_public_research.xlsx` | Public dataset — 26 published articles, 224 formulations |
| `NanoOracle_Dataset_v3_verified_cleaned_v2_train_benjamin.xlsx` | Lab training dataset |
| `NanoOracle_Dataset_v3_verified_cleaned_v2_test_3.xlsx` | Lab test dataset |

## Notebooks — execution order

| Notebook | Description |
|----------|-------------|
| `01_analysis_split.ipynb` | Exploratory analysis, outlier detection, source-aware train/test split |
| `02_pipeline_labo.ipynb` | Lab model — training (XGBoost + KFold CV) and inference |
| `03_pipeline_literature.ipynb` | Literature model — training (XGBoost + Optuna + KFold CV) and inference |

## Configuration

At the top of each notebook, set the `WORK_DIR` variable to your local project root:

```python
WORK_DIR = r'path/to/nanooracle'
```

## Results summary

### Lab model (02_pipeline_labo.ipynb)
- Size: MAE = 4.58 nm, R² = 0.78, Accuracy = 65%
- EE%: MAE = 4.71%, R² = 0.33, Accuracy = 52%
- PDI: MAE = 0.037, R² = 0.01, Accuracy = 71%
- Zeta: MAE = 2.24 mV, R² = -0.10, Accuracy = 50%

### Literature model (03_pipeline_literature.ipynb)
- Size: MAE = 13.6 nm, R² = 0.63, Accuracy = 30%
- EE%: MAE = 7.1%, R² = 0.58, Accuracy = 49%
- PDI: MAE = 0.050, R² = 0.33, Accuracy = 61%
- Zeta: MAE = 3.7 mV, R² = 0.20, Accuracy = 38%

## Authors

Pierre — D4Gen 2026
