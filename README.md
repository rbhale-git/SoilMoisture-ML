# BSE 405 ML Project — Satellite Soil Moisture Calibration

**Team:** Ronak Bhale, John Brooks, Elyzabeth Benitez Rivera, Max Rader

## Project Overview

Predicting AMSR satellite soil moisture (`sm_tgt`) from SMOS-ASCAT satellite data (`sm_aux`) and soil composition features using machine learning. The dataset covers Germany in 2013 on a 28x56 grid (0.25 deg resolution) with ~321K observations.

## Research Questions

1. Can ML improve prediction of AMSR soil moisture using SMOS-ASCAT and soil texture?
2. How well do models generalize across different geographic regions?
3. Which features most influence the prediction?

## Setup

1. Create a folder on Google Drive: `BSE405_SoilMoisture/`
2. Inside it, create three subfolders: `data/`, `figures/`, `results/`
3. Upload `updated_data.csv` to `data/`
4. Upload all 4 notebooks from `notebooks/` into the root `BSE405_SoilMoisture/` folder
5. Open each notebook in Google Colab and run them **in order**

## Notebooks

Run these sequentially — each one depends on the output of the previous.

| # | Notebook | Owner | What it does | Outputs |
|---|----------|-------|-------------|---------|
| 1 | `01_EDA.ipynb` | Ronak Bhale | Explores the dataset — summary stats, distributions, correlation heatmap, spatial map, temporal coverage | Figures saved to `figures/` |
| 2 | `02_feature_engineering.ipynb` | John Brooks | Creates new features (day_of_year, season, soil ratios, interaction terms) and spatial split labels | `data/processed_data.csv` |
| 3 | `03_modeling.ipynb` | Elyzabeth Benitez Rivera | Trains 4 models across 3 split strategies, evaluates with RMSE/R2/bias | `results/model_results.csv`, `results/predictions_spatial.csv`, `results/feature_importance.csv` |
| 4 | `04_analysis.ipynb` | Max Rader | Produces final visualizations — model comparison, overfitting analysis, error maps, feature importance | Figures saved to `figures/` |

## Models

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline |
| Random Forest | Non-linear, provides feature importance |
| SVM (RBF via Nystroem approximation) | Kernel approximation used for scalability on 321K rows |
| MLP (Multi-Layer Perceptron) | Neural network with (64, 32) hidden layers |

## Split Strategies

| Split | Description |
|-------|-------------|
| Random 80/20 | Standard baseline split |
| West/East | Train on West Germany (longitude < 10.5), test on East — primary spatial experiment |
| Spatial Block CV | 6 blocks (2 lat bands x 3 lon bands), leave-one-block-out — supplementary analysis |

## Engineered Features

| Feature | Formula |
|---------|---------|
| `day_of_year` | Extracted from timestamp (1-365) |
| `season` | Winter/Spring/Summer/Fall from month |
| `clay_sand_ratio` | clay_content / sand_content |
| `clay_silt_ratio` | clay_content / silt_content |
| `clay_x_sm_aux` | clay_content * sm_aux |

## Evaluation Metrics

- **RMSE** — Root Mean Squared Error
- **R2** — Coefficient of Determination
- **Bias** — Mean prediction error
- **Train vs Test comparison** — Overfitting detection

## Running the Pipeline

### Option 1: Single Python Script (Recommended)

`run_all.py` consolidates all 4 notebooks into one executable that runs the full pipeline end-to-end: EDA, feature engineering, modeling, and analysis.

**Prerequisites:**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Python 3.8+ required. All dependencies are standard data science libraries.

**Usage:**

```bash
# Clone the repo
git clone https://github.com/rbhale-git/SoilMoisture-ML.git
cd SoilMoisture-ML

# Run with default paths (looks for data/updated_data.csv in repo root)
python run_all.py

# Or specify a custom data directory
python run_all.py --data-dir /path/to/your/data

# Or specify both data and output directories
python run_all.py --data-dir /path/to/data --output-dir /path/to/output

# Or run on Google Colab (mounts Google Drive automatically)
python run_all.py --colab
```

**What it does:**

| Phase | Description | Time |
|-------|-------------|------|
| Phase 1: EDA | Summary stats, distributions, correlation heatmap, spatial/temporal plots | ~10 sec |
| Phase 2: Feature Engineering | Creates temporal, ratio, interaction features + spatial split labels | ~5 sec |
| Phase 3: Modeling | Trains 4 models x 3 split strategies, computes all metrics | ~5-10 min |
| Phase 4: Analysis | Generates all presentation-ready visualizations | ~10 sec |

**Outputs:**

After running, you'll find:

```
figures/
├── feature_distributions.png
├── correlation_heatmap.png
├── spatial_map.png
├── temporal_coverage.png
├── aux_vs_tgt_scatter.png
├── model_comparison.png
├── overfitting_analysis.png
├── feature_importance.png
├── spatial_error_map.png
├── residual_distribution.png
└── spatial_degradation.png

results/
├── model_results.csv          # All models x splits x metrics
├── predictions_spatial.csv    # Per-point predictions for error mapping
└── feature_importance.csv     # Random Forest feature importances

data/
└── processed_data.csv         # Cleaned data with engineered features
```

**Running on Google Colab:**

Open a new Colab notebook and run:

```python
!git clone https://github.com/rbhale-git/SoilMoisture-ML.git
%cd SoilMoisture-ML
!python run_all.py
```

All figures and results will be saved inside the cloned repo directory.

### Option 2: Individual Notebooks

See the [Notebooks](#notebooks) section above. Run in order on Google Colab: 01 → 02 → 03 → 04.

For detailed Colab setup instructions, see `docs/colab-setup-instructions.md`.

## Project Structure

```
SoilMoisture-ML/
├── README.md
├── run_all.py                  # Single-script pipeline (recommended)
├── BSE 405 Project Proposal Idea.md
├── data/
│   └── updated_data.csv        # Raw dataset (321K rows)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_analysis.ipynb
├── docs/
│   ├── plans/
│   │   ├── 2026-03-09-soil-moisture-ml-design.md
│   │   └── 2026-03-09-soil-moisture-ml-plan.md
│   ├── colab-setup-instructions.md
│   └── presentation-prompt.md
├── figures/                    # Generated by pipeline
└── results/                    # Generated by pipeline
```

## Data Source

Kaggle: [Soil Moisture Remote Sensing Data (Germany 2013)](https://www.kaggle.com/datasets/sathyanarayanrao89/soil-moisture-remote-sensing-data-germany-2013/)
