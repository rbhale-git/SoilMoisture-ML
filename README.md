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

| # | Notebook | What it does | Outputs |
|---|----------|-------------|---------|
| 1 | `01_EDA.ipynb` | Explores the dataset — summary stats, distributions, correlation heatmap, spatial map, temporal coverage | Figures saved to `figures/` |
| 2 | `02_feature_engineering.ipynb` | Creates new features (day_of_year, season, soil ratios, interaction terms) and spatial split labels | `data/processed_data.csv` |
| 3 | `03_modeling.ipynb` | Trains 4 models across 3 split strategies, evaluates with RMSE/R2/bias | `results/model_results.csv`, `results/predictions_spatial.csv`, `results/feature_importance.csv` |
| 4 | `04_analysis.ipynb` | Produces final visualizations — model comparison, overfitting analysis, error maps, feature importance | Figures saved to `figures/` |

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

## Project Structure

```
BSE 405 ML Project/
├── README.md
├── BSE 405 Project Proposal Idea.md
├── docs/plans/
│   ├── 2026-03-09-soil-moisture-ml-design.md
│   └── 2026-03-09-soil-moisture-ml-plan.md
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_analysis.ipynb
└── ML project datasets/
    └── .../7_SoilMoisturePrediction_.../data/updated_data.csv
```

## Data Source

Kaggle: Soil Moisture Remote Sensing Data (Germany 2013)
