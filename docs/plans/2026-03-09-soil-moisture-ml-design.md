# Soil Moisture ML Project — Design Document

**Date:** 2026-03-09
**Team:** Ronak Bhale, John Brooks, Elyzabeth Benitez Rivera, Max
**Course:** BSE 405
**Deadline:** March 18, 2026

## Project Overview

Spatial calibration of satellite soil moisture products using machine learning. Predict AMSR satellite soil moisture (`sm_tgt`) from SMOS-ASCAT satellite data (`sm_aux`) combined with soil composition and spatial features.

## Dataset

- **Source:** Kaggle — Soil Moisture Remote Sensing Data (Germany 2013)
- **File:** `updated_data.csv` (321,584 rows, 8 columns)
- **Grid:** 28 x 56 (0.25 deg resolution), daily measurements
- **Features:** time, latitude, longitude, clay_content, sand_content, silt_content, sm_aux
- **Target:** sm_tgt (soil moisture, m3/m3)

## Stack

- Google Colab
- Python, pandas, scikit-learn, matplotlib, seaborn

## Architecture: Parallel Core + Shared Data (Approach B)

Four notebooks with clear data handoff points:

### 01_EDA.ipynb

Explore data and produce presentation-ready figures.

Outputs:
- Dataset summary stats (.describe(), .info())
- Distribution plots for each feature (histograms)
- Correlation heatmap (all features vs sm_tgt)
- Spatial map of data points across Germany
- Temporal coverage plot (observations over time)
- Missing value analysis

### 02_feature_engineering.ipynb

Create features, define spatial splits, export clean dataset.

Engineered features:
- `day_of_year` — extracted from time (1-365)
- `season` — Winter/Spring/Summer/Fall
- `clay_sand_ratio` — clay_content / sand_content
- `clay_silt_ratio` — clay_content / silt_content
- `clay_x_sm_aux` — clay_content * sm_aux (interaction term)

Spatial split columns:
- `region` — "West" if longitude < ~10.5 deg, "East" otherwise
- `spatial_block` — ~6 blocks based on lat/lon ranges for block CV

Output: `processed_data.csv` with all original + engineered features + split labels.

### 03_modeling.ipynb

Train 4 models across 3 split strategies.

Models:
| Model | Why | Strengths | Weaknesses |
|---|---|---|---|
| Linear Regression | Baseline | Fast, interpretable | Assumes linearity |
| Random Forest | Non-linear, feature importance | Robust, no scaling needed | Can overfit, slower |
| SVM (RBF) | Complex boundaries | Handles non-linearity | Slow at 321K rows — use Nystroem approximation |
| MLPRegressor | Complex interactions | Flexible, learns non-linear mappings | Black box, needs tuning + scaling |

Split strategies:
1. Random 80/20 — standard baseline
2. West/East spatial split — train West, test East (primary spatial experiment)
3. Spatial block CV — ~6 blocks, k-fold holding out one block at a time (supplementary)

Metrics per experiment:
- RMSE (Root Mean Squared Error)
- R2 (Coefficient of Determination)
- Bias (mean error)
- Training vs testing performance (overfitting analysis)

Outputs:
- `model_results.csv` — all models x all splits x all metrics
- `predictions_spatial.csv` — per-point predictions with lat/lon for error mapping

### 04_analysis.ipynb

Produce final analysis and presentation-ready visualizations.

Outputs:
1. Model comparison bar chart — RMSE and R2 for all models across all splits
2. Overfitting analysis — train vs test R2 grouped bar chart
3. Spatial error maps — Germany grid colored by prediction error per split strategy
4. Feature importance plot — horizontal bar chart from Random Forest
5. Spatial CV comparison — performance degradation from random -> West/East -> block CV
6. Residual distribution — histogram of errors for best model

All figures saved as PNGs with labels, units, and legends.

## Research Questions

1. Can ML improve prediction of AMSR soil moisture (sm_tgt) using SMOS-ASCAT (sm_aux) and soil texture?
2. How well do models generalize across different geographic regions?
3. Which features most influence the prediction (soil composition vs satellite measurement)?
