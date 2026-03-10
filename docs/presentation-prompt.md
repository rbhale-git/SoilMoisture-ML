# Prompt: Create BSE 405 ML Project Proposal Presentation

## Context

You are helping create a **proposal presentation** for a BSE 405 Machine Learning course project at UW-Madison. The presentation is due **March 18, 2026** and must not exceed **12 minutes**. It will be followed by a Q&A session.

The project directory is: `C:\Users\ronak\Spring 2026\BSE 405 ML Project`
GitHub repo: https://github.com/rbhale-git/SoilMoisture-ML

## Team

- Ronak Bhale
- John Brooks
- Elyzabeth Benitez Rivera
- Max Rader

This is a team of 4 undergraduates. Each team member must present their own section.

## Project Summary

**Title:** Spatial Calibration of Satellite Soil Moisture Products Using Machine Learning

**Problem:** Two satellites measure soil moisture over Germany — SMOS-ASCAT (`sm_aux`) and AMSR (`sm_tgt`). The goal is to use machine learning to predict AMSR soil moisture from SMOS-ASCAT readings combined with soil composition and spatial features. This is a satellite calibration / data fusion problem.

**Dataset:** Kaggle — "Soil Moisture Remote Sensing Data (Germany 2013)"
- 321,584 rows, 8 columns
- 28 x 56 grid at 0.25 deg resolution, daily measurements across Germany in 2013
- Features: time, latitude, longitude, clay_content, sand_content, silt_content, sm_aux
- Target: sm_tgt (soil moisture in m3/m3)
- No missing values
- Mean sm_aux: 0.194, Mean sm_tgt: 0.412 (AMSR reads higher than SMOS-ASCAT)
- 1,166 unique grid points, 363 unique dates

**Engineered Features:**
- day_of_year (1-365 from timestamp)
- season (Winter/Spring/Summer/Fall)
- clay_sand_ratio (clay_content / sand_content)
- clay_silt_ratio (clay_content / silt_content)
- clay_x_sm_aux (clay_content x sm_aux interaction term)

**Models (4 total):**
| Model | Why chosen | Strengths | Weaknesses |
|-------|-----------|-----------|------------|
| Linear Regression | Baseline, interpretable | Fast, simple, good benchmark | Assumes linear relationships |
| Random Forest | Handles non-linear patterns, built-in feature importance | Robust, no scaling needed | Can overfit, slower to train |
| SVM (RBF via Nystroem approximation) | Complex decision boundaries | Handles non-linearity well | Slow on large data — Nystroem approximation used for scalability |
| MLP (Multi-Layer Perceptron) | Captures complex interactions | Flexible, learns non-linear mappings | Black box, needs tuning and scaling |

**Experimental Setup:**
- Environment: Google Colab, Python, scikit-learn, pandas, matplotlib, seaborn
- 3 split strategies:
  1. Random 80/20 split — standard baseline (Train: 257,267 / Test: 64,317)
  2. West/East spatial split — train on West Germany (longitude < 10.5 deg), test on East Germany (Train: 180,583 / Test: 141,001). Primary spatial generalization experiment.
  3. Spatial block cross-validation — 6 blocks (2 latitude bands x 3 longitude bands), leave-one-block-out. Supplementary analysis.

**Evaluation Metrics:**
- RMSE (Root Mean Squared Error)
- R2 (Coefficient of Determination)
- Bias (mean prediction error)
- Train vs test performance comparison (overfitting detection)

**Research Questions:**
1. Can ML improve prediction of AMSR soil moisture using SMOS-ASCAT and soil texture?
2. How well do models generalize across different geographic regions?
3. Which features most influence the prediction (soil composition vs satellite measurement)?

## Actual Results (from running the pipeline)

### Model Results Table

| Model | Split | Train RMSE | Test RMSE | Train R2 | Test R2 | Test Bias |
|-------|-------|-----------|----------|---------|--------|----------|
| Linear Regression | Random 80/20 | 0.1054 | 0.1054 | 0.091 | 0.091 | -0.0004 |
| Random Forest | Random 80/20 | 0.0206 | 0.0549 | 0.965 | 0.753 | -0.0004 |
| SVM (RBF approx) | Random 80/20 | 0.1004 | 0.1004 | 0.175 | 0.176 | -0.0004 |
| MLP | Random 80/20 | 0.0785 | 0.0792 | 0.496 | 0.487 | 0.006 |
| Linear Regression | West->East | 0.0925 | 0.1249 | 0.140 | -0.032 | -0.015 |
| Random Forest | West->East | 0.0191 | 0.1424 | 0.963 | -0.343 | -0.024 |
| SVM (RBF approx) | West->East | 0.0904 | 0.1280 | 0.179 | -0.085 | -0.020 |
| MLP | West->East | 0.0706 | 0.2250 | 0.499 | -2.350 | 0.021 |
| Linear Regression | Spatial Block CV | 0.1048 | 0.1062 | 0.098 | -0.093 | -0.002 |
| Random Forest | Spatial Block CV | 0.0201 | 0.0907 | 0.967 | 0.200 | 0.006 |
| SVM (RBF approx) | Spatial Block CV | 0.0997 | 0.1059 | 0.184 | -0.092 | -0.004 |
| MLP | Spatial Block CV | 0.0775 | 0.0996 | 0.506 | -0.034 | 0.014 |

### Key Findings

1. **Random Forest dominates on random split** (R2 = 0.753) but fails completely on spatial splits (R2 = -0.343 West->East). It memorizes location-specific patterns.
2. **All models fail on spatial generalization** — every model has negative R2 on West->East, meaning they predict worse than guessing the mean.
3. **Random Forest heavily overfits** — Train R2 = 0.965 vs Test R2 = 0.753 on random split.
4. **MLP collapses on West->East** (R2 = -2.35) — worst spatial performance.
5. **Linear Regression is the "best" spatial model** — least negative R2 on West->East (-0.032). Simpler models generalize better.
6. **Spatial Block CV confirms the pattern** — RF partially recovers (R2 = 0.20) since block CV averages across regions.

### Feature Importance (Random Forest)

| Feature | Importance |
|---------|-----------|
| day_of_year | 0.249 |
| longitude | 0.208 |
| latitude | 0.183 |
| sm_aux | 0.119 |
| clay_x_sm_aux | 0.075 |
| clay_sand_ratio | 0.057 |
| clay_silt_ratio | 0.035 |
| sand_content | 0.026 |
| silt_content | 0.025 |
| clay_content | 0.019 |
| season_Summer | 0.002 |
| season_Spring | 0.001 |
| season_Winter | 0.001 |

**Key insight:** Spatial/temporal features (day_of_year 25%, longitude 21%, latitude 18%) dominate over soil physics and satellite data. The RF is mostly learning "where and when" patterns, not transferable soil moisture relationships. sm_aux has only 12% importance despite being the core satellite measurement.

## Figures Available

The following figures have been generated and are available at `C:\Users\ronak\Spring 2026\BSE 405 ML Project\figures\`:

**EDA Figures (embed in Dataset Description slides):**
- `feature_distributions.png` — Histograms of clay, sand, silt, sm_aux, sm_tgt
- `correlation_heatmap.png` — 7x7 correlation matrix. Key: sm_aux vs sm_tgt = 0.25 (weak), sand vs silt = -0.97 (highly collinear)
- `spatial_map.png` — Germany grid showing observation density
- `temporal_coverage.png` — Daily observation count across 2013
- `aux_vs_tgt_scatter.png` — sm_aux vs sm_tgt scatter with 1:1 line showing calibration gap

**Analysis Figures (embed in Results/Evaluation slides):**
- `model_comparison.png` — Side-by-side RMSE and R2 bar charts across all models and splits
- `overfitting_analysis.png` — Train vs Test R2 grouped bars for random split
- `feature_importance.png` — Horizontal bar chart of RF feature importances
- `spatial_error_map.png` — Germany grid colored by prediction error (Linear Regression, West->East)
- `residual_distribution.png` — Histogram of prediction errors
- `spatial_degradation.png` — Line chart showing R2 dropping from random -> spatial splits for each model

## Assignment Requirements & Rubric

The presentation is graded on 14 criteria totaling 100 points:

### Technical Content (73 pts)

**1. Dataset Description (15 pts)** — HIGHEST WEIGHT
Detailed explanation of dataset selection (title, features, labels) and why it is suitable for the project.

**2. Problem Description (6 pts)**
Detailed explanation of the research topic and the significance of the problem.

**3. Proposed Approach (15 pts)** — HIGHEST WEIGHT
Detailed description of the chosen methods and techniques, at least 3 models. Must include strengths/weaknesses and justification.

**4. Experimental Setup (15 pts)** — HIGHEST WEIGHT
Detailed plan for experiment setting up and designing. Environment, tools, split strategies, configurations.

**5. Experimental Evaluation (9 pts)**
Detailed description of criteria and strategies for evaluating results.

**6. Project Timeline (3 pts)**
Detailed timeline table with milestones, deadlines, and task allocation.

**7. References (1 pt)**
Properly cite all data and literature sources.

**8. Layout (6 pts)**
Logically ordered, distinct sections, concise, minimal jargon.

**9. Supporting Material (3 pts)**
Graphs and figures must be interpretable, labeled with relevant units.

### Presentation & Delivery (27 pts)

**10. Presenters (3 pts)** — Roles and duties, each member presents.
**11. Presentation Clarity (6 pts)** — Clear, concise, evidence-backed.
**12. Presentation Skills (6 pts)** — Voice, gestures, eye contact.
**13. Time Management (6 pts)** — Must not exceed 12 minutes.
**14. Q&A Discussion (6 pts)** — Thoughtful responses with references.

## Task

Create a presentation slide deck using `python-pptx` that:

1. **Embeds the actual figures** from `C:\Users\ronak\Spring 2026\BSE 405 ML Project\figures\` into the relevant slides
2. **Includes the actual results** from the model results table above — real numbers, not placeholders
3. **Is structured to match the rubric sections exactly**
4. **Targets ~10-11 minutes** of content
5. **Includes speaker notes** for each slide with what to say (including references to the specific numbers)
6. **Assigns each section to a team member** and notes who presents what
7. **Emphasizes the three highest-weight sections** (Dataset Description, Proposed Approach, Experimental Setup — 15 pts each)

### Slide Structure

1. **Title slide** — project title, team names, course, date
2. **Team Roles** — who presents what section
3. **Problem Description** — satellite calibration, why it matters, research questions
4. **Dataset Overview** — source, size, structure, why suitable
5. **Dataset Details** — features table, embed `spatial_map.png` and `correlation_heatmap.png`
6. **EDA Highlights** — embed `feature_distributions.png`, `aux_vs_tgt_scatter.png`
7. **Proposed Approach** — 4 models, regression task, why each was chosen
8. **Model Details** — strengths/weaknesses table for all 4 models
9. **Experimental Setup** — environment, tools, pipeline structure
10. **Split Strategies** — random, West/East, block CV with diagram. Embed `spatial_map.png` showing the geographic split
11. **Evaluation Plan** — metrics (RMSE, R2, bias), overfitting analysis, spatial generalization
12. **Preliminary Results** — embed `model_comparison.png`, reference actual R2/RMSE numbers
13. **Spatial Generalization** — embed `spatial_degradation.png`, discuss negative R2 findings
14. **Feature Importance** — embed `feature_importance.png`, discuss that spatial/temporal features dominate
15. **Error Analysis** — embed `spatial_error_map.png` and `residual_distribution.png`
16. **Project Timeline** — milestones table with deadlines
17. **References** — Kaggle dataset, scikit-learn, relevant papers
18. **Q&A** — closing slide

### Format

Generate as a Python script using `python-pptx`. The script should:
- Use a clean, professional color scheme (dark blue headers, white backgrounds)
- Embed actual PNG figures from the figures/ directory
- Include proper slide layouts with titles and content
- Add speaker notes to each slide
- Save to `C:\Users\ronak\Spring 2026\BSE 405 ML Project\presentation\proposal_presentation.pptx`

### Reference Files

Read these for full context:
- `README.md` — project overview
- `docs/plans/2026-03-09-soil-moisture-ml-design.md` — full design document
- `results/model_results.csv` — actual model results
- `results/feature_importance.csv` — actual feature importances
- `figures/` — all 11 generated figures to embed
