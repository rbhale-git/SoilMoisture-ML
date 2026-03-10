# Prompt: Create BSE 405 ML Project Proposal Presentation

## Context

You are helping create a **proposal presentation** for a BSE 405 Machine Learning course project at UW-Madison. The presentation is due **March 18, 2026** and must not exceed **12 minutes**. It will be followed by a Q&A session.

The project directory is: `C:\Users\ronak\Spring 2026\BSE 405 ML Project`
GitHub repo: https://github.com/rbhale-git/SoilMoisture-ML

## Team

- Ronak Bhale
- John Brooks
- Elyzabeth Benitez Rivera
- Max

This is a team of 4 undergraduates. Each team member must present their own section.

## Project Summary

**Title:** Spatial Calibration of Satellite Soil Moisture Products Using Machine Learning

**Problem:** Two satellites measure soil moisture over Germany — SMOS-ASCAT (`sm_aux`) and AMSR (`sm_tgt`). The goal is to use machine learning to predict AMSR soil moisture from SMOS-ASCAT readings combined with soil composition and spatial features. This is a satellite calibration / data fusion problem.

**Dataset:** Kaggle — "Soil Moisture Remote Sensing Data (Germany 2013)"
- 321,584 rows, 8 columns
- 28 × 56 grid at 0.25° resolution, daily measurements across Germany in 2013
- Features: time, latitude, longitude, clay_content, sand_content, silt_content, sm_aux
- Target: sm_tgt (soil moisture in m³/m³)

**Engineered Features:**
- day_of_year (1-365 from timestamp)
- season (Winter/Spring/Summer/Fall)
- clay_sand_ratio (clay_content / sand_content)
- clay_silt_ratio (clay_content / silt_content)
- clay_x_sm_aux (clay_content × sm_aux interaction term)

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
  1. Random 80/20 split — standard baseline
  2. West/East spatial split — train on West Germany (longitude < 10.5°), test on East Germany. This is the primary spatial generalization experiment.
  3. Spatial block cross-validation — 6 blocks (2 latitude bands × 3 longitude bands), leave-one-block-out. Supplementary analysis.

**Evaluation Metrics:**
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- Bias (mean prediction error)
- Train vs test performance comparison (overfitting detection)

**Research Questions:**
1. Can ML improve prediction of AMSR soil moisture using SMOS-ASCAT and soil texture?
2. How well do models generalize across different geographic regions?
3. Which features most influence the prediction (soil composition vs satellite measurement)?

**Codebase Structure:**
- `notebooks/01_EDA.ipynb` — Exploratory data analysis
- `notebooks/02_feature_engineering.ipynb` — Feature creation and spatial splits
- `notebooks/03_modeling.ipynb` — Train 4 models across 3 splits
- `notebooks/04_analysis.ipynb` — Visualizations: model comparison, error maps, feature importance

**Key Visualizations the project produces:**
- Feature distribution histograms
- Correlation heatmap
- Spatial map of observations across Germany
- Temporal coverage plot
- sm_aux vs sm_tgt scatter plot
- Model comparison bar chart (RMSE + R² across splits)
- Overfitting analysis (train vs test R²)
- Feature importance horizontal bar chart (Random Forest)
- Spatial error map (Germany grid colored by prediction error)
- Residual distribution histogram
- Spatial generalization degradation chart (R² dropping across split strategies)

## Assignment Requirements & Rubric

The presentation is graded on 14 criteria totaling 100 points. The presentation MUST include all of the following sections:

### Technical Content (73 pts)

**1. Dataset Description (15 pts)** — HIGHEST WEIGHT
Detailed explanation of dataset selection (title, features, labels) and why it is suitable for the project. This needs to be thorough: describe each column, the spatial/temporal structure, data source, and why this dataset is a good fit for ML.

**2. Problem Description (6 pts)**
Detailed explanation of the research topic and the significance of the problem. Explain the satellite calibration problem, why it matters, and the research questions.

**3. Proposed Approach (15 pts)** — HIGHEST WEIGHT
Detailed description of the chosen methods and techniques, at least 3 models. Must include:
- Why each model is suitable for this problem domain
- Type of task (regression)
- Analysis of each model's strengths and weaknesses
- Justification of suitability for project objectives

**4. Experimental Setup (15 pts)** — HIGHEST WEIGHT
Detailed plan for experiment setting up and designing. Must include:
- Environment and tools (Google Colab, scikit-learn, etc.)
- Train/test splitting strategies (all 3 strategies)
- Configurations used

**5. Experimental Evaluation (9 pts)**
Detailed description of criteria and strategies for evaluating results. Cover RMSE, R², bias, overfitting analysis, and how spatial CV tests generalization.

**6. Project Timeline (3 pts)**
Detailed timeline table with milestones, deadlines, and allocation of future tasks among team members.

**7. References (1 pt)**
Properly cite all data and literature sources. At minimum:
- Kaggle dataset source
- scikit-learn documentation
- Any relevant remote sensing or ML papers

**8. Layout (6 pts)**
Information logically ordered, distinct sections, concise sentences, minimal jargon.

**9. Supporting Material (3 pts)**
Graphs and figures must be interpretable, labeled with relevant units, distinguishable.

### Presentation & Delivery (27 pts)

**10. Presenters (3 pts)**
Clearly outline the roles and duties of each team member. Each member must present their section.

**11. Presentation Clarity (6 pts)**
Project introduced clearly and concisely. All statements evidence-backed.

**12. Presentation Skills (6 pts)**
Clear voice, appropriate volume, gestures, eye contact.

**13. Time Management (6 pts)**
Must not exceed 12 minutes.

**14. Q&A Discussion (6 pts)**
Thoughtful and accurate responses, supported by references or calculations.

## Task

Create a presentation slide deck for this project proposal. Use a clean, professional template. The presentation should:

1. **Be structured to match the rubric sections exactly** — every rubric criterion should map to one or more slides
2. **Target ~10-11 minutes** of content (leaving buffer for the 12 min limit)
3. **Include speaker notes** for each slide with what to say
4. **Assign each section to a team member** and note who presents what
5. **Include placeholder references** to the visualizations the project will produce (e.g., "Insert correlation heatmap here")
6. **Be evidence-backed** — include specific numbers (321K rows, 8 columns, 28×56 grid, etc.)
7. **Emphasize the three highest-weight sections** (Dataset Description, Proposed Approach, Experimental Setup — 15 pts each)

### Suggested Slide Structure

1. Title slide (project title, team names, course)
2. Team roles and responsibilities
3. Problem description — satellite calibration, why it matters
4. Dataset overview — source, size, structure
5. Dataset details — features table, spatial/temporal properties
6. Supporting figure — show spatial map or correlation heatmap placeholder
7. Proposed approach overview — 4 models, regression task
8. Model comparison table — strengths/weaknesses/justification
9. Experimental setup — environment, tools
10. Split strategies — random, West/East, block CV (with diagram)
11. Evaluation plan — metrics, overfitting analysis, spatial generalization
12. Project timeline — milestones table
13. References
14. Q&A slide

### Format

Generate the presentation as a Python script using `python-pptx` that creates a .pptx file. The script should:
- Use a clean, professional color scheme
- Include proper slide layouts with titles and content
- Add speaker notes to each slide
- Save to `C:\Users\ronak\Spring 2026\BSE 405 ML Project\presentation\proposal_presentation.pptx`

### Reference Files

Read these files for full context:
- `docs/plans/2026-03-09-soil-moisture-ml-design.md` — full design document
- `docs/plans/2026-03-09-soil-moisture-ml-plan.md` — implementation plan
- `BSE 405 Project Proposal Idea.md` — original proposal notes
- `README.md` — project overview
- `notebooks/` — all 4 notebooks for code details
