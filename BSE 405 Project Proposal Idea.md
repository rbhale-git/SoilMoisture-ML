BSE 405 Project Proposal Idea
Satellite Soil Moisture Modelling Using Machine Learning

Project Title (Working)
Spatial Calibration of Satellite Soil Moisture Products Using Machine Learning

Dataset Overview (Dataset 7 – Soil Moisture)
	• Location: Germany
	• Year: 2013 (full year)
	• Spatial structure: 28 × 56 grid (0.25° resolution raster)
	• Temporal resolution: Daily measurements (irregular 1–6 day gaps)
Features:
	• Latitude
	• Longitude
	• Clay %
	• Sand %
	• Silt %
	• sm_aux (SMOS–ASCAT satellite product)
Target:
	• sm_tgt (AMSR satellite soil moisture, m³/m³)

Problem Framing
Two satellites measure soil moisture:
	• sm_aux (auxiliary input)
	• sm_tgt (target product)
Goal:
	Use machine learning to predict sm_tgt using sm_aux and soil composition features.
This becomes a satellite calibration / data fusion problem.

Research Questions
	1. Can machine learning improve the prediction of AMSR soil moisture (sm_tgt) using SMOS–ASCAT (sm_aux) and soil texture?
	2. How well do models generalize across different geographic regions?
	3. Which features most influence the prediction (soil composition vs satellite measurement)?

Proposed Methodology
1. Baseline Regression Models
We compare multiple supervised regression models:
	• Linear Regression
	• Random Forest
	• Support Vector Machine (RBF)
	• Multi-Layer Perceptron

2. Evaluation Metrics
	• RMSE (Root Mean Squared Error)
	• R² (Coefficient of Determination)
	• Bias analysis
We will also compare training vs testing performance to analyze overfitting.

3. Spatial Cross-Validation
Instead of a random train/test split:
	• Train on one geographic region
	• Test on a different region
This evaluates spatial generalization performance.
Possible strategies:
	• West vs East Germany
	• Spatial block cross-validation

4. Feature Engineering
Potential additions:
	• Soil texture ratios
	• Interaction terms (e.g., clay × sm_aux)
	• Day-of-year feature (seasonal effect)
	• Seasonal grouping (winter vs summer)

5. Model Interpretation
Using Random Forest feature importance:
	• Does soil texture significantly improve predictions?
	• Is sm_aux already highly predictive?
	• Do coordinates encode regional climate differences?

Expected Outputs
	• Exploratory Data Analysis
	• Model comparison results
	• Spatial validation analysis
	• Feature importance analysis
	• Error distribution maps across Germany
	• Final report and presentation

Why This Project Makes Sense
	• Strong alignment with remote sensing and satellite data concepts
	• Clear regression framework
	• Structured spatial dataset (grid-based)
	• Allows meaningful interpretation of results
	• Large dataset suitable for robust evaluation

Team Discussion Points
	1. Do we want to focus strictly on classical ML models?
	2. How advanced do we want spatial validation to be?
	3. How much feature engineering do we include?
	4. How should we divide responsibilities (EDA, modeling, validation, visualization)?
