# Project Synopsis: Research Question Answers

## 1. Can ML improve prediction of AMSR soil moisture using SMOS-ASCAT and soil texture?

**Partially — but only with non-linear models and only when training and testing on the same geographic distribution.**

The raw correlation between `sm_aux` and `sm_tgt` is only 0.25, meaning the SMOS-ASCAT satellite reading alone is a weak predictor of AMSR moisture. Adding soil texture and spatial/temporal features through ML does improve this:

- **Random Forest** achieves R2 = 0.753 on the random split — a substantial improvement over the 0.25 correlation baseline
- **MLP** reaches R2 = 0.487 — moderate improvement
- **SVM** manages R2 = 0.176 — marginal improvement
- **Linear Regression** gets R2 = 0.091 — almost no improvement, confirming the relationship is non-linear

However, these improvements collapse under spatial validation (see Q2), suggesting the models are largely learning location-specific patterns rather than true physical relationships between soil texture, satellite readings, and moisture.

## 2. How well do models generalize across different geographic regions?

**They don't.** This is the strongest finding of the project.

Every model produces **negative R2** on both spatial split strategies, meaning they predict worse than simply guessing the mean:

| Model | West->East R2 | Block CV R2 |
|-------|-------------|-------------|
| Linear Regression | -0.032 | -0.200 |
| Random Forest | -0.343 | -0.030 |
| SVM (RBF approx) | -0.085 | -0.188 |
| MLP | -2.350 | -0.720 |

**Key observations:**
- **Random Forest** drops from R2 = 0.753 (random) to -0.343 (West->East) — the most dramatic degradation. Its high random-split performance was an illusion created by memorizing location-specific patterns
- **MLP** catastrophically fails at -2.350 on West->East, making wildly wrong predictions on unseen geography
- **Linear Regression** actually degrades the least (-0.032), suggesting simpler models are more robust to distribution shift, even if their absolute performance is poor
- The **overfitting gap** in Random Forest (train R2 = 0.965 vs test R2 = 0.753 on random split) foreshadowed this failure — the model was memorizing, not learning

This demonstrates that standard random train/test splits can give a misleading picture of model performance on spatial data. Spatial cross-validation is essential for geoscience ML problems.

## 3. Which features most influence the prediction?

**Spatial and temporal features dominate — not soil physics or satellite data.**

| Feature | Importance | Category |
|---------|-----------|----------|
| day_of_year | 24.9% | Temporal |
| longitude | 20.8% | Spatial |
| latitude | 18.3% | Spatial |
| sm_aux | 11.9% | Satellite |
| clay_x_sm_aux | 7.5% | Engineered |
| clay_sand_ratio | 5.7% | Soil texture |
| clay_silt_ratio | 3.5% | Soil texture |
| sand/silt/clay content | ~7.0% | Soil texture |
| season features | ~0.4% | Temporal |

**Key insights:**

- **64% of importance comes from where and when** (day_of_year + longitude + latitude). The Random Forest is primarily learning "soil moisture is higher in this region at this time of year" — geographic and seasonal patterns, not transferable soil-moisture physics
- **sm_aux accounts for only 12%** despite being the core satellite measurement and the most physically meaningful predictor. This is surprisingly low and explains why models fail spatially — they barely use the one feature that should generalize
- **Soil texture features collectively contribute ~16%** — meaningful but secondary. The engineered features (ratios, interaction) are more useful than raw clay/sand/silt values
- **Season one-hot features are nearly useless** (0.4%) because `day_of_year` already captures seasonality with finer granularity

This explains the spatial generalization failure: the model relies on coordinates as proxies for regional climate differences. When tested on a new region, those coordinate-based patterns don't transfer.

## 4. Why Do These Models Fail?

The spatial generalization failure has five root causes:

### 1. Missing Environmental Features

The dataset contains only soil texture (clay, sand, silt) and one satellite reading (sm_aux). Real soil moisture depends on many variables we don't have:

- **Precipitation** — the primary driver of soil moisture variation
- **Temperature and evapotranspiration** — controls how fast water leaves the soil
- **Vegetation type and density** — affects water uptake and interception
- **Topography** — slope, elevation, and drainage patterns
- **Land use** — urban vs agricultural vs forest

Without these, models have no way to learn the physical processes that determine moisture. They compensate by memorizing location-specific patterns via coordinates.

### 2. Weak Core Signal

The correlation between `sm_aux` and `sm_tgt` is only 0.25. This means the SMOS-ASCAT satellite reading — the most physically meaningful input — explains only ~6% of variance in AMSR moisture. The models simply don't have a strong enough input signal to build a generalizable calibration function. Even a perfect model would struggle with this input.

### 3. Static Soil Texture

Clay, sand, and silt content are fixed properties at each grid point. They don't change over the year. This means soil texture can explain spatial variation (why one location is wetter than another on average) but cannot explain temporal variation (why a location is wetter today than yesterday). Since soil moisture is highly dynamic, static features have limited predictive power on their own.

### 4. Spatial Autocorrelation

Nearby grid points have similar soil moisture because they share the same weather, terrain, and soil conditions. In a random 80/20 split, the test set contains points that are neighbors of training points — the model can "cheat" by recognizing the location and recalling what nearby training points looked like. This inflates R² on random splits but provides no transferable learning. The West/East split eliminates this shortcut entirely.

### 5. Model Over-Flexibility

Random Forest and MLP have enough capacity to memorize the training data:
- **Random Forest** train R² = 0.965 vs test R² = 0.753 — the 0.21 gap signals heavy memorization
- **MLP** train R² = 0.499 vs test R² = 0.487 — less overfitting but still learns coordinate-dependent patterns

When moved to new geography, these memorized patterns become actively harmful (negative R²), predicting worse than the mean. Linear Regression, with its limited capacity, cannot memorize as much — which is why it degrades the least (-0.032 vs -0.343 for RF).

### What Would Help?

To build models that actually generalize spatially, the project would need:

1. **Weather/climate features** — precipitation, temperature, humidity at each grid point and time
2. **Vegetation indices** (NDVI) — satellite-derived plant activity data
3. **Physics-informed approaches** — models that encode known soil-water relationships as constraints
4. **Dropping coordinates as features** — forcing models to rely on physically meaningful inputs rather than location proxies
5. **Transfer learning** — pre-training on regions with dense ground-truth data, then fine-tuning on sparse regions

The current failure is not a model selection problem — it's a feature availability problem. No ML model can learn soil moisture physics from soil texture and a weak satellite signal alone.
