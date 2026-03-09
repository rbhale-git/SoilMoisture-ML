# Soil Moisture ML Project — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete ML pipeline to predict AMSR satellite soil moisture from SMOS-ASCAT data and soil features, with spatial cross-validation and feature importance analysis.

**Architecture:** Four Colab notebooks with CSV handoff — EDA, feature engineering, modeling, analysis. Each notebook is self-contained and reads/writes shared CSV files.

**Tech Stack:** Google Colab, Python 3, pandas, numpy, scikit-learn, matplotlib, seaborn

**Data path (local):** `ML project datasets/ML project datasets/7_SoilMoisturePrediction_Tabular_Classification/data/updated_data.csv`

---

### Task 1: Create project folder structure on Google Drive

**Files:**
- Create: `BSE405_SoilMoisture/` folder on Google Drive
- Create: `BSE405_SoilMoisture/data/` subfolder
- Create: `BSE405_SoilMoisture/figures/` subfolder
- Create: `BSE405_SoilMoisture/results/` subfolder
- Upload: `updated_data.csv` to `data/`

**Step 1:** Create folder structure on Google Drive manually or via Colab.

**Step 2:** Upload `updated_data.csv` (17.7 MB) to `BSE405_SoilMoisture/data/`.

**Step 3:** Verify the file is accessible:
```python
from google.colab import drive
drive.mount('/content/drive')

import os
DATA_DIR = '/content/drive/MyDrive/BSE405_SoilMoisture/data'
FIG_DIR = '/content/drive/MyDrive/BSE405_SoilMoisture/figures'
RESULTS_DIR = '/content/drive/MyDrive/BSE405_SoilMoisture/results'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(os.listdir(DATA_DIR))
```
Expected: `['updated_data.csv']`

---

### Task 2: 01_EDA.ipynb — Load and inspect data

**Files:**
- Create: `BSE405_SoilMoisture/01_EDA.ipynb`

**Step 1: Mount Drive and load data**
```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = '/content/drive/MyDrive/BSE405_SoilMoisture/data'
FIG_DIR = '/content/drive/MyDrive/BSE405_SoilMoisture/figures'

df = pd.read_csv(f'{DATA_DIR}/updated_data.csv', parse_dates=['time'])
print(f"Shape: {df.shape}")
df.head()
```
Expected: Shape (321584, 8), table with time, latitude, longitude, clay_content, sand_content, silt_content, sm_aux, sm_tgt.

**Step 2: Summary statistics**
```python
print("=== Data Types ===")
print(df.dtypes)
print("\n=== Summary Statistics ===")
df.describe()
```
Run and verify all 7 numeric columns have count, mean, std, min, max.

**Step 3: Missing value analysis**
```python
print("=== Missing Values ===")
print(df.isnull().sum())
print(f"\nTotal missing: {df.isnull().sum().sum()}")
print(f"Rows with any missing: {df.isnull().any(axis=1).sum()}")
```
Run and note which columns (if any) have missing values.

---

### Task 3: 01_EDA.ipynb — Distribution plots

**Step 1: Feature histograms**
```python
numeric_cols = ['clay_content', 'sand_content', 'silt_content', 'sm_aux', 'sm_tgt']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

axes[-1].axis('off')
plt.suptitle('Feature Distributions', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
```
Run and verify 5 histograms appear, saved to figures/.

**Step 2: Correlation heatmap**
```python
corr_cols = ['latitude', 'longitude', 'clay_content', 'sand_content', 'silt_content', 'sm_aux', 'sm_tgt']
corr = df[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', square=True)
plt.title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```
Run and verify heatmap shows correlations. Note sm_aux vs sm_tgt correlation.

---

### Task 4: 01_EDA.ipynb — Spatial and temporal plots

**Step 1: Spatial map of data points**
```python
# Get unique grid points
grid = df.groupby(['latitude', 'longitude']).size().reset_index(name='count')

plt.figure(figsize=(10, 8))
scatter = plt.scatter(grid['longitude'], grid['latitude'], c=grid['count'],
                      cmap='YlOrRd', s=20, edgecolors='gray', linewidth=0.3)
plt.colorbar(scatter, label='Number of Observations')
plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
plt.title('Spatial Distribution of Observations (Germany)')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/spatial_map.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Unique grid points: {len(grid)}")
print(f"Lat range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
print(f"Lon range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
```
Run and verify Germany-shaped scatter plot appears.

**Step 2: Temporal coverage**
```python
daily_count = df.groupby('time').size()

plt.figure(figsize=(12, 4))
plt.plot(daily_count.index, daily_count.values, linewidth=0.8)
plt.xlabel('Date')
plt.ylabel('Number of Observations')
plt.title('Temporal Coverage (2013)')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/temporal_coverage.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Date range: {df['time'].min()} to {df['time'].max()}")
print(f"Unique dates: {df['time'].nunique()}")
```
Run and verify time series plot spanning 2013.

**Step 3: sm_aux vs sm_tgt scatter**
```python
sample = df.sample(n=5000, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(sample['sm_aux'], sample['sm_tgt'], alpha=0.3, s=5)
plt.xlabel('sm_aux (SMOS-ASCAT, m³/m³)')
plt.ylabel('sm_tgt (AMSR, m³/m³)')
plt.title('Auxiliary vs Target Soil Moisture')
plt.plot([0, 1], [0, 1], 'r--', label='1:1 line')
plt.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/aux_vs_tgt_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
```
Run and verify scatter shows relationship between the two satellite products.

---

### Task 5: 02_feature_engineering.ipynb — Create features

**Files:**
- Create: `BSE405_SoilMoisture/02_feature_engineering.ipynb`

**Step 1: Load data**
```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

DATA_DIR = '/content/drive/MyDrive/BSE405_SoilMoisture/data'

df = pd.read_csv(f'{DATA_DIR}/updated_data.csv', parse_dates=['time'])
print(f"Original shape: {df.shape}")
df.head()
```

**Step 2: Temporal features**
```python
df['day_of_year'] = df['time'].dt.dayofyear

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['season'] = df['time'].dt.month.apply(get_season)

print(df[['time', 'day_of_year', 'season']].head(10))
print(f"\nSeason counts:\n{df['season'].value_counts()}")
```
Run and verify day_of_year is 1-365, seasons are balanced roughly by quarter.

**Step 3: Soil texture ratios and interaction**
```python
# Avoid division by zero
df['clay_sand_ratio'] = df['clay_content'] / df['sand_content'].replace(0, np.nan)
df['clay_silt_ratio'] = df['clay_content'] / df['silt_content'].replace(0, np.nan)
df['clay_x_sm_aux'] = df['clay_content'] * df['sm_aux']

print(df[['clay_sand_ratio', 'clay_silt_ratio', 'clay_x_sm_aux']].describe())
```
Run and verify no infinities, check for NaNs introduced.

**Step 4: Spatial split columns**
```python
# West/East split (~10.5 degrees longitude, approximate historical border)
df['region'] = np.where(df['longitude'] < 10.5, 'West', 'East')
print(f"Region counts:\n{df['region'].value_counts()}")

# Spatial block CV — divide into 6 blocks (2 lat bands x 3 lon bands)
lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

lat_bins = np.linspace(lat_min, lat_max + 0.01, 3)  # 2 lat bands
lon_bins = np.linspace(lon_min, lon_max + 0.01, 4)  # 3 lon bands

df['lat_band'] = pd.cut(df['latitude'], bins=lat_bins, labels=['S', 'N'])
df['lon_band'] = pd.cut(df['longitude'], bins=lon_bins, labels=['W', 'C', 'E'])
df['spatial_block'] = df['lat_band'].astype(str) + '_' + df['lon_band'].astype(str)

print(f"\nSpatial block counts:\n{df['spatial_block'].value_counts()}")
```
Run and verify 6 blocks with reasonable counts each. Region split is roughly balanced.

---

### Task 6: 02_feature_engineering.ipynb — Handle missing values and export

**Step 1: Handle missing values**
```python
print(f"Missing values before cleanup:\n{df.isnull().sum()}")
print(f"\nTotal rows before: {len(df)}")

# Drop rows with any NaN in feature or target columns
df_clean = df.dropna(subset=['sm_aux', 'sm_tgt', 'clay_sand_ratio', 'clay_silt_ratio'])
print(f"Total rows after: {len(df_clean)}")
print(f"Rows dropped: {len(df) - len(df_clean)}")
```
Run and verify minimal rows dropped.

**Step 2: Export processed data**
```python
# Drop helper columns, keep what modeling needs
df_clean = df_clean.drop(columns=['lat_band', 'lon_band'])

print(f"Final shape: {df_clean.shape}")
print(f"Columns: {list(df_clean.columns)}")

df_clean.to_csv(f'{DATA_DIR}/processed_data.csv', index=False)
print(f"\nSaved to {DATA_DIR}/processed_data.csv")
```
Expected columns: time, latitude, longitude, clay_content, sand_content, silt_content, sm_aux, sm_tgt, day_of_year, season, clay_sand_ratio, clay_silt_ratio, clay_x_sm_aux, region, spatial_block.

**Step 3: Verify export**
```python
verify = pd.read_csv(f'{DATA_DIR}/processed_data.csv')
print(f"Verified shape: {verify.shape}")
verify.head()
```

---

### Task 7: 03_modeling.ipynb — Setup and data preparation

**Files:**
- Create: `BSE405_SoilMoisture/03_modeling.ipynb`

**Step 1: Load processed data and define features**
```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/content/drive/MyDrive/BSE405_SoilMoisture/data'
RESULTS_DIR = '/content/drive/MyDrive/BSE405_SoilMoisture/results'

df = pd.read_csv(f'{DATA_DIR}/processed_data.csv')
print(f"Shape: {df.shape}")

# Define feature columns (exclude time, target, split labels)
FEATURE_COLS = ['latitude', 'longitude', 'clay_content', 'sand_content', 'silt_content',
                'sm_aux', 'day_of_year', 'clay_sand_ratio', 'clay_silt_ratio', 'clay_x_sm_aux']

# One-hot encode season
df = pd.get_dummies(df, columns=['season'], drop_first=True)
season_cols = [c for c in df.columns if c.startswith('season_')]
FEATURE_COLS += season_cols

TARGET = 'sm_tgt'

print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
```

**Step 2: Define model dictionary**
```python
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'SVM (RBF approx)': Pipeline([
        ('scaler', StandardScaler()),
        ('nystroem', Nystroem(kernel='rbf', n_components=100, random_state=42)),
        ('sgd', SGDRegressor(random_state=42, max_iter=1000))
    ]),
    'MLP': Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
    ])
}
```

**Step 3: Define evaluation function**
```python
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_bias': np.mean(y_test_pred - y_test),
    }
    return results, y_test_pred
```

---

### Task 8: 03_modeling.ipynb — Random split experiment

**Step 1: Random 80/20 split**
```python
from sklearn.model_selection import train_test_split

X = df[FEATURE_COLS].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
```

**Step 2: Run all models on random split**
```python
all_results = []

for name, model in models.items():
    print(f"Training {name}...")
    results, _ = evaluate_model(model, X_train, y_train, X_test, y_test)
    results['model'] = name
    results['split'] = 'Random 80/20'
    all_results.append(results)
    print(f"  Test RMSE: {results['test_rmse']:.4f}, Test R²: {results['test_r2']:.4f}")

print("\nRandom split complete.")
```
Run and verify all 4 models produce reasonable RMSE and R² values.

---

### Task 9: 03_modeling.ipynb — West/East spatial split experiment

**Step 1: Split by region**
```python
west_mask = df['region'] == 'West'
east_mask = df['region'] == 'East'

X_west = df.loc[west_mask, FEATURE_COLS].values
y_west = df.loc[west_mask, TARGET].values
X_east = df.loc[east_mask, FEATURE_COLS].values
y_east = df.loc[east_mask, TARGET].values

print(f"West (train): {X_west.shape[0]}, East (test): {X_east.shape[0]}")
```

**Step 2: Run all models on spatial split**
```python
# Store predictions for error mapping
spatial_predictions = df.loc[east_mask, ['latitude', 'longitude']].copy()

for name, model in models.items():
    print(f"Training {name} (West→East)...")
    results, y_pred = evaluate_model(model, X_west, y_west, X_east, y_east)
    results['model'] = name
    results['split'] = 'West→East'
    all_results.append(results)
    spatial_predictions[f'pred_{name}'] = y_pred
    print(f"  Test RMSE: {results['test_rmse']:.4f}, Test R²: {results['test_r2']:.4f}")

spatial_predictions['actual'] = y_east
print("\nWest→East split complete.")
```

---

### Task 10: 03_modeling.ipynb — Spatial block CV experiment

**Step 1: Run block CV**
```python
blocks = df['spatial_block'].unique()
print(f"Blocks: {blocks}")

for name, model_template in models.items():
    print(f"\nBlock CV for {name}...")
    block_results = []

    for block in blocks:
        test_mask = df['spatial_block'] == block
        train_mask = ~test_mask

        X_tr = df.loc[train_mask, FEATURE_COLS].values
        y_tr = df.loc[train_mask, TARGET].values
        X_te = df.loc[test_mask, FEATURE_COLS].values
        y_te = df.loc[test_mask, TARGET].values

        if len(X_te) == 0:
            continue

        # Clone model for each fold
        from sklearn.base import clone
        model = clone(model_template)
        res, _ = evaluate_model(model, X_tr, y_tr, X_te, y_te)
        block_results.append(res)
        print(f"  Block {block}: RMSE={res['test_rmse']:.4f}, R²={res['test_r2']:.4f}")

    # Average across blocks
    avg_results = {
        'train_rmse': np.mean([r['train_rmse'] for r in block_results]),
        'test_rmse': np.mean([r['test_rmse'] for r in block_results]),
        'train_r2': np.mean([r['train_r2'] for r in block_results]),
        'test_r2': np.mean([r['test_r2'] for r in block_results]),
        'test_bias': np.mean([r['test_bias'] for r in block_results]),
        'model': name,
        'split': 'Spatial Block CV'
    }
    all_results.append(avg_results)
    print(f"  Average: RMSE={avg_results['test_rmse']:.4f}, R²={avg_results['test_r2']:.4f}")
```

---

### Task 11: 03_modeling.ipynb — Save results

**Step 1: Save model results**
```python
results_df = pd.DataFrame(all_results)
results_df = results_df[['model', 'split', 'train_rmse', 'test_rmse', 'train_r2', 'test_r2', 'test_bias']]
results_df.to_csv(f'{RESULTS_DIR}/model_results.csv', index=False)
print(results_df.to_string(index=False))
```

**Step 2: Save spatial predictions**
```python
spatial_predictions.to_csv(f'{RESULTS_DIR}/predictions_spatial.csv', index=False)
print(f"Spatial predictions saved: {spatial_predictions.shape}")
```

**Step 3: Save feature importance from Random Forest**
```python
# Retrain RF on full random split to get feature importance
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

importance_df.to_csv(f'{RESULTS_DIR}/feature_importance.csv', index=False)
print(importance_df.to_string(index=False))
```

---

### Task 12: 04_analysis.ipynb — Model comparison visualizations

**Files:**
- Create: `BSE405_SoilMoisture/04_analysis.ipynb`

**Step 1: Load results**
```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = '/content/drive/MyDrive/BSE405_SoilMoisture/results'
FIG_DIR = '/content/drive/MyDrive/BSE405_SoilMoisture/figures'

results = pd.read_csv(f'{RESULTS_DIR}/model_results.csv')
predictions = pd.read_csv(f'{RESULTS_DIR}/predictions_spatial.csv')
importance = pd.read_csv(f'{RESULTS_DIR}/feature_importance.csv')

print(results.to_string(index=False))
```

**Step 2: Model comparison bar chart**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# RMSE comparison
pivot_rmse = results.pivot(index='model', columns='split', values='test_rmse')
pivot_rmse.plot(kind='bar', ax=axes[0], rot=15)
axes[0].set_title('Test RMSE by Model and Split Strategy')
axes[0].set_ylabel('RMSE (m³/m³)')
axes[0].legend(title='Split')

# R² comparison
pivot_r2 = results.pivot(index='model', columns='split', values='test_r2')
pivot_r2.plot(kind='bar', ax=axes[1], rot=15)
axes[1].set_title('Test R² by Model and Split Strategy')
axes[1].set_ylabel('R²')
axes[1].legend(title='Split')

plt.suptitle('Model Comparison Across Split Strategies', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Step 3: Overfitting analysis**
```python
fig, ax = plt.subplots(figsize=(10, 6))

# Filter to random split for clean comparison
random_results = results[results['split'] == 'Random 80/20']

x = np.arange(len(random_results))
width = 0.35

ax.bar(x - width/2, random_results['train_r2'], width, label='Train R²', color='steelblue')
ax.bar(x + width/2, random_results['test_r2'], width, label='Test R²', color='coral')

ax.set_xlabel('Model')
ax.set_ylabel('R²')
ax.set_title('Train vs Test R² (Random Split) — Overfitting Analysis')
ax.set_xticks(x)
ax.set_xticklabels(random_results['model'], rotation=15)
ax.legend()
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/overfitting_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### Task 13: 04_analysis.ipynb — Feature importance and error maps

**Step 1: Feature importance plot**
```python
plt.figure(figsize=(10, 6))
plt.barh(importance['feature'], importance['importance'], color='teal')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Step 2: Spatial error map (best model on West→East)**
```python
# Identify best model from results
best_model = results.loc[results[results['split'] == 'West→East']['test_rmse'].idxmin(), 'model']
print(f"Best model (West→East): {best_model}")

pred_col = f'pred_{best_model}'
predictions['error'] = predictions[pred_col] - predictions['actual']

plt.figure(figsize=(10, 8))
scatter = plt.scatter(predictions['longitude'], predictions['latitude'],
                      c=predictions['error'], cmap='RdBu_r', s=3, alpha=0.5,
                      vmin=-0.2, vmax=0.2)
plt.colorbar(scatter, label='Prediction Error (m³/m³)')
plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
plt.title(f'Spatial Error Map — {best_model} (Train: West, Test: East)')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/spatial_error_map.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Step 3: Residual distribution**
```python
plt.figure(figsize=(8, 5))
plt.hist(predictions['error'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(x=0, color='red', linestyle='--', label='Zero error')
plt.xlabel('Prediction Error (m³/m³)')
plt.ylabel('Frequency')
plt.title(f'Residual Distribution — {best_model} (West→East)')
plt.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/residual_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Mean error (bias): {predictions['error'].mean():.4f}")
print(f"Std of error: {predictions['error'].std():.4f}")
```

**Step 4: Spatial CV degradation chart**
```python
# Show how performance drops from random → spatial splits
split_order = ['Random 80/20', 'West→East', 'Spatial Block CV']

fig, ax = plt.subplots(figsize=(10, 6))

for model_name in results['model'].unique():
    model_data = results[results['model'] == model_name]
    model_data = model_data.set_index('split').loc[split_order]
    ax.plot(split_order, model_data['test_r2'], marker='o', label=model_name)

ax.set_xlabel('Split Strategy')
ax.set_ylabel('Test R²')
ax.set_title('Spatial Generalization — R² Degradation Across Split Strategies')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/spatial_degradation.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### Task Summary

| Task | Notebook | Description |
|------|----------|-------------|
| 1 | Setup | Create Google Drive folder, upload data |
| 2-4 | 01_EDA | Load data, distributions, correlation, spatial/temporal plots |
| 5-6 | 02_feature_engineering | Create features, spatial splits, export processed_data.csv |
| 7-11 | 03_modeling | Setup models, run 3 experiments, save results |
| 12-13 | 04_analysis | Model comparison, feature importance, error maps, residuals |
