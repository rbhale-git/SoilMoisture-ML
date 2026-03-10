# Colab Setup Instructions

## Option A: Clone from GitHub (Recommended)

No manual file setup needed. Each notebook's first cell handles everything.

1. Open Google Colab (colab.research.google.com)
2. File > Open Notebook > GitHub tab
3. Paste: `https://github.com/rbhale-git/SoilMoisture-ML`
4. Select the notebook you want to run
5. Before running, add a new cell at the top with:

```python
# Clone repo and set paths
import os
if not os.path.exists('/content/SoilMoisture-ML'):
    !git clone https://github.com/rbhale-git/SoilMoisture-ML.git

DATA_DIR = '/content/SoilMoisture-ML/data'
FIG_DIR = '/content/SoilMoisture-ML/figures'
RESULTS_DIR = '/content/SoilMoisture-ML/results'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
```

6. Then **comment out or skip** the Drive mount lines in the notebook's original first code cell (`drive.mount(...)`) and replace the `DATA_DIR`, `FIG_DIR`, `RESULTS_DIR` paths with the ones above.

7. Run notebooks in order: 01 → 02 → 03 → 04

**Note:** Colab sessions are temporary. If your runtime disconnects, you need to re-clone. Figures and results will be lost unless you download them or push back to GitHub.

---

## Option B: Google Drive Setup

Use this if you want persistent storage across Colab sessions.

### Step 1: Create folder structure on Google Drive

Create these folders in your Google Drive (My Drive):

```
My Drive/
└── BSE405_SoilMoisture/
    ├── data/
    ├── figures/
    └── results/
```

### Step 2: Upload the dataset

Upload `updated_data.csv` into `BSE405_SoilMoisture/data/`.

The file is available at:
- GitHub repo: `data/updated_data.csv`
- Or locally: `C:\Users\ronak\Spring 2026\BSE 405 ML Project\data\updated_data.csv`

### Step 3: Upload notebooks

Upload all 4 notebooks from the `notebooks/` folder into `BSE405_SoilMoisture/`:
- `01_EDA.ipynb`
- `02_feature_engineering.ipynb`
- `03_modeling.ipynb`
- `04_analysis.ipynb`

### Step 4: Open in Colab

1. Go to Google Drive
2. Navigate to `BSE405_SoilMoisture/`
3. Double-click any `.ipynb` file → it opens in Google Colab
4. Run cells in order

The notebooks already have `drive.mount('/content/drive')` and point to:
- `DATA_DIR = '/content/drive/MyDrive/BSE405_SoilMoisture/data'`
- `FIG_DIR = '/content/drive/MyDrive/BSE405_SoilMoisture/figures'`
- `RESULTS_DIR = '/content/drive/MyDrive/BSE405_SoilMoisture/results'`

### Step 5: Run notebooks in order

1. `01_EDA.ipynb` — no dependencies, just needs `updated_data.csv`
2. `02_feature_engineering.ipynb` — needs `updated_data.csv`, produces `processed_data.csv`
3. `03_modeling.ipynb` — needs `processed_data.csv`, produces `model_results.csv`, `predictions_spatial.csv`, `feature_importance.csv`
4. `04_analysis.ipynb` — needs all three result CSVs from step 3

---

## Troubleshooting

**"FileNotFoundError: updated_data.csv"**
- The data file isn't in the expected path. Check that `updated_data.csv` is in the `data/` folder.

**"Drive already mounted"**
- This is a warning, not an error. Ignore it or change to `drive.mount('/content/drive', force_remount=True)`.

**"processed_data.csv not found" (in notebook 03)**
- You need to run notebook 02 first. It creates this file.

**"model_results.csv not found" (in notebook 04)**
- You need to run notebook 03 first. It creates the result files.

**Session disconnected / files lost**
- Option A (GitHub clone): Re-run the clone cell and re-run all notebooks.
- Option B (Google Drive): Files persist on Drive. Just re-mount and continue.
