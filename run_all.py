"""
BSE 405 ML Project — Satellite Soil Moisture Calibration
Consolidated pipeline: EDA → Feature Engineering → Modeling → Analysis

Usage:
    python run_all.py                          # uses default paths
    python run_all.py --data-dir ./data        # custom data directory
    python run_all.py --colab                  # use Google Colab/Drive paths
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone


# ============================================================================
# Configuration
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Soil Moisture ML Pipeline')
    parser.add_argument('--data-dir', default=None, help='Directory containing updated_data.csv')
    parser.add_argument('--output-dir', default=None, help='Base output directory for figures/results')
    parser.add_argument('--colab', action='store_true', help='Use Google Colab/Drive paths')
    return parser.parse_args()


def setup_paths(args):
    if args.colab:
        from google.colab import drive
        drive.mount('/content/drive')
        base = '/content/drive/MyDrive/BSE405_SoilMoisture'
        data_dir = f'{base}/data'
        fig_dir = f'{base}/figures'
        results_dir = f'{base}/results'
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = args.data_dir or os.path.join(script_dir, 'data')
        output_base = args.output_dir or script_dir
        fig_dir = os.path.join(output_base, 'figures')
        results_dir = os.path.join(output_base, 'results')

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    return data_dir, fig_dir, results_dir


# ============================================================================
# Phase 1: Exploratory Data Analysis
# ============================================================================

def run_eda(df, fig_dir):
    print('\n' + '='*60)
    print('PHASE 1: Exploratory Data Analysis')
    print('='*60)

    # Summary statistics
    print(f"\nShape: {df.shape}")
    print(f"\n=== Data Types ===\n{df.dtypes}")
    print(f"\n=== Summary Statistics ===\n{df.describe()}")

    # Missing values
    print(f"\n=== Missing Values ===\n{df.isnull().sum()}")
    print(f"Total missing: {df.isnull().sum().sum()}")
    print(f"Rows with any missing: {df.isnull().any(axis=1).sum()}")

    # Feature distributions
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
    plt.savefig(f'{fig_dir}/feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: feature_distributions.png')

    # Correlation heatmap
    corr_cols = ['latitude', 'longitude', 'clay_content', 'sand_content', 'silt_content', 'sm_aux', 'sm_tgt']
    corr = df[corr_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', square=True)
    plt.title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: correlation_heatmap.png')

    # Spatial map
    grid = df.groupby(['latitude', 'longitude']).size().reset_index(name='count')
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(grid['longitude'], grid['latitude'], c=grid['count'],
                          cmap='YlOrRd', s=20, edgecolors='gray', linewidth=0.3)
    plt.colorbar(scatter, label='Number of Observations')
    plt.xlabel('Longitude (\u00b0)')
    plt.ylabel('Latitude (\u00b0)')
    plt.title('Spatial Distribution of Observations (Germany)')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/spatial_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: spatial_map.png  (Unique grid points: {len(grid)})')
    print(f'Lat range: {df["latitude"].min():.2f} to {df["latitude"].max():.2f}')
    print(f'Lon range: {df["longitude"].min():.2f} to {df["longitude"].max():.2f}')

    # Temporal coverage
    daily_count = df.groupby('time').size()
    plt.figure(figsize=(12, 4))
    plt.plot(daily_count.index, daily_count.values, linewidth=0.8)
    plt.xlabel('Date')
    plt.ylabel('Number of Observations')
    plt.title('Temporal Coverage (2013)')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/temporal_coverage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: temporal_coverage.png  (Date range: {df["time"].min()} to {df["time"].max()}, Unique dates: {df["time"].nunique()})')

    # sm_aux vs sm_tgt
    sample = df.sample(n=5000, random_state=42)
    plt.figure(figsize=(8, 6))
    plt.scatter(sample['sm_aux'], sample['sm_tgt'], alpha=0.3, s=5)
    plt.xlabel('sm_aux (SMOS-ASCAT, m\u00b3/m\u00b3)')
    plt.ylabel('sm_tgt (AMSR, m\u00b3/m\u00b3)')
    plt.title('Auxiliary vs Target Soil Moisture')
    plt.plot([0, 1], [0, 1], 'r--', label='1:1 line')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/aux_vs_tgt_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: aux_vs_tgt_scatter.png')


# ============================================================================
# Phase 2: Feature Engineering
# ============================================================================

def run_feature_engineering(df, data_dir):
    print('\n' + '='*60)
    print('PHASE 2: Feature Engineering')
    print('='*60)

    # Temporal features
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
    print(f"Season counts:\n{df['season'].value_counts()}")

    # Soil texture ratios and interaction
    df['clay_sand_ratio'] = df['clay_content'] / df['sand_content'].replace(0, np.nan)
    df['clay_silt_ratio'] = df['clay_content'] / df['silt_content'].replace(0, np.nan)
    df['clay_x_sm_aux'] = df['clay_content'] * df['sm_aux']
    print(f"\nEngineered features:\n{df[['clay_sand_ratio', 'clay_silt_ratio', 'clay_x_sm_aux']].describe()}")

    # Spatial splits
    df['region'] = np.where(df['longitude'] < 10.5, 'West', 'East')
    print(f"\nRegion counts:\n{df['region'].value_counts()}")

    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    lat_bins = np.linspace(lat_min, lat_max + 0.01, 3)
    lon_bins = np.linspace(lon_min, lon_max + 0.01, 4)
    df['lat_band'] = pd.cut(df['latitude'], bins=lat_bins, labels=['S', 'N'])
    df['lon_band'] = pd.cut(df['longitude'], bins=lon_bins, labels=['W', 'C', 'E'])
    df['spatial_block'] = df['lat_band'].astype(str) + '_' + df['lon_band'].astype(str)
    print(f"\nSpatial block counts:\n{df['spatial_block'].value_counts()}")

    # Handle missing values
    print(f"\nMissing values before cleanup:\n{df.isnull().sum()}")
    rows_before = len(df)
    df_clean = df.dropna(subset=['sm_aux', 'sm_tgt', 'clay_sand_ratio', 'clay_silt_ratio'])
    print(f"Rows before: {rows_before}, after: {len(df_clean)}, dropped: {rows_before - len(df_clean)}")

    # Drop helper columns and export
    df_clean = df_clean.drop(columns=['lat_band', 'lon_band'])
    output_path = f'{data_dir}/processed_data.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"\nSaved processed data: {df_clean.shape} -> {output_path}")
    print(f"Columns: {list(df_clean.columns)}")

    return df_clean


# ============================================================================
# Phase 3: Modeling
# ============================================================================

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


def run_modeling(df, results_dir):
    print('\n' + '='*60)
    print('PHASE 3: Modeling')
    print('='*60)

    # Define features
    FEATURE_COLS = ['latitude', 'longitude', 'clay_content', 'sand_content', 'silt_content',
                    'sm_aux', 'day_of_year', 'clay_sand_ratio', 'clay_silt_ratio', 'clay_x_sm_aux']

    df = pd.get_dummies(df, columns=['season'], drop_first=True)
    season_cols = [c for c in df.columns if c.startswith('season_')]
    FEATURE_COLS += season_cols
    TARGET = 'sm_tgt'
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # Define models
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

    all_results = []

    # Experiment 1: Random 80/20 split
    print('\n--- Experiment 1: Random 80/20 Split ---')
    X = df[FEATURE_COLS].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    for name, model in models.items():
        print(f"  Training {name}...")
        results, _ = evaluate_model(clone(model), X_train, y_train, X_test, y_test)
        results['model'] = name
        results['split'] = 'Random 80/20'
        all_results.append(results)
        print(f"    Test RMSE: {results['test_rmse']:.4f}, Test R2: {results['test_r2']:.4f}")

    # Experiment 2: West/East spatial split
    print('\n--- Experiment 2: West/East Spatial Split ---')
    west_mask = df['region'] == 'West'
    east_mask = df['region'] == 'East'
    X_west = df.loc[west_mask, FEATURE_COLS].values
    y_west = df.loc[west_mask, TARGET].values
    X_east = df.loc[east_mask, FEATURE_COLS].values
    y_east = df.loc[east_mask, TARGET].values
    print(f"West (train): {X_west.shape[0]}, East (test): {X_east.shape[0]}")

    spatial_predictions = df.loc[east_mask, ['latitude', 'longitude']].copy()

    for name, model in models.items():
        print(f"  Training {name} (West->East)...")
        results, y_pred = evaluate_model(clone(model), X_west, y_west, X_east, y_east)
        results['model'] = name
        results['split'] = 'West->East'
        all_results.append(results)
        spatial_predictions[f'pred_{name}'] = y_pred
        print(f"    Test RMSE: {results['test_rmse']:.4f}, Test R2: {results['test_r2']:.4f}")

    spatial_predictions['actual'] = y_east

    # Experiment 3: Spatial block CV
    print('\n--- Experiment 3: Spatial Block Cross-Validation ---')
    blocks = df['spatial_block'].unique()
    print(f"Blocks: {blocks}")

    for name, model_template in models.items():
        print(f"  Block CV for {name}...")
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

            model = clone(model_template)
            res, _ = evaluate_model(model, X_tr, y_tr, X_te, y_te)
            block_results.append(res)
            print(f"    Block {block}: RMSE={res['test_rmse']:.4f}, R2={res['test_r2']:.4f}")

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
        print(f"    Average: RMSE={avg_results['test_rmse']:.4f}, R2={avg_results['test_r2']:.4f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['model', 'split', 'train_rmse', 'test_rmse', 'train_r2', 'test_r2', 'test_bias']]
    results_df.to_csv(f'{results_dir}/model_results.csv', index=False)
    print(f'\n=== All Results ===\n{results_df.to_string(index=False)}')

    spatial_predictions.to_csv(f'{results_dir}/predictions_spatial.csv', index=False)
    print(f'\nSaved: model_results.csv, predictions_spatial.csv')

    # Feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importance_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(f'{results_dir}/feature_importance.csv', index=False)
    print(f'\nFeature Importance:\n{importance_df.to_string(index=False)}')

    return results_df, spatial_predictions, importance_df


# ============================================================================
# Phase 4: Analysis & Visualization
# ============================================================================

def run_analysis(results, predictions, importance, fig_dir):
    print('\n' + '='*60)
    print('PHASE 4: Analysis & Visualization')
    print('='*60)

    # Model comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pivot_rmse = results.pivot(index='model', columns='split', values='test_rmse')
    pivot_rmse.plot(kind='bar', ax=axes[0], rot=15)
    axes[0].set_title('Test RMSE by Model and Split Strategy')
    axes[0].set_ylabel('RMSE (m\u00b3/m\u00b3)')
    axes[0].legend(title='Split')

    pivot_r2 = results.pivot(index='model', columns='split', values='test_r2')
    pivot_r2.plot(kind='bar', ax=axes[1], rot=15)
    axes[1].set_title('Test R\u00b2 by Model and Split Strategy')
    axes[1].set_ylabel('R\u00b2')
    axes[1].legend(title='Split')

    plt.suptitle('Model Comparison Across Split Strategies', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: model_comparison.png')

    # Overfitting analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    random_results = results[results['split'] == 'Random 80/20']
    x = np.arange(len(random_results))
    width = 0.35
    ax.bar(x - width/2, random_results['train_r2'], width, label='Train R\u00b2', color='steelblue')
    ax.bar(x + width/2, random_results['test_r2'], width, label='Test R\u00b2', color='coral')
    ax.set_xlabel('Model')
    ax.set_ylabel('R\u00b2')
    ax.set_title('Train vs Test R\u00b2 (Random Split) \u2014 Overfitting Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(random_results['model'], rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/overfitting_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: overfitting_analysis.png')

    # Feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance['feature'], importance['importance'], color='teal')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: feature_importance.png')

    # Spatial error map
    best_model = results.loc[results[results['split'] == 'West->East']['test_rmse'].idxmin(), 'model']
    print(f'Best model (West->East): {best_model}')
    pred_col = f'pred_{best_model}'
    predictions['error'] = predictions[pred_col] - predictions['actual']

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(predictions['longitude'], predictions['latitude'],
                          c=predictions['error'], cmap='RdBu_r', s=3, alpha=0.5,
                          vmin=-0.2, vmax=0.2)
    plt.colorbar(scatter, label='Prediction Error (m\u00b3/m\u00b3)')
    plt.xlabel('Longitude (\u00b0)')
    plt.ylabel('Latitude (\u00b0)')
    plt.title(f'Spatial Error Map \u2014 {best_model} (Train: West, Test: East)')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/spatial_error_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: spatial_error_map.png')

    # Residual distribution
    plt.figure(figsize=(8, 5))
    plt.hist(predictions['error'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(x=0, color='red', linestyle='--', label='Zero error')
    plt.xlabel('Prediction Error (m\u00b3/m\u00b3)')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution \u2014 {best_model} (West\u2192East)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/residual_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: residual_distribution.png')
    print(f'Mean error (bias): {predictions["error"].mean():.4f}')
    print(f'Std of error: {predictions["error"].std():.4f}')

    # Spatial degradation chart
    split_order = ['Random 80/20', 'West->East', 'Spatial Block CV']
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name in results['model'].unique():
        model_data = results[results['model'] == model_name]
        model_data = model_data.set_index('split').loc[split_order]
        ax.plot(split_order, model_data['test_r2'], marker='o', label=model_name)
    ax.set_xlabel('Split Strategy')
    ax.set_ylabel('Test R\u00b2')
    ax.set_title('Spatial Generalization \u2014 R\u00b2 Degradation Across Split Strategies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/spatial_degradation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: spatial_degradation.png')


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    data_dir, fig_dir, results_dir = setup_paths(args)

    print(f'Data dir:    {data_dir}')
    print(f'Figures dir: {fig_dir}')
    print(f'Results dir: {results_dir}')

    # Load raw data
    csv_path = f'{data_dir}/updated_data.csv'
    if not os.path.exists(csv_path):
        print(f'\nERROR: {csv_path} not found.')
        print('Make sure updated_data.csv is in the data directory.')
        sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=['time'])

    # Run pipeline
    run_eda(df, fig_dir)
    df_processed = run_feature_engineering(df, data_dir)
    results_df, predictions, importance = run_modeling(df_processed, results_dir)
    run_analysis(results_df, predictions, importance, fig_dir)

    print('\n' + '='*60)
    print('PIPELINE COMPLETE')
    print(f'Figures saved to: {fig_dir}')
    print(f'Results saved to: {results_dir}')
    print('='*60)


if __name__ == '__main__':
    main()
