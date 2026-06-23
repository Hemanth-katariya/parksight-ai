"""
ParkSight — Full Dataset Pre-computation Script
=================================================
Run this ONCE on your local machine to process the entire 290k+ row
dataset. It generates lightweight pre-computed result files in the
data/ directory that the Streamlit app reads instantly on startup.

Usage:
    cd parksight
    .\\venv\\Scripts\\python.exe process_full_dataset.py

What it does:
    1. Loads the full 109MB raw CSV
    2. Runs data_pipeline to clean + feature-engineer all rows
    3. Saves processed violations as a compact Parquet file
    4. Trains the XGBoost model on the FULL dataset and caches it
    5. Generates predictions and saves them
    6. Computes EPI scores, DBSCAN clusters, H3 grid, congestion
    7. Saves ALL results into data/precomputed/ for Streamlit Cloud

After running, push the updated data/ directory to GitHub.
The deployed Streamlit app will load these pre-computed results
instantly with zero heavy computation.
"""

import os
import sys
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Ensure project root is on the path ──────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from src import data_pipeline, hotspot_engine, epi_scorer, osm_fetcher
from src import predictive_model, network_analysis

CACHE_DIR = os.path.join(ROOT, 'cache')
DATA_DIR = os.path.join(ROOT, 'data')
PRECOMPUTED_DIR = os.path.join(DATA_DIR, 'precomputed')
RAW_CSV = os.path.join(DATA_DIR, 'jan_to_may_police_violation_anonymized.csv')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PRECOMPUTED_DIR, exist_ok=True)


def main():
    # ════════════════════════════════════════════════════════════
    #  Step 1: Load & process the FULL raw CSV
    # ════════════════════════════════════════════════════════════
    logger.info('='*60)
    logger.info('STEP 1: Loading and processing the full raw CSV...')
    logger.info('='*60)

    if not os.path.exists(RAW_CSV):
        logger.error(f'Raw CSV not found at: {RAW_CSV}')
        logger.error('Please ensure the full dataset CSV is in the data/ directory.')
        sys.exit(1)

    # Delete old processed cache so data_pipeline regenerates from the full CSV
    processed_parquet = os.path.join(CACHE_DIR, 'violations_processed.parquet')
    if os.path.exists(processed_parquet):
        os.remove(processed_parquet)
        logger.info('Removed old processed cache to force regeneration.')

    df = data_pipeline.load_and_clean(RAW_CSV)
    logger.info(f'✅ Processed {len(df):,} rows from full dataset.')

    # ════════════════════════════════════════════════════════════
    #  Step 2: Save compact raw_dataset.parquet for Streamlit
    # ════════════════════════════════════════════════════════════
    logger.info('='*60)
    logger.info('STEP 2: Saving compact raw_dataset.parquet for Streamlit...')
    logger.info('='*60)

    raw_parquet = os.path.join(DATA_DIR, 'raw_dataset.parquet')
    df.to_parquet(raw_parquet, engine='pyarrow', index=False)
    size_mb = os.path.getsize(raw_parquet) / (1024 * 1024)
    logger.info(f'✅ Saved raw_dataset.parquet ({size_mb:.1f} MB) with all {len(df):,} rows.')

    # ════════════════════════════════════════════════════════════
    #  Step 3: Assign OSM road classes
    # ════════════════════════════════════════════════════════════
    logger.info('='*60)
    logger.info('STEP 3: Fetching OSM road network and assigning road weights...')
    logger.info('='*60)

    edges = osm_fetcher.get_bengaluru_road_network()
    df = osm_fetcher.assign_road_class_to_violations(df, edges)
    logger.info(f'✅ Road weights assigned to {len(df):,} rows.')

    # ════════════════════════════════════════════════════════════
    #  Step 4: Compute H3 grid & save
    # ════════════════════════════════════════════════════════════
    logger.info('='*60)
    logger.info('STEP 4: Computing H3 hexagonal grid...')
    logger.info('='*60)

    h3_grid = hotspot_engine.compute_h3_grid(df)
    h3_grid.to_parquet(os.path.join(PRECOMPUTED_DIR, 'h3_grid.parquet'), index=False)
    logger.info(f'✅ H3 grid computed and saved ({len(h3_grid)} hexagons).')

    # ════════════════════════════════════════════════════════════
    #  Step 5: Run DBSCAN clustering & save ONLY cluster_stats
    # ════════════════════════════════════════════════════════════
    logger.info('='*60)
    logger.info('STEP 5: Running DBSCAN density clustering...')
    logger.info('='*60)

    df_clustered, cluster_stats = hotspot_engine.run_dbscan_clustering(df)
    # Save only cluster_stats (tiny), NOT the full df_clustered (166MB)
    cluster_stats.to_parquet(os.path.join(PRECOMPUTED_DIR, 'cluster_stats.parquet'), index=False)
    cluster_count = len(cluster_stats) if cluster_stats is not None and not cluster_stats.empty else 0
    logger.info(f'✅ DBSCAN found {cluster_count} spatial clusters. Saved cluster_stats only.')

    # ════════════════════════════════════════════════════════════
    #  Step 6: Compute EPI scores & save
    # ════════════════════════════════════════════════════════════
    logger.info('='*60)
    logger.info('STEP 6: Computing Enforcement Priority Index (EPI) scores...')
    logger.info('='*60)

    junctions_epi = epi_scorer.compute_junction_epi(df)
    junctions_epi.to_parquet(os.path.join(PRECOMPUTED_DIR, 'epi_scores.parquet'), index=False)
    logger.info(f'✅ EPI scores computed for {len(junctions_epi)} junctions.')

    # ════════════════════════════════════════════════════════════
    #  Step 7: Train XGBoost model on FULL dataset
    # ════════════════════════════════════════════════════════════
    logger.info('='*60)
    logger.info('STEP 7: Training XGBoost model on full dataset...')
    logger.info('='*60)

    # Delete old model cache to force retraining
    model_cache = os.path.join(CACHE_DIR, 'xgb_model.pkl')
    metrics_cache = os.path.join(CACHE_DIR, 'model_metrics.pkl')
    for f in [model_cache, metrics_cache]:
        if os.path.exists(f):
            os.remove(f)

    xgb_result = predictive_model.train_model(df, force_retrain=True)
    metrics = xgb_result.get('metrics', {})

    # Save model and metrics to precomputed dir too
    with open(os.path.join(PRECOMPUTED_DIR, 'xgb_model.pkl'), 'wb') as f:
        pickle.dump(xgb_result['model'], f)
    with open(os.path.join(PRECOMPUTED_DIR, 'model_metrics.pkl'), 'wb') as f:
        pickle.dump({
            'metrics': metrics,
            'feature_importance': xgb_result.get('feature_importance', pd.DataFrame()),
        }, f)

    logger.info(f'✅ XGBoost trained — R²={metrics.get("test_r2", "N/A")}, '
                f'MAE={metrics.get("test_mae", "N/A")}, '
                f'Train={metrics.get("train_size", "?")}, '
                f'Test={metrics.get("test_size", "?")}')

    # ════════════════════════════════════════════════════════════
    #  Step 8: Generate predictions & save
    # ════════════════════════════════════════════════════════════
    logger.info('='*60)
    logger.info('STEP 8: Generating violation predictions...')
    logger.info('='*60)

    predictions = pd.DataFrame()
    if xgb_result.get('model') is not None:
        predictions = predictive_model.predict_future_violations(
            xgb_result['model'],
            xgb_result['training_data'],
        )
        predictions.to_parquet(os.path.join(PRECOMPUTED_DIR, 'predictions.parquet'), index=False)
        logger.info(f'✅ Predictions generated for {len(predictions)} junctions.')
    else:
        logger.warning('⚠️ No model available — skipping predictions.')

    # ════════════════════════════════════════════════════════════
    #  Step 9: Compute Network Centrality & Congestion Impact
    # ════════════════════════════════════════════════════════════
    logger.info('='*60)
    logger.info('STEP 9: Computing network centrality & congestion impact...')
    logger.info('='*60)

    centrality = network_analysis.compute_node_centrality(k=80)
    df_with_cis = network_analysis.compute_congestion_impact(df, centrality)
    congestion_summary = network_analysis.get_junction_congestion_summary(df_with_cis)
    congestion_summary.to_parquet(os.path.join(PRECOMPUTED_DIR, 'congestion_summary.parquet'), index=False)
    logger.info(f'✅ Congestion impact computed for {len(congestion_summary)} junctions.')

    # ════════════════════════════════════════════════════════════
    #  Step 10: Save a manifest
    # ════════════════════════════════════════════════════════════
    import json
    manifest = {
        'total_rows': len(df),
        'n_junctions': len(junctions_epi),
        'n_predictions': len(predictions),
        'n_clusters': cluster_count,
        'n_h3_cells': len(h3_grid),
        'model_r2': metrics.get('test_r2'),
        'model_mae': metrics.get('test_mae'),
        'precomputed': True,
    }
    with open(os.path.join(PRECOMPUTED_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    # ════════════════════════════════════════════════════════════
    #  DONE
    # ════════════════════════════════════════════════════════════
    logger.info('='*60)
    logger.info('🎉 ALL PRE-COMPUTATION COMPLETE!')
    logger.info('='*60)
    logger.info(f'Total rows processed: {len(df):,}')
    logger.info(f'Junctions with EPI: {len(junctions_epi)}')
    logger.info(f'Junctions with predictions: {len(predictions)}')
    logger.info(f'DBSCAN clusters: {cluster_count}')
    logger.info(f'XGBoost R² score: {metrics.get("test_r2", "N/A")}')

    # Show file sizes
    logger.info('')
    logger.info('Pre-computed files saved:')
    for fname in os.listdir(PRECOMPUTED_DIR):
        fpath = os.path.join(PRECOMPUTED_DIR, fname)
        size = os.path.getsize(fpath) / 1024
        logger.info(f'  {fname}: {size:.1f} KB')

    logger.info('')
    logger.info('Next steps:')
    logger.info('  1. git add data/precomputed/ data/raw_dataset.parquet app.py')
    logger.info('  2. git commit -m "Full 290k dataset with pre-computed AI results"')
    logger.info('  3. git push')
    logger.info('')
    logger.info('The Streamlit app will now load these results instantly! 🚀')


if __name__ == '__main__':
    main()
