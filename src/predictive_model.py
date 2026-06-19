"""
ParkSight — Predictive AI Model
XGBoost-based violation hotspot prediction engine.

Trains on historical violation patterns to forecast future violations
by junction, enabling proactive (not reactive) enforcement.
"""

import os
import logging
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
MODEL_CACHE = os.path.join(CACHE_DIR, 'xgb_model.pkl')
METRICS_CACHE = os.path.join(CACHE_DIR, 'model_metrics.pkl')


def _prepare_junction_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate violations to junction × date level and engineer
    temporal and spatial features for XGBoost training.

    Returns a DataFrame with one row per (junction, date) pair.
    """
    # Only use rows with named junctions
    jdf = df[df['junction_name'].notna()].copy()
    jdf['date'] = pd.to_datetime(jdf['date'])

    # ── Aggregate: daily counts per junction ────────────────
    # Ensure road_weight column exists before aggregation
    if 'road_weight' not in jdf.columns:
        jdf['road_weight'] = 0.5

    daily = (
        jdf.groupby(['junction_name', 'date'])
        .agg(
            violation_count=('id', 'count'),
            peak_violations=('is_peak', 'sum'),
            repeat_offenders=('is_repeat_offender', 'sum'),
            unique_vehicles=('vehicle_number', 'nunique'),
            avg_road_weight=('road_weight', 'mean'),
            avg_lat=('latitude', 'mean'),
            avg_lon=('longitude', 'mean'),
        )
        .reset_index()
    )

    # ── Temporal features ───────────────────────────────────
    daily['day_of_week'] = daily['date'].dt.dayofweek
    daily['month'] = daily['date'].dt.month
    daily['is_weekend'] = daily['day_of_week'].isin([5, 6]).astype(int)
    daily['day_of_month'] = daily['date'].dt.day

    # ── Peak fraction ───────────────────────────────────────
    daily['peak_fraction'] = (
        daily['peak_violations'] / daily['violation_count']
    ).fillna(0)

    # ── Repeat offender fraction ────────────────────────────
    daily['repeat_fraction'] = (
        daily['repeat_offenders'] / daily['violation_count']
    ).fillna(0)

    # ── Historical features per junction ────────────────────
    # Rolling averages (lag features)
    daily = daily.sort_values(['junction_name', 'date'])

    # Compute lag features grouped by junction
    for lag in [1, 3, 7]:
        daily[f'lag_{lag}d'] = (
            daily.groupby('junction_name')['violation_count']
            .shift(lag)
        )

    # Rolling 7-day mean and std
    daily['rolling_7d_mean'] = (
        daily.groupby('junction_name')['violation_count']
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    )
    daily['rolling_7d_std'] = (
        daily.groupby('junction_name')['violation_count']
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())
    ).fillna(0)

    # Junction-level historical average (overall mean for this junction)
    junction_means = daily.groupby('junction_name')['violation_count'].mean()
    daily['junction_historical_mean'] = daily['junction_name'].map(junction_means)

    # Drop rows with NaN lag features (first few days per junction)
    daily = daily.dropna(subset=['lag_1d'])

    logger.info('Prepared %d training rows from %d junctions',
                len(daily), daily['junction_name'].nunique())
    return daily


FEATURE_COLS = [
    'day_of_week', 'month', 'is_weekend', 'day_of_month',
    'avg_road_weight', 'peak_fraction', 'repeat_fraction',
    'lag_1d', 'lag_3d', 'lag_7d',
    'rolling_7d_mean', 'rolling_7d_std',
    'junction_historical_mean',
]


def train_model(df: pd.DataFrame, force_retrain: bool = False) -> dict:
    """
    Train (or load cached) XGBoost model for violation count prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Full violation dataframe with all features.
    force_retrain : bool
        If True, retrain even if cache exists.

    Returns
    -------
    dict with keys:
        'model': trained XGBRegressor
        'metrics': dict of train/test metrics (MAE, RMSE, R²)
        'feature_importance': pd.DataFrame with feature name + importance
        'training_data': prepared daily DataFrame
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check cache
    if not force_retrain and os.path.exists(MODEL_CACHE) and os.path.exists(METRICS_CACHE):
        logger.info('Loading cached XGBoost model')
        try:
            with open(MODEL_CACHE, 'rb') as f:
                model = pickle.load(f)
            with open(METRICS_CACHE, 'rb') as f:
                cached = pickle.load(f)
            # Still need training_data for predictions
            daily = _prepare_junction_daily(df)
            cached['model'] = model
            cached['training_data'] = daily
            return cached
        except Exception as e:
            logger.warning('Cache load failed (%s), retraining', e)

    # ── Prepare data ────────────────────────────────────────
    daily = _prepare_junction_daily(df)

    if len(daily) < 50:
        logger.warning('Insufficient data for training (%d rows)', len(daily))
        return {'model': None, 'metrics': {}, 'feature_importance': pd.DataFrame(), 'training_data': daily}

    # ── Time-based train/test split ─────────────────────────
    # Use last 20% of dates as test set
    dates_sorted = sorted(daily['date'].unique())
    split_idx = int(len(dates_sorted) * 0.8)
    split_date = dates_sorted[split_idx]

    train = daily[daily['date'] < split_date]
    test = daily[daily['date'] >= split_date]

    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train['violation_count']
    X_test = test[FEATURE_COLS].fillna(0)
    y_test = test['violation_count']

    logger.info('Training XGBoost: %d train / %d test rows', len(train), len(test))

    # ── Train XGBoost ───────────────────────────────────────
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ────────────────────────────────────────────
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        'train_mae': round(mean_absolute_error(y_train, y_pred_train), 2),
        'test_mae': round(mean_absolute_error(y_test, y_pred_test), 2),
        'train_rmse': round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 2),
        'test_rmse': round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2),
        'train_r2': round(r2_score(y_train, y_pred_train), 4),
        'test_r2': round(r2_score(y_test, y_pred_test), 4),
        'train_size': len(train),
        'test_size': len(test),
        'n_junctions': daily['junction_name'].nunique(),
        'date_range': f"{dates_sorted[0].strftime('%Y-%m-%d')} to {dates_sorted[-1].strftime('%Y-%m-%d')}",
        'split_date': split_date.strftime('%Y-%m-%d'),
    }

    logger.info(
        'Model trained — Test R²=%.4f, MAE=%.2f, RMSE=%.2f',
        metrics['test_r2'], metrics['test_mae'], metrics['test_rmse'],
    )

    # ── Feature importance ──────────────────────────────────
    importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    # ── Cache model + metrics ───────────────────────────────
    with open(MODEL_CACHE, 'wb') as f:
        pickle.dump(model, f)
    with open(METRICS_CACHE, 'wb') as f:
        pickle.dump({
            'metrics': metrics,
            'feature_importance': importance,
        }, f)

    return {
        'model': model,
        'metrics': metrics,
        'feature_importance': importance,
        'training_data': daily,
    }


def predict_future_violations(
    model: XGBRegressor,
    training_data: pd.DataFrame,
    target_date: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Predict violation counts for each junction on a target date.

    Uses the most recent lag features from training data to project
    forward. Returns a DataFrame of predicted violations per junction
    sorted by descending prediction.

    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    training_data : pd.DataFrame
        Output of _prepare_junction_daily (needed for lag features).
    target_date : pd.Timestamp, optional
        Date to predict for. Defaults to the day after the last
        training date.

    Returns
    -------
    pd.DataFrame with columns:
        junction_name, predicted_violations, risk_level, lat, lon, ...
    """
    if model is None:
        return pd.DataFrame()

    if target_date is None:
        target_date = training_data['date'].max() + pd.Timedelta(days=1)

    target_date = pd.Timestamp(target_date)

    # ── Build feature rows for each junction ────────────────
    junctions = training_data['junction_name'].unique()
    rows = []

    for jname in junctions:
        jdata = training_data[training_data['junction_name'] == jname].sort_values('date')
        if len(jdata) < 3:
            continue

        last = jdata.iloc[-1]
        recent_7 = jdata.tail(7)

        row = {
            'junction_name': jname,
            'day_of_week': target_date.dayofweek,
            'month': target_date.month,
            'is_weekend': int(target_date.dayofweek in [5, 6]),
            'day_of_month': target_date.day,
            'avg_road_weight': last['avg_road_weight'],
            'peak_fraction': last['peak_fraction'],
            'repeat_fraction': last['repeat_fraction'],
            'lag_1d': last['violation_count'],
            'lag_3d': jdata.tail(3)['violation_count'].iloc[0] if len(jdata) >= 3 else last['violation_count'],
            'lag_7d': jdata.tail(7)['violation_count'].iloc[0] if len(jdata) >= 7 else last['violation_count'],
            'rolling_7d_mean': recent_7['violation_count'].mean(),
            'rolling_7d_std': recent_7['violation_count'].std() if len(recent_7) > 1 else 0,
            'junction_historical_mean': last['junction_historical_mean'],
            'avg_lat': last['avg_lat'],
            'avg_lon': last['avg_lon'],
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    pred_df = pd.DataFrame(rows)
    X_pred = pred_df[FEATURE_COLS].fillna(0)

    # ── Predict ─────────────────────────────────────────────
    pred_df['predicted_violations'] = np.maximum(0, model.predict(X_pred)).round(1)

    # ── Risk level categorisation ───────────────────────────
    q75 = pred_df['predicted_violations'].quantile(0.75)
    q90 = pred_df['predicted_violations'].quantile(0.90)
    pred_df['risk_level'] = pd.cut(
        pred_df['predicted_violations'],
        bins=[-1, q75 * 0.5, q75, q90, float('inf')],
        labels=['Low', 'Medium', 'High', 'Critical'],
    )

    pred_df['prediction_date'] = target_date.strftime('%Y-%m-%d')

    # Sort by predicted violations descending
    pred_df = (
        pred_df.sort_values('predicted_violations', ascending=False)
        .reset_index(drop=True)
    )
    pred_df['rank'] = range(1, len(pred_df) + 1)

    logger.info(
        'Predicted violations for %d junctions on %s — top: %s (%.1f)',
        len(pred_df),
        target_date.strftime('%Y-%m-%d'),
        pred_df.iloc[0]['junction_name'] if len(pred_df) else 'N/A',
        pred_df.iloc[0]['predicted_violations'] if len(pred_df) else 0,
    )

    return pred_df
