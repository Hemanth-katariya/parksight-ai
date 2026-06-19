"""
ParkSight — Data Pipeline
Load, clean, and feature-engineer parking violation data.
"""

import os
import ast
import json
import logging

import pandas as pd

try:
    from src import h3_latlng_to_cell
except ImportError:
    import h3
    h3_latlng_to_cell = h3.geo_to_h3

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
PARQUET_PATH = os.path.join(CACHE_DIR, 'violations_processed.parquet')
DEFAULT_DATA = os.path.join(DATA_DIR, 'raw_dataset.parquet')


def _safe_parse_violation(x):
    """Parse JSON-like violation_type string into a list."""
    if pd.isna(x):
        return []
    try:
        result = ast.literal_eval(str(x))
        if isinstance(result, list):
            return result
        return [str(result)]
    except (ValueError, SyntaxError):
        return [str(x)] if x else []


def _safe_h3(lat, lon, resolution=8):
    """Safely compute H3 index, returning None on error."""
    try:
        return h3_latlng_to_cell(lat, lon, resolution)
    except Exception:
        return None


def load_and_clean(data_path: str = None) -> pd.DataFrame:
    """
    Load raw Data (Parquet/CSV), clean data, engineer features, and cache as parquet.

    Parameters
    ----------
    data_path : str, optional
        Path to the raw file. Defaults to data/raw_dataset.parquet

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-engineered dataframe.
    """
    if data_path is None:
        data_path = DEFAULT_DATA

    logger.info('Loading Data from %s', data_path)
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, low_memory=False)
    logger.info('Loaded %d rows', len(df))

    # ── Parse timestamps ────────────────────────────────────────
    df['created_datetime'] = pd.to_datetime(
        df['created_datetime'], utc=True, errors='coerce'
    )

    # ── Temporal features ───────────────────────────────────────
    df['hour'] = df['created_datetime'].dt.hour
    df['day_of_week'] = df['created_datetime'].dt.dayofweek
    df['month'] = df['created_datetime'].dt.month
    df['date'] = df['created_datetime'].dt.date
    df['is_peak'] = df['hour'].isin([7, 8, 9, 17, 18, 19, 20])

    # ── Parse violation_type ────────────────────────────────────
    _parsed = df['violation_type'].apply(_safe_parse_violation)
    df['primary_violation'] = _parsed.apply(
        lambda x: x[0] if x else 'UNKNOWN'
    )
    df['violation_count_per_record'] = _parsed.apply(len)
    # Store as JSON string (not Python list) for Parquet & Streamlit hash compat
    df['violation_list'] = _parsed.apply(json.dumps)

    # ── Clean junction_name ─────────────────────────────────────
    df.loc[df['junction_name'] == 'No Junction', 'junction_name'] = pd.NA

    # ── Drop invalid coordinates ────────────────────────────────
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
    logger.info('Rows after coordinate cleaning: %d', len(df))

    # ── H3 hexagonal index (resolution 8, ~460 m edge) ─────────
    df['h3_index'] = df.apply(
        lambda r: _safe_h3(r['latitude'], r['longitude'], 8), axis=1
    )
    df = df.dropna(subset=['h3_index'])

    # ── Repeat offenders ────────────────────────────────────────
    vehicle_counts = df.groupby('vehicle_number').size()
    df['is_repeat_offender'] = df['vehicle_number'].map(vehicle_counts) >= 3

    # ── Persist to parquet ──────────────────────────────────────
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(PARQUET_PATH, engine='pyarrow', index=False)
        logger.info('Saved processed data to %s', PARQUET_PATH)
    except Exception as e:
        logger.warning(f"Could not save processed data to parquet cache: {str(e)}")

    return df


def load_processed(data_path: str = None) -> pd.DataFrame:
    """
    Return processed dataframe — from parquet cache if available,
    otherwise run the full pipeline.
    """
    if os.path.exists(PARQUET_PATH):
        try:
            logger.info('Loading from parquet cache')
            df = pd.read_parquet(PARQUET_PATH)
            
            # Validate cache integrity
            if len(df) < 100 or 'created_datetime' not in df.columns:
                raise ValueError("Corrupt parquet cache")
                
            # Ensure boolean columns after parquet round-trip
            if 'is_peak' in df.columns:
                df['is_peak'] = df['is_peak'].astype(bool)
            if 'is_repeat_offender' in df.columns:
                df['is_repeat_offender'] = df['is_repeat_offender'].astype(bool)
            return df
        except Exception as e:
            logger.warning(f"Failed to load parquet cache ({str(e)}). Regenerating...")
            try:
                os.remove(PARQUET_PATH)
            except Exception:
                pass
                
    return load_and_clean(data_path)
