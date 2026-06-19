"""
ParkSight — Temporal Analysis & Patrol Recommendations
Time-based violation patterns and time-aware dispatch queue.
"""

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
PEAK_HOURS = {7, 8, 9, 17, 18, 19, 20}


def get_hourly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Violations aggregated by hour of day (0-23)."""
    hourly = (
        df.groupby('hour')
        .agg(violation_count=('id', 'count'))
        .reindex(range(24), fill_value=0)
        .reset_index()
    )
    hourly.columns = ['hour', 'violation_count']
    return hourly


def get_daily_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Violations aggregated by day of week."""
    daily = (
        df.groupby('day_of_week')
        .agg(violation_count=('id', 'count'))
        .reindex(range(7), fill_value=0)
        .reset_index()
    )
    daily['day_name'] = daily['day_of_week'].map(
        dict(enumerate(DAY_NAMES))
    )
    return daily[['day_name', 'violation_count']]


def get_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Daily violation time series."""
    monthly = (
        df.groupby('date')
        .agg(violation_count=('id', 'count'))
        .reset_index()
        .sort_values('date')
    )
    return monthly


def get_patrol_recommendation(
    df: pd.DataFrame,
    junctions_epi: pd.DataFrame,
    current_hour: int | None = None,
) -> pd.DataFrame:
    """
    Generate a time-aware patrol dispatch queue.

    Merges current-hour violation counts with the EPI score,
    optionally boosted during peak hours.

    Parameters
    ----------
    df : pd.DataFrame
        Full violation frame (must include 'hour', 'junction_name').
    junctions_epi : pd.DataFrame
        Output of ``epi_scorer.compute_junction_epi``.
    current_hour : int, optional
        Hour to evaluate (0-23). Defaults to now.

    Returns
    -------
    pd.DataFrame
        Top-10 junctions with patrol priority.
    """
    if current_hour is None:
        current_hour = datetime.now().hour

    is_peak_now = current_hour in PEAK_HOURS

    # Violations at this hour across all dates
    hour_df = df[df['hour'] == current_hour]
    hour_junctions = (
        hour_df[hour_df['junction_name'].notna()]
        .groupby('junction_name')
        .agg(hour_violations=('id', 'count'))
        .reset_index()
    )

    if junctions_epi.empty or hour_junctions.empty:
        logger.info('Insufficient data for patrol recommendation')
        return pd.DataFrame(columns=[
            'rank', 'junction_name', 'police_station', 'epi_score',
            'time_adjusted_priority', 'total_violations',
            'hour_violations', 'is_peak_now',
        ])

    merged = hour_junctions.merge(junctions_epi, on='junction_name', how='inner')

    if is_peak_now:
        merged['time_adjusted_priority'] = (merged['epi_score'] * 1.3).clip(upper=100).round(1)
    else:
        merged['time_adjusted_priority'] = merged['epi_score']

    merged = merged.sort_values('time_adjusted_priority', ascending=False).head(10)
    merged = merged.reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)
    merged['is_peak_now'] = is_peak_now

    cols = [
        'rank', 'junction_name', 'police_station', 'epi_score',
        'time_adjusted_priority', 'total_violations',
        'hour_violations', 'is_peak_now',
    ]
    return merged[[c for c in cols if c in merged.columns]]


def get_vehicle_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Top 10 vehicle types by violation count."""
    vb = (
        df.groupby('vehicle_type')
        .agg(violation_count=('id', 'count'))
        .sort_values('violation_count', ascending=False)
        .head(10)
        .reset_index()
    )
    return vb
