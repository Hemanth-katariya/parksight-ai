"""
ParkSight — Enforcement Priority Index (EPI) Scorer
Composite scoring system for junction-level enforcement prioritisation.

EPI = (Violation Density × 0.40)
    + (Peak Hour Weight × 0.30)
    + (Road Class Weight × 0.20)
    + (Repeat Offender Rate × 0.10)
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_junction_epi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Enforcement Priority Index for every named junction.

    Only junctions with a non-null, non-'No Junction' name are scored.
    If *road_weight* is missing from the input frame (e.g. OSM data
    unavailable), a fallback value of 0.5 is used.

    Returns a DataFrame sorted by descending EPI score with a
    1-based rank column.
    """
    # ── Filter to named junctions ───────────────────────────────
    jdf = df[df['junction_name'].notna()].copy()

    if jdf.empty:
        logger.warning('No named junctions found — returning empty EPI table')
        return pd.DataFrame()

    # Ensure road_weight exists (fallback if OSM was unavailable)
    if 'road_weight' not in jdf.columns:
        jdf['road_weight'] = 0.5

    # ── Aggregate per junction ──────────────────────────────────
    junctions = jdf.groupby('junction_name').agg(
        total_violations=('id', 'count'),
        peak_violations=('is_peak', 'sum'),
        repeat_offender_count=('is_repeat_offender', 'sum'),
        unique_vehicles=('vehicle_number', 'nunique'),
        avg_road_weight=('road_weight', 'mean'),
        lat=('latitude', 'mean'),
        lon=('longitude', 'mean'),
        police_station=(
            'police_station',
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
        ),
    ).reset_index()

    # ── Component scores (all 0 → 1) ───────────────────────────
    junctions['density_score'] = _normalize(junctions['total_violations'])

    junctions['peak_hour_weight'] = (
        junctions['peak_violations'] / junctions['total_violations']
    ).fillna(0)

    junctions['road_class_weight'] = junctions['avg_road_weight']

    raw_repeat = (
        junctions['repeat_offender_count'] / junctions['unique_vehicles']
    ).fillna(0)
    junctions['repeat_offender_rate'] = _normalize(raw_repeat)

    # ── Composite EPI ───────────────────────────────────────────
    junctions['epi'] = (
        junctions['density_score'] * 0.40
        + junctions['peak_hour_weight'] * 0.30
        + junctions['road_class_weight'] * 0.20
        + junctions['repeat_offender_rate'] * 0.10
    )
    junctions['epi_score'] = (junctions['epi'] * 100).round(1)

    # ── Rank ────────────────────────────────────────────────────
    junctions = junctions.sort_values('epi_score', ascending=False).reset_index(drop=True)
    junctions['rank'] = range(1, len(junctions) + 1)

    logger.info(
        'EPI computed for %d junctions — top: %s (%.1f)',
        len(junctions),
        junctions.iloc[0]['junction_name'] if len(junctions) else 'N/A',
        junctions.iloc[0]['epi_score'] if len(junctions) else 0,
    )
    return junctions
