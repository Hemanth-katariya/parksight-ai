"""
ParkSight — Hotspot Engine
H3 hexagonal binning and DBSCAN density clustering.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


def compute_h3_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate violations into H3 resolution-8 hexagonal grid cells.

    Returns a DataFrame with one row per H3 cell, including density
    and peak-hour scores.
    """
    grid = df.groupby('h3_index').agg(
        violation_count=('id', 'count'),
        peak_violations=('is_peak', 'sum'),
        repeat_offenders=('is_repeat_offender', 'sum'),
        unique_vehicles=('vehicle_number', 'nunique'),
        lat=('latitude', 'mean'),
        lon=('longitude', 'mean'),
    ).reset_index()

    # Normalised density score 0 → 1
    vmin = grid['violation_count'].min()
    vmax = grid['violation_count'].max()
    if vmax == vmin:
        grid['density_score'] = 0.5
    else:
        grid['density_score'] = (
            (grid['violation_count'] - vmin) / (vmax - vmin)
        )

    # Fraction of violations during peak hours
    grid['peak_hour_weight'] = (
        grid['peak_violations'] / grid['violation_count']
    ).fillna(0)

    logger.info('H3 grid computed — %d cells', len(grid))
    return grid


def run_dbscan_clustering(
    df: pd.DataFrame,
    eps_km: float = 0.3,
    min_samples: int = 50,
) -> tuple:
    """
    Run DBSCAN clustering on violation coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'latitude' and 'longitude' columns.
    eps_km : float
        Neighbourhood radius in kilometres.
    min_samples : int
        Minimum points to form a core sample.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Original df with 'cluster_id' column, and a cluster_stats
        summary DataFrame.
    """
    coords = np.radians(df[['latitude', 'longitude']].values)
    eps_rad = eps_km / 6371.0  # Earth radius in km

    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric='haversine',
    ).fit(coords)

    df = df.copy()
    df['cluster_id'] = db.labels_

    clustered = df[df['cluster_id'] >= 0]
    cluster_stats = (
        clustered.groupby('cluster_id')
        .agg(
            cluster_size=('id', 'count'),
            cluster_lat=('latitude', 'mean'),
            cluster_lon=('longitude', 'mean'),
            peak_fraction=('is_peak', 'mean'),
        )
        .reset_index()
    )

    logger.info(
        'DBSCAN found %d clusters from %d points',
        len(cluster_stats),
        len(clustered),
    )
    return df, cluster_stats
