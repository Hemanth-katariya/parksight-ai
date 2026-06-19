"""
ParkSight — OpenStreetMap Road Network Fetcher
Downloads and caches Bengaluru's driveable road network,
assigns road-class weights to violation records.
"""

import os
import pickle
import logging

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
ROAD_CACHE = os.path.join(CACHE_DIR, 'bengaluru_roads.pkl')

HIGHWAY_WEIGHTS = {
    'motorway': 1.0,
    'trunk': 1.0,
    'primary': 0.9,
    'secondary': 0.75,
    'tertiary': 0.6,
    'residential': 0.4,
    'service': 0.3,
    'unclassified': 0.3,
}


def _get_weight(hw) -> float:
    """Resolve highway tag (may be a list) to a numeric weight."""
    if isinstance(hw, list):
        hw = hw[0]
    hw_str = str(hw).lower()
    for key, weight in HIGHWAY_WEIGHTS.items():
        if key in hw_str:
            return weight
    return 0.3


def get_bengaluru_road_network() -> gpd.GeoDataFrame:
    """
    Fetch (or load from cache) Bengaluru's driveable road edges.

    Returns a GeoDataFrame with columns:
        geometry, highway, lanes, name, length, road_weight

    On network failure the function returns ``None`` and logs a warning.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Try cache first ─────────────────────────────────────────
    if os.path.exists(ROAD_CACHE):
        logger.info('Loading road network from cache')
        try:
            with open(ROAD_CACHE, 'rb') as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning('Cache load failed (%s) — re-downloading', exc)

    # ── Download via osmnx ──────────────────────────────────────
    try:
        import osmnx as ox

        logger.info('Downloading Bengaluru road network from OSM …')
        G = ox.graph_from_place(
            'Bengaluru, Karnataka, India',
            network_type='drive',
        )
        edges = ox.graph_to_gdfs(G, nodes=False)

        # Keep useful columns only
        keep_cols = ['geometry', 'highway', 'lanes', 'name', 'length']
        keep_cols = [c for c in keep_cols if c in edges.columns]
        edges = edges[keep_cols].copy()

        edges['road_weight'] = edges['highway'].apply(_get_weight)

        # Persist
        with open(ROAD_CACHE, 'wb') as f:
            pickle.dump(edges, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info('Road network cached (%d edges)', len(edges))
        return edges

    except Exception as exc:
        logger.warning(
            'OSM download failed: %s — road weights will use fallback (0.5)',
            exc,
        )
        return None


def assign_road_class_to_violations(
    df: pd.DataFrame,
    edges_gdf: gpd.GeoDataFrame | None,
) -> pd.DataFrame:
    """
    Spatially join each violation to its nearest road edge and
    copy over the *road_weight*.

    If *edges_gdf* is ``None`` (OSM unavailable), every row gets
    a fallback weight of 0.5.
    """
    df = df.copy()

    if edges_gdf is None or edges_gdf.empty:
        logger.info('No road network available — using fallback road_weight=0.5')
        df['road_weight'] = 0.5
        return df

    try:
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        violations_gdf = gpd.GeoDataFrame(
            df, geometry=geometry, crs='EPSG:4326'
        )

        # Ensure matching CRS
        if edges_gdf.crs is None:
            edges_gdf = edges_gdf.set_crs('EPSG:4326')
        elif edges_gdf.crs != violations_gdf.crs:
            edges_gdf = edges_gdf.to_crs('EPSG:4326')

        # Project to a metric CRS for accurate distance-based join
        try:
            violations_proj = violations_gdf.to_crs(epsg=32643)  # UTM 43N for Bengaluru
            edges_proj = edges_gdf[['geometry', 'road_weight']].to_crs(epsg=32643)
            joined = gpd.sjoin_nearest(
                violations_proj,
                edges_proj,
                how='left',
                max_distance=100,  # 100 metres
            )
        except Exception:
            # Fallback: geographic CRS join (less accurate but functional)
            joined = gpd.sjoin_nearest(
                violations_gdf,
                edges_gdf[['geometry', 'road_weight']],
                how='left',
                max_distance=0.001,  # ~100 m in degrees
            )

        # sjoin_nearest can produce duplicate rows — keep first match per violation
        joined = joined[~joined.index.duplicated(keep='first')]

        df['road_weight'] = joined['road_weight'].values
        df['road_weight'] = df['road_weight'].fillna(0.3)

        logger.info('Road weights assigned to %d violations', len(df))

    except Exception as exc:
        logger.warning('Spatial join failed (%s) — fallback road_weight=0.5', exc)
        df['road_weight'] = 0.5


    return df
