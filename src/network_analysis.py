"""
ParkSight — Network Congestion Analysis
Uses OSM road network graph theory (betweenness centrality) to
quantify how parking violations impact traffic flow.

A violation on a high-centrality road (one that many shortest paths
cross) causes far more congestion than one on a dead-end street.
"""

import os
import logging
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
GRAPH_CACHE = os.path.join(CACHE_DIR, 'bengaluru_graph.pkl')
CENTRALITY_CACHE = os.path.join(CACHE_DIR, 'edge_centrality.pkl')

# Lane reduction factor by road type — estimates how much of the road
# is blocked when a vehicle parks illegally on it.
LANE_REDUCTION = {
    'motorway': 0.05,       # Wide — parking barely blocks
    'trunk': 0.10,
    'primary': 0.15,
    'secondary': 0.25,
    'tertiary': 0.35,
    'residential': 0.50,    # Narrow — parking blocks half the road
    'service': 0.60,
    'unclassified': 0.40,
}


def _get_lane_reduction(hw) -> float:
    """Resolve highway tag (may be a list) to a lane reduction factor."""
    if isinstance(hw, list):
        hw = hw[0]
    hw_str = str(hw).lower()
    for key, factor in LANE_REDUCTION.items():
        if key in hw_str:
            return factor
    return 0.40  # default


def _get_or_download_graph():
    """
    Get the Bengaluru road network as a networkx DiGraph.
    Cache to disk for reuse.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(GRAPH_CACHE):
        logger.info('Loading road graph from cache')
        try:
            with open(GRAPH_CACHE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning('Graph cache load failed (%s), re-downloading', e)

    try:
        import osmnx as ox
        logger.info('Downloading Bengaluru road graph from OSM...')
        G = ox.graph_from_place(
            'Bengaluru, Karnataka, India',
            network_type='drive',
        )
        with open(GRAPH_CACHE, 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info('Road graph cached (%d nodes, %d edges)',
                     G.number_of_nodes(), G.number_of_edges())
        return G
    except Exception as e:
        logger.warning('OSM graph download failed: %s', e)
        return None


def compute_node_centrality(G=None, k: int = 80) -> dict:
    """
    Compute approximate betweenness centrality for the road network.

    Uses k random source nodes for speed (O(k*(V+E)) instead of O(V*E)).
    Results are cached to disk.

    Parameters
    ----------
    G : networkx.DiGraph, optional
        Road network graph. If None, downloads from OSM.
    k : int
        Number of random source nodes for approximation.

    Returns
    -------
    dict mapping node_id -> centrality score (0 to 1)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(CENTRALITY_CACHE):
        logger.info('Loading cached centrality scores')
        try:
            with open(CENTRALITY_CACHE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass

    if G is None:
        G = _get_or_download_graph()

    if G is None:
        logger.warning('No graph available — cannot compute centrality')
        return {}

    import networkx as nx

    n_nodes = G.number_of_nodes()

    # For very large graphs (>100K nodes), use degree centrality as proxy
    # (betweenness is O(k*V*E) and can take 10+ minutes on 155K nodes)
    if n_nodes > 100000:
        logger.info(
            'Graph too large (%d nodes) for betweenness — using degree centrality',
            n_nodes,
        )
        centrality = nx.degree_centrality(G)
    else:
        k = min(k, n_nodes)
        logger.info('Computing betweenness centrality (k=%d, nodes=%d)...', k, n_nodes)
        centrality = nx.betweenness_centrality(G, k=k, weight='length', normalized=True)

    # Normalise to [0, 1]
    max_c = max(centrality.values()) if centrality else 1
    if max_c > 0:
        centrality = {k: v / max_c for k, v in centrality.items()}

    with open(CENTRALITY_CACHE, 'wb') as f:
        pickle.dump(centrality, f)

    logger.info('Centrality computed and cached for %d nodes', len(centrality))
    return centrality


def compute_congestion_impact(
    df: pd.DataFrame,
    centrality: dict = None,
) -> pd.DataFrame:
    """
    Compute a Congestion Impact Score (CIS) for each violation record.

    CIS = node_centrality × lane_reduction × peak_multiplier

    Higher CIS means a violation that disrupts traffic flow more severely.

    Parameters
    ----------
    df : pd.DataFrame
        Violation dataframe with latitude, longitude, road_weight, is_peak.
    centrality : dict, optional
        Pre-computed node centrality. If None, computes it.

    Returns
    -------
    pd.DataFrame with added columns:
        - node_centrality: how important the nearest intersection is
        - lane_reduction: estimated road capacity reduction
        - congestion_impact: composite CIS score (0 to 1)
    """
    df = df.copy()

    # ── Get centrality ──────────────────────────────────────
    if centrality is None or len(centrality) == 0:
        logger.info('No centrality data — using road_weight as proxy')
        df['node_centrality'] = df.get('road_weight', 0.5)
    else:
        # Map each violation to the nearest graph node's centrality
        try:
            import osmnx as ox
            G = _get_or_download_graph()
            if G is not None:
                # Find nearest node for each violation point
                nearest_nodes = ox.nearest_nodes(
                    G,
                    df['longitude'].values,
                    df['latitude'].values,
                )
                df['nearest_node'] = nearest_nodes
                df['node_centrality'] = df['nearest_node'].map(centrality).fillna(0.1)
            else:
                df['node_centrality'] = df.get('road_weight', 0.5)
        except Exception as e:
            logger.warning('Nearest node mapping failed (%s) — using road_weight proxy', e)
            df['node_centrality'] = df.get('road_weight', 0.5)

    # ── Lane reduction factor ───────────────────────────────
    # Invert road_weight: high road_weight = major road = low lane reduction
    # Low road_weight = minor road = high lane reduction
    if 'road_weight' in df.columns:
        # Map: motorway(1.0)->0.05, service(0.3)->0.60
        df['lane_reduction'] = 0.65 - (df['road_weight'] * 0.55)
        df['lane_reduction'] = df['lane_reduction'].clip(0.05, 0.65)
    else:
        df['lane_reduction'] = 0.35

    # ── Peak multiplier ─────────────────────────────────────
    df['peak_multiplier'] = np.where(df['is_peak'], 1.5, 1.0)

    # ── Composite Congestion Impact Score ────────────────────
    raw_cis = (
        df['node_centrality'] *
        df['lane_reduction'] *
        df['peak_multiplier']
    )
    # Normalise to [0, 1]
    cis_min = raw_cis.min()
    cis_max = raw_cis.max()
    if cis_max > cis_min:
        df['congestion_impact'] = ((raw_cis - cis_min) / (cis_max - cis_min)).round(4)
    else:
        df['congestion_impact'] = 0.5

    logger.info(
        'Congestion impact computed for %d violations — '
        'mean=%.3f, p90=%.3f, max=%.3f',
        len(df),
        df['congestion_impact'].mean(),
        df['congestion_impact'].quantile(0.90),
        df['congestion_impact'].max(),
    )

    return df


def get_junction_congestion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate congestion impact to junction level.

    Returns a DataFrame of junctions sorted by mean congestion impact,
    with additional stats useful for the AI report.
    """
    jdf = df[df['junction_name'].notna()].copy()

    if jdf.empty or 'congestion_impact' not in jdf.columns:
        return pd.DataFrame()

    summary = (
        jdf.groupby('junction_name')
        .agg(
            mean_congestion=('congestion_impact', 'mean'),
            max_congestion=('congestion_impact', 'max'),
            total_violations=('id', 'count'),
            avg_centrality=('node_centrality', 'mean'),
            avg_lane_reduction=('lane_reduction', 'mean'),
            peak_fraction=('is_peak', 'mean'),
            lat=('latitude', 'mean'),
            lon=('longitude', 'mean'),
        )
        .reset_index()
        .sort_values('mean_congestion', ascending=False)
        .reset_index(drop=True)
    )

    # Congestion severity label
    q75 = summary['mean_congestion'].quantile(0.75)
    q90 = summary['mean_congestion'].quantile(0.90)
    summary['severity'] = pd.cut(
        summary['mean_congestion'],
        bins=[-1, q75 * 0.5, q75, q90, float('inf')],
        labels=['Low', 'Moderate', 'High', 'Severe'],
    )

    summary['rank'] = range(1, len(summary) + 1)

    return summary
