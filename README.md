# ParkSight 🚔

**Hybrid AI-driven parking violation intelligence for targeted traffic enforcement.**

ParkSight transforms 298,450 raw parking violation records from Bengaluru Traffic Police (Nov 2023 – Apr 2024) into actionable enforcement insights through **predictive AI (XGBoost)**, **network congestion analysis (Graph Theory)**, **spatial analytics**, and **Generative AI briefings (Google Gemini)**.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the violation CSV in data/
mkdir -p data
cp /path/to/jan_to_may_police_violation_anonymized.csv data/

# 3. Launch the dashboard
streamlit run app.py
```

## First Run

The first launch performs several one-time operations (all cached for subsequent runs):

1. **Data processing** — parses CSV, engineers features, computes H3 indices, caches as Parquet (~10 s)
2. **OSM download** — fetches Bengaluru's driveable road network from OpenStreetMap (~2 min)
3. **XGBoost model training** — trains violation prediction model on junction-level daily data (~15 s)
4. **Betweenness centrality** — computes node centrality on road graph to quantify congestion impact (~1–2 min)

If OSM download fails (no internet), the dashboard falls back to default road weights and proxy centrality.

## Architecture

| Component | Technique | Detail |
|-----------|-----------|--------|
| Spatial Indexing | **H3 Hexagons** | Resolution-8 cells (~460 m edge) for uniform spatial aggregation |
| Hotspot Detection | **DBSCAN** | Density-based clustering (eps = 300 m, min_samples = 50) |
| Priority Scoring | **EPI** | Composite Enforcement Priority Index (see below) |
| Road Context | **OpenStreetMap** | Road classification weights via `osmnx` |
| Patrol Queue | **Time-Aware** | Dispatch recommendations adjusted for current hour |
| **Predictive AI** | **XGBoost** | Junction-level daily violation forecasting with lag features |
| **Congestion Impact** | **Betweenness Centrality** | Network graph theory to quantify traffic flow disruption |
| **AI Briefings** | **Google Gemini** | Natural-language patrol briefings from prediction + congestion data |

### Enforcement Priority Index (EPI)

```
EPI = (Violation Density × 0.40)
    + (Peak Hour Weight  × 0.30)
    + (Road Class Weight × 0.20)
    + (Repeat Offender Rate × 0.10)
```

All components are normalised 0–1 before weighting. The final score is scaled to 0–100.

### XGBoost Prediction Model

**Features:** day_of_week, month, is_weekend, road_class_weight, peak_fraction, repeat_fraction, lag_1d, lag_3d, lag_7d, rolling_7d_mean, rolling_7d_std, junction_historical_mean

**Target:** Daily violation count per junction

**Split:** Time-based 80/20 train/test split

### Congestion Impact Score (CIS)

```
CIS = Node Centrality × Lane Reduction Factor × Peak Multiplier
```

- **Node Centrality**: Betweenness centrality from OSM road graph (how many shortest paths cross the intersection)
- **Lane Reduction**: Estimated capacity reduction by road type (motorway=0.05 → residential=0.50)
- **Peak Multiplier**: 1.5× during peak hours (7–9 AM, 5–8 PM)

## Dashboard Tabs

1. **🗺️ Hotspot Map** — Interactive Folium map with H3 hexagons, DBSCAN clusters, and EPI-scored junction markers.
2. **🎯 Enforcement Queue** — Time-aware patrol dispatch recommendations with full EPI ranking.
3. **📊 Temporal Patterns** — Hourly, daily, and monthly violation trends with insight callouts.
4. **🔍 Junction Deep-Dive** — Per-junction analytics including mini-map, hourly pattern, and vehicle breakdown.
5. **🔮 AI Predictions & Impact** — XGBoost predictions, congestion impact analysis, feature importance, and AI-generated patrol briefing.

## Data

298,450 parking violation records, Bengaluru, Nov 2023 – Apr 2024.
Source: Bengaluru Traffic Police (anonymised).

## Tech Stack

- **Data**: pandas, NumPy, PyArrow
- **Spatial**: H3, GeoPandas, Shapely, OSMnx
- **ML/AI**: XGBoost, scikit-learn (DBSCAN), SHAP
- **Network**: NetworkX (betweenness centrality via OSMnx graph)
- **GenAI**: Google Generative AI (Gemini 2.0 Flash)
- **Viz**: Folium, Plotly, Branca
- **App**: Streamlit, streamlit-folium
