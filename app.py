"""
ParkSight 🚔 — Parking Violation Intelligence Dashboard
Main Streamlit application.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime

import streamlit as st
import pandas as pd

# ── Ensure project root is on the path ──────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import data_pipeline, hotspot_engine, epi_scorer, osm_fetcher  # noqa: E402
from src import temporal_analysis, visualizer  # noqa: E402
from src import predictive_model, network_analysis, ai_reporter  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
#  Page Config
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ParkSight",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Apply Inter font ONLY to text-bearing elements — never override
       Material Design icon fonts used by Streamlit internals. */
    html, body,
    p, span, div, h1, h2, h3, h4, h5, h6,
    label, input, textarea, select, button,
    .stMarkdown, .stText, .stCaption,
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
    }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid rgba(108,92,231,0.25);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #a29bfe !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #dfe6e9 !important;
        font-weight: 800 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e0e1a 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] h1 {
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Tables — do NOT use overflow:hidden, it clips column menus */
    .stDataFrame {
        border-radius: 10px;
    }

    /* Ensure dataframe column-header dropdown menus render on top */
    [data-testid="stDataFrame"] [role="menu"],
    [data-testid="stDataFrame"] [data-baseweb="popover"],
    [data-testid="stDataFrame"] [data-baseweb="menu"] {
        z-index: 999 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════
#  Cached Data Loaders
# ════════════════════════════════════════════════════════════════

RAW_DATA_PATH = os.path.join(ROOT, 'data', 'raw_dataset.parquet')


@st.cache_data(show_spinner=False, ttl=3600)
def get_processed_data(data_path: str) -> pd.DataFrame:
    """Load (or build + cache) the cleaned violation dataframe."""
    return data_pipeline.load_processed(data_path)


@st.cache_data(show_spinner=False, ttl=3600)
def get_osm_roads():
    """Fetch Bengaluru road network (cached to disk)."""
    return osm_fetcher.get_bengaluru_road_network()


@st.cache_data(show_spinner=False, ttl=3600)
def get_h3_grid(_df_hash, df):
    return hotspot_engine.compute_h3_grid(df)


@st.cache_data(show_spinner=False, ttl=3600)
def get_dbscan(_df_hash, df):
    return hotspot_engine.run_dbscan_clustering(df)


@st.cache_data(show_spinner=False, ttl=3600)
def get_epi(_df_hash, df):
    return epi_scorer.compute_junction_epi(df)


@st.cache_resource(show_spinner=False, ttl=3600)
def get_xgb_model(_df_hash, df):
    """Train or load cached XGBoost violation prediction model."""
    return predictive_model.train_model(df)


@st.cache_resource(show_spinner=False, ttl=3600)
def get_centrality():
    """Compute or load cached node betweenness centrality."""
    return network_analysis.compute_node_centrality(k=80)


def get_congestion_impact(df, centrality):
    """Compute congestion impact for each violation (uses disk-cached centrality)."""
    return network_analysis.compute_congestion_impact(df, centrality)


# ════════════════════════════════════════════════════════════════
#  Data Loading (with progress bar)
# ════════════════════════════════════════════════════════════════

def load_all_data():
    """Master loader — returns all precomputed analytics objects with progress tracking."""

    data_file = RAW_DATA_PATH
    
    # 1. Fallback uploader if the dataset is missing
    if not os.path.exists(data_file):
        st.sidebar.error("Bundled dataset not found in `data/raw_dataset.parquet`.")
        uploaded = st.sidebar.file_uploader("Upload dataset (CSV or Parquet)", type=["csv", "parquet"])
        if uploaded is not None:
            import tempfile
            temp_path = os.path.join(tempfile.gettempdir(), 'uploaded_dataset.' + uploaded.name.split('.')[-1])
            try:
                with open(temp_path, 'wb') as f:
                    f.write(uploaded.getvalue())
                data_file = temp_path
            except Exception as upload_err:
                st.error(f"Failed to save uploaded file: {str(upload_err)}")
                st.stop()
        else:
            st.error(
                "### 📁 Data file not found\n\n"
                "The bundled dataset is missing. Please upload it via the sidebar to continue."
            )
            st.stop()

    # ── Progress bar for loading pipeline ─────────────────────
    progress = st.progress(0, text="\U0001F504 Initialising ParkSight...")

    progress.progress(5, text="\U0001F504 Loading & processing violation data...")
    df = get_processed_data(data_file)
    progress.progress(20, text="\u2705 Data loaded \u2014 Fetching road network...")

    edges = get_osm_roads()
    progress.progress(35, text="\u2705 Road network ready \u2014 Assigning road weights...")
    df = osm_fetcher.assign_road_class_to_violations(df, edges)
    progress.progress(45, text="\u2705 Road weights assigned \u2014 Computing H3 grid...")

    df_hash = hash(len(df))

    h3_grid = get_h3_grid(df_hash, df)
    progress.progress(55, text="\u2705 H3 grid done \u2014 Running DBSCAN clustering...")

    df_clustered, cluster_stats = get_dbscan(df_hash, df)
    progress.progress(65, text="\u2705 Clusters found \u2014 Computing EPI scores...")

    junctions_epi = get_epi(df_hash, df)
    progress.progress(70, text="\u2705 EPI ready \u2014 Training AI prediction model...")

    # ── AI Model Training ─────────────────────────────────────
    xgb_result = get_xgb_model(df_hash, df)
    progress.progress(85, text="\u2705 AI model trained \u2014 Computing congestion impact...")

    # ── Network Congestion Analysis ───────────────────────────
    centrality = get_centrality()
    df_with_cis = get_congestion_impact(df, centrality)
    congestion_summary = network_analysis.get_junction_congestion_summary(df_with_cis)
    progress.progress(95, text="\u2705 Congestion analysed \u2014 Generating predictions...")

    # ── Predictions ───────────────────────────────────────────
    predictions = pd.DataFrame()
    if xgb_result.get('model') is not None:
        predictions = predictive_model.predict_future_violations(
            xgb_result['model'],
            xgb_result['training_data'],
        )

    progress.progress(100, text="\u2705 All analytics + AI ready!")
    time.sleep(0.5)
    progress.empty()

    return {
        'df': df,
        'df_clustered': df_clustered,
        'h3_grid': h3_grid,
        'cluster_stats': cluster_stats,
        'junctions_epi': junctions_epi,
        'xgb_result': xgb_result,
        'predictions': predictions,
        'congestion_summary': congestion_summary,
        'df_with_cis': df_with_cis,
    }


# ════════════════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame):
    """Render sidebar filters and return filtered dataframe."""
    st.sidebar.markdown("# ParkSight 🚔")
    st.sidebar.caption("Parking Violation Intelligence")
    st.sidebar.markdown("---")

    # Police station filter
    stations = sorted(df['police_station'].dropna().unique())
    sel_stations = st.sidebar.multiselect(
        "🏢 Police Station", stations, default=[], key="filter_stations"
    )

    # Vehicle type filter
    vtypes = sorted(df['vehicle_type'].dropna().unique())
    sel_vtypes = st.sidebar.multiselect(
        "🚗 Vehicle Type", vtypes, default=[], key="filter_vtypes"
    )

    # Date range
    all_dates = pd.to_datetime(df['date'])
    min_date, max_date = all_dates.min().date(), all_dates.max().date()
    date_range = st.sidebar.slider(
        "📅 Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        key="filter_dates",
    )

    # About EPI
    with st.sidebar.expander("ℹ️ About EPI"):
        st.markdown(
            """
            **Enforcement Priority Index** ranks junctions by
            how urgently they need patrol attention.

            ```
            EPI = Density × 0.40
                + Peak Hour × 0.30
                + Road Class × 0.20
                + Repeat Rate × 0.10
            ```

            - **Density**: normalised violation count
            - **Peak Hour**: fraction during 7-9 AM / 5-8 PM
            - **Road Class**: OSM highway classification
            - **Repeat Rate**: fraction of repeat offenders

            Score range: **0 – 100**
            """
        )

    # ── Gemini API key ────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 🤖 AI Configuration")
    
    # Try to find a default key in secrets or env
    default_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            default_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    if not default_key and os.environ.get("GEMINI_API_KEY"):
        default_key = os.environ.get("GEMINI_API_KEY")
        
    gemini_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        value=default_key if default_key else "",
        placeholder="Using pre-configured key..." if default_key else "Paste your Gemini API key here",
        key="gemini_api_key",
        help="Optional. A pre-configured API key will be used if available. You can override it here by pasting your own.",
    )
    st.session_state['gemini_key'] = gemini_key if gemini_key else None

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Built with Streamlit \u00b7 Data: Bengaluru Traffic Police"
    )

    # ── Apply filters ─────────────────────────────────────────
    filtered = df.copy()
    if sel_stations:
        filtered = filtered[filtered['police_station'].isin(sel_stations)]
    if sel_vtypes:
        filtered = filtered[filtered['vehicle_type'].isin(sel_vtypes)]

    filtered_dates = pd.to_datetime(filtered['date'])
    filtered = filtered[
        (filtered_dates >= pd.Timestamp(date_range[0]))
        & (filtered_dates <= pd.Timestamp(date_range[1]))
    ]

    return filtered


# ════════════════════════════════════════════════════════════════
#  Tab 1 — Hotspot Map
# ════════════════════════════════════════════════════════════════

def tab_hotspot_map(df, h3_grid, junctions_epi, cluster_stats):
    """Render the Hotspot Map tab."""
    from streamlit_folium import st_folium

    # ── KPIs ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Violations", f"{len(df):,}")
    # DBSCAN runs on full dataset (too expensive per-filter);
    # show cluster count with a helpful tooltip via help=
    cluster_count = len(cluster_stats) if cluster_stats is not None and not cluster_stats.empty else 0
    c2.metric(
        "Hotspot Clusters",
        cluster_count,
        help="Spatial clusters computed on the full dataset (all dates/locations). "
             "Changing filters updates Violations, Peak%, and EPI rankings but not the cluster geometry.",
    )
    peak_pct = (
        df['is_peak'].sum() / len(df) * 100 if len(df) > 0 else 0
    )
    c3.metric("Peak-Hour %", f"{peak_pct:.1f}%")

    if junctions_epi is not None and not junctions_epi.empty:
        top_j = junctions_epi.iloc[0]
        c4.metric(
            "Top Junction",
            f"{top_j['junction_name'][:25]}",
            f"▲ EPI {top_j['epi_score']}",
        )
    else:
        c4.metric("Top Junction", "N/A")

    st.markdown("")

    # ── Map ───────────────────────────────────────────────────
    with st.spinner("🗺️ Rendering map…"):
        m = visualizer.build_hotspot_map(
            df, h3_grid, junctions_epi, cluster_stats
        )
        st_folium(m, height=550, use_container_width=True, returned_objects=[])

    # ── Bottom row ────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("#### 🏆 Top 10 Junctions")
        if junctions_epi is not None and not junctions_epi.empty:
            display_cols = ['rank', 'junction_name', 'total_violations', 'epi_score']
            display_cols = [c for c in display_cols if c in junctions_epi.columns]
            st.dataframe(
                junctions_epi[display_cols].head(10),
                use_container_width=True,
                hide_index=True,
                height=320,
                column_config={
                    "rank": st.column_config.NumberColumn("#", width="small"),
                    "junction_name": st.column_config.TextColumn("Junction Name", width="large"),
                    "total_violations": st.column_config.NumberColumn("Violations", format="%d"),
                    "epi_score": st.column_config.ProgressColumn(
                        "EPI Score",
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )
        else:
            st.info("No junction data available.")

    with right:
        st.markdown("#### 🚗 Violation by Vehicle Type")
        fig = visualizer.build_violation_pie(df)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  Tab 2 — Enforcement Queue
# ════════════════════════════════════════════════════════════════

def tab_enforcement_queue(df, junctions_epi):
    """Render the Enforcement Queue tab."""
    now = datetime.now()
    current_hour = now.hour
    is_peak = current_hour in {7, 8, 9, 17, 18, 19, 20}

    st.markdown("### 🎯 Patrol Dispatch Recommendations")
    status_col = "🔴 PEAK HOUR" if is_peak else "🟢 Off-Peak"
    st.markdown(
        f"**Current time:** {now.strftime('%H:%M')} — **Status:** {status_col}"
    )

    # ── NOW recommendations ───────────────────────────────────
    recs = temporal_analysis.get_patrol_recommendation(
        df, junctions_epi, current_hour=current_hour
    )

    if is_peak:
        st.warning("⚠️ Peak enforcement window active — priorities boosted by 30%")
    else:
        st.info("ℹ️ Off-peak period — standard priority scoring")

    if not recs.empty:
        st.dataframe(
            recs,
            use_container_width=True,
            hide_index=True,
            height=350,
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "junction_name": st.column_config.TextColumn("Junction", width="large"),
                "police_station": st.column_config.TextColumn("Police Station", width="medium"),
                "epi_score": st.column_config.ProgressColumn(
                    "EPI Score",
                    help="Enforcement Priority Index (0–100). Higher = more urgent patrol needed.",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                "time_adjusted_priority": st.column_config.NumberColumn(
                    "Adj. Priority",
                    help="EPI score boosted by 30% during peak hours",
                    format="%.1f",
                ),
                "total_violations": st.column_config.NumberColumn(
                    "Total Violations", format="%d"
                ),
                "hour_violations": st.column_config.NumberColumn(
                    "This Hour (Avg)",
                    help="Average violations recorded at this hour historically",
                    format="%d",
                ),
                "is_peak_now": st.column_config.CheckboxColumn(
                    "Peak Now?",
                    help="Is current hour within the peak enforcement window (7–9 AM / 5–8 PM)?",
                ),
            },
        )
    else:
        st.info("No recommendation data available for this hour.")

    st.markdown("---")

    # ── EPI Formula explainer FIRST to avoid title overlap with table ─
    st.markdown("### 📋 Full Junction EPI Rankings")
    st.caption(
        "EPI = Density × 0.40 + Peak Hour × 0.30 + Road Class × 0.20 + Repeat Rate × 0.10 "
        "| All components [0–1], final score scaled to 0–100."
    )

    with st.expander("📐 EPI Formula Details — click to expand", expanded=False):
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            st.code(
                "EPI = (Violation Density  × 0.40)\n"
                "    + (Peak Hour Weight   × 0.30)\n"
                "    + (Road Class Weight  × 0.20)\n"
                "    + (Repeat Offender Rate × 0.10)",
                language="",
            )
        with col_f2:
            formula_data = pd.DataFrame({
                "Component": ["Violation Density", "Peak Hour Weight", "Road Class Weight", "Repeat Offender Rate"],
                "Weight": ["40%", "30%", "20%", "10%"],
                "Source": [
                    "Violation count, min-max normalised",
                    "Fraction in 7-9 AM / 5-8 PM",
                    "OSM highway class (motorway=1.0 → service=0.3)",
                    "Repeat offenders ÷ unique vehicles",
                ],
            })
            st.dataframe(formula_data, use_container_width=True, hide_index=True, height=180)
        st.info(
            "**Score range: 0–100.** This scores *enforcement opportunity*, "
            "not direct congestion impact. No real-time traffic data is used."
        )

    st.markdown("")

    if junctions_epi is not None and not junctions_epi.empty:
        # Station filter within the tab
        stations_in_epi = sorted(
            junctions_epi['police_station'].dropna().unique()
        )
        sel_st = st.multiselect(
            "Filter by Police Station",
            stations_in_epi,
            default=[],
            key="epi_station_filter",
        )

        display = junctions_epi.copy()
        if sel_st:
            display = display[display['police_station'].isin(sel_st)]

        display_cols = [
            'rank', 'junction_name', 'police_station', 'epi_score',
            'total_violations', 'peak_hour_weight', 'road_class_weight',
            'repeat_offender_rate', 'density_score',
        ]
        display_cols = [c for c in display_cols if c in display.columns]

        st.dataframe(
            display[display_cols],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "junction_name": st.column_config.TextColumn("Junction Name", width="large"),
                "police_station": st.column_config.TextColumn("Police Station", width="medium"),
                "epi_score": st.column_config.ProgressColumn(
                    "EPI Score",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                "total_violations": st.column_config.NumberColumn("Violations", format="%d"),
                "peak_hour_weight": st.column_config.NumberColumn(
                    "Peak Wt.",
                    help="Fraction of violations during peak hours (0–1)",
                    format="%.2f",
                ),
                "road_class_weight": st.column_config.NumberColumn(
                    "Road Wt.",
                    help="OSM road class weight (motorway=1.0, service=0.3)",
                    format="%.2f",
                ),
                "repeat_offender_rate": st.column_config.NumberColumn(
                    "Repeat Rate",
                    help="Normalised ratio of repeat offenders (0–1)",
                    format="%.2f",
                ),
                "density_score": st.column_config.NumberColumn(
                    "Density",
                    help="Min-max normalised violation density (0–1)",
                    format="%.2f",
                ),
            },
        )
    else:
        st.info("No junction EPI data available.")


# ════════════════════════════════════════════════════════════════
#  Tab 3 — Temporal Patterns
# ════════════════════════════════════════════════════════════════

def tab_temporal_patterns(df):
    """Render the Temporal Patterns tab."""
    hourly = temporal_analysis.get_hourly_pattern(df)
    daily = temporal_analysis.get_daily_pattern(df)
    monthly = temporal_analysis.get_monthly_trend(df)
    vehicle = temporal_analysis.get_vehicle_breakdown(df)

    # ── Insight callouts ──────────────────────────────────────
    top3_hours = hourly.nlargest(3, 'violation_count')['hour'].tolist()
    top3_str = ", ".join(f"{h}:00" for h in top3_hours)
    top_day = daily.loc[daily['violation_count'].idxmax(), 'day_name']
    top_vehicle = vehicle.iloc[0]['vehicle_type'] if not vehicle.empty else 'N/A'

    i1, i2, i3 = st.columns(3)
    i1.info(f"🕐 **Peak hours:** {top3_str}")
    i2.info(f"📅 **Busiest day:** {top_day}")
    i3.info(f"🚗 **Top offender:** {top_vehicle}")

    st.markdown("")

    # Row 1: Hourly pattern
    st.plotly_chart(
        visualizer.build_hourly_chart(hourly),
        use_container_width=True,
    )

    # Row 2: Daily + Vehicle
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            visualizer.build_daily_chart(daily),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            visualizer.build_vehicle_chart(vehicle),
            use_container_width=True,
        )

    # Row 3: Monthly trend
    st.plotly_chart(
        visualizer.build_monthly_chart(monthly),
        use_container_width=True,
    )


# ════════════════════════════════════════════════════════════════
#  Tab 4 — Junction Deep-Dive
# ════════════════════════════════════════════════════════════════

def tab_junction_deep_dive(df, junctions_epi):
    """Render the Junction Deep-Dive tab."""
    from streamlit_folium import st_folium

    if junctions_epi is None or junctions_epi.empty:
        st.info("No junction EPI data available.")
        return

    # Junction selector
    junction_options = junctions_epi['junction_name'].tolist()
    selected = st.selectbox(
        "Select a Junction (sorted by EPI)",
        junction_options,
        key="deep_dive_junction",
    )

    jrow = junctions_epi[junctions_epi['junction_name'] == selected].iloc[0]
    jdf = df[df['junction_name'] == selected]

    if jdf.empty:
        st.warning(f"No violation records found for **{selected}**.")
        return

    # ── KPIs ──────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Violations", f"{jrow['total_violations']:,}")
    k2.metric("EPI Score", f"{jrow['epi_score']}")
    k3.metric("Rank", f"#{jrow['rank']}")

    st.markdown("")

    # ── Map + Charts ──────────────────────────────────────────
    map_col, chart_col = st.columns([1, 1])

    with map_col:
        st.markdown("##### 📍 Violation Locations")
        m = visualizer.build_junction_detail_map(df, selected)
        st_folium(m, height=380, use_container_width=True, returned_objects=[])

    with chart_col:
        st.markdown("##### ⏰ Hourly Pattern")
        j_hourly = temporal_analysis.get_hourly_pattern(jdf)
        st.plotly_chart(
            visualizer.build_hourly_chart(j_hourly),
            use_container_width=True,
        )

    # ── Bottom detail row ─────────────────────────────────────
    b1, b2 = st.columns(2)

    with b1:
        st.markdown("##### 🚗 Vehicle Breakdown")
        j_vehicle = temporal_analysis.get_vehicle_breakdown(jdf)
        if not j_vehicle.empty:
            st.plotly_chart(
                visualizer.build_vehicle_chart(j_vehicle),
                use_container_width=True,
            )

    with b2:
        st.markdown("##### 📊 Junction Details")

        # Top 5 violation types
        # violation_list is stored as a JSON string — parse back to list
        _vl = jdf['violation_list'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
        )
        viol_counts = (
            _vl.explode()
            .value_counts()
            .head(5)
            .reset_index(name='count')
        )
        viol_counts.columns = ['Violation Type', 'Count']
        st.dataframe(
            viol_counts,
            use_container_width=True,
            hide_index=True,
            height=220,
            column_config={
                "Violation Type": st.column_config.TextColumn("Violation Type", width="large"),
                "Count": st.column_config.NumberColumn("Count", format="%d"),
            },
        )

        # Repeat offenders
        repeat_count = jdf['is_repeat_offender'].sum()
        repeat_pct = (
            repeat_count / len(jdf) * 100 if len(jdf) > 0 else 0
        )
        st.markdown(
            f"**Repeat offenders:** {repeat_count:,} "
            f"({repeat_pct:.1f}% of violations)"
        )


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
#  Tab 5 — AI Predictions & Impact
# ════════════════════════════════════════════════════════════════

def tab_ai_predictions(df, predictions, xgb_result, congestion_summary, junctions_epi):
    """Render the AI Predictions & Impact tab."""
    from streamlit_folium import st_folium

    metrics = xgb_result.get('metrics', {})
    importance = xgb_result.get('feature_importance', pd.DataFrame())

    # ── AI Briefing at the top ────────────────────────────────
    st.markdown("### \U0001F916 AI-Generated Patrol Briefing")

    api_key = st.session_state.get('gemini_key', None)
    briefing = ai_reporter.generate_ai_briefing(
        predictions=predictions,
        congestion=congestion_summary,
        epi=junctions_epi,
        metrics=metrics,
        api_key=api_key,
    )

    # Display the AI report in a styled container
    source_badge = (
        "\U0001F7E2 Powered by " + briefing['source']
        if 'Gemini' in briefing['source']
        else "\U0001F7E1 " + briefing['source']
    )
    st.caption(f"{source_badge} | Generated: {briefing['timestamp']}")
    st.markdown(briefing['report'])

    # Download report button (PDF if fpdf2 is installed, otherwise markdown fallback)
    pdf_data = ai_reporter.export_briefing_to_pdf(briefing['report'])
    if pdf_data is not None:
        st.download_button(
            label="📥 Download AI Briefing Report (PDF)",
            data=pdf_data,
            file_name=f"parksight_ai_briefing_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )
    else:
        st.download_button(
            label="📥 Download AI Briefing Report (Markdown)",
            data=briefing['report'],
            file_name=f"parksight_ai_briefing_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )
        st.info("💡 **Tip:** Install `fpdf2` in your virtual environment to enable one-click PDF downloading! Run `pip install fpdf2` in your terminal.")


    if not api_key:
        st.info(
            "\U0001F511 **Tip:** Enter a Gemini API key in the sidebar to get "
            "AI-powered briefings from Google Gemini 2.0 Flash."
        )

    st.markdown("---")

    # ── Model Performance KPIs ────────────────────────────────
    st.markdown("### \U0001F4CA XGBoost Model Performance")
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "R\u00b2 Score",
            f"{metrics.get('test_r2', 0):.4f}",
            help="Coefficient of determination \u2014 1.0 is perfect. "
                 "Shows how much variance in violations the model explains.",
        )
        m2.metric(
            "MAE",
            f"{metrics.get('test_mae', 0):.2f}",
            help="Mean Absolute Error \u2014 average prediction error in violations/day.",
        )
        m3.metric(
            "RMSE",
            f"{metrics.get('test_rmse', 0):.2f}",
            help="Root Mean Square Error \u2014 penalises large errors more.",
        )
        m4.metric(
            "Junctions",
            f"{metrics.get('n_junctions', 0)}",
            help="Number of named junctions the model was trained on.",
        )

        st.caption(
            f"Trained on {metrics.get('train_size', '?')} samples | "
            f"Tested on {metrics.get('test_size', '?')} samples | "
            f"Period: {metrics.get('date_range', 'N/A')} | "
            f"Split: {metrics.get('split_date', 'N/A')}"
        )
    else:
        st.warning("Model training metrics not available.")

    st.markdown("")

    # ── Row 1: Prediction Map + Risk Pie ──────────────────────
    if predictions is not None and not predictions.empty:
        st.markdown("### \U0001F52E Predicted Violation Hotspots (Next Day)")

        map_col, chart_col = st.columns([3, 2])
        with map_col:
            pred_map = visualizer.build_prediction_map(predictions)
            st_folium(pred_map, height=450, use_container_width=True, returned_objects=[])

        with chart_col:
            st.plotly_chart(
                visualizer.build_prediction_risk_pie(predictions),
                use_container_width=True,
            )
            # Top 5 predictions table
            st.markdown("##### Top 5 Predicted Hotspots")
            top5_cols = ['rank', 'junction_name', 'predicted_violations', 'risk_level']
            top5_cols = [c for c in top5_cols if c in predictions.columns]
            st.dataframe(
                predictions[top5_cols].head(5),
                use_container_width=True,
                hide_index=True,
                height=200,
                column_config={
                    "rank": st.column_config.NumberColumn("#", width="small"),
                    "junction_name": st.column_config.TextColumn("Junction", width="large"),
                    "predicted_violations": st.column_config.NumberColumn(
                        "Predicted", format="%.0f",
                    ),
                    "risk_level": st.column_config.TextColumn("Risk"),
                },
            )
    else:
        st.info("No predictions available \u2014 model may need more data.")

    st.markdown("---")

    # ── Row 2: Feature Importance + Congestion Impact ─────────
    left, right = st.columns(2)

    with left:
        st.markdown("### \U0001F9E0 What Drives Violations? (Explainable AI)")
        if not importance.empty:
            st.plotly_chart(
                visualizer.build_feature_importance_chart(importance),
                use_container_width=True,
            )
            st.caption(
                "SHAP-equivalent feature importance from XGBoost \u2014 shows "
                "which factors most strongly predict parking violations."
            )
        else:
            st.info("Feature importance not available.")

    with right:
        st.markdown("### \U0001F6A7 Congestion Impact Analysis")
        if congestion_summary is not None and not congestion_summary.empty:
            st.plotly_chart(
                visualizer.build_congestion_impact_chart(congestion_summary),
                use_container_width=True,
            )
            st.caption(
                "Congestion Impact Score = Road Centrality \u00d7 Lane Reduction \u00d7 Peak Multiplier. "
                "Higher score = violation disrupts more traffic routes."
            )
        else:
            st.info("Congestion impact data not available.")

    # ── Full predictions table ────────────────────────────────
    if predictions is not None and not predictions.empty:
        st.markdown("---")
        with st.expander("\U0001F4CB Full Prediction Table \u2014 all junctions", expanded=False):
            pred_display_cols = [
                'rank', 'junction_name', 'predicted_violations',
                'risk_level', 'rolling_7d_mean', 'avg_road_weight',
            ]
            pred_display_cols = [c for c in pred_display_cols if c in predictions.columns]
            st.dataframe(
                predictions[pred_display_cols],
                use_container_width=True,
                hide_index=True,
                height=400,
                column_config={
                    "rank": st.column_config.NumberColumn("#", width="small"),
                    "junction_name": st.column_config.TextColumn("Junction", width="large"),
                    "predicted_violations": st.column_config.NumberColumn(
                        "Predicted Violations", format="%.1f",
                    ),
                    "risk_level": st.column_config.TextColumn("Risk Level"),
                    "rolling_7d_mean": st.column_config.NumberColumn(
                        "7d Avg", format="%.1f",
                        help="7-day rolling average of historical violations",
                    ),
                    "avg_road_weight": st.column_config.NumberColumn(
                        "Road Weight", format="%.2f",
                        help="OSM road class weight (1.0=motorway, 0.3=service)",
                    ),
                },
            )


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    # ── Load everything ───────────────────────────────────────
    data = load_all_data()

    df_full = data['df']
    df_clustered = data['df_clustered']
    h3_grid = data['h3_grid']
    cluster_stats = data['cluster_stats']
    junctions_epi = data['junctions_epi']
    xgb_result = data['xgb_result']
    predictions = data['predictions']
    congestion_summary = data['congestion_summary']

    # ── Sidebar filters ───────────────────────────────────────
    df_filtered = render_sidebar(df_full)

    if len(df_filtered) == 0:
        st.info("\U0001F50D No violations match the current filters. Adjust the sidebar.")
        st.stop()

    # Recompute grid & EPI on filtered data for responsive charts
    filt_hash = hash((len(df_filtered), tuple(df_filtered['id'].head(5))))
    h3_grid_f = get_h3_grid(filt_hash, df_filtered)
    junctions_epi_f = get_epi(filt_hash, df_filtered)

    # Use full-data clusters (DBSCAN is expensive & filter-invariant)
    cluster_stats_f = cluster_stats

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "\U0001F5FA\uFE0F Hotspot Map",
        "\U0001F3AF Enforcement Queue",
        "\U0001F4CA Temporal Patterns",
        "\U0001F50D Junction Deep-Dive",
        "\U0001F52E AI Predictions & Impact",
    ])

    with tab1:
        tab_hotspot_map(df_filtered, h3_grid_f, junctions_epi_f, cluster_stats_f)

    with tab2:
        tab_enforcement_queue(df_filtered, junctions_epi_f)

    with tab3:
        tab_temporal_patterns(df_filtered)

    with tab4:
        tab_junction_deep_dive(df_filtered, junctions_epi_f)

    with tab5:
        tab_ai_predictions(
            df_filtered, predictions, xgb_result,
            congestion_summary, junctions_epi_f,
        )


if __name__ == "__main__":
    main()
