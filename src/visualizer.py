"""
ParkSight — Visualiser
Folium map builders and Plotly chart factories.
"""

import logging

import folium
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from branca.colormap import LinearColormap

try:
    from src import h3_cell_to_boundary
except ImportError:
    import h3
    h3_cell_to_boundary = lambda idx, geo_json=False: h3.h3_to_geo_boundary(idx, geo_json=geo_json)

logger = logging.getLogger(__name__)

# ─── Colour palette ────────────────────────────────────────────────
_CM = LinearColormap(
    colors=['#00b894', '#fdcb6e', '#e17055', '#d63031'],
    vmin=0, vmax=1,
    caption='Violation Density',
)

_PLOTLY_TEMPLATE = 'plotly_dark'
_BG = '#0e1117'  # Streamlit dark background


# ═══════════════════════════════════════════════════════════════════
#  Folium Maps
# ═══════════════════════════════════════════════════════════════════

def build_hotspot_map(
    df: pd.DataFrame,
    h3_grid: pd.DataFrame,
    junctions_epi: pd.DataFrame,
    cluster_stats: pd.DataFrame,
) -> folium.Map:
    """
    Build a multi-layer Folium map with H3 hexagons, DBSCAN cluster
    markers, and EPI-scored junction pins.
    """
    m = folium.Map(
        location=[12.9716, 77.5946],
        zoom_start=12,
        tiles='CartoDB dark_matter',
        control_scale=True,
    )

    # ── Layer 1: H3 hexagons ────────────────────────────────────
    hex_layer = folium.FeatureGroup(name='H3 Violation Density', show=True)
    grid_filtered = h3_grid[h3_grid['violation_count'] > 10]

    for _, row in grid_filtered.iterrows():
        try:
            boundary = h3_cell_to_boundary(row['h3_index'], geo_json=True)
            color = _CM(row['density_score'])
            peak_pct = round(row['peak_hour_weight'] * 100, 1)
            folium.Polygon(
                locations=[(lat, lon) for lon, lat in boundary],
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.45,
                weight=1,
                tooltip=(
                    f"Violations: {row['violation_count']}"
                    f" | Peak hour: {peak_pct}%"
                ),
            ).add_to(hex_layer)
        except Exception:
            continue
    hex_layer.add_to(m)

    # ── Layer 2: DBSCAN cluster markers ─────────────────────────
    if cluster_stats is not None and not cluster_stats.empty:
        cluster_layer = folium.FeatureGroup(name='Hotspot Clusters', show=True)
        for _, row in cluster_stats.iterrows():
            radius = max(8, min(30, row['cluster_size'] / 100))
            folium.CircleMarker(
                location=[row['cluster_lat'], row['cluster_lon']],
                radius=radius,
                color='#ff4757',
                fill=True,
                fill_color='#ff6b6b',
                fill_opacity=0.6,
                weight=2,
                popup=folium.Popup(
                    f"<b>Hotspot Cluster #{int(row['cluster_id'])}</b><br>"
                    f"Size: {int(row['cluster_size'])} violations<br>"
                    f"Peak fraction: {row['peak_fraction']:.0%}",
                    max_width=250,
                ),
            ).add_to(cluster_layer)
        cluster_layer.add_to(m)

    # ── Layer 3: Top-20 EPI junction markers ────────────────────
    if junctions_epi is not None and not junctions_epi.empty:
        junction_layer = folium.FeatureGroup(name='EPI Junctions', show=True)
        top = junctions_epi.head(20)
        for _, row in top.iterrows():
            epi = row['epi_score']
            if epi > 75:
                bg = '#d63031'
            elif epi > 50:
                bg = '#e17055'
            else:
                bg = '#fdcb6e'

            icon = folium.DivIcon(
                html=(
                    f'<div style="'
                    f'background:{bg};color:#fff;padding:3px 7px;'
                    f'border-radius:12px;font-size:11px;font-weight:700;'
                    f'white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,.4);'
                    f'">{epi:.0f}</div>'
                ),
                icon_size=(42, 22),
                icon_anchor=(21, 11),
            )
            popup_html = (
                f"<b>{row['junction_name']}</b><br>"
                f"EPI Score: <b>{epi}</b><br>"
                f"Rank: #{row['rank']}<br>"
                f"Violations: {row['total_violations']}<br>"
                f"Peak %: {row['peak_hour_weight']:.0%}<br>"
                f"Road class: {row['road_class_weight']:.2f}<br>"
                f"Station: {row.get('police_station', 'N/A')}"
            )
            folium.Marker(
                location=[row['lat'], row['lon']],
                icon=icon,
                popup=folium.Popup(popup_html, max_width=280),
            ).add_to(junction_layer)
        junction_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def build_junction_detail_map(
    df: pd.DataFrame,
    junction_name: str,
) -> folium.Map:
    """Small map centred on a single junction with violation dots."""
    jdf = df[df['junction_name'] == junction_name]
    if jdf.empty:
        return folium.Map(location=[12.9716, 77.5946], zoom_start=13)

    centre = [jdf['latitude'].mean(), jdf['longitude'].mean()]
    m = folium.Map(location=centre, zoom_start=15, tiles='CartoDB dark_matter')

    for _, row in jdf.sample(min(500, len(jdf))).iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color='#6c5ce7',
            fill=True,
            fill_opacity=0.6,
            weight=0,
        ).add_to(m)

    return m


# ═══════════════════════════════════════════════════════════════════
#  Plotly Charts
# ═══════════════════════════════════════════════════════════════════

def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply consistent dark-theme styling."""
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(size=18, color='#dfe6e9')),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color='#b2bec3'),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def build_epi_bar_chart(junctions_epi: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of top-20 junctions by EPI score."""
    top = junctions_epi.head(20).sort_values('epi_score')
    fig = go.Figure(
        go.Bar(
            x=top['epi_score'],
            y=top['junction_name'],
            orientation='h',
            marker=dict(
                color=top['epi_score'],
                colorscale=[[0, '#00b894'], [0.5, '#fdcb6e'], [1, '#d63031']],
                showscale=True,
                colorbar=dict(title='EPI'),
            ),
            hovertemplate=(
                '<b>%{y}</b><br>EPI: %{x:.1f}<extra></extra>'
            ),
        )
    )
    fig.add_annotation(
        text=(
            'EPI = Density×0.40 + Peak×0.30 '
            '+ Road×0.20 + Repeat×0.10'
        ),
        xref='paper', yref='paper',
        x=0.98, y=-0.08,
        showarrow=False,
        font=dict(size=10, color='#636e72'),
    )
    return _base_layout(fig, 'Top 20 Junctions — Enforcement Priority Index')


def build_hourly_chart(hourly: pd.DataFrame) -> go.Figure:
    """Line chart with peak-hour shading."""
    fig = go.Figure()

    # Peak-hour shading
    for start, end in [(6.5, 9.5), (16.5, 20.5)]:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor='rgba(214,48,49,0.12)',
            line_width=0,
            annotation_text='Peak' if start < 10 else 'Peak',
            annotation_position='top',
            annotation_font_color='#d63031',
        )

    fig.add_trace(
        go.Scatter(
            x=hourly['hour'],
            y=hourly['violation_count'],
            mode='lines+markers',
            line=dict(color='#6c5ce7', width=3),
            marker=dict(size=7, color='#a29bfe'),
            fill='tozeroy',
            fillcolor='rgba(108,92,231,0.15)',
            hovertemplate='Hour %{x}:00 — %{y:,} violations<extra></extra>',
        )
    )
    fig.update_xaxes(dtick=1, title='Hour of Day')
    fig.update_yaxes(title='Violations')
    return _base_layout(fig, 'Violations by Hour of Day')


def build_daily_chart(daily: pd.DataFrame) -> go.Figure:
    """Bar chart of violations per day of week."""
    fig = go.Figure(
        go.Bar(
            x=daily['day_name'],
            y=daily['violation_count'],
            marker_color='#00cec9',
            hovertemplate='%{x}: %{y:,}<extra></extra>',
        )
    )
    fig.update_yaxes(title='Violations')
    return _base_layout(fig, 'Violations by Day of Week')


def build_vehicle_chart(vehicle: pd.DataFrame) -> go.Figure:
    """Bar chart of top vehicle types."""
    fig = go.Figure(
        go.Bar(
            x=vehicle['vehicle_type'],
            y=vehicle['violation_count'],
            marker_color='#fd79a8',
            hovertemplate='%{x}: %{y:,}<extra></extra>',
        )
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(title='Violations')
    return _base_layout(fig, 'Top Vehicle Types')


def build_monthly_chart(monthly: pd.DataFrame) -> go.Figure:
    """Area chart with 7-day rolling average overlay."""
    monthly = monthly.copy()
    monthly['date'] = pd.to_datetime(monthly['date'])
    monthly['rolling_7d'] = (
        monthly['violation_count'].rolling(7, min_periods=1).mean()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly['date'],
            y=monthly['violation_count'],
            name='Daily',
            mode='lines',
            line=dict(color='#636e72', width=1),
            fill='tozeroy',
            fillcolor='rgba(99,110,114,0.15)',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly['date'],
            y=monthly['rolling_7d'],
            name='7-day avg',
            mode='lines',
            line=dict(color='#e17055', width=3),
        )
    )
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Violations')
    return _base_layout(fig, 'Violation Trend (Nov 2023 – Apr 2024)')


def build_violation_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart of primary violation types."""
    vt = (
        df.groupby('primary_violation')
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(8)
    )
    fig = go.Figure(
        go.Pie(
            labels=vt['primary_violation'],
            values=vt['count'],
            hole=0.45,
            marker=dict(
                colors=px.colors.qualitative.Pastel,
            ),
            textinfo='label+percent',
            hovertemplate='%{label}: %{value:,}<extra></extra>',
        )
    )
    return _base_layout(fig, 'Violation Type Breakdown')


# ═══════════════════════════════════════════════════════════════════
#  AI Predictions — New Charts
# ═══════════════════════════════════════════════════════════════════

def build_prediction_map(predictions: pd.DataFrame) -> folium.Map:
    """
    Folium map showing predicted violation hotspots with risk-level
    colour coding (Critical=red, High=orange, Medium=yellow, Low=green).
    """
    m = folium.Map(
        location=[12.9716, 77.5946],
        zoom_start=12,
        tiles='CartoDB dark_matter',
        control_scale=True,
    )

    risk_colors = {
        'Critical': '#d63031',
        'High': '#e17055',
        'Medium': '#fdcb6e',
        'Low': '#00b894',
    }

    pred_layer = folium.FeatureGroup(name='Predicted Hotspots', show=True)

    for _, row in predictions.head(30).iterrows():
        if pd.isna(row.get('avg_lat')) or pd.isna(row.get('avg_lon')):
            continue

        risk = str(row.get('risk_level', 'Medium'))
        color = risk_colors.get(risk, '#fdcb6e')
        pred_val = row['predicted_violations']

        # Scale radius by predicted violations
        radius = max(8, min(35, pred_val / 2))

        folium.CircleMarker(
            location=[row['avg_lat'], row['avg_lon']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            weight=2,
            popup=folium.Popup(
                f"<b>{row['junction_name']}</b><br>"
                f"Predicted: <b>{pred_val:.0f}</b> violations<br>"
                f"Risk: <b>{risk}</b><br>"
                f"Rank: #{int(row['rank'])}",
                max_width=280,
            ),
            tooltip=f"{row['junction_name']}: {pred_val:.0f} predicted",
        ).add_to(pred_layer)

        # Add a label for top-10 predictions
        if row['rank'] <= 10:
            icon = folium.DivIcon(
                html=(
                    f'<div style="'
                    f'background:{color};color:#fff;padding:2px 6px;'
                    f'border-radius:10px;font-size:10px;font-weight:700;'
                    f'white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,.4);'
                    f'">{pred_val:.0f}</div>'
                ),
                icon_size=(40, 20),
                icon_anchor=(20, 10),
            )
            folium.Marker(
                location=[row['avg_lat'], row['avg_lon']],
                icon=icon,
            ).add_to(pred_layer)

    pred_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def build_feature_importance_chart(importance: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of XGBoost feature importances."""
    imp = importance.sort_values('importance', ascending=True)

    # Friendly feature names
    name_map = {
        'day_of_week': 'Day of Week',
        'month': 'Month',
        'is_weekend': 'Is Weekend',
        'day_of_month': 'Day of Month',
        'avg_road_weight': 'Road Class (OSM)',
        'peak_fraction': 'Peak Hour Fraction',
        'repeat_fraction': 'Repeat Offender Rate',
        'lag_1d': 'Yesterday\'s Violations',
        'lag_3d': 'Violations 3 Days Ago',
        'lag_7d': 'Violations 7 Days Ago',
        'rolling_7d_mean': '7-Day Rolling Average',
        'rolling_7d_std': '7-Day Volatility',
        'junction_historical_mean': 'Junction Historical Mean',
    }
    imp['feature_label'] = imp['feature'].map(name_map).fillna(imp['feature'])

    fig = go.Figure(
        go.Bar(
            x=imp['importance'],
            y=imp['feature_label'],
            orientation='h',
            marker=dict(
                color=imp['importance'],
                colorscale=[[0, '#6c5ce7'], [0.5, '#a29bfe'], [1, '#fd79a8']],
                showscale=False,
            ),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
        )
    )
    fig.update_xaxes(title='Feature Importance (XGBoost gain)')
    return _base_layout(fig, 'What Drives Parking Violations? — AI Feature Importance')


def build_congestion_impact_chart(congestion_summary: pd.DataFrame) -> go.Figure:
    """
    Bar chart of top junctions by mean congestion impact score,
    colour-coded by severity level.
    """
    top = congestion_summary.head(15).sort_values('mean_congestion', ascending=True)

    severity_colors = {
        'Severe': '#d63031',
        'High': '#e17055',
        'Moderate': '#fdcb6e',
        'Low': '#00b894',
    }
    colors = [severity_colors.get(str(s), '#636e72') for s in top['severity']]

    fig = go.Figure(
        go.Bar(
            x=top['mean_congestion'],
            y=top['junction_name'],
            orientation='h',
            marker=dict(color=colors),
            hovertemplate=(
                '<b>%{y}</b><br>'
                'Congestion Impact: %{x:.3f}<br>'
                '<extra></extra>'
            ),
        )
    )
    fig.update_xaxes(title='Mean Congestion Impact Score')
    return _base_layout(fig, 'Top 15 Traffic-Choking Junctions — Congestion Impact')


def build_prediction_risk_pie(predictions: pd.DataFrame) -> go.Figure:
    """Donut chart showing distribution of risk levels across predicted junctions."""
    risk_counts = predictions['risk_level'].value_counts().reset_index()
    risk_counts.columns = ['risk_level', 'count']

    color_map = {
        'Critical': '#d63031',
        'High': '#e17055',
        'Medium': '#fdcb6e',
        'Low': '#00b894',
    }
    colors = [color_map.get(str(r), '#636e72') for r in risk_counts['risk_level']]

    fig = go.Figure(
        go.Pie(
            labels=risk_counts['risk_level'],
            values=risk_counts['count'],
            hole=0.5,
            marker=dict(colors=colors),
            textinfo='label+percent',
            hovertemplate='%{label}: %{value} junctions<extra></extra>',
        )
    )
    return _base_layout(fig, 'Predicted Risk Distribution')
