"""
ParkSight — AI Reporter (Gemini-powered)
Generates natural-language patrol briefings by combining XGBoost
predictions, congestion impact scores, and EPI rankings into
a structured executive summary via Google's Gemini API.

Falls back to a sophisticated rule-based generator if the API
is unavailable or no key is provided.
"""

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def _build_data_summary(
    predictions: pd.DataFrame,
    congestion: pd.DataFrame,
    epi: pd.DataFrame,
    metrics: dict,
) -> str:
    """Build a structured data summary to feed into the LLM prompt."""
    parts = []

    # Model performance
    if metrics:
        parts.append(
            f"MODEL PERFORMANCE:\n"
            f"- XGBoost R² score: {metrics.get('test_r2', 'N/A')}\n"
            f"- Mean Absolute Error: {metrics.get('test_mae', 'N/A')} violations/day\n"
            f"- Training period: {metrics.get('date_range', 'N/A')}\n"
            f"- Junctions analysed: {metrics.get('n_junctions', 'N/A')}"
        )

    # Top predictions
    if predictions is not None and not predictions.empty:
        top5 = predictions.head(5)
        pred_lines = []
        for _, r in top5.iterrows():
            pred_lines.append(
                f"  #{int(r['rank'])} {r['junction_name']}: "
                f"{r['predicted_violations']:.0f} predicted violations "
                f"(Risk: {r['risk_level']})"
            )
        parts.append(
            f"TOP 5 PREDICTED HOTSPOTS (next day):\n" +
            "\n".join(pred_lines)
        )

    # Top congestion impact zones
    if congestion is not None and not congestion.empty:
        top5_c = congestion.head(5)
        cong_lines = []
        for _, r in top5_c.iterrows():
            cong_lines.append(
                f"  #{int(r['rank'])} {r['junction_name']}: "
                f"Impact={r['mean_congestion']:.3f}, "
                f"Centrality={r['avg_centrality']:.3f}, "
                f"Severity={r['severity']}"
            )
        parts.append(
            f"TOP 5 CONGESTION IMPACT ZONES:\n" +
            "\n".join(cong_lines)
        )

    # Top EPI junctions
    if epi is not None and not epi.empty:
        top5_e = epi.head(5)
        epi_lines = []
        for _, r in top5_e.iterrows():
            epi_lines.append(
                f"  #{int(r['rank'])} {r['junction_name']}: "
                f"EPI={r['epi_score']:.1f}, "
                f"Violations={int(r['total_violations'])}"
            )
        parts.append(
            f"TOP 5 ENFORCEMENT PRIORITY JUNCTIONS:\n" +
            "\n".join(epi_lines)
        )

    return "\n\n".join(parts)


def _build_prompt(data_summary: str) -> str:
    """Construct the system + user prompt for Gemini."""
    return f"""You are ParkSight AI, an intelligent traffic enforcement advisor for Bengaluru Traffic Police.

Based on the following data analysis from our AI prediction engine, write a concise, professional patrol briefing.

DATA:
{data_summary}

INSTRUCTIONS:
1. Start with a one-line alert headline (e.g., "🚨 HIGH ALERT: 3 Critical Hotspots Predicted")
2. Write 2-3 sentences summarising the overall situation
3. List the top 3 recommended patrol deployment actions with specific junction names and time recommendations
4. Add one insight about which road characteristics or time patterns drive the most violations
5. End with a confidence statement based on the model's R² score
6. Keep the entire briefing under 250 words
7. Use professional but accessible language suitable for a police operations room
8. Do NOT invent junction names or numbers — use ONLY what is provided in the data above"""


def generate_ai_briefing(
    predictions: pd.DataFrame = None,
    congestion: pd.DataFrame = None,
    epi: pd.DataFrame = None,
    metrics: dict = None,
    api_key: str = None,
) -> dict:
    """
    Generate an AI-powered patrol briefing.

    Tries Gemini API first; falls back to rule-based if API fails.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of predictive_model.predict_future_violations.
    congestion : pd.DataFrame
        Output of network_analysis.get_junction_congestion_summary.
    epi : pd.DataFrame
        Output of epi_scorer.compute_junction_epi.
    metrics : dict
        Model training metrics.
    api_key : str, optional
        Gemini API key. If None, uses rule-based fallback.

    Returns
    -------
    dict with keys:
        'report': str — the generated briefing text
        'source': str — 'gemini' or 'rule-based'
        'timestamp': str — generation timestamp
    """
    data_summary = _build_data_summary(predictions, congestion, epi, metrics)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    # ── Try Gemini API ──────────────────────────────────────
    if api_key:
        keys = [k.strip() for k in api_key.replace(";", ",").split(",") if k.strip()]
        last_error = ""
        for i, key in enumerate(keys):
            try:
                import google.generativeai as genai
                genai.configure(api_key=key)

                model = genai.GenerativeModel('gemini-2.0-flash')
                prompt = _build_prompt(data_summary)

                response = model.generate_content(prompt)
                report_text = response.text

                logger.info(f'AI briefing generated via Gemini API (Key #{i+1})')
                return {
                    'report': report_text,
                    'source': f'Google Gemini 2.0 Flash (Key #{i+1})',
                    'timestamp': timestamp,
                    'data_summary': data_summary,
                }
            except Exception as e:
                logger.warning(f'Gemini API failed for key #{i+1} (%s)', e)
                last_error = str(e)
        
        # If all keys failed
        report = _generate_rule_based(predictions, congestion, epi, metrics)
        return {
            'report': report,
            'source': 'Rule-based (Gemini API failed)',
            'timestamp': timestamp,
            'data_summary': data_summary,
            'error': f"All {len(keys)} API keys failed. Last error: {last_error}",
        }

    # ── Rule-based fallback ─────────────────────────────────
    report = _generate_rule_based(predictions, congestion, epi, metrics)
    return {
        'report': report,
        'source': 'Rule-based (no API key)',
        'timestamp': timestamp,
        'data_summary': data_summary,
    }


def _generate_rule_based(
    predictions: pd.DataFrame,
    congestion: pd.DataFrame,
    epi: pd.DataFrame,
    metrics: dict,
) -> str:
    """
    Sophisticated rule-based report that mimics LLM output.
    Used when no API key is provided or API call fails.
    """
    lines = []

    # ── Headline ────────────────────────────────────────────
    critical_count = 0
    if predictions is not None and not predictions.empty:
        critical_count = (predictions['risk_level'] == 'Critical').sum()

    if critical_count >= 3:
        lines.append(f"🚨 **HIGH ALERT: {critical_count} Critical Hotspots Predicted**")
    elif critical_count >= 1:
        lines.append(f"⚠️ **ALERT: {critical_count} Critical Hotspot(s) Detected**")
    else:
        lines.append("ℹ️ **Status: Standard Enforcement Advisory**")

    lines.append("")

    # ── Situation summary ───────────────────────────────────
    if predictions is not None and not predictions.empty:
        top_j = predictions.iloc[0]['junction_name']
        top_v = predictions.iloc[0]['predicted_violations']
        total_pred = predictions['predicted_violations'].sum()
        lines.append(
            f"Our AI prediction model forecasts approximately **{total_pred:.0f} total "
            f"violations** across {len(predictions)} monitored junctions for the next "
            f"operational period. **{top_j}** is projected as the highest-risk zone "
            f"with **{top_v:.0f} predicted violations**."
        )
    lines.append("")

    # ── Congestion insight ──────────────────────────────────
    if congestion is not None and not congestion.empty:
        severe = congestion[congestion['severity'] == 'Severe']
        if not severe.empty:
            severe_names = ", ".join(severe['junction_name'].head(3).tolist())
            lines.append(
                f"**Congestion Impact Analysis:** Junctions **{severe_names}** "
                f"show the highest traffic flow disruption based on road network "
                f"centrality analysis. Illegal parking at these locations affects "
                f"the maximum number of alternative traffic routes."
            )
        else:
            top_c = congestion.iloc[0]
            lines.append(
                f"**Congestion Impact Analysis:** **{top_c['junction_name']}** "
                f"ranks highest in congestion impact (score: {top_c['mean_congestion']:.3f}) "
                f"due to its high road network centrality."
            )
    lines.append("")

    # ── Patrol recommendations ──────────────────────────────
    lines.append("### 📋 Recommended Patrol Deployment")
    lines.append("")

    if predictions is not None and not predictions.empty:
        for i, (_, r) in enumerate(predictions.head(3).iterrows(), 1):
            risk = r['risk_level']
            emoji = "🔴" if risk in ['Critical', 'High'] else "🟡" if risk == 'Medium' else "🟢"
            lines.append(
                f"{emoji} **Priority {i}: {r['junction_name']}** — "
                f"Deploy patrol during peak hours (7–9 AM, 5–8 PM). "
                f"Predicted violations: {r['predicted_violations']:.0f}. "
                f"Risk level: **{risk}**."
            )
    lines.append("")

    # ── Model confidence ────────────────────────────────────
    if metrics:
        r2 = metrics.get('test_r2', 0)
        confidence = "High" if r2 >= 0.7 else "Moderate" if r2 >= 0.4 else "Low"
        lines.append(
            f"**Model Confidence:** {confidence} (R² = {r2:.4f}). "
            f"The model explains {r2*100:.1f}% of the variance in historical "
            f"violation patterns across {metrics.get('n_junctions', 'N/A')} junctions."
        )

    lines.append("")
    lines.append(f"*Report generated by ParkSight AI — {datetime.now().strftime('%d %b %Y, %H:%M')}*")

    return "\n".join(lines)


def export_briefing_to_pdf(report_text: str) -> bytes:
    """
    Convert the markdown briefing report text into a clean PDF document bytes.
    Returns None if fpdf2 is not installed.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(108, 92, 231)  # Slate Purple Theme color
    pdf.cell(0, 10, "ParkSight AI Briefing Report", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(45, 52, 54)  # Professional dark gray

    # Strip out emojis, non-ASCII quotes/dashes and double asterisks to keep standard PDF encoding safe
    replacements = {
        "🚨": "[ALERT]",
        "⚠️": "[WARNING]",
        "ℹ️": "[INFO]",
        "🏆": "[TOP]",
        "🚗": "[VEHICLE]",
        "📋": "[DEPLOYMENT]",
        "🔮": "[PREDICTION]",
        "🔴": "[HIGH]",
        "🟡": "[MEDIUM]",
        "🟢": "[LOW]",
        "**": "",
        "—": "-",  # Replace em-dash
        "–": "-",  # Replace en-dash
        "“": '"',  # Replace smart quotes
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    
    clean_text = report_text
    for emoji, text in replacements.items():
        clean_text = clean_text.replace(emoji, text)

    # Coerce to latin-1 encoding by ignoring any other remaining complex unicode characters
    try:
        clean_text = clean_text.encode('latin-1', 'replace').decode('latin-1')
    except Exception:
        pass

    # Standard A4 width is 210mm. With 10mm margins, printable width is 190mm.
    # Replace empty lines with vertical spacing, and handle normal text with explicit width.
    for line in clean_text.split("\n"):
        stripped_line = line.strip()
        if not stripped_line:
            pdf.ln(4)
        else:
            # Use 190 width explicitly instead of 0 to avoid wrap margin calculation exceptions
            pdf.multi_cell(190, 6, stripped_line)

    return bytes(pdf.output())

