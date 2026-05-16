"""Shipment history — stored in session state for this demo."""
import sys
from pathlib import Path

root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="History · PharmaShield",
    page_icon="🗂️",
    layout="wide",
)
st.title("🗂️ Shipment History")
st.caption("Analysis history for this session")

# ── Guard: no analysis yet ─────────────────────────────────────────────────────

if 'last_assessment' not in st.session_state or st.session_state.last_assessment is None:
    st.info(
        "No shipments analyzed yet in this session. "
        "Go to **Route Planner** to analyze a shipment."
    )
    st.stop()

a = st.session_state.last_assessment

# ── Overview card ──────────────────────────────────────────────────────────────

origin_name = a.route.segments[0].origin_name
dest_name   = a.route.segments[-1].destination_name

RISK_COLORS = {
    'Low': '#22c55e', 'Medium': '#f59e0b',
    'High': '#ef4444', 'Critical': '#dc2626',
}
color = RISK_COLORS.get(a.aggregated_risk_label, '#94a3b8')

st.markdown(
    f"""
    <div style="background:{color}22;border:2px solid {color};
                border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;">
        <b style="color:{color};font-size:1.1rem">
            {a.aggregated_risk_label} Risk — {a.aggregated_risk_score}/100
        </b>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span style="color:#e2e8f0">{origin_name} → {dest_name}</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span style="color:#94a3b8">{a.pharma_class.upper()} · {a.weight_kg:.0f} kg</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Summary metrics ────────────────────────────────────────────────────────────

st.subheader("Most Recent Analysis")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Route",      f"{origin_name} → {dest_name}")
col2.metric("Risk Score", f"{a.aggregated_risk_score}/100")
col3.metric("Risk Label", a.aggregated_risk_label)
col4.metric("Packaging",  a.packaging.packaging_type.upper())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pharma Class",    a.pharma_class.upper())
c2.metric("Weight",          f"{a.weight_kg:.0f} kg")
c3.metric("Total Distance",  f"{a.route.total_distance_km:,.0f} km")
c4.metric("Transit Time",    f"{a.route.total_duration_hours:.0f}h")

# ── Segment breakdown ──────────────────────────────────────────────────────────

st.divider()
st.subheader("Segment Breakdown")

rows = []
for sr in a.segment_risks:
    rows.append({
        'Segment':       f"{sr.segment.origin_name} → {sr.segment.destination_name}",
        'Mode':          sr.segment.mode.upper(),
        'Distance (km)': sr.segment.distance_km,
        'Duration (h)':  sr.segment.duration_hours,
        'Risk Score':    sr.risk_score,
        'Risk Label':    sr.risk_label,
        'Max Temp (C)':  sr.weather['max_temp_c'],
        'Geo Risk':      sr.geo_risk_destination['risk_score'],
        'Weather':       sr.weather.get('dominant_condition', 'unknown').title(),
    })

seg_df = pd.DataFrame(rows)
st.dataframe(seg_df, use_container_width=True, hide_index=True)

# ── Packaging detail ───────────────────────────────────────────────────────────

st.divider()
st.subheader("Packaging Recommendation")

pkg = a.packaging
p1, p2, p3 = st.columns(3)
p1.metric("Type",          pkg.packaging_type.upper())
p2.metric("Min Autonomy",  f"{pkg.min_autonomy_hours}h")
p3.metric("Rule",          pkg.rule_id)

st.info(pkg.justification)

# ── Risk factors ───────────────────────────────────────────────────────────────

st.divider()
st.subheader("Top Risk Factors (worst segment)")

worst = max(a.segment_risks, key=lambda sr: sr.risk_score)
st.caption(
    f"Worst segment: **{worst.segment.origin_name} → {worst.segment.destination_name}** "
    f"({worst.segment.mode.upper()}) — {worst.risk_score}/100"
)
for factor in worst.top_risk_factors:
    st.markdown(f"- {factor}")

st.divider()
st.caption(
    "Note: History persists only for the current browser session. "
    "In a production deployment, this would be stored in a database."
)
