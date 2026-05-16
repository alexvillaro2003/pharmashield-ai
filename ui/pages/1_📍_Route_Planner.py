import sys
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Route Planner · PharmaShield",
    page_icon="📍",
    layout="wide",
)

# ── Project root on sys.path so core.* imports work from the pages/ subdirectory
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json
import folium
from streamlit_folium import st_folium
from core.risk_engine import assess_shipment, ShipmentRiskAssessment
from core.routing import plan_route

# ── Constants ──────────────────────────────────────────────────────────────────

MODE_COLORS = {
    'road': '#f59e0b',
    'sea':  '#3b82f6',
    'air':  '#8b5cf6',
    'rail': '#22c55e',
}
MODE_EMOJI = {'road': '🚛', 'sea': '🚢', 'air': '✈️', 'rail': '🚂'}
RISK_EMOJI = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴', 'Critical': '🚨'}
PKG_EMOJI  = {'none': '📦', 'passive': '❄️', 'active': '⚡'}
PKG_COLOR  = {'none': '#22c55e', 'passive': '#3b82f6', 'active': '#f59e0b'}

PHARMA_OPTIONS = {
    'crt':       '🟢 CRT (15-25°C) — Tablets, capsules',
    'cool':      '🔵 Cool (2-8°C) — Vaccines, insulin',
    'frozen':    '🟣 Frozen (-20 to -15°C) — Some biologics',
    'ultracold': '⚫ Ultra-Cold (-80 to -60°C) — mRNA vaccines',
}


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data
def load_nodes() -> dict:
    with open(_root / 'data' / 'master' / 'nodes.json', encoding='utf-8') as f:
        return {n['id']: n for n in json.load(f)['nodes']}


@st.cache_data
def load_pharma_classes() -> dict:
    with open(_root / 'data' / 'master' / 'pharma_classes.json', encoding='utf-8') as f:
        return {p['id']: p for p in json.load(f)['pharma_classes']}


# ── Map rendering ──────────────────────────────────────────────────────────────

def _build_map(assessment: ShipmentRiskAssessment) -> folium.Map:
    route = assessment.route
    all_lats = [s.origin_coords[0] for s in route.segments] + \
               [route.segments[-1].destination_coords[0]]
    all_lons = [s.origin_coords[1] for s in route.segments] + \
               [route.segments[-1].destination_coords[1]]
    center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]

    m = folium.Map(location=center, zoom_start=3, tiles='CartoDB dark_matter')

    for i, (seg, sr) in enumerate(zip(route.segments, assessment.segment_risks)):
        color = MODE_COLORS.get(seg.mode, '#ffffff')

        folium.PolyLine(
            locations=[list(seg.origin_coords), list(seg.destination_coords)],
            color=color,
            weight=4,
            opacity=0.85,
            tooltip=(
                f"{seg.mode.upper()}: {seg.origin_name} → {seg.destination_name} "
                f"| Risk: {sr.risk_score}/100"
            ),
        ).add_to(m)

        if i == 0:
            folium.Marker(
                list(seg.origin_coords),
                popup=f"<b>ORIGIN</b><br>{seg.origin_name}",
                icon=folium.Icon(color='green', icon='play'),
            ).add_to(m)

        folium.CircleMarker(
            list(seg.destination_coords),
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=(
                f"<b>{seg.destination_name}</b><br>"
                f"Risk: {sr.risk_score}/100 ({sr.risk_label})<br>"
                f"Weather: {sr.weather['max_temp_c']}°C max"
            ),
        ).add_to(m)

    last = route.segments[-1]
    folium.Marker(
        list(last.destination_coords),
        popup=f"<b>DESTINATION</b><br>{last.destination_name}",
        icon=folium.Icon(color='red', icon='flag'),
    ).add_to(m)

    legend = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(0,0,0,0.8);padding:10px 15px;
                border-radius:8px;font-size:12px;color:white;">
        <b>Transport Mode</b><br>
        <span style="color:#f59e0b">━━</span> Road &nbsp;
        <span style="color:#3b82f6">━━</span> Sea &nbsp;
        <span style="color:#8b5cf6">━━</span> Air &nbsp;
        <span style="color:#22c55e">━━</span> Rail
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    return m


# ── Results renderer ───────────────────────────────────────────────────────────

def _render_results(assessment: ShipmentRiskAssessment, col_map) -> None:
    route = assessment.route

    with col_map:
        st_folium(_build_map(assessment), width=None, height=450, returned_objects=[])
        st.divider()

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric(
            "🎯 Overall Risk",
            f"{assessment.aggregated_risk_score}/100",
            delta=assessment.aggregated_risk_label,
            delta_color="inverse",
        )
        mcol2.metric("📏 Total Distance", f"{route.total_distance_km:,.0f} km")
        mcol3.metric("⏱️ Transit Time",   f"{route.total_duration_hours:.0f}h")
        mcol4.metric("🔄 Handovers",      str(route.num_handovers))

    # Segment detail — full width below the two-column layout
    st.divider()
    st.subheader("📦 Segment Analysis")

    for i, (seg, sr) in enumerate(zip(route.segments, assessment.segment_risks)):
        label = (
            f"{MODE_EMOJI.get(seg.mode, '📦')} Segment {i+1}: "
            f"{seg.origin_name} → {seg.destination_name} "
            f"| {seg.mode.upper()} "
            f"| {RISK_EMOJI.get(sr.risk_label, '⚪')} {sr.risk_score}/100 {sr.risk_label}"
        )
        with st.expander(label, expanded=(i == 0)):
            c1, c2, c3 = st.columns(3)
            c1.metric("Distance",   f"{seg.distance_km:,.0f} km")
            c2.metric("Duration",   f"{seg.duration_hours:.1f}h")
            c3.metric("Risk Score", f"{sr.risk_score}/100")

            c4, c5, c6 = st.columns(3)
            c4.metric("Max Temp",       f"{sr.weather['max_temp_c']}°C")
            c5.metric(
                "Geo Risk Origin",
                f"{sr.geo_risk_origin['risk_score']}/10 {sr.geo_risk_origin['risk_label']}",
            )
            c6.metric(
                "Geo Risk Dest",
                f"{sr.geo_risk_destination['risk_score']}/10 {sr.geo_risk_destination['risk_label']}",
            )

            st.markdown("**Top Risk Factors:**")
            for factor in sr.top_risk_factors:
                st.markdown(f"- {factor}")

            if sr.weather.get('dominant_condition') != 'unknown':
                st.caption(
                    f"Weather: {sr.weather['dominant_condition'].title()} "
                    f"| Source: {sr.weather['source']}"
                )

    # Packaging recommendation
    st.divider()
    st.subheader("📦 Packaging Recommendation")

    pkg = assessment.packaging
    pc1, pc2 = st.columns([1, 2])

    with pc1:
        st.markdown(
            f"""
            <div style="background:{PKG_COLOR.get(pkg.packaging_type,'#333')};
                        border-radius:12px;padding:1.5rem;
                        text-align:center;color:white;">
                <div style="font-size:2.5rem">{PKG_EMOJI.get(pkg.packaging_type,'📦')}</div>
                <div style="font-size:1.4rem;font-weight:700;text-transform:uppercase">
                    {pkg.packaging_type} Packaging
                </div>
                <div style="font-size:1rem;opacity:0.9">
                    Min. {pkg.min_autonomy_hours}h autonomy
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with pc2:
        st.markdown(f"**Rule applied:** {pkg.rule_name} (`{pkg.rule_id}`)")
        st.markdown("**Justification:**")
        st.info(pkg.justification)


# ── Page layout ────────────────────────────────────────────────────────────────

st.title("📍 Route Planner")
st.caption("Analyze pharmaceutical shipment risk across any global route")

col_form, col_map = st.columns([1, 2])

nodes         = load_nodes()
pharma_classes = load_pharma_classes()

node_options = sorted(
    [
        (f"{n['name']} ({n['type'].replace('_', ' ').title()}, {n['country']})", nid)
        for nid, n in nodes.items()
    ],
    key=lambda x: x[0],
)
node_labels = [x[0] for x in node_options]
node_ids    = [x[1] for x in node_options]

default_origin_idx = node_ids.index('brussels') if 'brussels' in node_ids else 0
default_dest_idx   = node_ids.index('shanghai')  if 'shanghai'  in node_ids else 1

with col_form:
    st.subheader("Shipment Details")

    origin_label      = st.selectbox("🛫 Origin",      node_labels, index=default_origin_idx)
    origin_id         = node_ids[node_labels.index(origin_label)]

    destination_label = st.selectbox("🛬 Destination", node_labels, index=default_dest_idx)
    destination_id    = node_ids[node_labels.index(destination_label)]

    if origin_id == destination_id:
        st.warning("Origin and destination must be different.")

    st.divider()

    pharma_values = list(PHARMA_OPTIONS.values())
    pharma_keys   = list(PHARMA_OPTIONS.keys())
    pharma_label  = st.selectbox("💊 Product Type", pharma_values)
    pharma_id     = pharma_keys[pharma_values.index(pharma_label)]

    st.divider()

    weight_kg          = st.number_input(
        "⚖️ Weight (kg)", min_value=1.0, max_value=5000.0, value=200.0, step=10.0
    )
    carrier_reliability = st.slider(
        "🚚 Carrier Reliability",
        min_value=0.60, max_value=0.99, value=0.85, step=0.01,
        help="Historical on-time delivery rate of the carrier",
    )

    st.divider()

    analyze_btn = st.button(
        "🔍 Analyze Shipment",
        type="primary",
        use_container_width=True,
        disabled=(origin_id == destination_id),
    )

# ── Session state + analysis ───────────────────────────────────────────────────

if 'last_assessment' not in st.session_state:
    st.session_state.last_assessment = None

if analyze_btn and origin_id != destination_id:
    with st.spinner("🔄 Analyzing route... fetching weather & geopolitical data..."):
        try:
            st.session_state.last_assessment = assess_shipment(
                origin_id=origin_id,
                destination_id=destination_id,
                pharma_class=pharma_id,
                weight_kg=weight_kg,
                carrier_reliability=carrier_reliability,
            )
        except Exception as exc:
            st.error(f"Error analyzing shipment: {exc}")
            st.stop()

if st.session_state.last_assessment:
    _render_results(st.session_state.last_assessment, col_map)
else:
    with col_map:
        st.info(
            "Configure your shipment on the left and click **Analyze Shipment** "
            "to see the risk assessment and route map."
        )
