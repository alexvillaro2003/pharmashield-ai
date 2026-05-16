"""Batch shipment analysis via CSV upload."""
import sys
from pathlib import Path

root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import pandas as pd
import streamlit as st
from core.risk_engine import assess_shipment

st.set_page_config(
    page_title="Batch Upload · PharmaShield",
    page_icon="📁",
    layout="wide",
)
st.title("📁 Batch Shipment Analysis")
st.caption("Upload a CSV file to analyze multiple shipments at once")

# ── Template download ──────────────────────────────────────────────────────────

template_df = pd.DataFrame([
    {
        'origin_id': 'brussels', 'destination_id': 'shanghai',
        'pharma_class': 'cool', 'weight_kg': 200, 'carrier_reliability': 0.85,
    },
    {
        'origin_id': 'rotterdam', 'destination_id': 'los_angeles',
        'pharma_class': 'crt', 'weight_kg': 500, 'carrier_reliability': 0.90,
    },
    {
        'origin_id': 'frankfurt_airport', 'destination_id': 'jfk_airport',
        'pharma_class': 'ultracold', 'weight_kg': 80, 'carrier_reliability': 0.95,
    },
])

st.subheader("1. Download Template")
st.caption("Fill in the template with your shipments and upload it below.")
st.download_button(
    "⬇️ Download CSV Template",
    data=template_df.to_csv(index=False),
    file_name="pharmashield_batch_template.csv",
    mime="text/csv",
)

with st.expander("Valid values reference"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**pharma_class** — one of:
- `crt` · `cool` · `frozen` · `ultracold`

**carrier_reliability** — float between 0.60 and 0.99
""")
    with col_b:
        st.markdown("""
**origin_id / destination_id** — node IDs from nodes.json, e.g.:
`brussels`, `rotterdam`, `frankfurt_airport`, `schiphol`,
`shanghai`, `singapore`, `jfk_airport`, `los_angeles`, `sydney`, ...
""")

# ── Upload & analyze ───────────────────────────────────────────────────────────

st.divider()
st.subheader("2. Upload Your CSV")
uploaded = st.file_uploader("Upload filled CSV", type=['csv'])

if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
st.write(f"Found **{len(df)} shipment(s)** to analyze.")
st.dataframe(df, use_container_width=True)

required_cols = {'origin_id', 'destination_id', 'pharma_class', 'weight_kg', 'carrier_reliability'}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

if not st.button("🔍 Analyze All Shipments", type="primary"):
    st.stop()

# ── Run batch analysis ─────────────────────────────────────────────────────────

results = []
progress = st.progress(0)
status   = st.empty()

for idx, (_, row) in enumerate(df.iterrows()):
    status.text(f"Analyzing {row['origin_id']} → {row['destination_id']} ({idx+1}/{len(df)})...")
    try:
        a = assess_shipment(
            origin_id=str(row['origin_id']),
            destination_id=str(row['destination_id']),
            pharma_class=str(row['pharma_class']),
            weight_kg=float(row['weight_kg']),
            carrier_reliability=float(row['carrier_reliability']),
        )
        results.append({
            'Origin':         a.route.segments[0].origin_name,
            'Destination':    a.route.segments[-1].destination_name,
            'Pharma Class':   str(row['pharma_class']).upper(),
            'Risk Score':     a.aggregated_risk_score,
            'Risk Label':     a.aggregated_risk_label,
            'Packaging':      a.packaging.packaging_type.upper(),
            'Autonomy (h)':   a.packaging.min_autonomy_hours,
            'Transit (h)':    round(a.route.total_duration_hours, 1),
            'Distance (km)':  round(a.route.total_distance_km, 0),
            'Segments':       len(a.route.segments),
        })
    except Exception as exc:
        results.append({
            'Origin':       str(row['origin_id']),
            'Destination':  str(row['destination_id']),
            'Pharma Class': str(row['pharma_class']).upper(),
            'Risk Score':   None,
            'Risk Label':   f'ERROR: {exc}',
            'Packaging':    '-',
            'Autonomy (h)': None,
            'Transit (h)':  None,
            'Distance (km)': None,
            'Segments':     None,
        })
    progress.progress((idx + 1) / len(df))

status.empty()
progress.empty()

# ── Results ────────────────────────────────────────────────────────────────────

results_df = pd.DataFrame(results)
errors = results_df['Risk Label'].str.startswith('ERROR', na=False).sum()

if errors:
    st.warning(f"{errors} shipment(s) failed — see Risk Label column for details.")
else:
    st.success(f"✅ All {len(results_df)} shipments analyzed successfully.")

st.dataframe(results_df, use_container_width=True)

# Summary stats
ok = results_df[~results_df['Risk Score'].isna()]
if len(ok):
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Risk Score", f"{ok['Risk Score'].mean():.1f}/100")
    c2.metric("Highest Risk",
              f"{ok.loc[ok['Risk Score'].idxmax(), 'Origin']} → "
              f"{ok.loc[ok['Risk Score'].idxmax(), 'Destination']}")
    c3.metric("Active Packaging Required",
              f"{(ok['Packaging'] == 'ACTIVE').sum()}/{len(ok)}")

st.download_button(
    "⬇️ Download Results CSV",
    data=results_df.to_csv(index=False),
    file_name="pharmashield_batch_results.csv",
    mime="text/csv",
)
