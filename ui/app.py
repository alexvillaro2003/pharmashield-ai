import streamlit as st

st.set_page_config(
    page_title="PharmaShield AI",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .css-1d391kg { padding-top: 1rem; }

    [data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2d3148;
        border-radius: 8px;
        padding: 1rem;
    }

    .risk-low      { color: #22c55e; font-weight: 700; }
    .risk-medium   { color: #f59e0b; font-weight: 700; }
    .risk-high     { color: #ef4444; font-weight: 700; }
    .risk-critical { color: #dc2626; font-weight: 700; font-size: 1.1em; }

    .segment-card {
        background: #1a1d27;
        border: 1px solid #2d3148;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    footer { visibility: hidden; }

    .main-header {
        background: linear-gradient(135deg, #1a1d27 0%, #2d3148 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #7c6af7;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/snowflake.png", width=60)
    st.title("PharmaShield AI")
    st.caption("Cold Chain Risk Intelligence")
    st.divider()
    st.markdown("""
**Navigate:**
- 📍 Route Planner
- 🔍 Explainability
- 📊 Model Evaluation
- 🛡️ Ethics & GDPR
""")
    st.divider()
    st.caption("Thomas More University · AI Tools 2025-26")
    st.caption("Team: Alejandro · Juan · Álvaro · Mateo")

st.markdown("""
<div class="main-header">
    <h1>🧊 PharmaShield AI</h1>
    <p style="color: #94a3b8; margin: 0;">
        Cold Chain Risk Prediction for Pharmaceutical Shipments
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 📍 Route Planner")
    st.write("Analyze any global pharmaceutical shipment route. Get real-time risk scores per segment and packaging recommendations.")

with col2:
    st.markdown("### 🤖 AI-Powered")
    st.write("Random Forest model trained on 10,000 synthetic pharma shipments. Features include temperature buffer, autonomy gap, carrier reliability.")

with col3:
    st.markdown("### 🌍 Live Data")
    st.write("Real-time weather via Open-Meteo. Geopolitical risk via GDELT. Both APIs cached to ensure fast response times.")

st.divider()
st.info("👈 Use the sidebar to navigate between pages. Start with **Route Planner** to analyze a shipment.")
