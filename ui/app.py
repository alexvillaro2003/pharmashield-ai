import streamlit as st

st.set_page_config(
    page_title="PharmaShield AI",
    page_icon="🧊",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/ice.png",
        width=64,
    )
    st.markdown("## PharmaShield AI")
    st.markdown(
        "Cold-chain risk prediction for pharmaceutical shipments. "
        "Use the pages below to plan routes, inspect model explanations, "
        "evaluate performance, and manage batch jobs."
    )
    st.divider()
    st.markdown("**Pages**")
    st.markdown(
        "- 📍 Route Planner\n"
        "- 🔍 Explainability\n"
        "- 📊 Model Evaluation\n"
        "- 📁 Batch Upload\n"
        "- 🗂️ History\n"
        "- 🛡️ Ethics & GDPR"
    )
    st.divider()
    st.caption("Thomas More University · AI Tools 2025–2026")

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("# 🧊 PharmaShield AI")
st.markdown("### Cold Chain Risk Prediction for Pharmaceutical Shipments")
st.divider()

# ── Description ───────────────────────────────────────────────────────────────
col_desc, col_how = st.columns([3, 2], gap="large")

with col_desc:
    st.markdown("#### What is PharmaShield?")
    st.markdown(
        """
        PharmaShield predicts the **probability of thermal excursion** along a
        multimodal pharmaceutical shipment route (road → air → road) and
        recommends the most appropriate packaging solution.

        It combines:
        - **Machine learning** (XGBoost) trained on synthetic cold-chain data
        - **Real-time weather signals** along the route
        - **Geopolitical delay risk** from open event data (GDELT)
        - **SHAP explainability** so every prediction is auditable
        - **Interactive Folium maps** for visual route inspection
        """
    )

with col_how:
    st.markdown("#### How to use")
    st.markdown(
        """
        **Step 1 — Define your route**
        Go to *Route Planner*, enter origin, destination, transport modes,
        and the drug's temperature requirements.

        **Step 2 — Review the risk prediction**
        The model returns a risk score and the top contributing factors,
        visualised on an interactive map.

        **Step 3 — Act on the recommendation**
        PharmaShield suggests active or passive packaging and flags the
        highest-risk leg of the journey.
        """
    )

st.divider()

# ── Team credits ──────────────────────────────────────────────────────────────
st.markdown("#### Team")
c1, c2, c3, c4 = st.columns(4)
for col, name in zip(
    [c1, c2, c3, c4],
    ["_(Name 1)_", "_(Name 2)_", "_(Name 3)_", "_(Name 4)_"],
):
    with col:
        st.info(name)

st.caption("Thomas More University · AI Tools 2025–2026 · Semester 2")
