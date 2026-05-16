import sys
from pathlib import Path

root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import streamlit as st

st.set_page_config(
    page_title="Ethics & GDPR · PharmaShield",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Ethics, Privacy & GDPR")
st.caption(
    "Transparency about how PharmaShield AI works, its limitations, "
    "and data protection practices"
)


# ── Section 1 — Data & model transparency ─────────────────────────────────────

st.subheader("📊 Data & Model Transparency")
st.markdown("""
**Training data**: PharmaShield's predictive model was trained on a **synthetic dataset** of 10,000
pharmaceutical shipments generated using domain-validated probability distributions
(pharma class sensitivity, carrier reliability benchmarks, cold chain incident rates from GDP literature).

This approach was chosen because real pharmaceutical cold chain data is commercially
sensitive and not publicly available. In a production deployment, the synthetic dataset
would be replaced with historical shipment data from the client organisation.

**What this means for the model's predictions**: The model captures realistic relationships
between features and excursion risk, but its absolute probability values are calibrated to
synthetic distributions — not to any specific company's operational history.
Predictions should be used as **decision-support signals**, not as definitive verdicts.
""")


# ── Section 2 — Biases & limitations ─────────────────────────────────────────

st.divider()
st.subheader("⚖️ Known Biases & Limitations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**Geographic bias**
- Node coverage is stronger for Europe and major Asian hubs
- African and Central Asian corridors are underrepresented
- Routing heuristics may not reflect actual carrier networks in all regions

**Temporal bias**
- Weather forecasts use real-time Open-Meteo data (accurate, 16-day horizon)
- Geopolitical risk (GDELT) may underreport incidents in countries with low English-language media coverage
- Seasonal patterns in the synthetic data use simplified latitude-temperature models
""")

with col2:
    st.markdown("""
**Model limitations**
- AUC ~0.70 reflects an intentional stochastic component in synthetic data generation
  (Bernoulli sampling). Real cold chain failures have irreducible randomness.
- The model does not account for individual product formulation stability
- Packaging autonomy is modelled as a binary threshold, not a continuous degradation curve

**Human oversight requirement**
- All packaging and routing recommendations must be reviewed by a qualified
  pharmacist or cold chain specialist before operational use
- The system is explicitly designed as **decision-support**, not autonomous decision-making
""")


# ── Section 3 — GDPR ─────────────────────────────────────────────────────────

st.divider()
st.subheader("🔒 GDPR & Privacy")
st.markdown("""
**Data processed by PharmaShield**:

| Data type | Personal? | How handled |
|---|---|---|
| Shipment origin / destination | No — logistics infrastructure nodes | Used for routing calculation only |
| Product type & weight | No — operational data | Used for risk scoring only |
| Carrier reliability score | Potentially — if linked to individual drivers | Pseudonymised; only aggregate score used |
| Weather data (Open-Meteo) | No | Fetched per coordinates; no user data sent |
| Geopolitical data (GDELT) | No | Country-level only; no individual tracking |
| Shipment history (local cache) | Potentially | Stored locally only; user controls deletion |

**No personal data** is transmitted to external APIs. Open-Meteo and GDELT receive
only geographic coordinates and country names respectively.

In a production context, full GDPR compliance would require a Data Protection Impact
Assessment (DPIA) and formal data processing agreements with carrier partners.
""")


# ── Section 4 — AI transparency ───────────────────────────────────────────────

st.divider()
st.subheader("🤖 AI Usage Transparency")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
**AI tools used in this project**

- **Claude (Anthropic)** — research synthesis, code scaffolding, documentation
- **scikit-learn & XGBoost** — model training and evaluation
- **SHAP** — model explainability (feature attribution)
- **Open-Meteo API** — real-time weather forecasts
- **GDELT v2 API** — geopolitical risk signals from global news
""")

with col_b:
    st.markdown("""
**Human contributions**

- Problem framing and domain validation (pharma cold chain requirements)
- Dataset design and synthetic data generation logic
- Feature engineering decisions and regulatory alignment
- Ethical review and limitation documentation
- UX design, testing, and presentation
- Final validation of all AI-generated outputs against domain knowledge
""")

st.info(
    "Per the course requirement: *AI tools supported our work — "
    "they did not replace our judgement.*"
)


# ── Section 5 — Regulatory context ────────────────────────────────────────────

st.divider()
st.subheader("📋 Regulatory Context")
st.markdown("""
PharmaShield's packaging recommendations are aligned with the following standards:

| Standard | Scope |
|---|---|
| **EU GDP Guidelines 2013/C 343/01** | Good Distribution Practice for medicinal products |
| **WHO TRS 961 Annex 9** | Model guidance for temperature-sensitive products |
| **USP &lt;1079&gt;** | Good Storage and Shipping Practices |
| **USP &lt;1150&gt;** | Pharmaceutical Stability (CRT definition) |
| **IATA TCR Chapter 17** | Perishable cargo regulations (frozen/ultracold air) |
| **ISTA 7E** | Thermal performance testing for temperature-controlled distribution |

These references are used to define temperature ranges and packaging decision thresholds.
They do not constitute legal or regulatory approval of PharmaShield's output.
""")

st.divider()
st.caption(
    "PharmaShield AI · Thomas More University · AI Tools AY 2025-2026 · "
    "Team: Alejandro · Juan · Álvaro · Mateo"
)
