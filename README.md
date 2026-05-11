# PharmaShield AI — Cold Chain Risk Prediction for Pharma Shipments

PharmaShield is a Streamlit-based decision-support tool that predicts the risk of thermal excursion in multimodal pharmaceutical shipments and recommends the optimal packaging solution (active or passive). It combines ML models (XGBoost + scikit-learn), explainability (SHAP), real-time weather and geopolitical data, and interactive route visualisation (Folium) into a single, auditable interface designed for cold-chain logistics managers.

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic training dataset
python ml/generate_dataset.py

# 4. Train the risk model
python ml/train.py

# 5. Launch the app
streamlit run ui/app.py
```

---

## Project Structure

```
pharmashield/
├── requirements.txt          # Python dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml           # Dark theme configuration
├── data/
│   ├── master/               # Reference data (routes, airports, depots)
│   ├── synthetic/            # Generated training data
│   └── cache/
│       ├── weather/          # Cached weather API responses
│       └── gdelt/            # Cached geopolitical event data
├── core/
│   ├── routing.py            # Route parsing & segment builder
│   ├── risk_engine.py        # Feature assembly & model inference
│   ├── packaging.py          # Packaging recommendation logic
│   └── apis/
│       ├── weather.py        # OpenWeatherMap / Open-Meteo client
│       └── geopolitics.py    # GDELT client for delay risk signals
├── ml/
│   ├── generate_dataset.py   # Synthetic dataset generator
│   ├── train.py              # Model training & serialisation
│   └── model.pkl             # Trained XGBoost model (auto-generated)
├── ui/
│   ├── app.py                # Streamlit entry point
│   ├── pages/                # Multi-page navigation (auto-detected)
│   │   ├── 1_📍_Route_Planner.py
│   │   ├── 2_🔍_Explainability.py
│   │   ├── 3_📊_Model_Evaluation.py
│   │   ├── 4_📁_Batch_Upload.py
│   │   ├── 5_🗂️_History.py
│   │   └── 6_🛡️_Ethics_GDPR.py
│   └── components/
│       ├── map_view.py       # Folium map component
│       ├── route_panel.py    # Route input sidebar panel
│       └── packaging_card.py # Packaging recommendation card
└── tests/
    ├── test_routing.py
    └── test_packaging.py
```

---

## Team

| Role | Name |
|------|------|
| Developer | _(Name 1)_ |
| Developer | _(Name 2)_ |
| Developer | _(Name 3)_ |
| Developer | _(Name 4)_ |

---

## Acknowledgments

Thomas More University · AI Tools 2025–2026 · Semester 2
