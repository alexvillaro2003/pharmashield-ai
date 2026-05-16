# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pharmashield-ai is a decision-support tool for pharmaceutical cold-chain logistics. It predicts excursion risk for temperature-sensitive drug shipments, recommends GDP-compliant packaging, and explains predictions via SHAP.

## Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Data & model pipeline (run in order)
python ml/generate_dataset.py  # generates data/synthetic/shipments.csv
python ml/train.py             # trains and serializes model to ml/model.joblib

# Run the app
streamlit run ui/app.py

# No test suite or linter configured yet
```

## Architecture

The system has four layers that must be built out in dependency order:

**Master data** (`data/master/`) — JSON source-of-truth fixtures loaded by all other layers:
- `nodes.json`: 32 transport hubs (seaports, airports, inland depots) with coordinates, country, and risk scores
- `pharma_classes.json`: temperature classes (CRT / Cool / Frozen / UltraCold) with regulatory references (GDP, USP, Ph.Eur., WHO)
- `packaging_rules.json`: 12 packaging recommendation rules keyed by temperature class and route risk level

**ML pipeline** (`ml/`):
- `generate_dataset.py` is complete — generates 10K synthetic shipments with realistic climate modeling (latitude/season), geopolitical risk profiles, carrier risk, and excursion labels
- `train.py` is a TODO stub — should train XGBoost on `data/synthetic/shipments.csv` and serialize to `ml/model.joblib`

**Core logic** (`core/`) — all stubs, to be implemented:
- `routing.py`: parse multimodal routes (air/sea/road) using nodes from master data
- `risk_engine.py`: assemble features from route + live APIs, run model inference
- `packaging.py`: match packaging rules to predicted risk level and pharma class
- `apis/weather.py`: OpenWeatherMap or Open-Meteo client
- `apis/geopolitics.py`: GDELT client for geopolitical risk scoring

**UI** (`ui/`) — Streamlit multi-page app:
- `app.py`: entry point with sidebar navigation
- `pages/`: 6 sub-pages (route planner, explainability, model evaluation, batch upload, history, ethics/GDPR) — most are placeholder stubs
- `components/`: reusable widgets (map view via Folium, route input panel, packaging card)

## Data Flow

```
User inputs route → routing.py selects nodes →
risk_engine.py fetches weather + geopolitics, assembles features →
XGBoost predicts excursion probability →
packaging.py recommends packaging →
SHAP explains prediction →
Folium map + Streamlit UI renders result
```

## Key Conventions

- Pharma temperature classes and packaging rules are regulatory-aligned — always reference the constants in `data/master/pharma_classes.json` and `data/master/packaging_rules.json` rather than hardcoding values.
- The synthetic dataset generator (`ml/generate_dataset.py`) documents the feature schema — use it as the authoritative feature reference when implementing `risk_engine.py`.
- Node risk scores in `nodes.json` combine infrastructure quality and geopolitical stability; treat them as inputs to the feature vector, not final predictions.
