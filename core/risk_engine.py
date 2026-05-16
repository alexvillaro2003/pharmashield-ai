"""Pharmaceutical shipment risk engine.

Combines:
- Trained ML model (predicts P(temperature_excursion) per segment)
- Real-time weather (Open-Meteo) per segment destination
- Geopolitical risk (GDELT) per country crossed
- Packaging recommendation from packaging.py

Returns per-segment risk scores and an aggregated shipment assessment.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path as _Path

# Allow running as a plain script: python core/risk_engine.py
_root = _Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core.routing import Route, Segment, plan_route
from core.packaging import recommend_packaging, PackagingRecommendation
from core.apis.weather import get_weather_forecast
from core.apis.geopolitics import get_geopolitical_risk

_MODEL_PATH    = Path(__file__).parent.parent / 'ml' / 'model.pkl'
_METADATA_PATH = Path(__file__).parent.parent / 'ml' / 'model_metadata.json'
_PHARMA_PATH   = Path(__file__).parent.parent / 'data' / 'master' / 'pharma_classes.json'

# Module-level cache so repeated calls in the same process don't reload from disk
_model_cache: dict = {}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class SegmentRisk:
    segment: Segment
    weather: dict
    geo_risk_origin: dict
    geo_risk_destination: dict
    excursion_probability: float   # 0.0 – 1.0
    risk_score: float              # 0 – 100
    risk_label: str                # 'Low' | 'Medium' | 'High' | 'Critical'
    top_risk_factors: list[str]


@dataclass
class ShipmentRiskAssessment:
    route: Route
    pharma_class: str
    weight_kg: float
    carrier_reliability: float
    segment_risks: list[SegmentRisk]
    aggregated_risk_score: float
    aggregated_risk_label: str
    packaging: PackagingRecommendation
    summary: str


# ── Artifact loading ───────────────────────────────────────────────────────────

def _load_artifacts() -> tuple:
    """Load and cache model + metadata + pharma classes."""
    if not _model_cache:
        _model_cache['model'] = joblib.load(_MODEL_PATH)
        with open(_METADATA_PATH, encoding='utf-8') as f:
            _model_cache['metadata'] = json.load(f)
        with open(_PHARMA_PATH, encoding='utf-8') as f:
            _model_cache['pharma_classes'] = {
                p['id']: p for p in json.load(f)['pharma_classes']
            }
    return _model_cache['model'], _model_cache['metadata'], _model_cache['pharma_classes']


# ── Feature engineering ────────────────────────────────────────────────────────

def _encode(classes: list[str], value: str) -> int:
    """Return LabelEncoder index for value; falls back to 0 if unseen."""
    try:
        return classes.index(value)
    except ValueError:
        return 0


def _current_season() -> str:
    month = datetime.now().month
    if month in (12, 1, 2): return 'Winter'
    if month in (3, 4, 5):  return 'Spring'
    if month in (6, 7, 8):  return 'Summer'
    return 'Autumn'


def _build_feature_row(
    segment: Segment,
    pharma_class: str,
    weight_kg: float,
    weather: dict,
    geo_risk: float,
    carrier_reliability: float,
    num_handovers: int,
    packaging_type: str,
    packaging_autonomy: int,
    metadata: dict,
    pharma_classes: dict,
) -> pd.DataFrame:
    """Build a single-row DataFrame matching the exact feature schema the model expects."""
    pharma = pharma_classes[pharma_class]
    encoders = metadata['label_encoders']   # dict[str, list[str]]

    temp_buffer   = min(pharma['temp_max'] - weather['max_temp_c'],
                        weather['min_temp_c'] - pharma['temp_min'])
    autonomy_gap  = packaging_autonomy - segment.duration_hours
    is_intercon   = 1 if segment.distance_km > 5000 else 0

    features = {
        'distance_km':                 segment.distance_km,
        'weight_kg':                   weight_kg,
        'transit_duration_hours':      segment.duration_hours,
        'num_handovers':               num_handovers,
        'max_ambient_temp_c_expected': weather['max_temp_c'],
        'min_ambient_temp_c_expected': weather['min_temp_c'],
        'geopolitical_risk_score':     geo_risk,
        'carrier_reliability_score':   carrier_reliability,
        'packaging_autonomy_hours':    float(packaging_autonomy),
        'temp_buffer':                 temp_buffer,
        'autonomy_gap':                autonomy_gap,
        'is_intercontinental':         is_intercon,
        'pharma_class_id_enc':         _encode(encoders['pharma_class_id'],      pharma_class),
        'primary_transport_mode_enc':  _encode(encoders['primary_transport_mode'], segment.mode),
        'packaging_type_used_enc':     _encode(encoders['packaging_type_used'],  packaging_type),
        'season_enc':                  _encode(encoders['season'],               _current_season()),
    }

    feature_names = metadata['feature_names']
    return pd.DataFrame([{col: features.get(col, 0) for col in feature_names}])[feature_names]


# ── Risk helpers ───────────────────────────────────────────────────────────────

def _score_to_label(score: float) -> str:
    if score < 25: return 'Low'
    if score < 50: return 'Medium'
    if score < 75: return 'High'
    return 'Critical'


def _top_risk_factors(features: pd.DataFrame, pharma: dict) -> list[str]:
    row = features.iloc[0]
    factors = []

    if row['temp_buffer'] < 0:
        factors.append(
            f"Ambient temp outside pharma range "
            f"({pharma['temp_min']}-{pharma['temp_max']}C)"
        )
    if row['autonomy_gap'] < 0:
        factors.append(
            f"Packaging autonomy ({row['packaging_autonomy_hours']:.0f}h) "
            f"below transit duration ({row['transit_duration_hours']:.0f}h)"
        )
    if row['carrier_reliability_score'] < 0.75:
        factors.append(
            f"Low carrier reliability ({row['carrier_reliability_score']:.2f})"
        )
    if row['geopolitical_risk_score'] > 6:
        factors.append(
            f"High geopolitical risk ({row['geopolitical_risk_score']:.1f}/10)"
        )
    if row['num_handovers'] >= 3:
        factors.append(f"Multiple handovers ({int(row['num_handovers'])})")
    if row['transit_duration_hours'] > 200:
        factors.append(
            f"Long transit duration ({row['transit_duration_hours']:.0f}h)"
        )
    if not factors:
        factors.append("All risk factors within acceptable ranges")

    return factors[:3]


# ── Public API ─────────────────────────────────────────────────────────────────

def assess_shipment(
    origin_id: str,
    destination_id: str,
    pharma_class: str,
    weight_kg: float = 200.0,
    carrier_reliability: float = 0.85,
) -> ShipmentRiskAssessment:
    """Assess full cold-chain risk for a pharma shipment.

    Args:
        origin_id:           Node ID from nodes.json (e.g. 'brussels').
        destination_id:      Node ID from nodes.json (e.g. 'shanghai').
        pharma_class:        'crt', 'cool', 'frozen', or 'ultracold'.
        weight_kg:           Shipment weight in kg.
        carrier_reliability: Carrier reliability score 0.0-1.0.

    Returns:
        ShipmentRiskAssessment with per-segment risks and packaging recommendation.
    """
    model, metadata, pharma_classes = _load_artifacts()

    route = plan_route(origin_id, destination_id)

    # Collect weather and geo-risk for every segment
    segment_weathers: list[dict] = []
    segment_geo_orig: list[dict] = []
    segment_geo_dest: list[dict] = []
    all_max_temps: list[float]   = []
    all_min_temps: list[float]   = []

    for seg in route.segments:
        wx = get_weather_forecast(seg.destination_coords[0], seg.destination_coords[1])
        segment_weathers.append(wx)
        all_max_temps.append(wx['max_temp_c'])
        all_min_temps.append(wx['min_temp_c'])

        geo_o = get_geopolitical_risk(seg.origin_country)
        time.sleep(0.3)   # avoid GDELT 429 on back-to-back calls
        geo_d = get_geopolitical_risk(seg.destination_country)
        time.sleep(0.3)
        segment_geo_orig.append(geo_o)
        segment_geo_dest.append(geo_d)

    # Packaging recommendation uses worst-case ambient across the whole route
    max_temp = max(all_max_temps) if all_max_temps else 20.0
    min_temp = min(all_min_temps) if all_min_temps else 10.0
    packaging = recommend_packaging(
        pharma_class=pharma_class,
        duration_hours=route.total_duration_hours,
        max_ambient_temp_c=max_temp,
        min_ambient_temp_c=min_temp,
    )

    # Per-segment ML inference
    segment_risks: list[SegmentRisk] = []
    for seg, wx, geo_o, geo_d in zip(
        route.segments, segment_weathers, segment_geo_orig, segment_geo_dest
    ):
        avg_geo = (geo_o['risk_score'] + geo_d['risk_score']) / 2

        feat_df = _build_feature_row(
            segment=seg,
            pharma_class=pharma_class,
            weight_kg=weight_kg,
            weather=wx,
            geo_risk=avg_geo,
            carrier_reliability=carrier_reliability,
            num_handovers=route.num_handovers,
            packaging_type=packaging.packaging_type,
            packaging_autonomy=packaging.min_autonomy_hours,
            metadata=metadata,
            pharma_classes=pharma_classes,
        )

        prob       = float(model.predict_proba(feat_df)[0][1])
        risk_score = round(prob * 100, 1)

        segment_risks.append(SegmentRisk(
            segment=seg,
            weather=wx,
            geo_risk_origin=geo_o,
            geo_risk_destination=geo_d,
            excursion_probability=prob,
            risk_score=risk_score,
            risk_label=_score_to_label(risk_score),
            top_risk_factors=_top_risk_factors(feat_df, pharma_classes[pharma_class]),
        ))

    # Aggregate: duration-weighted average, but never below 85% of worst segment
    if segment_risks:
        durations    = [s.segment.duration_hours for s in segment_risks]
        total_dur    = sum(durations)
        weighted     = sum(s.risk_score * d for s, d in zip(segment_risks, durations)) / total_dur
        worst        = max(s.risk_score for s in segment_risks)
        aggregated   = round(max(weighted, worst * 0.85), 1)
    else:
        aggregated = 0.0

    summary = (
        f"Shipment {route.segments[0].origin_name} -> {route.segments[-1].destination_name} "
        f"({pharma_class.upper()}, {weight_kg:.0f} kg, {route.total_duration_hours:.0f}h transit). "
        f"Aggregated risk: {aggregated}/100 ({_score_to_label(aggregated)}). "
        f"Recommended packaging: {packaging.packaging_type.upper()} "
        f"(autonomy >= {packaging.min_autonomy_hours}h)."
    )

    return ShipmentRiskAssessment(
        route=route,
        pharma_class=pharma_class,
        weight_kg=weight_kg,
        carrier_reliability=carrier_reliability,
        segment_risks=segment_risks,
        aggregated_risk_score=aggregated,
        aggregated_risk_label=_score_to_label(aggregated),
        packaging=packaging,
        summary=summary,
    )


if __name__ == '__main__':
    sep = '=' * 70

    print(f"\n{sep}")
    print("TEST 1: Brussels -> Shanghai, Cool vaccine")
    print(sep)
    r1 = assess_shipment('brussels', 'shanghai', 'cool', weight_kg=150,
                         carrier_reliability=0.85)
    print(r1.summary)
    print(f"\nSegments ({len(r1.segment_risks)}):")
    for sr in r1.segment_risks:
        s = sr.segment
        print(f"  [{s.mode:5s}] {s.origin_name} -> {s.destination_name}: "
              f"risk={sr.risk_score} ({sr.risk_label}), "
              f"weather_max={sr.weather['max_temp_c']}C, "
              f"geo_dest={sr.geo_risk_destination['risk_score']}")
        print(f"    Factors: {sr.top_risk_factors[:2]}")
    print(f"\nPackaging: {r1.packaging.packaging_type} | "
          f"{r1.packaging.min_autonomy_hours}h | {r1.packaging.rule_id}")
    print(f"Justification: {r1.packaging.justification[:80]}...")

    print(f"\n{sep}")
    print("TEST 2: Frankfurt -> JFK, UltraCold mRNA vaccine")
    print(sep)
    r2 = assess_shipment('frankfurt_airport', 'jfk_airport', 'ultracold',
                         weight_kg=80, carrier_reliability=0.95)
    print(r2.summary)
    print(f"\nPackaging: {r2.packaging.packaging_type} | {r2.packaging.min_autonomy_hours}h")
    if r2.segment_risks:
        sr = r2.segment_risks[0]
        print(f"Segment risk: {sr.risk_score} ({sr.risk_label})")
        print(f"Factors: {sr.top_risk_factors}")

    print(f"\n{sep}")
    print("TEST 3: Brussels -> Madrid, CRT tablets")
    print(sep)
    r3 = assess_shipment('brussels', 'madrid', 'crt', weight_kg=300,
                         carrier_reliability=0.80)
    print(r3.summary)
    if r3.segment_risks:
        sr = r3.segment_risks[0]
        print(f"Segment risk: {sr.risk_score} ({sr.risk_label})")
        print(f"Packaging: {r3.packaging.packaging_type} | {r3.packaging.rule_id}")
