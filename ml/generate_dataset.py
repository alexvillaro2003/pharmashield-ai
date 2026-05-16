import json
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
NUM_SHIPMENTS = 10_000
OUTPUT_PATH = Path('data/synthetic/shipments_train.csv')

# ── Mappings ──────────────────────────────────────────────────────────────────

CONTINENT_MAP = {
    'BE': 'Europe', 'NL': 'Europe', 'DE': 'Europe', 'FR': 'Europe', 'ES': 'Europe',
    'CN': 'Asia',   'HK': 'Asia',   'SG': 'Asia',   'KR': 'Asia',   'JP': 'Asia',
    'AE': 'Asia',   'QA': 'Asia',   'IN': 'Asia',   'TH': 'Asia',   'TW': 'Asia',   'MY': 'Asia',
    'US': 'Americas', 'BR': 'Americas',
    'AU': 'Oceania',
}

COUNTRY_GEO_RISK_BASE = {
    'BE': 1.5, 'NL': 1.5, 'DE': 1.5, 'FR': 2.0, 'ES': 2.0,
    'CN': 5.0, 'HK': 4.0, 'SG': 2.0, 'KR': 3.0, 'JP': 1.5,
    'AE': 3.5, 'QA': 3.5, 'IN': 5.5, 'TH': 4.5, 'TW': 3.5, 'MY': 3.0,
    'US': 2.5, 'BR': 6.0, 'AU': 1.5,
}

COUNTRY_CARRIER_BASE = {
    'BE': 0.92, 'NL': 0.93, 'DE': 0.94, 'FR': 0.88, 'ES': 0.85,
    'CN': 0.78, 'HK': 0.84, 'SG': 0.90, 'KR': 0.86, 'JP': 0.92,
    'AE': 0.87, 'QA': 0.85, 'IN': 0.72, 'TH': 0.76, 'TW': 0.84, 'MY': 0.82,
    'US': 0.89, 'BR': 0.70, 'AU': 0.88,
}

PRODUCT_NAMES = {
    'crt':       ['Aspirin 500mg', 'Ibuprofen 400mg', 'Paracetamol 1g', 'Amoxicillin 500mg',
                  'Metformin 850mg', 'Atorvastatin 20mg', 'Omeprazole 20mg', 'Lisinopril 10mg'],
    'cool':      ['Insulin Glargine', 'Influenza Vaccine', 'Adalimumab (Humira)',
                  'Bevacizumab (Avastin)', 'Trastuzumab (Herceptin)', 'Hepatitis B Vaccine'],
    'frozen':    ['Varicella Vaccine', 'Zoster Vaccine (Shingrix)', 'Plasma-Derived Factor VIII',
                  'Frozen Biologic RX-8', 'Oncology Prep KX-7'],
    'ultracold': ['mRNA COVID-19 Vaccine', 'Gene Therapy AAV-9',
                  'CAR-T Cell Therapy', 'mRNA Influenza Vaccine NG'],
}

MAJOR_HUBS = {
    'rotterdam', 'shanghai', 'singapore', 'dubai_port', 'jfk_airport', 'los_angeles_lax',
    'frankfurt_airport', 'schiphol', 'dubai_airport', 'hong_kong', 'busan', 'tokyo_narita',
}

PHARMA_WEIGHTS = {'crt': 0.50, 'cool': 0.35, 'frozen': 0.12, 'ultracold': 0.03}
BASE_P         = {'crt': 0.05, 'cool': 0.15, 'frozen': 0.25, 'ultracold': 0.40}


# ── Geographic helpers ────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def get_climate(lat, month):
    """Return (max_ambient_temp, min_ambient_temp) realistic for given lat and month."""
    abs_lat = abs(lat)
    if abs_lat <= 23.5:
        t_mean, amplitude = 30.0, 3.0
    elif abs_lat <= 40:
        t_mean, amplitude = 20.0, 8.0
    elif abs_lat <= 55:
        t_mean, amplitude = 12.0, 12.0
    else:
        t_mean, amplitude = 2.0, 14.0

    # NH peaks July, SH peaks January
    peak_month = 7 if lat >= 0 else 1
    phase  = (month - peak_month) / 12.0 * 2 * np.pi
    t_base = t_mean + amplitude * np.cos(phase)

    max_temp = t_base + np.random.normal(2.0, 1.5)
    min_temp = max_temp - np.random.uniform(6, 12)
    return float(max_temp), float(min_temp)


def get_season(lat, month):
    if lat >= 0:
        if month in (12, 1, 2):  return 'Winter'
        if month in (3, 4, 5):   return 'Spring'
        if month in (6, 7, 8):   return 'Summer'
        return 'Autumn'
    else:
        if month in (12, 1, 2):  return 'Summer'
        if month in (3, 4, 5):   return 'Autumn'
        if month in (6, 7, 8):   return 'Winter'
        return 'Spring'


def determine_transport_mode(distance_km, origin, dest):
    orig_cont = CONTINENT_MAP.get(origin['country'], 'Unknown')
    dest_cont = CONTINENT_MAP.get(dest['country'], 'Unknown')

    can_sea  = 'sea'  in origin.get('modes_out', []) and 'sea'  in dest.get('modes_in', [])
    can_air  = 'air'  in origin.get('modes_out', []) and 'air'  in dest.get('modes_in', [])
    can_rail = 'rail' in origin.get('modes_out', []) and 'rail' in dest.get('modes_in', [])

    if orig_cont == dest_cont:
        if distance_km < 1500:
            return 'road'
        if can_sea and (distance_km > 2500 or np.random.random() < 0.55):
            return 'sea'
        return 'rail' if can_rail else 'road'

    # Intercontinental (inland nodes are assumed to truck to nearest port)
    continents = frozenset({orig_cont, dest_cont})
    # Europe ↔ Asia: Silk Road rail possible
    if continents == frozenset({'Europe', 'Asia'}) and can_rail and np.random.random() < 0.12:
        return 'rail'
    if can_air and distance_km > 6000 and np.random.random() < 0.30:
        return 'air'
    if can_air and np.random.random() < 0.15:
        return 'air'
    return 'sea'  # default: truck to port pre-haul implicit


def calculate_transit_duration(distance_km, mode):
    speed    = {'sea': 25,  'air': 800, 'rail': 80, 'road': 60}[mode]
    handling = {'sea': 48,  'air': 24,  'rail': 6,  'road': 2}[mode]
    noise    = {'sea': 24,  'air': 6,   'rail': 8,  'road': 3}[mode]
    return max(2.0, distance_km / speed + handling + np.random.normal(0, noise))


def determine_packaging(pharma_id, pharma, duration_h, max_ambient, min_ambient):
    outside = max_ambient > pharma['temp_max'] or min_ambient < pharma['temp_min']

    if pharma_id == 'crt':
        if not outside and duration_h <= 240:
            return 'none', 0.0
        if outside and duration_h <= 48:
            return 'passive', np.random.uniform(48, 72)
        return 'active', np.random.uniform(144, 240)

    if pharma_id == 'cool':
        if duration_h <= 48:
            return 'passive', np.random.uniform(48, 72)
        if duration_h <= 72:
            return 'passive', np.random.uniform(72, 100)
        return 'active', np.random.uniform(120, 200)

    if pharma_id == 'frozen':
        if duration_h <= 48:
            return 'passive', np.random.uniform(48, 72)
        if duration_h <= 96:
            return 'passive', np.random.uniform(96, 120)
        return 'active', np.random.uniform(120, 200)

    # ultracold
    if duration_h <= 24:
        return 'passive', np.random.uniform(24, 48)
    return 'active', np.random.uniform(120, 240)


def calculate_excursion_probability(pharma_id, pharma, max_ambient, min_ambient,
                                    duration_h, pkg_type, autonomy_h,
                                    carrier_reliability, geo_risk, handovers):
    # No packaging + outside range → certain excursion
    if pkg_type == 'none' and (max_ambient > pharma['temp_max'] or min_ambient < pharma['temp_min']):
        return 1.0

    p = BASE_P[pharma_id]

    # 1. Duration vs packaging autonomy
    if autonomy_h > 0 and duration_h > autonomy_h:
        ratio = duration_h / autonomy_h
        p *= 1.2 if ratio < 3 else 1.4

    # 2. Temperature buffer: min(class_max - max_ambient, min_ambient - class_min)
    #    Positive = comfortable margin, negative = ambient outside class range
    temp_buffer = min(pharma['temp_max'] - max_ambient, min_ambient - pharma['temp_min'])
    if temp_buffer < -20:
        p *= 1.6
    elif temp_buffer < -10:
        p *= 1.4
    elif temp_buffer < 0:
        p *= 1.2
    elif temp_buffer > 10:
        p *= 0.6

    # 3. Carrier reliability
    if carrier_reliability < 0.7:
        p *= 1.5
    elif carrier_reliability > 0.9:
        p *= 0.5

    # 4. Geopolitical risk
    if geo_risk > 7:
        p *= 1.3

    # 5. Handovers (additive)
    p += 0.05 * handovers

    return float(min(p, 1.0))


# ── Data loader ───────────────────────────────────────────────────────────────

def load_master_data():
    base = Path('data/master')
    with open(base / 'nodes.json') as f:
        nodes_data = json.load(f)
    with open(base / 'pharma_classes.json') as f:
        pharma_data = json.load(f)
    with open(base / 'packaging_rules.json') as f:
        rules_data = json.load(f)
    return nodes_data, pharma_data, rules_data


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_shipments(nodes_data, pharma_data, num_shipments):
    np.random.seed(SEED)

    nodes = nodes_data['nodes']
    pharma_classes = {p['id']: p for p in pharma_data['pharma_classes']}

    # Node importance weights: seaport > airport > inland_hub; major hubs boosted
    raw_w = np.array([
        ({'seaport': 4, 'airport': 3, 'inland_hub': 2}.get(n['type'], 2) +
         (2 if n['id'] in MAJOR_HUBS else 0))
        for n in nodes
    ], dtype=float)
    weights = raw_w / raw_w.sum()

    pharma_ids   = list(PHARMA_WEIGHTS.keys())
    pharma_probs = list(PHARMA_WEIGHTS.values())
    node_indices = np.arange(len(nodes))

    records = []
    for i in range(num_shipments):
        # Pharma class
        pharma_id = np.random.choice(pharma_ids, p=pharma_probs)
        pharma    = pharma_classes[pharma_id]

        # Origin / destination (must differ)
        while True:
            oi = np.random.choice(node_indices, p=weights)
            di = np.random.choice(node_indices, p=weights)
            if oi != di:
                break
        origin = nodes[oi]
        dest   = nodes[di]

        olat, olon = origin['coords']['lat'], origin['coords']['lon']
        dlat, dlon = dest['coords']['lat'],   dest['coords']['lon']

        # Route
        distance_km = haversine(olat, olon, dlat, dlon)
        mode        = determine_transport_mode(distance_km, origin, dest)
        duration_h  = calculate_transit_duration(distance_km, mode)

        # Handovers
        lo, hi = {'road': (0, 1), 'rail': (1, 2), 'sea': (1, 3), 'air': (1, 2)}[mode]
        num_handovers = int(np.random.randint(lo, hi + 1))

        # Climate at destination
        month      = int(np.random.randint(1, 13))
        max_amb, min_amb = get_climate(dlat, month)
        season     = get_season(dlat, month)

        # Geopolitical risk (worst of origin/destination)
        geo_base = max(
            COUNTRY_GEO_RISK_BASE.get(origin['country'], 3.0),
            COUNTRY_GEO_RISK_BASE.get(dest['country'],   3.0),
        )
        geo_risk = float(np.clip(np.random.normal(geo_base, 1.5), 0.0, 10.0))

        # Carrier reliability (weakest link)
        carrier_base = min(
            COUNTRY_CARRIER_BASE.get(origin['country'], 0.80),
            COUNTRY_CARRIER_BASE.get(dest['country'],   0.80),
        )
        carrier_reliability = float(np.clip(np.random.normal(carrier_base, 0.05), 0.60, 0.99))

        # Packaging
        pkg_type, autonomy_h = determine_packaging(
            pharma_id, pharma, duration_h, max_amb, min_amb
        )

        # Target
        excursion_prob      = calculate_excursion_probability(
            pharma_id, pharma, max_amb, min_amb,
            duration_h, pkg_type, autonomy_h,
            carrier_reliability, geo_risk, num_handovers,
        )
        temperature_excursion = int(np.random.binomial(1, excursion_prob))

        # Product name
        names        = PRODUCT_NAMES[pharma_id]
        product_name = names[np.random.randint(len(names))]

        # Weight
        weight_kg = float(np.clip(np.random.lognormal(5.3, 1.5), 10, 2000))

        records.append({
            'shipment_id':                 f'SHP-{i + 1:05d}',
            'pharma_class_id':             pharma_id,
            'product_name':                product_name,
            'weight_kg':                   round(weight_kg, 1),
            'origin_id':                   origin['id'],
            'destination_id':              dest['id'],
            'distance_km':                 round(distance_km, 1),
            'primary_transport_mode':      mode,
            'transit_duration_hours':      round(duration_h, 1),
            'num_handovers':               num_handovers,
            'max_ambient_temp_c_expected': round(max_amb, 1),
            'min_ambient_temp_c_expected': round(min_amb, 1),
            'season':                      season,
            'geopolitical_risk_score':     round(geo_risk, 2),
            'carrier_reliability_score':   round(carrier_reliability, 3),
            'packaging_type_used':         pkg_type,
            'packaging_autonomy_hours':    round(autonomy_h, 1),
            'temperature_excursion':       temperature_excursion,
        })

    return pd.DataFrame(records)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    nodes_data, pharma_data, rules_data = load_master_data()
    df = generate_shipments(nodes_data, pharma_data, NUM_SHIPMENTS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[OK] Dataset generado: {len(df)} filas x {len(df.columns)} cols")
    print(f"\nDistribución del target:")
    print(df['temperature_excursion'].value_counts(normalize=True).round(3))
    print(f"\nDistribución por pharma_class:")
    print(df['pharma_class_id'].value_counts())
    print(f"\nModos de transporte:")
    print(df['primary_transport_mode'].value_counts())
    print(f"\nGuardado en: {OUTPUT_PATH}")
