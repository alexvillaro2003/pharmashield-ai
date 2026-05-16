"""Multimodal pharma shipment routing heuristic.

Given two node IDs from data/master/nodes.json, returns a Route composed of
one or more Segments, each with a transport mode, distance, and duration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

CONTINENT_MAP: dict[str, str] = {
    'BE': 'EU', 'NL': 'EU', 'DE': 'EU', 'FR': 'EU', 'ES': 'EU', 'GB': 'EU',
    'CN': 'AS', 'HK': 'AS', 'SG': 'AS', 'KR': 'AS', 'JP': 'AS',
    'AE': 'AS', 'IN': 'AS', 'TH': 'AS', 'TW': 'AS', 'MY': 'AS', 'QA': 'AS',
    'US': 'AM', 'BR': 'AM', 'CA': 'AM', 'MX': 'AM',
    'AU': 'OC',
}

SPEED_KMH: dict[str, float] = {
    'road': 60.0,
    'rail': 80.0,
    'sea':  25.0,
    'air': 750.0,
}

# Loading/unloading + customs time added per segment
HANDLING_HOURS: dict[str, float] = {
    'road':  2.0,
    'rail':  4.0,
    'sea':  24.0,
    'air':   8.0,
}

_DATA_PATH = Path(__file__).parent.parent / 'data' / 'master' / 'nodes.json'
_nodes_cache: dict[str, dict] | None = None


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Segment:
    origin_id: str
    destination_id: str
    origin_name: str
    destination_name: str
    origin_coords: tuple[float, float]       # (lat, lon)
    destination_coords: tuple[float, float]  # (lat, lon)
    mode: str                                # 'road' | 'sea' | 'air' | 'rail'
    distance_km: float
    duration_hours: float
    origin_country: str
    destination_country: str


@dataclass
class Route:
    origin_id: str
    destination_id: str
    segments: list[Segment]
    total_distance_km: float
    total_duration_hours: float
    primary_mode: str       # mode of the longest segment by distance
    num_handovers: int      # len(segments) - 1
    is_intercontinental: bool


# ── Node loading ───────────────────────────────────────────────────────────────

def _load_nodes() -> dict[str, dict]:
    global _nodes_cache
    if _nodes_cache is None:
        with open(_DATA_PATH) as f:
            data = json.load(f)
        _nodes_cache = {n['id']: n for n in data['nodes']}
    return _nodes_cache


# ── Geometry ───────────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ── Node lookup ────────────────────────────────────────────────────────────────

def find_nearest_node(node_id: str, node_type: str, nodes: dict[str, dict]) -> str:
    """Return the id of the nearest node of `node_type`, excluding `node_id` itself."""
    source = nodes[node_id]
    slat = source['coords']['lat']
    slon = source['coords']['lon']

    best_id, best_dist = None, math.inf
    for nid, node in nodes.items():
        if nid == node_id or node['type'] != node_type:
            continue
        d = haversine(slat, slon, node['coords']['lat'], node['coords']['lon'])
        if d < best_dist:
            best_dist = d
            best_id = nid
    return best_id


# ── Segment factory ────────────────────────────────────────────────────────────

def _make_segment(orig: dict, dest: dict, mode: str, distance_km: float) -> Segment:
    duration = distance_km / SPEED_KMH[mode] + HANDLING_HOURS[mode]
    return Segment(
        origin_id=orig['id'],
        destination_id=dest['id'],
        origin_name=orig['name'],
        destination_name=dest['name'],
        origin_coords=(orig['coords']['lat'], orig['coords']['lon']),
        destination_coords=(dest['coords']['lat'], dest['coords']['lon']),
        mode=mode,
        distance_km=round(distance_km, 1),
        duration_hours=round(duration, 2),
        origin_country=orig['country'],
        destination_country=dest['country'],
    )


# ── Route assembly ─────────────────────────────────────────────────────────────

def _build_route(origin_id: str, destination_id: str,
                 segments: list[Segment], orig: dict, dest: dict) -> Route:
    total_dist = sum(s.distance_km for s in segments)
    total_dur  = sum(s.duration_hours for s in segments)
    primary    = max(segments, key=lambda s: s.distance_km).mode
    orig_cont  = CONTINENT_MAP.get(orig['country'], 'XX')
    dest_cont  = CONTINENT_MAP.get(dest['country'], 'XX')
    return Route(
        origin_id=origin_id,
        destination_id=destination_id,
        segments=segments,
        total_distance_km=round(total_dist, 1),
        total_duration_hours=round(total_dur, 2),
        primary_mode=primary,
        num_handovers=len(segments) - 1,
        is_intercontinental=(orig_cont != dest_cont),
    )


# ── Mode selection ─────────────────────────────────────────────────────────────

def _choose_main_mode(orig: dict, dest: dict) -> str:
    """Decide between 'air' and 'sea' for the intercontinental main leg."""
    # Both airports -> always fly
    if orig['type'] == 'airport' and dest['type'] == 'airport':
        return 'air'
    # Destination has no sea intake (e.g. airport) -> must use air
    if 'sea' not in dest.get('modes_in', []):
        return 'air'
    # Origin has no air output (seaport / inland_hub) -> default to sea
    if 'air' not in orig.get('modes_out', []):
        return 'sea'
    # Origin is an airport -> prefer air
    if orig['type'] == 'airport':
        return 'air'
    return 'sea'


def _build_air_route(orig: dict, dest: dict, nodes: dict[str, dict]) -> list[Segment]:
    """Construct an air route, adding road feeder legs as needed."""
    segments: list[Segment] = []

    # Feeder to nearest airport at origin side
    if orig['type'] != 'airport':
        ap_orig = nodes[find_nearest_node(orig['id'], 'airport', nodes)]
        d = haversine(orig['coords']['lat'], orig['coords']['lon'],
                      ap_orig['coords']['lat'], ap_orig['coords']['lon'])
        segments.append(_make_segment(orig, ap_orig, 'road', d))
        air_orig = ap_orig
    else:
        air_orig = orig

    # Feeder from nearest airport at destination side
    if dest['type'] != 'airport':
        ap_dest = nodes[find_nearest_node(dest['id'], 'airport', nodes)]
        d_dest  = haversine(ap_dest['coords']['lat'], ap_dest['coords']['lon'],
                            dest['coords']['lat'], dest['coords']['lon'])
        feeder_dest = _make_segment(ap_dest, dest, 'road', d_dest)
        air_dest = ap_dest
    else:
        air_dest    = dest
        feeder_dest = None

    d_air = haversine(air_orig['coords']['lat'], air_orig['coords']['lon'],
                      air_dest['coords']['lat'],  air_dest['coords']['lon'])
    segments.append(_make_segment(air_orig, air_dest, 'air', d_air))

    if feeder_dest:
        segments.append(feeder_dest)
    return segments


def _build_sea_route(orig: dict, dest: dict, nodes: dict[str, dict]) -> list[Segment]:
    """Construct a sea route, adding road feeder legs for non-seaport endpoints."""
    segments: list[Segment] = []

    # Feeder to nearest seaport at origin side
    if orig['type'] != 'seaport':
        port_orig = nodes[find_nearest_node(orig['id'], 'seaport', nodes)]
        d = haversine(orig['coords']['lat'], orig['coords']['lon'],
                      port_orig['coords']['lat'], port_orig['coords']['lon'])
        segments.append(_make_segment(orig, port_orig, 'road', d))
        sea_orig = port_orig
    else:
        sea_orig = orig

    # Feeder from nearest seaport at destination side
    if dest['type'] != 'seaport':
        port_dest = nodes[find_nearest_node(dest['id'], 'seaport', nodes)]
        d_dest    = haversine(port_dest['coords']['lat'], port_dest['coords']['lon'],
                              dest['coords']['lat'],      dest['coords']['lon'])
        feeder_dest = _make_segment(port_dest, dest, 'road', d_dest)
        sea_dest    = port_dest
    else:
        sea_dest    = dest
        feeder_dest = None

    d_sea = haversine(sea_orig['coords']['lat'], sea_orig['coords']['lon'],
                      sea_dest['coords']['lat'],  sea_dest['coords']['lon'])
    segments.append(_make_segment(sea_orig, sea_dest, 'sea', d_sea))

    if feeder_dest:
        segments.append(feeder_dest)
    return segments


# ── Routing strategies ─────────────────────────────────────────────────────────

def _route_same_continent(orig: dict, dest: dict, distance: float,
                           nodes: dict[str, dict]) -> list[Segment]:
    if distance < 1500:
        return [_make_segment(orig, dest, 'road', distance)]

    if distance <= 4000:
        # Rail for European pairs, road otherwise
        if CONTINENT_MAP.get(orig['country']) == 'EU':
            return [_make_segment(orig, dest, 'rail', distance)]
        return [_make_segment(orig, dest, 'road', distance)]

    # Unusually long intra-continental leg: sea if both endpoints support it
    if ('sea' in orig.get('modes_out', []) and 'sea' in dest.get('modes_in', [])):
        return [_make_segment(orig, dest, 'sea', distance)]
    return [_make_segment(orig, dest, 'road', distance)]


def _route_intercontinental(orig: dict, dest: dict, nodes: dict[str, dict]) -> list[Segment]:
    if _choose_main_mode(orig, dest) == 'air':
        return _build_air_route(orig, dest, nodes)
    return _build_sea_route(orig, dest, nodes)


# ── Public API ─────────────────────────────────────────────────────────────────

def plan_route(origin_id: str, destination_id: str) -> Route:
    """Plan a multimodal pharma shipment route between two nodes.

    Args:
        origin_id: Node ID from nodes.json (e.g. 'brussels')
        destination_id: Node ID from nodes.json (e.g. 'shanghai')

    Returns:
        Route with all segments and aggregated metrics.

    Raises:
        ValueError: if either ID is not present in nodes.json.
    """
    nodes = _load_nodes()

    if origin_id not in nodes:
        raise ValueError(f"Unknown origin node: '{origin_id}'. Valid IDs: {sorted(nodes)}")
    if destination_id not in nodes:
        raise ValueError(f"Unknown destination node: '{destination_id}'. Valid IDs: {sorted(nodes)}")

    orig = nodes[origin_id]
    dest = nodes[destination_id]

    orig_cont = CONTINENT_MAP.get(orig['country'], 'XX')
    dest_cont = CONTINENT_MAP.get(dest['country'], 'XX')

    direct_dist = haversine(
        orig['coords']['lat'], orig['coords']['lon'],
        dest['coords']['lat'], dest['coords']['lon'],
    )

    if orig_cont == dest_cont:
        segments = _route_same_continent(orig, dest, direct_dist, nodes)
    else:
        segments = _route_intercontinental(orig, dest, nodes)

    return _build_route(origin_id, destination_id, segments, orig, dest)


# ── Manual smoke test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_cases = [
        ('brussels',         'shanghai'),
        ('rotterdam',        'los_angeles'),
        ('frankfurt_airport','jfk_airport'),
        ('brussels',         'madrid'),
        ('schiphol',         'dubai_airport'),
        ('singapore',        'sydney'),
    ]

    for origin, dest_id in test_cases:
        route = plan_route(origin, dest_id)
        print(f"\n{'=' * 60}")
        print(f"Route: {origin} -> {dest_id}")
        print(f"Total: {route.total_distance_km:.0f} km, "
              f"{route.total_duration_hours:.1f}h, "
              f"{route.num_handovers} handovers")
        print("Segments:")
        for s in route.segments:
            print(f"  [{s.mode:5s}] {s.origin_name} -> {s.destination_name}: "
                  f"{s.distance_km:.0f} km, {s.duration_hours:.1f}h")
