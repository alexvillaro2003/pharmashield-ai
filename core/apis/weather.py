"""Open-Meteo weather forecast wrapper.

No API key required. Results cached for 6 hours in data/cache/weather/.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import requests

_CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache' / 'weather'
_CACHE_TTL_SECONDS = 6 * 3600
_TIMEOUT = 10

# WMO Weather Interpretation Codes → human label
_WMO_MAP: list[tuple[range, str]] = [
    (range(0,   2),  'sunny'),
    (range(2,   4),  'cloudy'),
    (range(51,  68), 'rainy'),
    (range(71,  78), 'snowy'),
    (range(80, 100), 'stormy'),
]

_FALLBACK = {
    'max_temp_c':          20.0,
    'min_temp_c':          10.0,
    'mean_temp_c':         15.0,
    'dominant_condition': 'unknown',
    'source':             'fallback',
    'cached':             False,
}


def _wmo_to_condition(code: int) -> str:
    for r, label in _WMO_MAP:
        if code in r:
            return label
    return 'cloudy'


def _dominant_condition(codes: list[int]) -> str:
    from collections import Counter
    labels = [_wmo_to_condition(c) for c in codes]
    return Counter(labels).most_common(1)[0][0]


def _cache_path(lat: float, lon: float) -> Path:
    return _CACHE_DIR / f"weather_{lat:.2f}_{lon:.2f}.json"


def _load_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        age = datetime.now(timezone.utc).timestamp() - data.get('_ts', 0)
        if age < _CACHE_TTL_SECONDS:
            return data
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _save_cache(path: Path, payload: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload['_ts'] = datetime.now(timezone.utc).timestamp()
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def get_weather_forecast(lat: float, lon: float, days_ahead: int = 16) -> dict:
    """Return temperature forecast for (lat, lon) via Open-Meteo.

    Args:
        lat, lon:    Coordinates of the point of interest.
        days_ahead:  Forecast window (max 16 for the free tier).

    Returns:
        Dict with max/min/mean temps, dominant weather condition, and metadata.
        Falls back to safe defaults if the API is unreachable.
    """
    days_ahead = min(days_ahead, 16)
    cache_path = _cache_path(lat, lon)

    cached = _load_cache(cache_path)
    if cached:
        cached['cached'] = True
        cached.pop('_ts', None)
        return cached

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,weathercode"
        f"&timezone=auto&forecast_days={days_ahead}"
    )

    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()

        daily     = body['daily']
        max_temps = daily['temperature_2m_max']
        min_temps = daily['temperature_2m_min']
        codes     = daily['weathercode']

        all_temps  = [t for t in max_temps + min_temps if t is not None]
        max_temp_c = max(t for t in max_temps if t is not None)
        min_temp_c = min(t for t in min_temps if t is not None)
        mean_temp  = sum(all_temps) / len(all_temps)

        result = {
            'max_temp_c':          round(max_temp_c, 1),
            'min_temp_c':          round(min_temp_c, 1),
            'mean_temp_c':         round(mean_temp, 1),
            'dominant_condition': _dominant_condition([c for c in codes if c is not None]),
            'source':             'open-meteo',
            'coords':             {'lat': lat, 'lon': lon},
            'cached':             False,
        }
        _save_cache(cache_path, {k: v for k, v in result.items() if k != 'cached'})
        return result

    except Exception as exc:
        warnings.warn(f"[weather] Open-Meteo unavailable ({exc}); using fallback defaults.")
        return {**_FALLBACK, 'coords': {'lat': lat, 'lon': lon}}


if __name__ == '__main__':
    tests = [
        ('Shanghai',   30.6275, 122.0703),
        ('Antwerp',    51.3197,   4.4003),
        ('Singapore',   1.2641, 103.8230),
    ]
    for name, lat, lon in tests:
        result = get_weather_forecast(lat=lat, lon=lon)
        print(
            f"{name}: max={result['max_temp_c']}C  min={result['min_temp_c']}C  "
            f"mean={result['mean_temp_c']}C  condition={result['dominant_condition']}  "
            f"source={result['source']}  cached={result['cached']}"
        )
