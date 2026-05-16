"""GDELT v2 geopolitical risk wrapper.

No API key required. Results cached for 12 hours in data/cache/gdelt/.
Falls back gracefully if GDELT is unreachable.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import requests

_CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache' / 'gdelt'
_CACHE_TTL_SECONDS = 12 * 3600
_TIMEOUT = 15

COUNTRY_NAMES = {
    'BE': 'Belgium', 'NL': 'Netherlands', 'DE': 'Germany', 'FR': 'France',
    'ES': 'Spain', 'CN': 'China', 'HK': 'Hong Kong', 'SG': 'Singapore',
    'KR': 'South Korea', 'JP': 'Japan', 'AE': 'United Arab Emirates',
    'IN': 'India', 'TH': 'Thailand', 'TW': 'Taiwan', 'MY': 'Malaysia',
    'QA': 'Qatar', 'US': 'United States', 'BR': 'Brazil', 'AU': 'Australia'
}


def _score_from_count(count: int) -> float:
    if count == 0:   return 1.0
    if count <= 5:   return 2.5
    if count <= 10:  return 4.5
    if count <= 15:  return 6.5
    if count <= 20:  return 8.0
    return 9.5


def _label_from_score(score: float) -> str:
    if score < 3:  return 'Low'
    if score < 6:  return 'Medium'
    if score < 8:  return 'High'
    return 'Critical'


def _cache_path(country_code: str) -> Path:
    return _CACHE_DIR / f"gdelt_{country_code}.json"


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


def get_geopolitical_risk(country_code: str) -> dict:
    """Return geopolitical risk score for a country via GDELT v2.

    Args:
        country_code: ISO 2-letter code (e.g. 'CN', 'AE', 'US')

    Returns:
        Dict with risk_score (0-10), risk_label, article_count, interpretation.
        Falls back to safe defaults if GDELT is unreachable.
    """
    country_name = COUNTRY_NAMES.get(country_code, country_code)
    cache_path = _cache_path(country_code)

    cached = _load_cache(cache_path)
    if cached:
        cached['cached'] = True
        cached.pop('_ts', None)
        return cached

    query = f"{country_name.replace(' ', '+')}+logistics+supply+chain"
    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={query}&mode=artlist&maxrecords=25"
        f"&timespan=7d&format=json&tone=negative&sourcelang=english"
    )

    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        count = len(body.get('articles', []))

        score = _score_from_count(count)
        label = _label_from_score(score)

        result = {
            'country_code':   country_code,
            'risk_score':     score,
            'risk_label':     label,
            'article_count':  count,
            'interpretation': (
                f"Based on {count} logistics-related news articles "
                f"in the past 7 days, {country_name} shows {label.lower()} "
                f"geopolitical risk for supply chain operations."
            ),
            'source':  'gdelt',
            'cached':  False,
        }
        _save_cache(cache_path, {k: v for k, v in result.items() if k != 'cached'})
        return result

    except Exception as exc:
        warnings.warn(f"[geopolitics] GDELT unavailable ({exc}); using fallback.")
        return {
            'country_code':   country_code,
            'risk_score':     3.0,
            'risk_label':     'Low',
            'article_count':  0,
            'interpretation': (
                f"Risk data unavailable for {country_name}. "
                f"Using conservative default score."
            ),
            'source':  'fallback',
            'cached':  False,
        }


if __name__ == '__main__':
    for country in ['CN', 'AE', 'US', 'BE', 'SG']:
        result = get_geopolitical_risk(country)
        print(
            f"{country}: score={result['risk_score']} ({result['risk_label']}) "
            f"-- {result['article_count']} articles -- source={result['source']} "
            f"-- cached={result['cached']}"
        )
