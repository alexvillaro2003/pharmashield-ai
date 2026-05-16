"""Pharmaceutical packaging recommender based on declarative rules.

Reads data/master/packaging_rules.json and applies the first matching rule.
Rules are ordered from most specific to most general; the first match wins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_RULES_PATH  = Path(__file__).parent.parent / 'data' / 'master' / 'packaging_rules.json'
_PHARMA_PATH = Path(__file__).parent.parent / 'data' / 'master' / 'pharma_classes.json'


@dataclass
class PackagingRecommendation:
    needs_packaging: bool
    packaging_type: str        # 'none' | 'passive' | 'active'
    min_autonomy_hours: int
    justification: str
    rule_id: str
    rule_name: str
    pharma_class: str
    duration_hours: float
    max_ambient_temp_c: float


def _load_rules() -> list[dict]:
    with open(_RULES_PATH, encoding='utf-8') as f:
        return json.load(f)['rules']


def _load_pharma_classes() -> dict:
    with open(_PHARMA_PATH, encoding='utf-8') as f:
        return {p['id']: p for p in json.load(f)['pharma_classes']}


def _check_condition(value: float, condition: dict) -> bool:
    if 'min' in condition and value < condition['min']:
        return False
    if 'max' in condition and value > condition['max']:
        return False
    return True


def _rule_matches(rule: dict, pharma_class: str, duration_hours: float,
                  max_ambient_temp_c: float, ambient_outside_range: bool) -> bool:
    conds = rule['conditions']

    if conds.get('pharma_class') != pharma_class:
        return False

    if 'duration_hours' in conds:
        if not _check_condition(duration_hours, conds['duration_hours']):
            return False

    if 'max_ambient_temp_c_expected' in conds:
        if not _check_condition(max_ambient_temp_c, conds['max_ambient_temp_c_expected']):
            return False

    if 'ambient_outside_class_range' in conds:
        if conds['ambient_outside_class_range'] != ambient_outside_range:
            return False

    return True


def recommend_packaging(
    pharma_class: str,
    duration_hours: float,
    max_ambient_temp_c: float,
    min_ambient_temp_c: float,
) -> PackagingRecommendation:
    """Recommend packaging based on pharma class, route duration, and ambient conditions.

    Args:
        pharma_class:       'crt', 'cool', 'frozen', or 'ultracold'
        duration_hours:     Total route transit time in hours.
        max_ambient_temp_c: Maximum expected ambient temperature along the route.
        min_ambient_temp_c: Minimum expected ambient temperature along the route.

    Returns:
        PackagingRecommendation with type, required autonomy, and GDP justification.
        Defaults to conservative active packaging when no rule matches.
    """
    rules = _load_rules()
    pharma_classes = _load_pharma_classes()

    if pharma_class not in pharma_classes:
        raise ValueError(
            f"Unknown pharma_class: '{pharma_class}'. "
            f"Valid values: {list(pharma_classes)}"
        )

    pharma = pharma_classes[pharma_class]
    ambient_outside_range = (
        max_ambient_temp_c > pharma['temp_max']
        or min_ambient_temp_c < pharma['temp_min']
    )

    for rule in rules:
        if _rule_matches(rule, pharma_class, duration_hours,
                         max_ambient_temp_c, ambient_outside_range):
            rec = rule['recommendation']
            return PackagingRecommendation(
                needs_packaging=rec['needs_packaging'],
                packaging_type=rec['packaging_type'],
                min_autonomy_hours=rec['min_autonomy_hours'],
                justification=rec['justification'],
                rule_id=rule['id'],
                rule_name=rule['name'],
                pharma_class=pharma_class,
                duration_hours=duration_hours,
                max_ambient_temp_c=max_ambient_temp_c,
            )

    # Conservative fallback: no rule matched -> active with 20% safety margin
    safety_hours = int(duration_hours * 1.2)
    return PackagingRecommendation(
        needs_packaging=True,
        packaging_type='active',
        min_autonomy_hours=safety_hours,
        justification=(
            f"No specific rule matched for {pharma_class.upper()} at "
            f"{max_ambient_temp_c}C over {duration_hours:.0f}h. "
            f"Defaulting to active packaging with 20% safety margin."
        ),
        rule_id='fallback',
        rule_name='Conservative fallback',
        pharma_class=pharma_class,
        duration_hours=duration_hours,
        max_ambient_temp_c=max_ambient_temp_c,
    )


if __name__ == '__main__':
    test_cases = [
        # (pharma_class, duration_h, max_temp, min_temp, description)
        ('crt',       72,   22, 18, 'CRT con clima normal'),
        ('crt',       72,   35, 30, 'CRT con clima tropical'),
        ('cool',      48,   28, 22, 'Cool corto, clima calido'),
        ('cool',      96,   25, 18, 'Cool medio, clima templado'),
        ('cool',      200,  30, 22, 'Cool muy largo, tropical'),
        ('frozen',    48,    5, -5, 'Frozen corto'),
        ('frozen',    120,  10, -3, 'Frozen largo'),
        ('ultracold', 18,   10, -2, 'UltraCold muy corto'),
        ('ultracold', 96,   15,  0, 'UltraCold largo'),
    ]

    print(f"{'Case':40} {'Type':8} {'Autonomy':>10}  {'Rule'}")
    print('-' * 100)
    for pharma, dur, max_t, min_t, desc in test_cases:
        rec = recommend_packaging(pharma, dur, max_t, min_t)
        print(f"{desc:40} {rec.packaging_type:8} {rec.min_autonomy_hours:>7}h   {rec.rule_id}")
