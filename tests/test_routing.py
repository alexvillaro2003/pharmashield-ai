"""Basic tests for core/routing.py."""

import pytest
from core.routing import plan_route

TEST_PAIRS = [
    ('brussels',          'shanghai'),
    ('rotterdam',         'los_angeles'),
    ('frankfurt_airport', 'jfk_airport'),
    ('brussels',          'madrid'),
    ('schiphol',          'dubai_airport'),
    ('singapore',         'sydney'),
]


def test_brussels_shanghai_has_multiple_segments():
    route = plan_route('brussels', 'shanghai')
    assert len(route.segments) >= 2


def test_brussels_madrid_is_single_road_segment():
    route = plan_route('brussels', 'madrid')
    assert len(route.segments) == 1
    assert route.segments[0].mode == 'road'


def test_all_routes_have_positive_distance():
    for orig, dest in TEST_PAIRS:
        route = plan_route(orig, dest)
        assert route.total_distance_km > 0, f"{orig}->{dest}: distance must be > 0"


def test_no_segment_has_negative_duration():
    for orig, dest in TEST_PAIRS:
        route = plan_route(orig, dest)
        for seg in route.segments:
            assert seg.duration_hours >= 0, (
                f"{orig}->{dest}: segment {seg.origin_id}->{seg.destination_id} "
                f"has duration {seg.duration_hours}"
            )


def test_invalid_node_raises_value_error():
    with pytest.raises(ValueError):
        plan_route('nonexistent_node', 'brussels')
    with pytest.raises(ValueError):
        plan_route('brussels', 'nonexistent_node')


def test_num_handovers_equals_segments_minus_one():
    for orig, dest in TEST_PAIRS:
        route = plan_route(orig, dest)
        assert route.num_handovers == len(route.segments) - 1, (
            f"{orig}->{dest}: handovers mismatch"
        )


def test_intercontinental_flag():
    assert plan_route('brussels', 'shanghai').is_intercontinental is True
    assert plan_route('brussels', 'madrid').is_intercontinental is False
