"""The win / loss / tie rule and the test-day lock."""

import json

import pytest

import rewind.compare.final as final
from rewind.compare.final import verdict


def r(lo, hi):
    return {"min": lo, "max": hi, "mean": (lo + hi) / 2}


@pytest.mark.parametrize("seq, dcn, metric, expected", [
    (r(.670, .672), r(.666, .668), "auc", "win"),     # worst sequence seed > best DCN seed
    (r(.664, .666), r(.667, .668), "auc", "loss"),
    (r(.666, .669), r(.667, .668), "auc", "tie"),     # overlapping ranges
    (r(.950, .952), r(.955, .957), "ne", "win"),      # lower NE is better
    (r(.958, .960), r(.955, .957), "ne", "loss"),
    (r(.990, 1.01), r(1.05, 1.07), "calibration", "win"),   # straddles 1 vs 5-7% too high
    (r(0.90, 0.92), r(0.97, 1.02), "calibration", "loss"),
    (r(0.95, 1.07), r(0.98, 1.03), "calibration", "tie"),
])
def test_verdict(seq, dcn, metric, expected):
    assert verdict(seq, dcn, metric) == expected


def test_test_day_is_locked_and_reruns_are_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "LOCK", tmp_path / "lock.json")
    with pytest.raises(SystemExit, match="locked"):
        final.guard_test_day(opened=False, reason=None)
    final.guard_test_day(opened=True, reason=None)  # first opening: allowed, recorded
    with pytest.raises(SystemExit, match="already opened"):
        final.guard_test_day(opened=True, reason=None)
    final.guard_test_day(opened=True, reason="fixed a bug in scoring")
    log = json.loads((tmp_path / "lock.json").read_text())
    assert log["opened_at_utc"] and log["reruns"][0]["reason"] == "fixed a bug in scoring"
