"""Rolling-origin schedule: origins match docs/protocol.md section 5 and training never touches a test target."""
import numpy as np
import pytest

from forecasting_study.splits import DAILY, MONTHLY, SCHEDULES, Schedule, iter_splits


def test_monthly_origins_match_protocol():
    assert MONTHLY.test_origins == (351, 357, 363, 369, 375, 381, 387, 393)
    assert MONTHLY.val_origins == (300, 306, 312, 318, 324)
    assert MONTHLY.horizons == (1, 3, 6, 12) and MONTHLY.m == 12 and MONTHLY.T == 405


def test_daily_origins_match_protocol():
    assert DAILY.test_origins == (872, 900, 928, 956, 984, 1012, 1040, 1068)
    assert DAILY.val_origins == (732, 760, 788, 816, 844)
    assert DAILY.horizons == (1, 7, 14, 28) and DAILY.m == 7 and DAILY.T == 1096


@pytest.mark.parametrize("sched", list(SCHEDULES.values()), ids=lambda s: s.name)
def test_at_least_five_origins_and_last_target_is_T(sched):
    assert len(sched.test_origins) >= 5
    assert sched.test_origins[-1] + sched.H == sched.T


def test_monthly_origins_map_to_expected_dates():
    import pandas as pd
    months = pd.date_range("1992-01-01", periods=MONTHLY.T, freq="MS")
    assert months[MONTHLY.test_origins[0] - 1].strftime("%Y-%m") == "2021-03"
    assert months[MONTHLY.test_origins[-1] - 1].strftime("%Y-%m") == "2024-09"
    assert months[MONTHLY.T - 1].strftime("%Y-%m") == "2025-09"
    assert months[MONTHLY.val_origins[0] - 1].strftime("%Y-%m") == "2016-12"
    assert months[MONTHLY.val_origins[-1] - 1].strftime("%Y-%m") == "2018-12"


@pytest.mark.parametrize("sched", list(SCHEDULES.values()), ids=lambda s: s.name)
@pytest.mark.parametrize("stage", ["validation", "test"])
def test_training_window_never_includes_its_target_or_later(sched, stage):
    n = 0
    for origin, h, train, target in iter_splits(sched, stage):
        n += 1
        assert train[0] == 0 and train[-1] + 1 == origin           # expanding window y_1..y_origin
        assert np.array_equal(train, np.arange(origin))             # contiguous
        assert train[-1] + 1 < target                               # last training index (1-based) < target index
        assert target == origin + h <= sched.T
    assert n == len(sched.horizons) * len(getattr(sched, "val_origins" if stage == "validation" else "test_origins"))


@pytest.mark.parametrize("sched", list(SCHEDULES.values()), ids=lambda s: s.name)
def test_no_training_window_includes_test_period(sched):
    """Validation training windows and validation targets stay before the first test target; test training windows
    contain only data up to their own origin (so never any later test target)."""
    first_test_target_possible = sched.test_origins[0] + 1
    for origin, h, train, target in iter_splits(sched, "validation"):
        assert train[-1] + 1 < first_test_target_possible
        assert target <= sched.test_origins[0]
    test_targets = {t for _, _, _, t in iter_splits(sched, "test")}
    for origin, h, train, target in iter_splits(sched, "test"):
        later_targets = {x for x in test_targets if x > origin}
        assert not (set((train + 1).tolist()) & later_targets)      # 1-based indices in training vs later targets


def test_validation_and_test_targets_are_disjoint():
    for sched in SCHEDULES.values():
        val_t = {t for _, _, _, t in iter_splits(sched, "validation")}
        test_t = {t for _, _, _, t in iter_splits(sched, "test")}
        assert max(val_t) <= sched.test_origins[0]
        assert not (val_t & {t for t in test_t if t > sched.test_origins[0]})


def test_validate_catches_bad_schedule():
    bad = Schedule("bad", T=100, m=4, horizons=(1, 2), spacing=5, val_origins=(40, 45),
                   test_origins=(60, 65, 70, 75, 80), n_cal=10)  # last origin + H = 82 != 100
    with pytest.raises(AssertionError):
        bad.validate()
    overlap = Schedule("overlap", T=100, m=4, horizons=(1, 2), spacing=5, val_origins=(60, 65, 70, 75, 80),
                       test_origins=(78, 83, 88, 93, 98), n_cal=10)
    with pytest.raises(AssertionError):
        overlap.validate()
