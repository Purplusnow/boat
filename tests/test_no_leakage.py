"""누수 방지 회귀 테스트.

경주 후에야 정해지는 값이 피처에 섞이면, 검증 지표는 좋아지고 실전 성적만
나빠진다. 그 격차는 원인을 짚기 어려워 오래 남는다. 그래서 '결과에서 온 값'이
피처 목록에 들어갈 수 없다는 것을 기계가 지키게 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boatai import features as ft  # noqa: E402

# 경주 후에 정해지는 값들. 하나라도 피처에 들어가면 안 된다.
FORBIDDEN = {
    "ord", "ord_note", "y_win", "y_top2", "y_top3",
    "win_payout", "payout", "has_result",
    "p_win", "p_top2", "p_top3", "p_win_norm", "pred_rank",
}


def _frame() -> pd.DataFrame:
    """피처 목록을 뽑기에 충분한 최소 프레임."""
    rows = []
    for lane in range(1, 7):
        rows.append({
            "race_key": "2026-05-1-01", "lane": lane, "racer_nm": f"선수{lane}",
            "racer_grd": "A1", "race_class": "일반", "st_method": "002", "sex": "남",
            "age": 30 + lane, "weight": 52.0 + lane, "avg_rank": 3.0,
            "high_rate": 30.0, "avg_acdnt_scr": 0.2, "f_cnt": 0, "l_cnt": 0,
            "tms6_avg_rank_scr": 5.0, "tms6_avg_scr": 5.0, "tms6_win_ratio": 15.0,
            "tms6_high_rate": 30.0, "tms6_high3_rate": 45.0, "tms6_avg_st": 0.20,
            "bf_dd_recd_scr": 5.0, "mm6_race_cnt": 30, "thdd_race_no": 3,
            "mot_avg_rank_scr": 4.5, "mot_high_rate": 30.0, "mot_high3_rate": 40.0,
            "mot_prev_avg": 3.0, "boat_avg_rank_scr": 4.5, "boat_high_rate": 30.0,
            "recent_avg": 3.5, "recent_cnt": 8, "field_size": 6,
            "stnd_yr": 2026, "week_tcnt": 5, "day_tcnt": 1, "race_no": 1,
            "race_ymd": "20260128", "ord": lane, "win_payout": 1.7,
            **{f"recent{i}": 3 for i in range(1, 9)},
            **{f"course{c}_high_rate": 50.0 for c in range(1, 7)},
            **{f"course{c}_cnt": 6 for c in range(1, 7)},
        })
    df = pd.DataFrame(rows)
    df["own_course_high_rate"] = ft._own_course(df)
    df["own_course_cnt"] = ft._own_course_cnt(df)
    return ft.add_relative(df)


def test_feature_columns_exclude_outcome_fields():
    cols = set(ft.feature_columns(_frame()))
    leaked = cols & FORBIDDEN
    assert not leaked, f"결과에서 온 값이 피처에 들어갔다: {sorted(leaked)}"


def test_feature_columns_have_no_duplicates():
    # 중복이 있으면 학습·추론 행렬의 열 수가 어긋나 조용히 틀린 예측이 나온다.
    cols = ft.feature_columns(_frame())
    assert len(cols) == len(set(cols))


def test_own_course_picks_the_lane_actually_drawn():
    df = _frame()
    # 3코스 값만 다르게 두면, 3번 정만 그 값을 가져가야 한다.
    df["course3_high_rate"] = 77.0
    got = ft._own_course(df)
    assert got[df["lane"] == 3].iloc[0] == 77.0
    assert got[df["lane"] == 1].iloc[0] == 50.0


def test_relative_features_are_zero_when_all_equal():
    # 전원이 같은 값이면 편차는 0이고, 표준편차 0으로 나눠 inf 를 만들지 않는다.
    df = _frame()
    assert (df["rel_mot_high_rate"] == 0).all()
    assert df["z_mot_high_rate"].isna().all()
