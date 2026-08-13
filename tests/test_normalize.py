"""정규화 규칙 회귀 테스트.

경정 API 는 한 필드에 값 두 개를 붙여 준다. 파싱이 조용히 틀리면 모델은 문자열
범주를 배우고, 그 사실은 지표가 나빠지는 것 외에는 어디에도 드러나지 않는다.
실제 응답에서 그대로 옮긴 값으로 고정해 둔다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boatai.kboat import normalize as nz  # noqa: E402


def test_race_key_is_zero_padded():
    # 회차·경주번호는 자릿수가 다르면 문자열 정렬이 시간순과 어긋난다.
    assert nz.race_key("2026", 5, 1, "01") == "2026-05-1-01"
    assert nz.race_key(2026, 33, 2, 17) == "2026-33-2-17"
    assert nz.race_key("2026", 5, 1, "01") < nz.race_key("2026", 33, 2, "17")


def test_coords_round_trip():
    key = nz.race_key(2026, 33, 2, 17)
    assert nz.coords_from_key(key) == {
        "race_key": key, "stnd_yr": 2026, "week_tcnt": 33,
        "day_tcnt": 2, "race_no": 17}


def test_ratio_count_keeps_denominator():
    # "60.0/5" 는 5회 중 3회 연대. 모수를 버리면 3회짜리 100% 와 30회짜리
    # 80% 가 같은 무게로 들어간다.
    assert nz.parse_ratio_count("60.0/5") == (60.0, 5)
    assert nz.parse_ratio_count("100.0/3") == (100.0, 3)
    assert nz.parse_ratio_count("") == (None, None)


def test_fl_counts():
    assert nz.parse_fl("F0L0") == (0, 0)
    assert nz.parse_fl("F1L2") == (1, 2)
    assert nz.parse_fl("") == (None, None)


def test_recent_ranks_drops_spacing():
    assert nz.parse_recent_ranks("4346 3222") == [4, 3, 4, 6, 3, 2, 2, 2]
    assert nz.parse_recent_ranks("") == []


def test_prev_motor_rider():
    assert nz.parse_prev_motor_rider("김은지/444") == ("김은지", [4, 4, 4])


def test_pool_value_splits_combo_and_payout():
    # 실제 응답: 2026년 5회차 1일차 1R (1착 정3, 2착 정6, 3착 정5)
    assert nz.parse_pool_value("③1.7") == ("3", 1.7)
    assert nz.parse_pool_value("③-⑥8.3") == ("3-6", 8.3)
    assert nz.parse_pool_value("③-⑥-⑤22.2") == ("3-6-5", 22.2)
    assert nz.parse_pool_value("") == (None, None)


def test_result_top3_reads_lane_and_name():
    rec = {"stnd_yr": "2026", "week_tcnt": 5, "day_tcnt": "1", "race_no": "01",
           "rank1": "③고정환", "rank2": "⑥곽현성", "rank3": "⑤손유정"}
    rows = nz.result_top3(rec)
    assert [(r["lane"], r["racer_nm"], r["ord"]) for r in rows] == [
        (3, "고정환", 1), (6, "곽현성", 2), (5, "손유정", 3)]


def test_entry_row_reads_lane_from_race_reg_no():
    # race_reg_no 는 명세상 '선수번호'지만 실제 값은 정번(1~6)이다.
    rec = {"stnd_yr": "2026", "week_tcnt": "5", "day_tcnt": "1", "race_no": "01",
           "race_reg_no": "1", "racer_nm": "문주엽", "racer_grd_cd": "A1",
           "motor_no": "078", "boat_no": "056", "fl_tcnt": "F0L0",
           "tms_8_rank_ord_no": "4346 3222",
           "mm_6_1_race_high_rank_ratio": "100.0/3"}
    row = nz.entry_row(rec)
    assert row["lane"] == 1
    assert row["motor_no"] == 78 and row["boat_no"] == 56
    assert row["course1_high_rate"] == 100.0 and row["course1_cnt"] == 3
    assert row["recent1"] == 4 and row["recent_cnt"] == 8


def test_result_rows_keep_unmatched_lane_as_none():
    # 정번을 못 붙여도 행을 버리지 않는다. 버리면 착순이 통째로 사라진다.
    recs = [{"stnd_year": "2026", "tms": "5", "day_ord": "1", "race_no": "01",
             "race_rank": "1", "racer_no": "14-001", "racer_nm": "고정환",
             "race_day": "20260128", "mbr_nm": "미사리"}]
    rows = nz.result_rows(recs, {})
    assert len(rows) == 1 and rows[0]["lane"] is None
    assert rows[0]["ord"] == 1 and rows[0]["race_ymd"] == "20260128"
