"""전개 시뮬레이션 회귀 테스트.

물리가 틀리면 전개는 그럴듯한 거짓말이 된다. 손으로 검산할 수 있는 성질만
고정한다 — 상수의 정확한 값이 아니라 **부호와 순서**다. 그것이 이 모듈이
주장하는 전부이기도 하다.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boatai import simulate as sm  # noqa: E402


def _runners(over=None):
    """여섯 척. 기본은 전원 동일 조건이고, 정번별로 필요한 것만 덮어쓴다."""
    over = over or {}
    rs = []
    for lane in range(1, 7):
        r = {
            "lane": lane, "racer_nm": f"선수{lane}", "racer_grd": "A1",
            # **정번별 사전확률을 그대로 준다.** 시뮬레이션은 모델 확률에서
            # 코스 몫을 나눠 '기량'을 얻으므로, 전원 1/6 을 주면 6코스가
            # '사전확률 6% 인데 16.7% 를 받은 굉장한 선수'가 되어 버린다.
            # 사전확률을 그대로 주어야 기량이 전원 동일해진다.
            "p_win": sm.LANE_PRIOR[lane],
            "tms6_avg_st": 0.16, "tms6_high3_rate": 50.0,
            "own_course_rate": 40.0, "mot_high_rate": 30.0, "boat_high_rate": 30.0,
            "pred_rank": lane,
        }
        r.update(over.get(lane, {}))
        rs.append(r)
    return rs


# ── 물리 ───────────────────────────────────────────────────────

def test_bigger_radius_is_slower_to_traverse_but_exits_faster():
    """선회반경–속도 상충. 이 한 줄이 경정 전법의 전부다."""
    t_in, t_out = sm.turn_time(38.0, 0.0), sm.turn_time(50.0, 0.0)
    v_in, v_out = sm.turn_speed(38.0, 0.0), sm.turn_speed(50.0, 0.0)
    assert t_out > t_in, "바깥으로 크게 돌면 통과 시간이 더 걸려야 한다"
    assert v_out > v_in, "바깥으로 크게 돌면 더 빠른 속도로 나와야 한다"


def test_turn_speed_follows_sqrt_law():
    # v(r) = √(A·r) — 반경이 4배면 속도는 2배.
    assert math.isclose(sm.turn_speed(40.0, 0.0) / sm.turn_speed(10.0, 0.0), 2.0,
                        rel_tol=1e-6)


def test_turn_speed_is_capped_by_straight_speed():
    # 반경이 아무리 커도 직선 최고 속도를 넘을 수는 없다.
    assert sm.turn_speed(10_000.0, 0.0) == sm.V_STRAIGHT


def test_better_turn_skill_is_faster_at_same_radius():
    assert sm.turn_time(40.0, 1.0) < sm.turn_time(40.0, -1.0)


# ── 전개 ───────────────────────────────────────────────────────

def test_inner_lane_wins_more_when_all_else_equal():
    """조건이 같으면 1코스가 유리해야 한다. 실제 경정이 그렇다(36% vs 4%)."""
    sim = sm.simulate(_runners(), n_sims=600, seed=1)
    by_lane = {b["lane"]: b["sim_win"] for b in sim["per_boat"]}
    assert by_lane[1] > by_lane[6]
    assert by_lane[1] > by_lane[3] > by_lane[6]


def test_bad_start_loses_the_inside_line():
    """1코스라도 스타트가 나쁘면 최내측을 잡지 못한다."""
    good = sm.simulate(_runners(), n_sims=600, seed=2)
    bad = sm.simulate(_runners({1: {"tms6_avg_st": 0.60}}), n_sims=600, seed=2)
    g = next(b["turn1_inside"] for b in good["per_boat"] if b["lane"] == 1)
    d = next(b["turn1_inside"] for b in bad["per_boat"] if b["lane"] == 1)
    assert d < g


def test_probabilities_sum_to_one():
    sim = sm.simulate(_runners(), n_sims=400, seed=3)
    assert math.isclose(sum(b["sim_win"] for b in sim["per_boat"]), 1.0, abs_tol=1e-9)
    assert math.isclose(sum(b["turn1_inside"] for b in sim["per_boat"]), 1.0, abs_tol=1e-9)


def test_script_matches_its_own_tactic_and_winner():
    """머리말의 전법과 대본의 전법이 반드시 같은 것을 가리켜야 한다."""
    sim = sm.simulate(_runners({1: {"p_win": 0.6}}), n_sims=600, seed=4)
    assert f"전법은 {sim['top_tactic']}" in sim["script"][-1]["text"]
    win = sim["script_winner"]
    assert f"{win['lane']}번 {win['racer_nm']} 승리" in sim["script"][-1]["text"]


def test_model_probability_anchors_the_simulation():
    """모델이 강하게 미는 배는 시뮬레이션에서도 크게 올라와야 한다.

    다만 **1위까지 요구하지는 않는다.** 6코스는 선회 기하가 정면으로 불리하고,
    그 둘이 맞설 때 시뮬레이션이 모델과 다른 답을 내는 것은 오류가 아니라
    정보다 — 화면은 두 값을 나란히 보여주고, 순위는 모델이 정한다고 적는다.
    """
    base = sm.simulate(_runners(), n_sims=800, seed=5)
    push = sm.simulate(_runners({6: {"p_win": 0.75}}), n_sims=800, seed=5)
    b6 = next(b["sim_win"] for b in base["per_boat"] if b["lane"] == 6)
    p6 = next(b["sim_win"] for b in push["per_boat"] if b["lane"] == 6)
    assert p6 > b6 * 1.5, "모델이 강하게 밀면 시뮬레이션 승률도 뚜렷이 올라야 한다"


def test_missing_stats_do_not_crash_or_dominate():
    """신인(관측값 없음)이 '스타트 완벽한 선수'가 되어선 안 된다."""
    rs = _runners({4: {"tms6_avg_st": None, "mot_high_rate": None,
                         "own_course_rate": None, "tms6_high3_rate": None}})
    sim = sm.simulate(rs, n_sims=400, seed=6)
    by_lane = {b["lane"]: b["sim_win"] for b in sim["per_boat"]}
    assert by_lane[4] < by_lane[1]


def test_confidence_labels_track_dominance():
    flat = sm.simulate(_runners(), n_sims=600, seed=7)
    domin = sm.simulate(_runners({1: {"p_win": 0.85}}), n_sims=600, seed=7)
    assert domin["confidence"]["score"] > flat["confidence"]["score"]


# ── 주행 궤적 / 드리프트 ────────────────────────────────────────

def test_trace_covers_every_course_segment():
    import numpy as np
    boats = sm.build_boats(_runners())
    res = sm._one_race(boats, np.random.default_rng(1), trace=True)
    tr = res["trace"]
    # 경계 수 = 구간 수 + 1. 어긋나면 재생기가 엉뚱한 구간을 그린다.
    assert len(tr["t"][0]) == len(sm.COURSE) + 1 == len(sm.COURSE_MARKS)
    assert len(tr["t"]) == len(boats)


def test_course_totals_real_travel_distance():
    """구간 합은 **실제 주행거리**여야 한다. 공식 1800m 가 아니다.

    공식 1주회 600m 는 부표를 스치듯 도는 이론선이고, 실제로는 선회 반경만큼
    더 돈다(약 2,177m). 둘을 섞으면 완주 시간이 실제보다 20% 짧게 나온다.
    """
    assert abs(sm.COURSE_MARKS[-1] - sm.REAL_LAP_M * sm.LAPS) < 1e-6
    assert sm.COURSE_MARKS[-1] > sm.LAP_M * sm.LAPS      # 실제가 공식보다 길다
    assert sum(1 for k, _ in sm.COURSE if k == "T") == sm.LAPS * 2


def test_turn_marks_are_300m_apart():
    # 부표 간격은 공표값이고, 1주회 공식 600m 가 여기서 나온다.
    assert sm.MARK_GAP == 300.0
    assert sm.LAP_M == sm.MARK_GAP * 2


def test_turn_speed_is_physically_plausible():
    """최내측 선회가 실제 경정의 속도·횡가속 범위 안에 있어야 한다.

    반경을 너무 작게 잡으면 배가 1g 넘게 버티는 물체가 되고, 크게 잡으면
    선회가 직선처럼 되어 전법이 사라진다.
    """
    v = sm.turn_speed(sm.R_INNER, 0.0)
    assert 9.0 <= v <= 15.0, "최내측 선회 속도 33~54km/h 범위"
    g = v * v / sm.R_INNER / 9.8
    assert 0.4 <= g <= 0.9, "횡가속이 물 위 활주체의 범위를 벗어났다"


def test_times_increase_along_the_course():
    import numpy as np
    boats = sm.build_boats(_runners())
    tr = sm._one_race(boats, np.random.default_rng(2), trace=True)["trace"]
    for row in tr["t"]:
        assert all(b >= a for a, b in zip(row, row[1:])), "시각이 뒤로 갈 수 없다"


def test_inner_line_drifts_more():
    """안쪽을 파고든 배가 더 크게 밀려야 한다.

    선회 속도가 낮을수록 직선에서 싣고 온 속도를 더 많이 깎아야 하고, 그 깎기를
    선미를 던져서 한다. 이 부호가 뒤집히면 화면에서 바깥 배가 더 미끄러진다.
    """
    import numpy as np
    boats = sm.build_boats(_runners())
    res = sm._one_race(boats, np.random.default_rng(3), trace=True)
    beta_turn1 = [row[2] for row in res["trace"]["beta"]]     # 구간 2 = 1턴
    inner = int(np.argmin(res["radius"]))
    outer = int(np.argmax(res["radius"]))
    assert beta_turn1[inner] > beta_turn1[outer]


def test_slip_angle_stays_within_physical_bound():
    import numpy as np
    boats = sm.build_boats(_runners())
    tr = sm._one_race(boats, np.random.default_rng(4), trace=True)["trace"]
    flat = [v for row in tr["beta"] for v in row]
    assert min(flat) >= 0.0
    assert max(flat) <= math.degrees(sm.BETA_MAX) + 1e-6


def test_player_payload_is_self_contained():
    """재생기는 이 payload 만으로 코스를 다시 그릴 수 있어야 한다."""
    sim = sm.simulate(_runners(), n_sims=200, seed=5)
    pl = sim["player"]
    assert set(pl) == {"course", "boats", "duration"}
    assert set(pl["course"]) >= {"straight", "r_inner", "lap", "laps",
                                 "start_offset", "marks", "kinds"}
    assert len(pl["boats"]) == 6
    for b in pl["boats"]:
        assert len(b["t"]) == len(b["off"]) == len(b["beta"]) == len(pl["course"]["marks"])
    # 완주 순위는 1~6 이 한 번씩
    assert sorted(b["finish"] for b in pl["boats"]) == [1, 2, 3, 4, 5, 6]
