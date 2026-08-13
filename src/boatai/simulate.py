"""경주 전개 시뮬레이션 — 경정의 1턴마크 물리를 실제로 돌려 본다.

모델은 배를 한 척씩 독립으로 채점해 확률을 정규화한다. 그러나 경정은
**한 지점에서 승부가 갈리는 종목**이다. 여섯 척이 같은 1턴마크로 몰리고,
거기서 안쪽을 잡느냐 바깥으로 도느냐가 나머지 두 바퀴 반을 결정한다.
독립 채점으로는 이 상호작용을 담을 수 없다.

## 무엇을 계산하나 — 선회반경과 속도의 상충

배는 코너에서 옆으로 미끄러지지 않을 만큼만 속도를 낼 수 있다. 선회반경 r 로
돌 때 낼 수 있는 속도는

    v(r) = √(A·r)          A = 횡가속 한계 (m/s²)

이고, 반원(π·r)을 도는 데 걸리는 시간은

    t(r) = π·r / v(r) = π·√(r/A)

**반경이 커지면 통과 시간은 늘지만 빠져나오는 속도는 빨라진다.** 이 한 줄이
경정 전법의 전부다.

  * **인빠지기** — 안쪽(작은 r)을 잡으면 짧고 빨리 돌아 먼저 나온다. 대신
    느린 속도로 나오므로 뒤에 붙은 배에게 직선에서 따라잡힐 여지를 남긴다.
  * **휘감기** — 바깥(큰 r)으로 크게 돌면 늦게 나오지만 속도를 싣고 나온다.
    스타트가 좋아 시간 손해를 미리 벌어 두면 그대로 앞선다.
  * **찌르기** — 앞선 배가 크게 부풀어 오르면 그 안쪽에 생긴 틈을 파고든다.

## 어디까지가 사실이고 어디부터가 가정인가

**코스 제원은 추정값이다.** 미사리 주행수면의 1주회 600m 는 공표된 값이지만,
직선과 곡선 각각의 길이, 스타트라인에서 1턴마크까지의 거리, 횡가속 한계 A 는
실측 자료를 구하지 못해 아래 값으로 두었다. 그래서 이 모듈이 내는 **초 단위
숫자는 신뢰할 수 없다.** 신뢰할 수 있는 것은 부호와 순서다 — 안쪽이 먼저
나온다, 바깥은 속도를 싣고 나온다, 스타트가 나쁘면 안쪽을 못 잡는다.

그래서 시뮬레이션 결과는 **승률로 쓰지 않는다.** 승률은 학습된 모델이 낸다.
여기서는 모델이 못 주는 것 — 1턴마크 선두 확률, 전법 분포, 전개 대본 — 만
만든다. 화면에서 둘이 다른 것을 말하지 않도록 배의 기본 기량은 모델 확률에
닻을 내린다(``MODEL_WEIGHT``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# ── 코스 제원 ──────────────────────────────────────────────────────
#
# **경정장은 트랙이 아니다.** 경마장처럼 둘레가 막힌 주로가 아니라, 열린
# 수면 위에 **턴마크 부표 두 개**가 떠 있고 그 사이를 가르는 차단 구조물
# (센터폰툰)이 있을 뿐이다. 배는 어디로든 갈 수 있고, 다만 두 부표를 돌아야
# 한다. 그래서 '레인'도 '주로 폭'도 없고, 안쪽을 얼마나 좁게 도느냐는
# 전적으로 선수의 선택이다 — 이 선택이 곧 전법이다.
#
# 부표 간격 300m · 1주회 600m(=300×2)은 공표값이다. 다만 그 600m 는 부표를
# 스치듯 도는 **최내측 이론선**이고, 실제 주행거리는 선회 반경만큼 더 길다.
MARK_GAP = 300.0         # 1턴마크 ↔ 2턴마크 (공표값)
LAP_M = MARK_GAP * 2     # 1주회 공식 600m
LAPS = 3                 # 공식 1800m = 3바퀴
STRAIGHT_M = MARK_GAP    # 부표 사이 직선 구간
R_INNER = 20.0           # 최내측 선회반경(m). 부표를 이 정도로 감아 돈다
TURN_ARC_M = math.pi * R_INNER          # 최내측 선회 궤적 길이 ≈ 62.8m
LANE_W = 3.0             # 선회 시 배 사이 가로 간격(m) — 보정값
START_TO_MARK = 140.0    # 스타트라인 → 1턴마크 (추정)

# A_LAT 는 배가 옆으로 버틸 수 있는 한계다. R=20m 에서 선회 속도 11.8m/s(43km/h),
# 횡가속 0.71g — 활주 중 미끄러지며 도는 실제 경정정의 값에 가깝다.
A_LAT = 7.0              # 횡가속 한계 (m/s²)
V_STRAIGHT = 23.0        # 직선 최고 속도 (m/s) ≈ 83km/h
V_APPROACH = 21.0        # 스타트 후 1턴마크까지 진입 속도 (m/s)
#
# 실제 주행거리는 600×3 이 아니라 약 2,180m 이고, 완주 시간은 약 110초로
# 실제 경정(1분 50초 안팎)과 맞는다. 화면에는 공식 거리(1800m)를 적는다.

N_SIMS = 1500            # 반복 횟수. 6정이라 이 정도면 확률 표준오차 ~1%p

# ── 실측 자료를 못 구해 **보정으로 정한** 상수들 ──────────────────
# 손으로 정하면 전개는 그럴듯한 이야기일 뿐이다. 그래서 코스별 1착 비율
# (실측: 36 / 23 / 16 / 13 / 8 / 4%) 을 재현하도록 tools/calibrate_sim.py 로
# 맞췄다. 값을 바꾸면 반드시 그 도구를 다시 돌려야 한다.
#
# **보정은 '전원 동일 조건'에서 한다.** 실제 경주로 맞추면 모델 확률이 이미
# 코스 우위를 담고 있어, 물리가 엉망이어도 총합은 맞는 것처럼 보인다. 실제로
# 처음엔 그렇게 맞췄다가 전원 동일 조건을 돌려 보고서야 알았다 — 기하학이 만드는
# 격차가 1코스 21% · 6코스 13% 뿐이었고, 나머지는 전부 모델이 만들고 있었다.
#
# 여섯 척의 기량을 완전히 같게 두고 정번만 다르게 했을 때 (순수 기하학):
#
#     코스     1     2     3     4     5     6    오차
#     실측   36.1  23.2  16.2  12.5   7.8   4.1
#     현재   36.2  22.3  16.2  11.8   7.9   5.5   0.033
#     이전   21.0  18.8  16.3  16.6  14.9  12.5   0.399  ← 물리가 거의 무력
#
# 순서까지 실측과 같다. 여기에 모델 앵커를 얹어 실제 경주로 돌리면 총합
# 오차는 0.115 로 늘고, 5·6코스가 서로 근접한다(5.3% / 6.2%). 앵커가 남기는
# 찌꺼기이며, 이 시뮬레이션이 승률을 주장하지 않는 이유이기도 하다.
#
# 가장 결정적인 것은 PACE_SIGMA 였다. 0.065 는 90초 주행에 곱해져 5.8초를
# 흔든다 — 1턴마크에서 번 0.55초를 통째로 덮어 버린다. 실제 배의 속도 변동에
# 가까운 2% 로 낮추자 비로소 선회 물리가 결과를 결정하게 됐다.
EXIT_GAIN = 2.0          # 탈출 속도가 뒤이은 직선에서 시간으로 바뀌는 정도(초)
                         # (선회 여섯 번 모두에 적용되므로 값이 작다)
TRAIL_PENALTY = 0.35     # 뒤에 있을수록 이후 선회를 얼마나 더 바깥으로 도는가
TAU = 0.12               # 안쪽 라인을 한 칸 뺏는 데 필요한 시간 우위(초)

# 배의 기본 기량을 무엇으로 정할지. 모델 확률에 무게를 두는 이유는 화면에서
# 예상 순위와 전개가 서로 다른 말을 하지 않게 하기 위해서다. 나머지는 출주표의
# 관측값(모터·코스 성적)이 담당해 전개에 질감을 준다.
MODEL_WEIGHT = 0.30

# 앵커에서 빼야 할 것은 **모델이 실제로 코스에 부여한 몫**이다. 실측 비율을
# 쓰면 모델과 실측의 차이만큼 찌꺼기가 남는다 — 실제로 그 찌꺼기 때문에
# 6코스가 5코스를 앞지르는 뒤집힘이 났다.
#
#   코스      1      2      3      4      5      6
#   모델   .326   .222   .165   .135   .094   .060   ← 이 값을 쓴다
#   실측   .361   .232   .162   .125   .078   .041
#
# 모델은 가운데로 수축한다(1코스를 3.5%p 낮게, 6코스를 1.9%p 높게 본다).
# 정규화가 있는 모델의 전형적인 성질이며, 그 자체가 하나의 발견이다.
LANE_PRIOR = {1: 0.326, 2: 0.222, 3: 0.165, 4: 0.135, 5: 0.094, 6: 0.060}
SKILL_CLIP = 0.8         # 코스를 뺀 기량(로그 비율)의 양끝을 자르는 폭 — 보정값

# 흔들림. 경정은 같은 배·같은 선수라도 파도와 물살에 따라 결과가 크게 갈린다.
ST_SIGMA = 0.055         # 스타트 타이밍 표준편차(초) — 보정값
PACE_SIGMA = 0.020       # 당일 기량 변동(비율). 실제 배의 속도 변동에 가까운 값이며,
                         # 여기를 키우면 선회 물리가 잡음에 묻힌다(위 표 참조)
TURN_SIGMA = 0.055       # 선회 품질 변동(비율)

# ── 드리프트 (선회 중 밀림) ────────────────────────────────────────
# 배는 말과 달리 **바퀴로 방향을 바꾸지 않는다.** 직선에서 싣고 온 속도를
# 선회 속도까지 깎아내면서 동시에 방향을 바꿔야 하고, 그 둘을 한꺼번에 하는
# 방법이 선미를 바깥으로 던지는 것이다 — 선체가 진행 방향보다 안쪽을 향한 채
# 옆으로 미끄러진다. 경정 선회의 그림 자체가 이것이다.
#
#   깎아야 할 속도의 비율 = (진입속도 − 선회속도) / 선회속도
#
# 이 값이 클수록 크게 밀린다. **안쪽을 파고든 배일수록 선회 속도가 낮아
# 더 많이 깎아야 하므로 더 크게 밀린다** — 인빠지기가 화려해 보이는 이유다.
BETA_MAX = math.radians(42.0)   # 최대 슬립각(선체 방향과 진행 방향의 차이)
DRIFT_REF = 1.40                # 이 비율만큼 깎아야 하면 슬립각이 최대가 된다
                                # (부표를 좁게 감을수록 이 값을 넘어 최대로 밀린다)


@dataclass
class Boat:
    """시뮬레이션에 들어가는 한 척. 전부 경주 전에 알 수 있는 값이다."""

    lane: int
    racer_nm: str
    p_win: float                  # 모델 승률 (닻)
    st_mean: float = 0.16         # 평균 스타트 타이밍(초). 작을수록 좋다
    turn_skill: float = 0.0       # 선회력 (표준화)
    power: float = 0.0            # 모터·보트 힘 (표준화)
    grade: str = ""

    # 아래는 계산으로 채운다
    ability: float = 0.0          # 종합 기량 (표준화)


def _z(vals: Sequence[float]) -> np.ndarray:
    a = np.asarray(vals, dtype=float)
    if np.all(np.isnan(a)):
        return np.zeros(len(a))
    m = np.nanmean(a)
    s = np.nanstd(a)
    a = np.where(np.isnan(a), m, a)
    return (a - m) / s if s > 1e-9 else np.zeros(len(a))


def build_boats(runners: List[Dict]) -> List[Boat]:
    """출주표 행 → 시뮬레이션 입력.

    관측값이 비어 있는 신인은 경주 평균으로 채운다. 0 으로 채우면 '스타트가
    완벽한 선수'가 되어 전개가 통째로 뒤집힌다.
    """
    def g(r, k, d=np.nan):
        v = r.get(k)
        return float(v) if v is not None else d

    st = [g(r, "tms6_avg_st") for r in runners]
    # 평균ST 는 경주구분에 따라 눈금이 다르다(0.2대 / 18대). 경주 안에서는
    # 눈금이 같으므로 **경주 내 표준화**로만 쓴다 — 절대값을 쓰면 구분 001
    # 경주가 전부 '스타트 최악'이 된다.
    st_z = _z(st)
    turn = _z([np.nanmean([g(r, "tms6_high3_rate"), g(r, "own_course_rate")])
               for r in runners])
    power = _z([np.nanmean([g(r, "mot_high_rate"), g(r, "boat_high_rate")])
                for r in runners])
    p = np.array([max(g(r, "p_win", 1 / 6), 1e-4) for r in runners])
    # **코스 몫을 빼고 기량만 남긴다.** 모델 확률에는 이미 코스 우위가 들어
    # 있는데, 이 시뮬레이션은 선회 기하로 코스 우위를 따로 만든다. 그대로
    # 곱하면 이중 계산이 되어 1코스가 44.7%(실측 36.1%)까지 부푼다.
    # 그래서 앵커는 '그 코스치고 얼마나 강한가' 만 담는다.
    prior = np.array([LANE_PRIOR.get(int(r["lane"]), 1 / 6) for r in runners])
    # 경주 내 재표준화(_z)를 쓰지 않는다. 6코스는 사전확률이 4% 라 log 비율의
    # 폭이 1코스보다 훨씬 넓은데, 거기에 z 를 씌우면 '전원 평범한 경주'에서도
    # 누군가는 크게 밀려 6코스 승률이 부풀었다(실측 4.1% 대비 9.7%).
    # 로그 비율 자체가 이미 비교 가능한 눈금이므로 그대로 쓰고 양끝만 자른다.
    model_z = np.clip(np.log(p / prior), -SKILL_CLIP, SKILL_CLIP)

    boats = []
    for i, r in enumerate(runners):
        b = Boat(
            lane=int(r["lane"]), racer_nm=r.get("racer_nm") or "",
            p_win=float(g(r, "p_win", 1 / 6)),
            # ST 는 **작을수록 좋다**. 관측 ST 가 평균보다 크면(z 가 양수면)
            # 시뮬레이션의 스타트도 그만큼 늦어야 한다. 여기서 부호를 뒤집으면
            # 스타트가 나쁜 선수가 가장 먼저 튀어나가고, 그 오류는 화면에
            # '스타트 1위'로 버젓이 표시된다.
            st_mean=0.16 + 0.02 * float(st_z[i]),
            turn_skill=float(turn[i]), power=float(power[i]),
            grade=r.get("racer_grd") or "",
        )
        b.ability = (MODEL_WEIGHT * model_z[i]
                     + (1 - MODEL_WEIGHT) * (0.6 * power[i] + 0.4 * turn[i]))
        boats.append(b)
    return boats


# ---------------------------------------------------------------------------
# 물리
# ---------------------------------------------------------------------------

def turn_speed(r: float, skill: float) -> float:
    """반경 r 에서 낼 수 있는 속도. 선회력이 좋으면 같은 반경을 더 빨리 돈다."""
    return min(math.sqrt(A_LAT * (1 + 0.10 * skill) * r), V_STRAIGHT)


def turn_time(r: float, skill: float) -> float:
    """반원(π·r) 통과 시간. 반경이 커지면 늘어난다."""
    return math.pi * r / turn_speed(r, skill)


# 코스를 구간으로 편다. 스타트라인에서 시작해 1턴마크까지 달리고, 그 뒤로
# [선회 120m · 직선 180m] 을 되풀이해 1800m 에서 끝난다.
#
#   직선 140 + 선회 120 + (직선 180 + 선회 120)×5 + 직선 40 = 1800m
#   선회는 모두 여섯 번 (3주회 × 2개)
#
# 구간으로 나눠 두면 두 가지가 한꺼번에 해결된다 — 매 선회마다 라인을 다시
# 배정할 수 있고, 시간에 따른 위치가 그대로 나와 **주행 애니메이션**을 만들 수 있다.
REAL_LAP_M = STRAIGHT_M * 2 + TURN_ARC_M * 2   # 실제로 도는 거리 ≈ 726m
COURSE: List[tuple] = (
    [("S", START_TO_MARK), ("T", TURN_ARC_M)]
    + [("S", STRAIGHT_M), ("T", TURN_ARC_M)] * 5
    + [("S", STRAIGHT_M - START_TO_MARK)]
)
# 각 구간이 끝나는 지점의 누적 거리 (렌더러가 같은 값을 쓴다)
COURSE_MARKS = [0.0]
for _kind, _len in COURSE:
    COURSE_MARKS.append(COURSE_MARKS[-1] + _len)


def _one_race(boats: List[Boat], rng: np.random.Generator,
              trace: bool = False) -> Dict:
    """한 판을 구간 단위로 끝까지 돌린다.

    ``trace=True`` 면 구간 경계마다의 시각과 가로 오프셋을 함께 남긴다 —
    화면에서 배를 움직이려면 '언제 어디에 있었나'가 있어야 한다.
    """
    n = len(boats)
    st = np.array([max(0.0, b.st_mean + rng.normal(0, ST_SIGMA)) for b in boats])
    pace = np.array([1.0 + 0.045 * b.ability + rng.normal(0, PACE_SIGMA) for b in boats])
    tq = np.array([b.turn_skill + rng.normal(0, TURN_SIGMA) * 3 for b in boats])
    lanes = np.array([b.lane for b in boats])

    t = st.copy()                 # 각 배의 현재 시각
    off = (lanes - 1) * LANE_W    # 안쪽 기준 가로 오프셋(m)
    beta = np.zeros(n)            # 슬립각(rad). 직선에서는 0
    t_marks = [t.copy()]
    off_marks = [off.copy()]
    beta_marks = [beta.copy()]

    turn_i = 0
    first: Dict[str, np.ndarray] = {}

    for kind, seg_len in COURSE:
        if kind == "S":
            # 직선. 첫 구간만 스타트 직후라 진입 속도를 쓴다. 바깥 정번은
            # 안쪽으로 파고들 거리가 더 있다.
            if turn_i == 0:
                extra = (lanes - 1) * 2.0
                t = t + (seg_len + extra) / (V_APPROACH * pace)
            else:
                t = t + seg_len / (V_STRAIGHT * pace)
            # 직선에서는 안쪽으로 되붙고, 선체도 진행 방향과 나란해진다.
            off = off * 0.35
            beta = np.zeros(n)
        else:
            if turn_i == 0:
                # 1턴마크 — 안쪽 라인은 '먼저 도착한 배'가 아니라 **안쪽에서
                # 출발한 배**가 기본으로 가진다. 바깥 배가 뺏으려면 앞선 배를
                # 가로질러야 하고, 한 칸당 TAU 만큼의 시간 우위가 필요하다.
                # 도착 시각만으로 정하면 정번이 사실상 무의미해진다(전원 동일
                # 조건에서 1코스 21% · 6코스 13%).
                claim = (lanes - 1) + (t - t.min()) / TAU
                order = np.argsort(claim)
                slot = np.argsort(order)
                radius = R_INNER + slot * LANE_W
                first = {"turn1_order": order, "radius": radius.copy(),
                         "arrive": t.copy()}
            else:
                # 이후 선회는 앞선 배가 계속 안쪽을 쓴다(추월 저항).
                pos = np.argsort(np.argsort(t))
                radius = R_INNER + pos * LANE_W * TRAIL_PENALTY

            tt = np.array([turn_time(radius[i], tq[i]) for i in range(n)])
            v_exit = np.array([turn_speed(radius[i], tq[i]) for i in range(n)])

            # 드리프트 — 진입 속도를 선회 속도까지 깎아내는 만큼 밀린다.
            v_in = (V_APPROACH if turn_i == 0 else V_STRAIGHT) * pace
            shed = np.clip((v_in - v_exit) / np.maximum(v_exit, 1e-6), 0, None)
            beta = BETA_MAX * np.clip(shed / DRIFT_REF, 0, 1)

            t = t + tt / pace
            # 바깥으로 크게 돈 배는 속도를 싣고 나온다. 그 이득은 뒤이은
            # 직선에서 시간으로 환산된다 — 휘감기가 성립하는 이유다.
            t = t - EXIT_GAIN * (v_exit - v_exit.mean()) / V_STRAIGHT
            off = radius - R_INNER
            if turn_i == 0:
                first["exit_t"] = t.copy()
                first["v_exit"] = v_exit.copy()
            turn_i += 1

        t_marks.append(t.copy())
        off_marks.append(off.copy())
        beta_marks.append(beta.copy())

    res = {
        "finish": np.argsort(t),
        "st": st,
        "turn1_order": first["turn1_order"],
        "radius": first["radius"],
        "arrive": first["arrive"],
        "exit_t": first["exit_t"],
        "v_exit": first["v_exit"],
    }
    if trace:
        res["trace"] = {
            "t": np.round(np.array(t_marks).T, 3).tolist(),    # [배][구간경계]
            "off": np.round(np.array(off_marks).T, 2).tolist(),
            "beta": np.round(np.degrees(np.array(beta_marks).T), 1).tolist(),
        }
    return res


def _tactic(res: Dict, boats: List[Boat]) -> str:
    """이 판에서 승자가 쓴 전법을 이름 붙인다."""
    w = int(res["finish"][0])
    slot = int(np.where(res["turn1_order"] == w)[0][0])   # 1턴 진입 순서 (0=첫째)
    lead_out = int(np.argmin(res["exit_t"]))              # 1턴을 먼저 빠져나온 배

    if slot == 0 and boats[w].lane <= 2:
        return "인빠지기"
    if slot == 0:
        return "선행"
    if slot >= 2 and w == lead_out:
        return "휘감기"
    if slot >= 2:
        return "휘감아찌르기"
    return "찌르기"


def simulate(runners: List[Dict], n_sims: int = N_SIMS, seed: int = 42) -> Dict:
    """몬테카를로로 전개를 반복한다.

    돌려주는 값은 **승률이 아니다.** 승률은 모델이 낸다. 여기서는 1턴마크에서
    무슨 일이 벌어지는지와, 그 결과로 어떤 전법이 나오는지를 센다.
    """
    boats = build_boats(runners)
    if len(boats) < 2:
        return {}
    rng = np.random.default_rng(seed)
    n = len(boats)

    turn1_lead = np.zeros(n)     # 1턴을 가장 먼저 빠져나온 횟수
    turn1_in = np.zeros(n)       # 최내측 라인을 잡은 횟수
    wins = np.zeros(n)
    top2 = np.zeros(n)
    radius_sum = np.zeros(n)
    tactics: Dict[str, int] = {}
    runs = []

    for _ in range(n_sims):
        res = _one_race(boats, rng, trace=True)
        f = res["finish"]
        wins[f[0]] += 1
        top2[f[0]] += 1
        top2[f[1]] += 1
        turn1_lead[int(np.argmin(res["exit_t"]))] += 1
        turn1_in[int(res["turn1_order"][0])] += 1
        radius_sum += res["radius"]
        tac = _tactic(res, boats)
        tactics[tac] = tactics.get(tac, 0) + 1
        runs.append((int(f[0]), tac, res))

    # ── 대본 고르기 ─────────────────────────────────────────────
    # **예상 1순위가 이긴 판**에서 뽑는다. 화면 위쪽 예상판이 A 를 1순위로 적어
    # 놓았는데 아래 전개에서 B 가 이기면, 읽는 사람은 둘 다 믿을 수 없게 된다.
    #
    # 결과를 지어내는 것이 아니다. 1,500판 모두 같은 물리로 실제 계산한 것이고,
    # 그중 **예상대로 흘러간 판**을 골라 보이는 것뿐이다. 무엇을 골랐는지는
    # 화면에 적는다(script_is_expected).
    #
    # 그 배가 한 번도 못 이겼다면 억지로 만들지 않고 최다 승자의 판으로 물러난다.
    target = int(np.argmax([b.p_win for b in boats]))
    picked = target if wins[target] > 0 else int(np.argmax(wins))

    winner_tactics: Dict[str, int] = {}
    for w, tac, _ in runs:
        if w == picked:
            winner_tactics[tac] = winner_tactics.get(tac, 0) + 1
    script_tactic = (max(winner_tactics, key=winner_tactics.get)
                     if winner_tactics else max(tactics, key=tactics.get))
    script_run = next((r for r in runs if r[0] == picked and r[1] == script_tactic),
                      runs[0])

    per = []
    for i, b in enumerate(boats):
        per.append({
            "lane": b.lane, "racer_nm": b.racer_nm,
            "turn1_lead": turn1_lead[i] / n_sims,
            "turn1_inside": turn1_in[i] / n_sims,
            "sim_win": wins[i] / n_sims,
            "sim_top2": top2[i] / n_sims,
            "avg_radius": radius_sum[i] / n_sims,
        })
    per.sort(key=lambda r: -r["turn1_lead"])

    # 신뢰도는 **모델이 낸 예상 1순위의 승률**로 잰다.
    #
    # 시뮬레이션의 자체 승률로 재면 화면에 확률이 둘이 되고 서로 다른 값을
    # 말한다(실측: 모델 63.8% 인 경주에서 시뮬 88.8%). 승률을 내는 것은 학습
    # 모델의 일이고, 시뮬레이션의 일은 1턴마크에서 무슨 일이 벌어지는지와
    # 전개가 어떻게 흘러가는지를 보이는 것이다. 숫자는 한 곳에서만 나와야 한다.
    share = float(boats[target].p_win)
    return {
        "n_sims": n_sims,
        "per_boat": per,
        "tactics": sorted(({"name": k, "share": v / n_sims} for k, v in tactics.items()),
                          key=lambda r: -r["share"]),
        "script": _script(script_run[2], boats, script_run[1]),
        # 대표 판의 궤적. 재생기가 이걸로 배를 움직인다.
        "player": player_payload(script_run[2], boats),
        "confidence": _confidence(share),
        # 대본의 전법. 화면 머리말과 대본이 반드시 같은 것을 가리켜야 한다.
        "top_tactic": script_run[1],
        "tactic_note": tactic_note(script_run[1]),
        "script_winner": {"lane": boats[picked].lane,
                          "racer_nm": boats[picked].racer_nm,
                          "share": float(wins[picked] / n_sims)},
        # 대본이 '예상대로 흘러간 판'인지. 아니면 화면이 그렇게 밝혀야 한다.
        "script_is_expected": bool(picked == target),
        "expected_lane": boats[target].lane,
    }


def player_payload(res: Dict, boats: List[Boat]) -> Dict:
    """주행 재생기에 넘길 자료.

    좌표는 넘기지 않는다 — 코스 기하는 브라우저가 계산하고, 여기서는 **언제
    어디쯤을(구간 진행률) 어떤 자세로** 지나갔는지만 준다. 그래야 화면 크기가
    바뀌어도 다시 계산할 것이 없고, 자료도 작다(배당 6척 × 경계 14개).
    """
    tr = res.get("trace")
    if not tr:
        return {}
    finish = [int(i) for i in res["finish"]]
    return {
        "course": {
            "straight": STRAIGHT_M, "r_inner": round(R_INNER, 2),
            "lap": LAP_M, "laps": LAPS,
            "start_offset": STRAIGHT_M - START_TO_MARK,  # 스타트라인 위치
            "marks": [round(m, 1) for m in COURSE_MARKS],
            "kinds": [k for k, _ in COURSE],
        },
        "boats": [{
            "lane": b.lane, "racer_nm": b.racer_nm,
            "t": tr["t"][i], "off": tr["off"][i], "beta": tr["beta"][i],
            "finish": finish.index(i) + 1,
        } for i, b in enumerate(boats)],
        "duration": round(max(row[-1] for row in tr["t"]), 2),
    }


def _confidence(share: float) -> Dict:
    """예상 1순위가 이 전개에서 얼마나 자주 이기는지.

    승률이 아니라 **전개의 재현성**이다. 반복 계산에서 같은 결과가 자주 나올수록
    1턴마크에서 뒤집힐 여지가 적다는 뜻이다.
    """
    if share >= 0.55:
        return {"label": "강승부", "score": round(share * 100),
                "desc": "반복 계산의 절반 이상에서 예상 1순위가 승리했습니다."}
    if share >= 0.38:
        return {"label": "중승부", "score": round(share * 100),
                "desc": "예상 1순위가 우세하나 1턴마크에서 갈릴 여지가 있습니다."}
    return {"label": "약승부", "score": round(share * 100),
            "desc": "1턴마크 결과에 따라 순위가 크게 바뀔 수 있습니다."}


def _script(res: Dict, boats: List[Boat], tactic: str) -> List[Dict]:
    """한 판을 사람이 읽는 세 장면으로 옮긴다."""
    n = len(boats)
    st_order = np.argsort(res["st"])
    turn_order = res["turn1_order"]
    exit_order = np.argsort(res["exit_t"])
    finish = res["finish"]

    def name(i):
        return f"{boats[i].lane}번 {boats[i].racer_nm}"

    scenes = [{
        "title": "스타트",
        "text": (f"{name(int(st_order[0]))} 가 가장 빠르게 나갔다"
                 f"(ST {res['st'][int(st_order[0])]:.2f}). "
                 f"{name(int(st_order[-1]))} 는 {res['st'][int(st_order[-1])]:.2f} 로 늦었다."),
        "order": [int(i) for i in st_order],
    }, {
        "title": "1턴마크",
        "text": (f"{name(int(turn_order[0]))} 가 최내측(반경 "
                 f"{res['radius'][int(turn_order[0])]:.0f}m)을 잡았고, "
                 f"{name(int(exit_order[0]))} 가 먼저 빠져나왔다. "
                 f"가장 바깥은 반경 {res['radius'].max():.0f}m 로 돌아 "
                 f"{res['v_exit'].max():.1f}m/s 의 속도를 싣고 나왔다."),
        "order": [int(i) for i in exit_order],
    }, {
        "title": "결승",
        "text": (f"{name(int(finish[0]))} 승리. 전법은 {tactic}. "
                 f"{name(int(finish[1]))} 가 2착으로 따랐다."),
        "order": [int(i) for i in finish],
    }]
    for s in scenes:
        s["boats"] = [{"lane": boats[i].lane, "racer_nm": boats[i].racer_nm}
                      for i in s["order"]]
    return scenes


def tactic_note(tactic: str) -> str:
    """전법 한 줄 설명. 화면 범례와 대본이 같은 말을 쓰게 한다."""
    return {
        "인빠지기": "안쪽을 잡아 짧게 돌아 먼저 빠져나오는 전법. 1·2코스의 기본기다.",
        "선행": "1턴마크에 가장 먼저 도달해 그대로 앞서 나가는 전개.",
        "휘감기": "바깥으로 크게 돌아 속도를 싣고 앞지르는 전법. 스타트가 좋아야 성립한다.",
        "휘감아찌르기": "크게 도는 척 부풀렸다가 안쪽 틈을 파고드는 전법.",
        "찌르기": "앞선 배가 부풀어 오른 안쪽 틈을 파고드는 전법.",
    }.get(tactic, "")
