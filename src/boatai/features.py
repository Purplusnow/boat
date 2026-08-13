"""피처 생성.

경정의 특수성이 피처 설계를 크게 좌우한다.

* **출주 정수가 항상 6이고 코스가 곧 정번이다.** 경마처럼 두수가 8~16으로
  흔들리지 않으므로 경주 간 비교가 쉽고, 대신 코스(1~6)가 압도적인 단일
  피처다 — 1코스는 인쪽에서 최단거리로 선회한다.
* **모터가 회차마다 추첨으로 배정된다.** 선수 기량과 별개인 장비 운이 결과를
  크게 가르므로, 모터 성적은 선수 성적만큼 중요한 축이다.
* **출주표가 이미 잘 만들어진 롤링 지표를 준다.** 최근 6회차 성적, 코스별
  6개월 연대율, 모터·보트 연대율이 전부 경주 전 공개값이다. 우리가 과거
  결과에서 다시 계산할 필요가 없고, 계산하지 않으므로 **누수 위험도 없다**.

한 경주에서 이기는 배는 정확히 하나다. 따라서 절대값만큼이나 **같은 경주 안의
상대 위치**가 중요하다. 모터2연대율 35%는 다른 다섯이 20%대면 강점이지만
40%대면 약점이다. 그래서 주요 지표마다 경주 내 편차와 순위를 함께 만든다.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# 경주 내 상대 위치를 함께 만들 지표들. 절대값만으로는 '이 경주에서 셋 중
# 누가 나은가'를 모델이 배우기 어렵다.
RELATIVE = [
    "avg_rank", "high_rate", "tms6_avg_rank_scr", "tms6_avg_scr",
    "tms6_win_ratio", "tms6_high_rate", "tms6_high3_rate", "tms6_avg_st",
    "mot_avg_rank_scr", "mot_high_rate", "mot_high3_rate",
    "boat_avg_rank_scr", "boat_high_rate",
    "own_course_high_rate", "recent_avg", "avg_acdnt_scr",
]

# st_method(경주구분)를 반드시 범주형으로 넣는다. 단순한 분류 축이 아니라
# **측정 체계 자체가 갈리는 축**이기 때문이다. 실측:
#
#   002  180,600행  평균ST 0.00~1.33 (평균 0.27)   ← 대부분
#   001   11,736행  평균ST 17.0~20.1 (평균 18.04)
#
# 같은 tms6_avg_st 컬럼에 두 자가 섞여 있어, 절대값만 보면 001 경주의 선수는
# 전부 '스타트가 70배 나쁜' 것으로 읽힌다. 다행히 한 경주는 항상 한 방식이므로
# 경주 내 상대 피처(rel_/z_/rank_)는 그대로 유효하고, 트리 모델은 이 범주로
# 먼저 갈라 절대값을 따로 배울 수 있다. 이 컬럼을 빼면 그 보정 수단이 사라진다.
CATEGORICAL = ["lane", "racer_grd", "race_class", "st_method", "sex"]

BASE_NUMERIC = [
    "age", "weight", "avg_rank", "high_rate", "avg_acdnt_scr", "f_cnt", "l_cnt",
    "tms6_avg_rank_scr", "tms6_avg_scr", "tms6_win_ratio", "tms6_high_rate",
    "tms6_high3_rate", "tms6_avg_st", "bf_dd_recd_scr", "mm6_race_cnt",
    "thdd_race_no",
    "mot_avg_rank_scr", "mot_high_rate", "mot_high3_rate", "mot_prev_avg",
    "boat_avg_rank_scr", "boat_high_rate",
    "own_course_high_rate", "own_course_cnt",
    "recent1", "recent2", "recent3", "recent4",
    "recent5", "recent6", "recent7", "recent8", "recent_avg", "recent_cnt",
    "field_size",
]

TRAIN_SQL = """
SELECT
    e.*,
    r.race_ymd, r.stnd_yr, r.week_tcnt, r.day_tcnt, r.race_no,
    r.field_size, r.has_result,
    res.ord,
    -- 단승 배당(적중 조합의 배당)은 경주 단위 값이다. 회수율 계산에만 쓰고
    -- 피처로는 절대 넣지 않는다 — 경주 후에야 정해지는 값이다.
    win.payout AS win_payout
FROM entries e
JOIN races r      ON r.race_key = e.race_key
LEFT JOIN results res ON res.race_key = e.race_key AND res.lane = e.lane
LEFT JOIN payoffs win ON win.race_key = e.race_key AND win.pool = '단승'
"""


def _own_course(df: pd.DataFrame) -> pd.Series:
    """자기 정번에 해당하는 코스별 6개월 연대율만 뽑는다.

    출주표는 여섯 코스 값을 모두 주지만, 오늘 이 선수가 타는 자리는 하나다.
    나머지 다섯은 '이 선수가 다른 자리에서 어떤가'라는 다른 정보라 따로 둔다.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for lane in range(1, 7):
        col = f"course{lane}_high_rate"
        if col in df:
            m = df["lane"] == lane
            out[m] = pd.to_numeric(df.loc[m, col], errors="coerce")
    return out


def _own_course_cnt(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for lane in range(1, 7):
        col = f"course{lane}_cnt"
        if col in df:
            m = df["lane"] == lane
            out[m] = pd.to_numeric(df.loc[m, col], errors="coerce")
    return out


def add_relative(df: pd.DataFrame) -> pd.DataFrame:
    """경주 내 편차와 순위를 붙인다.

    편차(값 - 경주평균)와 순위를 둘 다 만드는 이유는 서로 다른 것을 말하기
    때문이다. 편차는 '얼마나 앞서는가', 순위는 '몇 번째인가'다. 여섯 명 중
    모터가 압도적인 한 명이 있는 경주와 고만고만한 경주를 순위만으로는
    구분할 수 없다.
    """
    g = df.groupby("race_key")
    for col in RELATIVE:
        if col not in df:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        mean = v.groupby(df["race_key"]).transform("mean")
        std = v.groupby(df["race_key"]).transform("std")
        df[f"rel_{col}"] = v - mean
        # 표준편차가 0(전원 동일)이면 편차도 0이다. 나눗셈으로 inf 를 만들지 않는다.
        df[f"z_{col}"] = (v - mean) / std.replace(0, np.nan)
        df[f"rank_{col}"] = v.groupby(df["race_key"]).rank(ascending=False, method="average")
    return df


def build_frame(conn: sqlite3.Connection, *, with_labels: bool = True) -> pd.DataFrame:
    """학습·추론 공통 프레임."""
    df = pd.read_sql_query(TRAIN_SQL, conn)
    if df.empty:
        return df

    for c in BASE_NUMERIC + ["ord", "lane"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["own_course_high_rate"] = _own_course(df)
    df["own_course_cnt"] = _own_course_cnt(df)
    df = add_relative(df)

    # 시간 정렬 키. 경주일자가 없는 행(착순 미수집)은 좌표로 대신한다 —
    # 워크포워드 검증에서 순서가 틀리면 미래로 과거를 맞히게 된다.
    ymd = pd.to_datetime(df["race_ymd"], format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(
        df["stnd_yr"].astype("Int64").astype(str) + "0101", format="%Y%m%d", errors="coerce")
    df["race_date"] = ymd.fillna(fallback)
    df["order_key"] = (df["stnd_yr"].astype("Int64").astype(str).str.zfill(4) +
                       df["week_tcnt"].astype("Int64").astype(str).str.zfill(2) +
                       df["day_tcnt"].astype("Int64").astype(str).str.zfill(1) +
                       df["race_no"].astype("Int64").astype(str).str.zfill(2))

    if with_labels:
        o = df["ord"]
        df["y_win"] = (o == 1).astype(float).where(o.notna())
        df["y_top2"] = (o <= 2).astype(float).where(o.notna())
        df["y_top3"] = (o <= 3).astype(float).where(o.notna())
    return df


def build_training_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """레이블이 온전한 경주만 남긴 학습 프레임.

    1착이 정확히 한 척으로 확인되는 경주만 쓴다. 실격·재경주로 착순이 깨진
    경주를 그대로 넣으면 '이긴 배가 없는 경주'가 음성 표본만 잔뜩 만든다.
    """
    df = build_frame(conn, with_labels=True)
    if df.empty:
        return df
    ok = df.groupby("race_key")["y_win"].transform(lambda s: (s == 1).sum() == 1)
    df = df[ok & df["ord"].notna()].copy()
    log.info("학습 프레임 %d행 / %d경주 (%s ~ %s)", len(df), df["race_key"].nunique(),
             str(df["race_date"].min())[:10], str(df["race_date"].max())[:10])
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    """실제로 존재하는 피처 컬럼만 고정된 순서로 돌려준다.

    순서를 고정하는 이유는 모델을 피클로 저장했다가 다시 쓰기 때문이다. 학습과
    추론의 컬럼 순서가 어긋나면 오류 없이 **틀린 예측**이 나온다.
    """
    cols = [c for c in BASE_NUMERIC if c in df]
    cols += [f"{p}{c}" for c in RELATIVE for p in ("rel_", "z_", "rank_")
             if f"{p}{c}" in df]
    cols += [c for c in CATEGORICAL if c in df]
    return cols
