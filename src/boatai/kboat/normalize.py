"""API 응답을 DB 행으로 정규화한다.

경정 API 는 사람이 읽을 화면을 그대로 내보낸 필드가 많다. 숫자가 문자열에
붙어 있고("①-②8.3"), 비율과 모수가 한 칸에 들어 있고("60.0/5"), 카운터가
코드로 뭉쳐 있다("F0L0"). 이런 값을 파서 없이 그대로 피처에 넣으면 조용히
문자열 범주가 되어, 모델은 "60.0/5" 와 "60.0/6" 을 완전히 다른 값으로 배운다.

파싱 규칙을 여기 한 곳에 모으는 이유는 같은 값을 두 군데서 다르게 읽는 사고를
막기 위해서다. 원본은 ``raw_json`` 에 그대로 남기므로, 이 규칙이 틀렸다는 게
나중에 드러나도 API 를 다시 때리지 않고 로컬에서 다시 만들 수 있다.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# 경정 성적표는 정번을 원 문자로 쓴다. ①고정환 / ③-⑥8.3 처럼 배당·착순 어디에나
# 섞여 나오므로 한 곳에서 숫자로 되돌린다.
CIRCLED = {c: i for i, c in enumerate("①②③④⑤⑥⑦⑧⑨", start=1)}


def _s(v: Any) -> str:
    return "" if v is None else str(v).strip()


def to_int(v: Any) -> Optional[int]:
    t = _s(v)
    if not t:
        return None
    m = re.search(r"-?\d+", t)
    return int(m.group()) if m else None


def to_float(v: Any) -> Optional[float]:
    t = _s(v)
    if not t:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    return float(m.group()) if m else None


def race_key(stnd_yr: Any, week: Any, day: Any, race_no: Any) -> str:
    """경주 식별자. API 좌표(연도·회차·일차·경주번호) 그대로가 가장 안전하다.

    날짜를 키로 쓰지 않는 이유는 착순 API 에만 날짜가 있고 나머지 셋은 좌표만
    주기 때문이다. 좌표를 키로 두면 어느 API 로 들어온 조각이든 같은 행에 붙는다.
    """
    return f"{to_int(stnd_yr)}-{to_int(week):02d}-{to_int(day)}-{to_int(race_no):02d}"


def coords_from_key(key: str) -> Dict[str, Any]:
    """``"2026-05-1-01"`` → 좌표 컬럼들.

    착순 API 로만 알게 되는 경주(출주표가 없는 경주)도 races 에 넣어야 하는데,
    좌표 컬럼은 NOT NULL 이다. 키에 이미 좌표가 들어 있으므로 되짚어 채운다 —
    출주표가 없다고 그 경주를 통째로 버리면 착순도 함께 사라진다.
    """
    parts = key.split("-")
    if len(parts) != 4:
        return {}
    return {"race_key": key, "stnd_yr": int(parts[0]), "week_tcnt": int(parts[1]),
            "day_tcnt": int(parts[2]), "race_no": int(parts[3])}


# ---------------------------------------------------------------------------
# 출주표 (경주 전 확정 정보)
# ---------------------------------------------------------------------------

def parse_ratio_count(v: Any) -> tuple:
    """``"60.0/5"`` → (60.0, 5). 코스별 연대율이 이 형태로 온다.

    비율만 쓰고 모수를 버리면 3회 중 3회 연대한 100% 와 30회 중 24회의 80% 가
    같은 무게로 들어간다. 경정은 6개월 코스별 출주가 한 자릿수인 경우가 흔해
    이 차이가 크다 — 모수를 함께 남겨 모델이 스스로 신뢰도를 배우게 한다.
    """
    t = _s(v)
    if not t:
        return None, None
    if "/" in t:
        a, _, b = t.partition("/")
        return to_float(a), to_int(b)
    return to_float(t), None


def parse_fl(v: Any) -> tuple:
    """``"F0L0"`` → (0, 0). F=플라잉(사전출발), L=출발지연.

    경정에서 F 는 실격이고 제재도 따른다. 위반 이력이 있는 선수는 스타트를
    보수적으로 잡는 경향이 있어, 사고 그 자체보다 **출발 성향의 대리 지표**로서
    값이 있다.
    """
    t = _s(v).upper()
    if not t:
        return None, None
    f = re.search(r"F(\d+)", t)
    l = re.search(r"L(\d+)", t)
    return (int(f.group(1)) if f else None, int(l.group(1)) if l else None)


def parse_recent_ranks(v: Any) -> List[int]:
    """``"4346 3222"`` → [4,3,4,6,3,2,2,2]. 최근 8경주 착순.

    공백은 회차 구분이다. 착순만 뽑아 순서대로 돌려준다 (앞이 오래된 쪽).
    """
    return [int(c) for c in re.sub(r"\D", "", _s(v))]


def parse_prev_motor_rider(v: Any) -> tuple:
    """``"김은지/444"`` → ("김은지", [4,4,4]).

    이 모터를 직전 회차에 탄 선수와 그 성적이다. 모터는 회차마다 추첨으로
    바뀌므로, 같은 모터를 방금 탄 사람의 결과는 **모터 컨디션의 최신 관측**이다.
    누적 2연대율보다 신선한 신호가 될 수 있다.
    """
    t = _s(v)
    if not t or "/" not in t:
        return (t or None), []
    name, _, ranks = t.partition("/")
    return name.strip() or None, [int(c) for c in re.sub(r"\D", "", ranks)]


def entry_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """출주표 한 줄 → entries 행."""
    key = race_key(rec.get("stnd_yr"), rec.get("week_tcnt"),
                   rec.get("day_tcnt"), rec.get("race_no"))
    f_cnt, l_cnt = parse_fl(rec.get("fl_tcnt"))
    prev_name, prev_ranks = parse_prev_motor_rider(rec.get("mot_bf_racer_1_no"))
    recent = parse_recent_ranks(rec.get("tms_8_rank_ord_no"))

    row: Dict[str, Any] = {
        "race_key": key,
        # race_reg_no 는 명세에 '선수번호'로 적혀 있지만 실제 값은 1~6 이고
        # 경주마다 한 번씩 나온다 — **정번(=코스)** 이다. 경정에서 코스는
        # 단일 최강 피처이므로 이름에 속아 버리면 안 된다.
        "lane": to_int(rec.get("race_reg_no")),
        "racer_nm": _s(rec.get("racer_nm")) or None,
        "racer_grd": _s(rec.get("racer_grd_cd")) or None,
        "sex": _s(rec.get("sex_cd")) or None,
        "age": to_int(rec.get("racer_age")),
        "weight": to_float(rec.get("wght")),
        "post_time": _s(rec.get("dptre_tm")) or None,
        "race_class": _s(rec.get("rmk2_rank")) or None,
        "st_method": _s(rec.get("st_mthd_cd")) or None,
        "color_nm": _s(rec.get("color_nm")) or None,

        # 선수 폼
        "avg_rank": to_float(rec.get("avg_rank")),
        "high_rate": to_float(rec.get("high_rate")),
        "avg_acdnt_scr": to_float(rec.get("avg_acdnt_scr")),
        "f_cnt": f_cnt,
        "l_cnt": l_cnt,
        "tms6_avg_rank_scr": to_float(rec.get("tms_6_avg_rank_scr")),
        "tms6_avg_scr": to_float(rec.get("tms_6_avg_scr")),
        "tms6_win_ratio": to_float(rec.get("tms_6_win_ratio")),
        "tms6_high_rate": to_float(rec.get("tms_6_high_rank_ratio")),
        "tms6_high3_rate": to_float(rec.get("tms_6_high_3_rank_ratio")),
        "tms6_avg_st": to_float(rec.get("tms_6_avg_strt_tm")),
        "bf_dd_recd_scr": to_float(rec.get("bf_dd_recd_scr")),
        "mm6_race_cnt": to_int(rec.get("mm_6_race_tcnt")),
        "thdd_race_no": to_int(rec.get("thdd_race_no")),

        # 장비
        "motor_no": to_int(rec.get("motor_no")),
        "mot_avg_rank_scr": to_float(rec.get("mot_avg_rank_scr")),
        "mot_high_rate": to_float(rec.get("mot_high_rank_ratio")),
        "mot_high3_rate": to_float(rec.get("mot_high_3_rank_ratio")),
        "mot_prev_racer": prev_name,
        "boat_no": to_int(rec.get("boat_no")),
        "boat_avg_rank_scr": to_float(rec.get("boat_avg_rank_scr")),
        "boat_high_rate": to_float(rec.get("boat_high_rank_ratio")),
    }

    # 코스별 6개월 연대율 — 비율과 모수를 따로 남긴다.
    for lane in range(1, 7):
        ratio, cnt = parse_ratio_count(rec.get(f"mm_6_{lane}_race_high_rank_ratio"))
        row[f"course{lane}_high_rate"] = ratio
        row[f"course{lane}_cnt"] = cnt

    # 최근 8경주 착순. 개별 값과 요약을 함께 남긴다 — 요약만 두면 '최근 3경주'
    # 같은 다른 창을 나중에 못 만든다.
    for i, r in enumerate(recent[:8], start=1):
        row[f"recent{i}"] = r
    if recent:
        row["recent_avg"] = sum(recent) / len(recent)
        row["recent_cnt"] = len(recent)
    if prev_ranks:
        row["mot_prev_avg"] = sum(prev_ranks) / len(prev_ranks)
    return row


def race_row_from_entry(rec: Dict[str, Any]) -> Dict[str, Any]:
    """출주표 한 줄에서 경주 단위 메타를 뽑는다."""
    return {
        "race_key": race_key(rec.get("stnd_yr"), rec.get("week_tcnt"),
                             rec.get("day_tcnt"), rec.get("race_no")),
        "stnd_yr": to_int(rec.get("stnd_yr")),
        "week_tcnt": to_int(rec.get("week_tcnt")),
        "day_tcnt": to_int(rec.get("day_tcnt")),
        "race_no": to_int(rec.get("race_no")),
        "post_time": _s(rec.get("dptre_tm")) or None,
        "race_class": _s(rec.get("rmk2_rank")) or None,
        "st_method": _s(rec.get("st_mthd_cd")) or None,
    }


# ---------------------------------------------------------------------------
# 착순 (경주결과순위)
# ---------------------------------------------------------------------------

def result_rows(records: List[Dict[str, Any]],
                lane_by_name: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    """착순 응답 → results 행.

    착순 API 는 정번을 주지 않는다(선수번호·선수명·착순뿐). 정번은 출주표에만
    있으므로 **경주 안에서 선수명으로 이어 붙인다**. 한 경주에 여섯 명뿐이라
    동명이인 충돌은 사실상 없고, 이어붙이지 못한 행은 정번을 비워 둔 채 남긴다 —
    조용히 버리면 착순이 통째로 사라진 것을 아무도 모른다.
    """
    out = []
    for rec in records:
        key = race_key(rec.get("stnd_year") or rec.get("stnd_yr"),
                       rec.get("tms") or rec.get("week_tcnt"),
                       rec.get("day_ord") or rec.get("day_tcnt"),
                       rec.get("race_no"))
        name = _s(rec.get("racer_nm"))
        out.append({
            "race_key": key,
            "racer_no": _s(rec.get("racer_no")) or None,
            "racer_nm": name or None,
            "lane": lane_by_name.get(key, {}).get(name),
            # 착순은 실격·사고 때 숫자가 아닐 수 있다. 숫자만 ord 로 두고
            # 원문은 note 에 남긴다 — 실격을 6착으로 바꿔 세면 안 된다.
            "ord": to_int(rec.get("race_rank")),
            "ord_note": _s(rec.get("race_rank")) or None,
            "race_ymd": _s(rec.get("race_day")) or None,
            "venue": _s(rec.get("mbr_nm")) or None,
        })
    return out


# ---------------------------------------------------------------------------
# 배당
# ---------------------------------------------------------------------------

# 승식 코드. 경정은 일곱 승식을 발매하지만, 공개 API 로 배당을 받을 수 있는
# 것은 여섯이다 — **삼쌍승(순서까지 맞히는 3연단)은 어느 API 에도 없다.**
# 없는 것을 '불발'로 세지 않도록 판정에서 아예 제외한다.
POOL_FIELDS = {
    "pool1_val": "단승",
    "pool2_1_val": "연승1",
    "pool2_2_val": "연승2",
    "pool4_val": "쌍승",
    "pool5_val": "복승",
    "pool6_val": "삼복승",
}
# 경주결과 API 는 같은 배당을 다른 필드명으로 준다 (연승 둘째가 pool3_val).
RESULT_POOL_FIELDS = {
    "pool1_val": "단승",
    "pool2_val": "연승1",
    "pool3_val": "연승2",
    "pool4_val": "쌍승",
    "pool5_val": "복승",
    "pool6_val": "삼복승",
}


def parse_pool_value(v: Any) -> tuple:
    """``"③-⑥8.3"`` → ("3-6", 8.3). 숫자만 오면 조합은 None.

    경주결과 API 는 조합과 배당을 한 문자열로 붙여 주고, 배당률 API 는 배당만
    준다. 두 소스를 같은 표에 넣기 위해 여기서 형태를 맞춘다.
    """
    t = _s(v)
    if not t:
        return None, None
    lanes = [str(CIRCLED[c]) for c in t if c in CIRCLED]
    payout = to_float(re.sub(r"[①-⑨\-\s]", "", t))
    return ("-".join(lanes) if lanes else None), payout


def payoff_rows(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """배당률 응답 한 줄 → payoffs 행들 (승식별)."""
    key = race_key(rec.get("stnd_yr"), rec.get("week_tcnt"),
                   rec.get("day_tcnt"), rec.get("race_no"))
    out = []
    for field, pool in POOL_FIELDS.items():
        combo, payout = parse_pool_value(rec.get(field))
        if payout is None:
            continue
        out.append({"race_key": key, "pool": pool, "combo": combo, "payout": payout})
    return out


def result_payoff_rows(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """경주결과 응답 한 줄 → payoffs 행들. 조합(정번)까지 함께 온다."""
    key = race_key(rec.get("stnd_yr"), rec.get("week_tcnt"),
                   rec.get("day_tcnt"), rec.get("race_no"))
    out = []
    for field, pool in RESULT_POOL_FIELDS.items():
        combo, payout = parse_pool_value(rec.get(field))
        if payout is None:
            continue
        out.append({"race_key": key, "pool": pool, "combo": combo, "payout": payout})
    return out


def result_top3(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """경주결과의 ``rank1~3`` ("③고정환") → 착순 행. 정번이 함께 온다.

    착순 API 가 못 들어온 경주라도 1~3착은 이걸로 채울 수 있다. 예측 평가는
    대부분 3착 이내만 보므로 실질적인 대체재가 된다.
    """
    key = race_key(rec.get("stnd_yr"), rec.get("week_tcnt"),
                   rec.get("day_tcnt"), rec.get("race_no"))
    out = []
    for i in (1, 2, 3):
        t = _s(rec.get(f"rank{i}"))
        if not t:
            continue
        lane = next((CIRCLED[c] for c in t if c in CIRCLED), None)
        name = re.sub(r"[①-⑨]", "", t).strip()
        out.append({"race_key": key, "lane": lane, "racer_nm": name or None, "ord": i})
    return out


def lane_map(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """{race_key: {선수명: 정번}}. 착순을 정번에 잇기 위한 색인."""
    out: Dict[str, Dict[str, int]] = {}
    for e in entries:
        if e.get("racer_nm") and e.get("lane"):
            out.setdefault(e["race_key"], {})[e["racer_nm"]] = e["lane"]
    return out
