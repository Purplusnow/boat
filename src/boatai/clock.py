"""도메인 시각은 항상 한국 시간으로 읽는다.

경정 경주일과 발주 시각은 KST 로 발표된다. ``date.today()`` 를 그대로 쓰면
UTC 환경에서만 날짜가 하루 어긋나고, 그 어긋남은 조용하다 — 예상이 하루 늦게
만들어지거나, 이미 끝난 경주가 '아직 발주 전'으로 취급돼 예측이 다시 쓰인다.

한국은 서머타임이 없으므로 고정 오프셋으로 충분하고, tzdata 설치 여부에
의존하지 않는다.
"""

from __future__ import annotations

import datetime as dt

KST = dt.timezone(dt.timedelta(hours=9), "KST")

# 경정은 미사리 한 곳에서만 시행하고, 개최 요일은 수·목이 기본이다.
# (경륜이 금·토·일을 쓰는 것과 갈린다.) 특별 편성으로 화요일이 붙는 회차가
# 있으므로 '후보'로만 쓰고, 실제 개최 여부는 항상 API 응답으로 확인한다.
RACE_WEEKDAYS = (1, 2, 3)  # 화, 수, 목


def now_kst() -> dt.datetime:
    """현재 한국 시각 (naive — DB·API 문자열과 그대로 비교하기 위함)."""
    return dt.datetime.now(dt.timezone.utc).astimezone(KST).replace(tzinfo=None)


def today_kst() -> dt.date:
    """오늘 (한국 기준 경주일)."""
    return now_kst().date()


def recent_race_dates(n: int = 8, today: dt.date | None = None) -> list[str]:
    """최근 개최 후보일(화·수·목)을 최신순 ``YYYYMMDD`` 로."""
    today = today or today_kst()
    out: list[str] = []
    d = today
    while len(out) < n:
        if d.weekday() in RACE_WEEKDAYS:
            out.append(d.strftime("%Y%m%d"))
        d -= dt.timedelta(days=1)
    return out
