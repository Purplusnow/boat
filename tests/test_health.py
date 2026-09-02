"""자체 점검 회귀 테스트.

여기서 막으려는 것은 **조용한 실패**다. 자동 실행이 초록불로 끝났는데 그날
예상이 비어 있던 일이 실제로 있었다. 그래서 아래 네 가지 상황이 반드시
빨간불이 되는지 기계가 지키게 한다.

점검 자체가 고장 나면 아무 경보도 안 뜨는데, 그건 점검이 없는 것보다 나쁘다 —
지켜지고 있다고 믿게 되기 때문이다.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boatai import health  # noqa: E402
from boatai.kboat.store import session  # noqa: E402

NOW = dt.datetime(2026, 9, 2, 9, 0)      # 수요일 오전 9시
VERSION = "v1"


def _db(tmp_path):
    return session(tmp_path / "t.sqlite")


def _race(conn, *, key, ymd, no, post, entries=6, preds=0, ords=0):
    """경주 하나를 원하는 상태로 심는다."""
    yr, wtc, dtc, _ = key.split("-")
    conn.execute(
        "INSERT INTO races(race_key, stnd_yr, week_tcnt, day_tcnt, race_no, "
        "race_ymd, post_time, field_size) VALUES(?,?,?,?,?,?,?,?)",
        (key, int(yr), int(wtc), int(dtc), no, ymd, post, entries))
    for lane in range(1, entries + 1):
        conn.execute("INSERT INTO entries(race_key, lane, racer_nm) VALUES(?,?,?)",
                     (key, lane, f"선수{lane}"))
    for lane in range(1, preds + 1):
        conn.execute(
            "INSERT INTO predictions(race_key, lane, racer_nm, p_win, pred_rank, "
            "model_version) VALUES(?,?,?,?,?,?)",
            (key, lane, f"선수{lane}", 0.3, lane, VERSION))
    for lane in range(1, ords + 1):
        conn.execute("INSERT INTO results(race_key, lane, racer_nm, ord) VALUES(?,?,?,?)",
                     (key, lane, f"선수{lane}", lane))
    conn.commit()


def _kinds(conn):
    return {i.kind for i in health.check(conn, NOW, version=VERSION)}


def test_출주표가_있는데_예상이_없으면_경보(tmp_path):
    with _db(tmp_path) as conn:
        _race(conn, key="2026-36-1-05", ymd="20260902", no=5, post="13:12")
        assert "발주 전 예상 없음" in _kinds(conn)


def test_예상이_있으면_조용하다(tmp_path):
    with _db(tmp_path) as conn:
        _race(conn, key="2026-36-1-05", ymd="20260902", no=5, post="13:12", preds=6)
        assert _kinds(conn) == set()


def test_발주_직전인데_출주표가_없으면_경보(tmp_path):
    """수집이 막힌 상태. 러너에서 포털 연결이 끊겼을 때 실제로 이렇게 된다."""
    with _db(tmp_path) as conn:
        _race(conn, key="2026-36-1-01", ymd="20260902", no=1, post="11:40", entries=0)
        assert "출주표 미수집" in _kinds(conn)


def test_발주까지_여유가_있으면_출주표가_없어도_조용하다(tmp_path):
    """출주표는 원래 늦게 올라온다. 앞날 것까지 경보로 세면 매주 거짓 경보가 뜬다."""
    with _db(tmp_path) as conn:
        _race(conn, key="2026-36-2-01", ymd="20260903", no=1, post="11:40", entries=0)
        assert _kinds(conn) == set()


def test_예상_없이_발주가_지나면_경보(tmp_path):
    """이건 기다려도 해결되지 않는다 — 그 경주의 실전 기록은 영영 못 만든다."""
    with _db(tmp_path) as conn:
        _race(conn, key="2026-36-1-01", ymd="20260902", no=1, post="08:00")
        assert "발주 놓침" in _kinds(conn)


def test_이틀_지난_경주에_착순이_없으면_경보(tmp_path):
    with _db(tmp_path) as conn:
        _race(conn, key="2026-35-2-01", ymd="20260827", no=1, post="11:40", preds=6)
        assert "결과 미수집" in _kinds(conn)


def test_어제_경주는_착순이_없어도_조용하다(tmp_path):
    """결과는 하루 뒤에 공개된다. 그 지연까지 경보로 세면 매일 빨간불이 된다."""
    with _db(tmp_path) as conn:
        _race(conn, key="2026-36-0-01", ymd="20260901", no=1, post="11:40", preds=6)
        assert "결과 미수집" not in _kinds(conn)


def test_경보는_모두_치명으로_표시된다(tmp_path):
    """--strict 가 걸러 내지 못하면 자동화가 빨간불을 못 낸다."""
    with _db(tmp_path) as conn:
        _race(conn, key="2026-36-1-05", ymd="20260902", no=5, post="13:12")
        issues = health.check(conn, NOW, version=VERSION)
        assert issues and all(i.fatal for i in issues)
