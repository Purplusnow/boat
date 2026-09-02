"""파이프라인 자체 점검.

``audit`` 은 **자료**가 있어야 할 자리에 있는지 본다. 여기서는 **우리가 제때 할
일을 했는지** 본다 — 발주 전에 예상을 만들었는가, 결과를 받아왔는가.

둘을 가른 이유는 실패의 성격이 다르기 때문이다. 자료 결손은 포털이 늦으면
저절로 메워지지만, 발주가 지나 버린 경주의 예상은 **영영 만들 수 없다.**
그건 기다린다고 해결되지 않으므로 사람이 알아야 한다.

그리고 이 점검이 필요한 진짜 이유는, **자동 실행이 성공했다고 말하면서 실패하기
때문이다.** 실제로 겪은 것들이다.

* 러너에서 포털로 나가는 연결이 세 번 다 끊겨 새 개최일 출주표가 안 들어왔는데,
  '기존 자료로 진행' 하며 초록불로 끝났다. 그날 예상이 통째로 비었다.
* 예약 실행이 아예 뜨지 않은 날이 있었다 (GitHub 예약은 보장이 아니다).

두 경우 모두 배포는 성공한다. 화면만 비어 있다. 그래서 **빌드 성공과 별개로**
결손을 세고, 있으면 빨간불을 낸다.

    python -m boatai.health            # 사람이 읽는 요약
    python -m boatai.health --strict   # 결손이 있으면 1 로 끝낸다
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3
import sys
from typing import List, NamedTuple, Optional

from .clock import now_kst
from .kboat.store import session
from .model import MODEL_VERSION

# 발주가 이만큼 남았는데 출주표가 없으면 수집이 막힌 것으로 본다. 출주표는 보통
# 전날 밤에 올라오지만 당일 아침인 적도 있어, 넉넉히 두되 손쓸 시간은 남긴다.
ENTRY_DEADLINE = dt.timedelta(hours=3)

# 결과는 하루 뒤에 공개된다. 이틀이 지나도 없으면 지연이 아니라 결손이다.
RESULT_GRACE_DAYS = 2

ROW_SQL = """
SELECT g.race_key, g.race_ymd, g.race_no, g.post_time,
       (SELECT COUNT(*) FROM entries e
         WHERE e.race_key = g.race_key)                       AS n_entry,
       (SELECT COUNT(*) FROM results r
         WHERE r.race_key = g.race_key AND r.ord IS NOT NULL)  AS n_ord,
       (SELECT COUNT(*) FROM predictions p
         WHERE p.race_key = g.race_key AND p.model_version = ?) AS n_pred
  FROM races g
 WHERE g.race_ymd IS NOT NULL AND g.race_ymd >= ?
 ORDER BY g.race_ymd, g.race_no
"""


class Issue(NamedTuple):
    fatal: bool          # True 면 빨간불 — 사람이 봐야 한다
    kind: str
    detail: str


def _post_at(ymd: str, hhmm: Optional[str]) -> Optional[dt.datetime]:
    if not ymd or not hhmm:
        return None
    try:
        return dt.datetime.strptime(f"{str(ymd)[:8]} {str(hhmm)[:5]}", "%Y%m%d %H:%M")
    except ValueError:
        return None


def check(conn: sqlite3.Connection, now: Optional[dt.datetime] = None,
          *, version: str = MODEL_VERSION, days: int = 7) -> List[Issue]:
    """최근 며칠과 앞날을 훑어 '했어야 하는데 안 한 것' 을 찾는다."""
    now = now or now_kst()
    since = (now - dt.timedelta(days=days)).strftime("%Y%m%d")
    today = now.strftime("%Y%m%d")

    rows = conn.execute(ROW_SQL, (version, since)).fetchall()
    issues: List[Issue] = []

    # 같은 종류를 경주마다 한 줄씩 내면 로그가 묻힌다. 개최일 단위로 센다.
    missing_pred: dict = {}
    missed_post: dict = {}
    no_entry_soon: dict = {}
    stale_result: dict = {}

    for r in rows:
        ymd = r["race_ymd"]
        post = _post_at(ymd, r["post_time"])

        # ── 아직 달리지 않은 경주 ─────────────────────────────
        if post and post > now:
            if r["n_entry"] and not r["n_pred"]:
                # 출주표가 있는데 예상이 없다. 지금이라도 만들 수 있다.
                missing_pred.setdefault(ymd, 0)
                missing_pred[ymd] += 1
            elif not r["n_entry"] and post - now <= ENTRY_DEADLINE:
                # 발주가 코앞인데 출주표가 없다 — 수집이 막혔다는 뜻이다.
                no_entry_soon.setdefault(ymd, 0)
                no_entry_soon[ymd] += 1
            continue

        # ── 이미 발주가 지난 경주 ─────────────────────────────
        #
        # 예상이 없으면 그 경주의 실전 기록은 영영 만들 수 없다. 다만 어제
        # 이전 것까지 계속 빨간불을 내면 경보가 굳어 아무도 안 보게 되므로,
        # **오늘 놓친 것만** 띄운다. 지난 것은 사후 예상이 목록을 메운다.
        if post and ymd == today and not r["n_pred"]:
            missed_post.setdefault(ymd, 0)
            missed_post[ymd] += 1

        # 결과는 하루 늦게 들어온다. 이틀이 지나도 없으면 결손이다.
        if ymd < (now - dt.timedelta(days=RESULT_GRACE_DAYS)).strftime("%Y%m%d") \
                and r["n_entry"] and not r["n_ord"]:
            stale_result.setdefault(ymd, 0)
            stale_result[ymd] += 1

    for ymd, n in sorted(missing_pred.items()):
        issues.append(Issue(True, "발주 전 예상 없음",
                            f"{ymd} {n}경주 — 출주표가 있는데 예상이 없다. "
                            "지금 만들면 아직 늦지 않았다."))
    for ymd, n in sorted(no_entry_soon.items()):
        issues.append(Issue(True, "출주표 미수집",
                            f"{ymd} {n}경주 — 발주 3시간 안인데 출주표가 없다. "
                            "수집이 막혔을 가능성이 높다."))
    for ymd, n in sorted(missed_post.items()):
        issues.append(Issue(True, "발주 놓침",
                            f"{ymd} {n}경주 — 예상 없이 발주가 지났다. "
                            "이 경주의 실전 기록은 만들 수 없다."))
    for ymd, n in sorted(stale_result.items()):
        issues.append(Issue(True, "결과 미수집",
                            f"{ymd} {n}경주 — 이틀이 지났는데 착순이 없다."))
    return issues


def summary(conn: sqlite3.Connection, now: Optional[dt.datetime] = None,
            *, version: str = MODEL_VERSION) -> str:
    """지금 무엇이 준비돼 있는지 한눈에 보는 표. 결손이 없어도 찍는다."""
    now = now or now_kst()
    since = (now - dt.timedelta(days=3)).strftime("%Y%m%d")
    rows = conn.execute(ROW_SQL, (version, since)).fetchall()
    days: dict = {}
    for r in rows:
        d = days.setdefault(r["race_ymd"], {"n": 0, "entry": 0, "pred": 0, "ord": 0})
        d["n"] += 1
        d["entry"] += 1 if r["n_entry"] else 0
        d["pred"] += 1 if r["n_pred"] else 0
        d["ord"] += 1 if r["n_ord"] else 0
    out = ["  개최일      경주  출주표  예상  결과"]
    for ymd in sorted(days):
        d = days[ymd]
        out.append(f"  {ymd}  {d['n']:>4}  {d['entry']:>5}  {d['pred']:>4}  {d['ord']:>4}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="파이프라인 자체 점검")
    ap.add_argument("--db", default="data/boatai.sqlite")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--strict", action="store_true",
                    help="결손이 있으면 1 로 끝낸다 (자동화에서 빨갛게 뜬다)")
    args = ap.parse_args(argv)

    with session(args.db) as conn:
        issues = check(conn, days=args.days)
        table = summary(conn)

    print("자체 점검 — 최근 상태")
    print(table)
    print()

    if not issues:
        print("할 일이 밀린 것 없음")
        return 0

    for it in issues:
        print(f"  ▸ {it.kind} — {it.detail}")
        # GitHub Actions 로그에서 눈에 띄게 한다.
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::error title={it.kind}::{it.detail}")
    print()
    return 1 if args.strict and any(i.fatal for i in issues) else 0


if __name__ == "__main__":
    sys.exit(main())
