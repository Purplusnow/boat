"""수집기 — 오픈API → SQLite.

**연 단위로 일괄 조회한다.** 네 개 API 모두 ``stnd_yr`` 하나만 주면 그 해 전체를
돌려주므로(2025년 기준 출주표 10,608행), 회차·일차를 하나씩 도는 것보다 호출이
수십 배 적다. 24년 백필이 700회 남짓으로 끝나 개발계정 일일 한도(1만) 안에
여유 있게 들어간다. 좌표를 하나씩 도는 방식은 같은 자료를 받는 데 8천 회를
쓰므로, 하루를 통째로 태우고도 다 못 받는다.

배당은 **경주결과 API 로 받는다.** 배당률 API 는 (연도·회차·일차·경주번호)가
전부 필수라 경주 하나에 한 번씩 불러야 하는데, 경주결과 API 는 같은 배당을
적중 조합(정번)까지 붙여 연 단위로 준다. 배당률 API 는 결손을 메울 때만 쓴다.

    python -m boatai.kboat.collect backfill --from-year 2015
    python -m boatai.kboat.collect daily
    python -m boatai.kboat.collect payoff --year 2026     # 결손 보충
    python -m boatai.kboat.collect status
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from typing import Dict, List, Optional

from ..clock import today_kst
from . import normalize as nz
from .client import KboatApiError, KboatClient
from .endpoints import APPROVED, REGISTRY, to_api_params
from .store import (DEFAULT_DB, already_fetched, counts, dumps, log_fetch,
                    mark_has_result, session, sync_field_size, upsert)

log = logging.getLogger(__name__)

# 경정 개최 자료가 오픈API 로 소급되는 하한. 그 이전은 조회해도 0건이다.
FIRST_YEAR = 2002


def _fetch(client: KboatClient, key: str, params: Dict) -> List[dict]:
    ep = REGISTRY[key]
    return client.fetch(ep.path, to_api_params(key, params), rows=1000, max_pages=60)


# ---------------------------------------------------------------------------
# 연 단위 수집
# ---------------------------------------------------------------------------

def collect_year(client: KboatClient, conn: sqlite3.Connection, year: int,
                 *, force: bool = False) -> Dict[str, int]:
    """한 해치를 통째로 받아 저장한다."""
    done: Dict[str, int] = {}

    # ── 출주표 → races + entries ──────────────────────────────────────
    coord = f"race_card:{year}"
    if force or not already_fetched(conn, "race_card", coord):
        cards = _fetch(client, "race_card", {"stnd_yr": year})
        races = {}
        entries = []
        for rec in cards:
            r = nz.race_row_from_entry(rec)
            races[r["race_key"]] = r
            e = nz.entry_row(rec)
            if e.get("lane") is None:
                # 정번이 없으면 행을 식별할 수 없다. 조용히 버리지 않고 센다.
                continue
            e["raw_json"] = dumps(rec)
            entries.append(e)
        upsert(conn, "races", list(races.values()), ["race_key"])
        upsert(conn, "entries", entries, ["race_key", "lane"])
        log_fetch(conn, "race_card", coord, len(cards))
        conn.commit()
        done["출주표"] = len(entries)
        log.info("  출주표 %d행 → 경주 %d개", len(entries), len(races))

    # ── 착순 → results (+ races.race_ymd) ─────────────────────────────
    coord = f"race_rank:{year}"
    if force or not already_fetched(conn, "race_rank", coord):
        ranks = _fetch(client, "race_rank", {"stnd_yr": year})
        # 정번은 출주표에만 있다. 이 해의 (경주, 선수명) → 정번 색인을 DB 에서
        # 만들어 붙인다. 출주표를 먼저 수집해야 하는 이유가 여기 있다.
        lane_by_name: Dict[str, Dict[str, int]] = {}
        for row in conn.execute(
                "SELECT race_key, racer_nm, lane FROM entries WHERE race_key LIKE ?",
                (f"{year}-%",)):
            if row["racer_nm"]:
                lane_by_name.setdefault(row["race_key"], {})[row["racer_nm"]] = row["lane"]

        rows = nz.result_rows(ranks, lane_by_name)
        # 경주일자는 경주 단위 정보다. 착순 API 에만 있으므로 여기서 races 로 올린다.
        race_dates = {r["race_key"]: {**nz.coords_from_key(r["race_key"]),
                                      "race_ymd": r["race_ymd"]}
                      for r in rows if r.get("race_ymd")}
        upsert(conn, "races", list(race_dates.values()), ["race_key"])

        keep = [r for r in rows if r.get("lane") is not None]
        orphan = len(rows) - len(keep)
        for r in keep:
            r["raw_json"] = None
        upsert(conn, "results", keep, ["race_key", "lane"])
        log_fetch(conn, "race_rank", coord, len(ranks))
        conn.commit()
        done["착순"] = len(keep)
        # 정번을 못 붙인 착순은 출주표가 없다는 뜻이다. 조용히 넘기면 그 경주는
        # 평가에서 통째로 빠지므로 반드시 눈에 띄게 남긴다.
        if orphan:
            log.warning("  착순 %d행이 출주표와 이어지지 않음 (정번 미상)", orphan)
        log.info("  착순 %d행 (경주일자 %d개 갱신)", len(keep), len(race_dates))

    # ── 경주결과 → payoffs (+ 1~3착 보강) ────────────────────────────
    coord = f"race_result:{year}"
    if force or not already_fetched(conn, "race_result", coord):
        results = _fetch(client, "race_result", {"stnd_yr": year})
        payoffs, top3 = [], []
        for rec in results:
            payoffs.extend(nz.result_payoff_rows(rec))
            top3.extend(nz.result_top3(rec))
        upsert(conn, "payoffs", payoffs, ["race_key", "pool"])
        # 착순 API 가 비어 있던 경주의 1~3착을 메운다. upsert 가 빈 값으로
        # 덮어쓰지 않으므로 이미 있는 착순은 그대로 남는다.
        upsert(conn, "results", [r for r in top3 if r.get("lane")], ["race_key", "lane"])
        log_fetch(conn, "race_result", coord, len(results))
        conn.commit()
        done["배당"] = len(payoffs)
        log.info("  배당 %d행 / 1~3착 보강 %d행", len(payoffs), len(top3))

    # ── 선수 회차별 성적 ──────────────────────────────────────────────
    coord = f"racer_tms:{year}"
    if force or not already_fetched(conn, "racer_tms", coord):
        tms = _fetch(client, "racer_tms", {"stnd_yr": year})
        rows = []
        for rec in tms:
            if not rec.get("racer_no"):
                continue
            rows.append({
                "stnd_yr": nz.to_int(rec.get("stnd_yr")),
                "week_tcnt": nz.to_int(rec.get("week_tcnt")),
                "racer_no": nz._s(rec.get("racer_no")),
                "racer_nm": nz._s(rec.get("racer_nm")) or None,
                "race_tcnt": nz.to_int(rec.get("race_tcnt")),
                **{f"rank{i}_tcnt": nz.to_int(rec.get(f"rank{i}_tcnt")) for i in range(1, 7)},
                "avg_rank": nz.to_float(rec.get("avg_rank")),
                "avg_acdnt_scr": nz.to_float(rec.get("avg_acdnt_scr")),
                "avg_scr": nz.to_float(rec.get("avg_scr")),
                "avg_strt_tm": nz.to_float(rec.get("avg_strt_tm")),
                "win_ratio": nz.to_float(rec.get("win_ratio")),
                "high_rate": nz.to_float(rec.get("high_rate")),
                "high_3_rank_ratio": nz.to_float(rec.get("high_3_rank_ratio")),
            })
        upsert(conn, "racer_tms", rows, ["stnd_yr", "week_tcnt", "racer_no"])
        log_fetch(conn, "racer_tms", coord, len(tms))
        conn.commit()
        done["선수성적"] = len(rows)

    mark_has_result(conn)
    sync_field_size(conn)
    conn.commit()
    return done


def backfill(client: KboatClient, conn: sqlite3.Connection,
             from_year: int, to_year: int, *, force: bool = False) -> None:
    for year in range(from_year, to_year + 1):
        log.info("── %d년 ──", year)
        try:
            got = collect_year(client, conn, year, force=force)
        except KboatApiError as e:
            # 호출 한도는 오늘 더 해 봐야 소용이 없다. 여기서 멈추고 내일
            # 이어가는 편이, 남은 해를 반쯤 받아 두는 것보다 낫다.
            if e.code in ("22", "429"):
                log.error("호출 한도 초과 — 중단합니다. 내일 같은 명령으로 이어집니다.")
                raise
            log.error("  %d년 실패: %s", year, e)
            continue
        if not got:
            log.info("  이미 수집됨 (--force 로 다시 받을 수 있습니다)")


# ---------------------------------------------------------------------------
# 배당 결손 보충
# ---------------------------------------------------------------------------

def fill_payoffs(client: KboatClient, conn: sqlite3.Connection, year: int,
                 limit: int = 400) -> int:
    """배당이 비어 있는 경주를 배당률 API 로 하나씩 메운다.

    경주결과 API 는 최근 회차가 늦게 반영된다(실측: 착순은 당일 들어오는데
    결과·배당은 회차 하나만큼 뒤진다). 그 구간을 이걸로 메운다. 경주당 한 번씩
    부르므로 호출이 비싸다 — limit 으로 상한을 두고, **잘라낸 만큼을 로그로
    남긴다**. 조용히 자르면 '배당이 원래 없는 경주'로 오해된다.
    """
    rows = conn.execute(
        "SELECT r.race_key, r.stnd_yr, r.week_tcnt, r.day_tcnt, r.race_no "
        "FROM races r LEFT JOIN payoffs p ON p.race_key = r.race_key "
        "WHERE r.stnd_yr = ? AND r.has_result = 1 AND p.race_key IS NULL "
        "ORDER BY r.week_tcnt DESC, r.day_tcnt DESC, r.race_no",
        (year,)).fetchall()
    if not rows:
        log.info("배당 결손 없음 (%d년)", year)
        return 0

    log.info("배당 결손 %d경주 — 최대 %d경주까지 보충합니다", len(rows), limit)
    n = 0
    for row in rows[:limit]:
        try:
            recs = _fetch(client, "payoff", {
                "stnd_yr": row["stnd_yr"], "week_tcnt": row["week_tcnt"],
                "day_tcnt": row["day_tcnt"], "race_no": row["race_no"]})
        except KboatApiError as e:
            if e.code in ("22", "429"):
                log.error("호출 한도 초과 — %d경주까지 보충하고 중단합니다", n)
                break
            log.warning("  %s 실패: %s", row["race_key"], e)
            continue
        payoffs = []
        for rec in recs:
            payoffs.extend(nz.payoff_rows(rec))
        if payoffs:
            upsert(conn, "payoffs", payoffs, ["race_key", "pool"])
            n += 1
    conn.commit()
    if len(rows) > limit:
        log.warning("결손 %d경주 중 %d경주만 시도했습니다 — 남은 %d경주는 다시 실행하세요",
                    len(rows), limit, len(rows) - limit)
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="경정 오픈API 수집")
    ap.add_argument("command",
                choices=["backfill", "daily", "payoff", "prune", "status"])
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--from-year", type=int, default=FIRST_YEAR)
    ap.add_argument("--to-year", type=int, default=today_kst().year)
    ap.add_argument("--year", type=int, default=today_kst().year)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--force", action="store_true", help="이미 받은 해도 다시 받는다")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    missing = [k for k in ("race_card", "race_rank", "race_result") if k not in APPROVED]
    if missing:
        print(f"승인되지 않은 필수 API 가 있습니다: {missing}", file=sys.stderr)
        return 2

    with session(args.db) as conn:
        if args.command == "status":
            for k, v in counts(conn).items():
                print(f"  {k:<12} {v:>9,}")
            row = conn.execute(
                "SELECT MIN(race_ymd) a, MAX(race_ymd) b, COUNT(*) n FROM races "
                "WHERE race_ymd IS NOT NULL").fetchone()
            print(f"\n  경주일자 {row['a']} ~ {row['b']} ({row['n']:,}경주)")
            return 0

        client = KboatClient.from_env()
        if args.command == "backfill":
            backfill(client, conn, args.from_year, args.to_year, force=args.force)
        elif args.command == "daily":
            # 올해만, 항상 다시 받는다. 최근 회차는 착순·배당이 나중에 채워지므로
            # '이미 받았다'고 건너뛰면 영영 비어 있게 된다.
            collect_year(client, conn, args.year, force=True)
        elif args.command == "prune":
            # 원본 응답은 파싱 규칙을 고칠 때만 쓸모가 있다. 오래된 것을 비워
            # DB 를 옮길 수 있는 크기로 유지한다(자동 배포의 캐시 용량 문제).
            from .store import prune_raw_json
            print(f"raw_json 정리 {prune_raw_json(conn):,}행")
        elif args.command == "payoff":
            n = fill_payoffs(client, conn, args.year, args.limit)
            print(f"배당 보충 {n}경주")

        for k, v in counts(conn).items():
            print(f"  {k:<12} {v:>9,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
