"""SQLite 저장소.

경주 단위 자료는 재수집이 잦으므로 모든 쓰기를 upsert 로 처리한다. 원본 응답은
``raw_json`` 에 그대로 남겨서, 나중에 파싱 규칙이 틀렸다는 게 드러나도 API 를
다시 때리지 않고 로컬에서 재정규화할 수 있게 한다 — 경정 API 는 필드 하나에
값 두 개를 붙여 주는 곳이 많아 규칙이 바뀔 여지가 특히 크다.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from ..clock import today_kst

DEFAULT_DB = Path("data/boatai.sqlite")

SCHEMA = """
PRAGMA journal_mode=WAL;

-- 경주 단위 메타. 경정은 미사리 한 곳뿐이라 경마장 축이 없고, 대신
-- (연도·회차·일차·경주번호) 좌표가 그 자리를 대신한다.
CREATE TABLE IF NOT EXISTS races (
    race_key    TEXT PRIMARY KEY,
    stnd_yr     INTEGER NOT NULL,
    week_tcnt   INTEGER NOT NULL,
    day_tcnt    INTEGER NOT NULL,
    race_no     INTEGER NOT NULL,
    race_ymd    TEXT,            -- 실제 경주일자 (착순 API 에만 있다)
    post_time   TEXT,            -- 발주 시각. 예상을 더 이상 고치지 않는 기준선이다.
    race_class  TEXT,            -- 일반/특선 등
    st_method   TEXT,            -- 경주구분(온라인/일반 등)
    field_size  INTEGER,
    has_result  INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_races_ymd  ON races(race_ymd);
CREATE INDEX IF NOT EXISTS idx_races_coord ON races(stnd_yr, week_tcnt, day_tcnt);

-- 출주표: 경주 전에 확정되는 정보. 피처는 전부 여기서 나온다.
CREATE TABLE IF NOT EXISTS entries (
    race_key    TEXT NOT NULL,
    lane        INTEGER NOT NULL,     -- 정번(=코스) 1~6
    racer_nm    TEXT,
    racer_grd   TEXT,
    sex         TEXT,
    age         INTEGER,
    weight      REAL,
    post_time   TEXT,
    race_class  TEXT,
    st_method   TEXT,
    color_nm    TEXT,

    avg_rank          REAL,
    high_rate         REAL,
    avg_acdnt_scr     REAL,
    f_cnt             INTEGER,
    l_cnt             INTEGER,
    tms6_avg_rank_scr REAL,
    tms6_avg_scr      REAL,
    tms6_win_ratio    REAL,
    tms6_high_rate    REAL,
    tms6_high3_rate   REAL,
    tms6_avg_st       REAL,
    bf_dd_recd_scr    REAL,
    mm6_race_cnt      INTEGER,
    thdd_race_no      INTEGER,

    motor_no          INTEGER,
    mot_avg_rank_scr  REAL,
    mot_high_rate     REAL,
    mot_high3_rate    REAL,
    mot_prev_racer    TEXT,
    mot_prev_avg      REAL,
    boat_no           INTEGER,
    boat_avg_rank_scr REAL,
    boat_high_rate    REAL,

    course1_high_rate REAL, course1_cnt INTEGER,
    course2_high_rate REAL, course2_cnt INTEGER,
    course3_high_rate REAL, course3_cnt INTEGER,
    course4_high_rate REAL, course4_cnt INTEGER,
    course5_high_rate REAL, course5_cnt INTEGER,
    course6_high_rate REAL, course6_cnt INTEGER,

    recent1 INTEGER, recent2 INTEGER, recent3 INTEGER, recent4 INTEGER,
    recent5 INTEGER, recent6 INTEGER, recent7 INTEGER, recent8 INTEGER,
    recent_avg REAL, recent_cnt INTEGER,

    raw_json    TEXT,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (race_key, lane),
    FOREIGN KEY (race_key) REFERENCES races(race_key)
);
CREATE INDEX IF NOT EXISTS idx_entries_racer ON entries(racer_nm);
CREATE INDEX IF NOT EXISTS idx_entries_motor ON entries(motor_no);

-- 착순. 실격·사고로 숫자가 아닐 수 있어 원문을 ord_note 에 남긴다.
CREATE TABLE IF NOT EXISTS results (
    race_key   TEXT NOT NULL,
    lane       INTEGER NOT NULL,
    racer_no   TEXT,
    racer_nm   TEXT,
    ord        INTEGER,
    ord_note   TEXT,
    raw_json   TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (race_key, lane)
);
CREATE INDEX IF NOT EXISTS idx_results_ord ON results(ord);
CREATE INDEX IF NOT EXISTS idx_results_racer ON results(racer_no);

-- 승식별 확정배당. 우리 추천 조합이 실제로 얼마를 돌려줬는지 검증하는 근거다.
-- combo 는 적중 조합의 정번(오름차순 정렬은 순서 없는 승식만).
CREATE TABLE IF NOT EXISTS payoffs (
    race_key   TEXT NOT NULL,
    pool       TEXT NOT NULL,     -- 단승·연승1·연승2·쌍승·복승·삼복승
    combo      TEXT,
    payout     REAL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (race_key, pool)
);
CREATE INDEX IF NOT EXISTS idx_payoffs_race ON payoffs(race_key);

-- 예측: 발주 전에 만들어 고정한다. 공개(기록) 후 수정하지 않는 것이
-- 적중률 숫자를 믿을 수 있게 하는 유일한 근거다.
CREATE TABLE IF NOT EXISTS predictions (
    race_key      TEXT NOT NULL,
    lane          INTEGER NOT NULL,
    racer_nm      TEXT,
    p_win         REAL NOT NULL,
    p_top2        REAL,
    p_top3        REAL,
    pred_rank     INTEGER,
    model_version TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (race_key, lane, model_version)
);
CREATE INDEX IF NOT EXISTS idx_pred_race ON predictions(race_key);

-- 전개 시뮬레이션. **예상과 함께 확정 저장한다** — 나중에 다시 돌리면 그때의 상수와
-- 난수로 다른 전개가 나와, 발주 전에 화면에 있던 것과 달라진다. 모의(v1-oos)
-- 기록은 여기 넣지 않고 빌드 때 계산한다. 저장할 가치가 있는 것은 공개분뿐이다.
CREATE TABLE IF NOT EXISTS simulations (
    race_key   TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,      -- simulate() 결과 전체 (JSON)
    conf_label TEXT,
    conf_score INTEGER,
    top_tactic TEXT,
    n_sims     INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 선수 회차별 성적 스냅샷. 회차 단위라 '그 시점까지의 폼'을 누수 없이 쓸 수 있다.
CREATE TABLE IF NOT EXISTS racer_tms (
    stnd_yr    INTEGER NOT NULL,
    week_tcnt  INTEGER NOT NULL,
    racer_no   TEXT NOT NULL,
    racer_nm   TEXT,
    race_tcnt  INTEGER,
    rank1_tcnt INTEGER, rank2_tcnt INTEGER, rank3_tcnt INTEGER,
    rank4_tcnt INTEGER, rank5_tcnt INTEGER, rank6_tcnt INTEGER,
    avg_rank   REAL, avg_acdnt_scr REAL, avg_scr REAL, avg_strt_tm REAL,
    win_ratio  REAL, high_rate REAL, high_3_rank_ratio REAL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (stnd_yr, week_tcnt, racer_no)
);

-- 수집 이력 (증분 수집용). 0건이었던 좌표도 '확인 완료'로 남긴다 —
-- 개최가 없던 회차를 매번 다시 묻는 것은 일일 호출량의 낭비다.
CREATE TABLE IF NOT EXISTS fetch_log (
    endpoint   TEXT NOT NULL,
    coord      TEXT NOT NULL,
    n_records  INTEGER NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (endpoint, coord)
);
"""


def connect(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def session(path: Path | str = DEFAULT_DB) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        init_db(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]


def upsert(conn: sqlite3.Connection, table: str, rows: Iterable[Dict[str, Any]],
           key_cols: List[str]) -> int:
    """존재하는 컬럼만 골라 upsert. 스키마에 없는 키는 조용히 버린다.

    **빈 값으로는 덮어쓰지 않는다.** 같은 행을 여러 API 가 조각조각 채운다 —
    발주시각은 출주표에만, 경주일자는 착순에만 있다. 들어온 값이 비었다고 기존
    값을 지우면 나중 수집이 앞선 수집을 무효로 만든다.
    """
    rows = [r for r in rows if r]
    if not rows:
        return 0
    cols = [c for c in _table_columns(conn, table) if c != "updated_at"]
    usable = [c for c in cols if any(c in r for r in rows)]
    if not usable:
        return 0

    placeholders = ",".join("?" for _ in usable)
    update_cols = [c for c in usable if c not in key_cols]
    set_clause = ", ".join(f"{c}=COALESCE(excluded.{c}, {table}.{c})" for c in update_cols)
    if "updated_at" in _table_columns(conn, table):
        set_clause = (set_clause + ", " if set_clause else "") + "updated_at=datetime('now')"

    sql = (f"INSERT INTO {table} ({','.join(usable)}) VALUES ({placeholders}) "
           f"ON CONFLICT({','.join(key_cols)}) DO UPDATE SET {set_clause}")
    conn.executemany(sql, [tuple(r.get(c) for c in usable) for r in rows])
    return len(rows)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def log_fetch(conn: sqlite3.Connection, endpoint: str, coord: str, n: int) -> None:
    conn.execute(
        "INSERT INTO fetch_log(endpoint,coord,n_records) VALUES(?,?,?) "
        "ON CONFLICT(endpoint,coord) DO UPDATE SET "
        "n_records=excluded.n_records, fetched_at=datetime('now')",
        (endpoint, coord, n),
    )


def already_fetched(conn: sqlite3.Connection, endpoint: str, coord: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM fetch_log WHERE endpoint=? AND coord=?", (endpoint, coord)
    ).fetchone()
    return row is not None


def mark_has_result(conn: sqlite3.Connection) -> int:
    """착순이 들어온 경주에 has_result 를 세운다."""
    cur = conn.execute(
        "UPDATE races SET has_result=1 WHERE has_result=0 AND race_key IN "
        "(SELECT race_key FROM results WHERE ord IS NOT NULL)")
    return cur.rowcount or 0


def sync_field_size(conn: sqlite3.Connection) -> int:
    """출주 정수(보통 6)를 채운다. 결항·기권이 있으면 6보다 작다."""
    cur = conn.execute(
        "UPDATE races SET field_size = ("
        "  SELECT COUNT(*) FROM entries e WHERE e.race_key = races.race_key) "
        "WHERE field_size IS NULL OR field_size = 0")
    return cur.rowcount or 0


def prune_raw_json(conn: sqlite3.Connection, keep_days: int = 365) -> int:
    """오래된 행의 ``raw_json`` 을 비운다.

    raw_json 의 가치는 파싱 규칙을 고치는 초기에 집중되는 반면 용량은 계속
    늘어난다. 최근 구간만 남겨 DB 를 옮길 수 있는 크기로 유지한다.
    """
    cutoff = (today_kst() - dt.timedelta(days=keep_days)).strftime("%Y%m%d")
    total = 0
    for table in ("entries", "results"):
        cur = conn.execute(
            f"UPDATE {table} SET raw_json = NULL WHERE raw_json IS NOT NULL AND race_key IN "
            f"(SELECT race_key FROM races WHERE race_ymd IS NOT NULL AND race_ymd < ?)",
            (cutoff,))
        total += cur.rowcount or 0
    conn.commit()
    conn.execute("VACUUM")
    return total


def counts(conn: sqlite3.Connection) -> Dict[str, int]:
    out = {}
    for t in ("races", "entries", "results", "payoffs", "predictions",
              "simulations", "racer_tms"):
        try:
            out[t] = conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
        except sqlite3.Error:
            out[t] = 0
    return out
