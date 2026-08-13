"""정적 사이트 생성기.

DB → Jinja2 → ``dist/`` 정적 HTML. 서버가 없으므로 전부 빌드 타임에 확정한다.

광고·애널리틱스·SEO 태그는 아직 두지 않는다. 대신 이 사이트가 반드시 해야
하는 일은 따로 있다.

* **예상이 언제 만들어졌는지 화면에 적는다.** 발주 전에 확정 저장한 기록과
  시간순 교차검증으로 사후 산출한 기록은 뜻이 전혀 다르다. 표에서 구분되지
  않으면 읽는 사람이 둘을 같은 것으로 받아들인다.
* **빗나간 예상을 지우지 않는다.** 결과 아카이브는 적중·불발을 그대로 싣는다.
* **표본 수를 함께 적는다.** 30경주짜리 60% 와 3,000경주짜리 48% 는 다른 말이다.

    python -m boatai.site --db data/boatai.sqlite --out dist
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .clock import now_kst, today_kst
from .marks import MARK_MEANING, MARK_THRESHOLDS, assign_marks
from .kboat.store import session
from .simulate import simulate as run_simulation, tactic_note
from .strategy import build_report as strategy_report
from .verify import BET_ORDER, build_report

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path("templates")
STATIC_DIR = Path("static")

# 발주 전에 확정 저장한 공개 기록 / 시간순 교차검증으로 산출한 검증 기록.
LIVE_VERSION = "v1"
OOS_VERSION = "v1-oos"
VERSION_LABEL = {
    LIVE_VERSION: ("공개", "발주 전에 확정 저장한 예상"),
    OOS_VERSION: ("검증", "시간순 교차검증으로 산출한 과거 재현 기록"),
}

WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]

RACE_SQL = """
SELECT r.race_key, r.race_ymd, r.stnd_yr, r.week_tcnt, r.day_tcnt, r.race_no,
       r.post_time, r.race_class, r.st_method, r.field_size,
       COALESCE(r.has_result, 0) AS has_result,
       (SELECT COUNT(*) FROM predictions p WHERE p.race_key = r.race_key
                                       AND p.model_version = ?)  AS n_pred,
       (SELECT COUNT(DISTINCT v.pool) FROM payoffs v WHERE v.race_key = r.race_key) AS n_pool
FROM races r
WHERE EXISTS (SELECT 1 FROM predictions p WHERE p.race_key = r.race_key
                                      AND p.model_version = ?)
-- 경주는 **항상 1R 부터** 늘어놓는다. 하루 안의 순서는 시행 순서이고,
-- 그것이 사람이 경주를 찾는 순서다. 날짜만 최신이 위로 온다.
ORDER BY r.race_ymd DESC, r.race_no ASC
"""

RUNNER_SQL = """
SELECT p.lane, p.racer_nm, p.p_win, p.p_top2, p.p_top3, p.pred_rank,
       e.racer_grd, e.age, e.weight, e.motor_no, e.boat_no,
       e.avg_rank, e.high_rate, e.avg_acdnt_scr, e.f_cnt, e.l_cnt,
       e.tms6_win_ratio, e.tms6_high_rate, e.tms6_high3_rate, e.tms6_avg_st,
       e.mot_high_rate, e.mot_high3_rate, e.boat_high_rate,
       e.recent_avg, e.mm6_race_cnt,
       e.course1_high_rate, e.course2_high_rate, e.course3_high_rate,
       e.course4_high_rate, e.course5_high_rate, e.course6_high_rate,
       e.course1_cnt, e.course2_cnt, e.course3_cnt,
       e.course4_cnt, e.course5_cnt, e.course6_cnt,
       res.ord
FROM predictions p
LEFT JOIN entries e   ON e.race_key = p.race_key AND e.lane = p.lane
LEFT JOIN results res ON res.race_key = p.race_key AND res.lane = p.lane
WHERE p.race_key = ? AND p.model_version = ?
ORDER BY p.lane
"""


def _dict(row: sqlite3.Row) -> Dict:
    return {k: row[k] for k in row.keys()}


def fmt_date(ymd: Optional[str]) -> str:
    """``"20260813"`` → ``"2026-08-13 (목)"``"""
    if not ymd or len(str(ymd)) < 8:
        return ""
    s = str(ymd)
    try:
        import datetime as dt
        d = dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return s
    return f"{d.isoformat()} ({WEEKDAY_KO[d.weekday()]})"


# ---------------------------------------------------------------------------
# 조회
# ---------------------------------------------------------------------------

def load_races(conn: sqlite3.Connection, version: str) -> List[Dict]:
    rows = [_dict(r) for r in conn.execute(RACE_SQL, (version, version))]
    for r in rows:
        r["date_label"] = fmt_date(r["race_ymd"])
        r["version"] = version
    return rows


def load_runners(conn: sqlite3.Connection, race_key: str, version: str) -> List[Dict]:
    runners = [_dict(r) for r in conn.execute(RUNNER_SQL, (race_key, version))]
    if not runners:
        return []
    # 기호는 예측 순위에서 나온다. 규칙은 marks 한 곳에만 있다.
    assign_marks(runners)
    for r in runners:
        lane = r.get("lane")
        # 오늘 타는 자리의 6개월 연대율만 앞으로 꺼낸다. 나머지 다섯 코스 값은
        # '다른 자리에서 어떤가'라는 다른 정보라 표에 싣지 않는다.
        r["own_course_rate"] = r.get(f"course{lane}_high_rate")
        r["own_course_cnt"] = r.get(f"course{lane}_cnt")
        r["is_winner"] = (r.get("ord") == 1)
    return sorted(runners, key=lambda x: x["lane"])


def load_payoffs(conn: sqlite3.Connection, race_key: str) -> List[Dict]:
    rows = [_dict(r) for r in conn.execute(
        "SELECT pool, combo, payout FROM payoffs WHERE race_key = ?", (race_key,))]
    order = {"단승": 0, "연승1": 1, "연승2": 2, "쌍승": 3, "복승": 4, "삼복승": 5}
    return sorted(rows, key=lambda r: order.get(r["pool"], 99))


def betting_combos(runners: List[Dict]) -> List[Dict]:
    """모델 확률을 승식 형식으로 옮겨 적는다.

    구매 권유가 아니라 확률의 다른 표기이며, 화면에도 그렇게 적는다. 회수율이
    100% 를 넘지 않는다는 사실을 같은 화면에 함께 둔다.
    """
    picks = [r["lane"] for r in sorted(runners, key=lambda x: x["pred_rank"] or 99)]
    if len(picks) < 3:
        return []
    return [
        {"label": "단승", "value": f"{picks[0]}", "note": "1순위"},
        {"label": "복승", "value": f"{picks[0]}-{picks[1]}", "note": "1·2순위 (순서 무관)"},
        {"label": "쌍승", "value": f"{picks[0]}→{picks[1]}", "note": "1·2순위 (순서까지)"},
        {"label": "삼복승", "value": f"{picks[0]}-{picks[1]}-{picks[2]}",
         "note": "1~3순위 (순서 무관)"},
        {"label": "복승 박스", "value": "-".join(str(x) for x in sorted(picks[:3])),
         "note": "상위 3정에서 2정 — 3통"},
    ]


def load_simulation(conn: sqlite3.Connection, race_key: str,
                    runners: List[Dict]) -> Dict:
    """전개 시뮬레이션. 저장된 것이 있으면 그것을 쓴다.

    공개(v1) 예상은 발주 전에 시뮬레이션까지 함께 확정 저장한다. 검증(v1-oos)
    기록은 설명용이므로 빌드 때 계산한다 — 1만 6천 경주를 미리 돌려 저장할
    이유가 없고, 저장한다고 그 숫자가 더 참이 되지도 않는다.
    """
    row = conn.execute("SELECT payload FROM simulations WHERE race_key=?",
                       (race_key,)).fetchone()
    if row:
        try:
            sim = json.loads(row["payload"])
            sim["frozen"] = True
            return sim
        except ValueError:
            pass
    sim = run_simulation(runners, n_sims=800)
    if sim:
        sim["frozen"] = False
    return sim


# 체크 포인트로 올릴 항목.
#   (표시 이름, 컬럼, 큰 값이 좋은가, 단위, 표본 컬럼, 최소 표본)
# 표본 하한을 두는 이유: 코스 연대율은 그 자리 출주가 한두 번뿐인 선수가
# 100% 로 찍힌다. 그걸 '이 코스 성적 최고'로 올리면 표가 거짓말을 한다.
FOCUS_SPECS = [
    ("모터 2연대율 최고", "mot_high_rate", True, "%", None, 0),
    ("최근 6회차 연대율 최고", "tms6_high_rate", True, "%", "mm6_race_cnt", 6),
    ("이 코스 성적 최고", "own_course_rate", True, "%", "own_course_cnt", 4),
    ("평균 착순 최상", "avg_rank", False, "", None, 0),
    ("보트 연대율 최고", "boat_high_rate", True, "%", None, 0),
]


def focus_points(runners: List[Dict]) -> List[Dict]:
    """'무엇을 보고 골랐나' 를 항목별로 뒤집어 보여준다.

    같은 표를 배 중심이 아니라 항목 중심으로 한 번 더 보여주면 훑어보기가
    빨라진다. 값이 비었거나 전원이 같으면 그 항목은 아무것도 말하지 않으므로
    내보내지 않는다. 표본이 얕은 값도 마찬가지다 — 1회 출주 100% 는 정보가
    아니라 잡음이다.
    """
    out = []
    for label, col, bigger, unit, cnt_col, min_n in FOCUS_SPECS:
        vals = []
        for r in runners:
            v = r.get(col)
            if v is None:
                continue
            if cnt_col and (r.get(cnt_col) or 0) < min_n:
                continue
            vals.append((v, r))
        if len(vals) < 2:
            continue
        nums = [v for v, _ in vals]
        if max(nums) == min(nums):
            continue
        best = (max if bigger else min)(vals, key=lambda x: x[0])
        n = best[1].get(cnt_col) if cnt_col else None
        out.append({
            "label": label, "lane": best[1]["lane"], "racer_nm": best[1]["racer_nm"],
            "value": f"{best[0]:g}{unit}",
            "sample": f"{int(n)}회" if n else "",
        })
    return out


def recent_form(runners: List[Dict]) -> List[Dict]:
    """선수별 최근 8경주 착순. 표가 아니라 눈으로 읽는 띠로 만든다."""
    out = []
    for r in sorted(runners, key=lambda x: x["pred_rank"] or 99):
        ranks = [r.get(f"recent{i}") for i in range(1, 9)]
        ranks = [int(v) for v in ranks if v is not None]
        out.append({
            "lane": r["lane"], "racer_nm": r["racer_nm"], "mark": r.get("mark"),
            "grade": r.get("racer_grd"),
            # 앞이 오래된 쪽이다. 최근이 오른쪽에 오도록 그대로 둔다.
            "ranks": ranks,
            "avg": (sum(ranks) / len(ranks)) if ranks else None,
        })
    return out


def race_picks(runners: List[Dict], n: int = 3) -> List[Dict]:
    """상위 n정. **결과와 무관하게** 항상 만든다.

    1순위만 적으면 목록에서 '이 경주를 어떻게 보는가'가 안 보인다. 승식은
    대부분 두세 정의 조합이라, 목록 단계에서 상위 3정이 함께 보여야 훑어보는
    사람이 경주를 고를 수 있다.

    적중 여부(outcome)와 분리해 두어야 '아직 안 달린 경주'와 '빗나간 경주'가
    같은 빈칸으로 보이지 않는다.
    """
    if not runners:
        return []
    ordered = sorted(runners, key=lambda x: x["pred_rank"] or 99)[:n]
    return [{"lane": r["lane"], "racer": r["racer_nm"], "mark": r.get("mark"),
             "p_win": r.get("p_win")} for r in ordered]


def race_outcome(runners: List[Dict]) -> Dict:
    """경주가 끝났으면 우리 예상이 어땠는지 한 줄로 요약한다."""
    if not any(r.get("ord") for r in runners):
        return {}
    by_rank = sorted(runners, key=lambda x: x["pred_rank"] or 99)
    top1 = by_rank[0]
    order = {r["pred_rank"]: r.get("ord") for r in by_rank}
    top3 = {order.get(i) for i in (1, 2, 3)} - {None}
    return {
        "top1_lane": top1["lane"],
        "top1_racer": top1["racer_nm"],
        "top1_ord": top1.get("ord"),
        "hit_win": top1.get("ord") == 1,
        "hit_place": bool(top1.get("ord")) and top1["ord"] <= 2,
        "hit_quinella": {order.get(1), order.get(2)} == {1, 2},
        "hit_trio": top3 == {1, 2, 3},
    }


# 자금 곡선에 쓸 계열. 여섯 규칙을 다 그리면 선이 엉켜 아무것도 안 읽힌다.
# 질문이 향하는 셋만 남긴다 — 정액(기준선), 마틴게일(실패하면 두 배),
# 손실회수형(이번에 맞히면 다 덮는 크기). 나머지는 아래 표에 그대로 있다.
CURVE_SERIES = ["정액", "마틴게일(2배)", "손실회수형"]
# dataviz 검증 통과 팔레트(카테고리 슬롯 1·2·3). 라이트 모드에서 aqua 가
# 대비 3:1 미만이라 **직접 라벨을 반드시 단다**(relief rule).
CURVE_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"]

CHART_W, CHART_H = 760, 300
PAD_L, PAD_R, PAD_T, PAD_B = 62, 96, 16, 34


def build_curve_chart(stage: Dict, start_bankroll: float) -> Dict:
    """자금 곡선을 SVG 좌표로 바꾼다.

    빌드 타임에 좌표를 확정한다 — 브라우저에서 계산할 이유가 없고, 자바스크립트가
    막힌 환경에서도 그림은 보여야 한다.
    """
    runs = {r["name"]: r for r in stage["runs"]}
    series = [runs[n] for n in CURVE_SERIES if n in runs and runs[n].get("curve")]
    if not series:
        return {}

    max_x = max(max(p[0] for p in s["curve"]) for s in series)
    max_y = max(start_bankroll, max(max(p[1] for p in s["curve"]) for s in series))
    max_x = max(max_x, 1)

    def sx(x): return PAD_L + (x / max_x) * (CHART_W - PAD_L - PAD_R)
    def sy(y): return CHART_H - PAD_B - (y / max_y) * (CHART_H - PAD_T - PAD_B)

    out = []
    for i, s in enumerate(series):
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in s["curve"])
        last = s["curve"][-1]
        out.append({
            "name": s["name"], "color": CURVE_COLORS[i % len(CURVE_COLORS)],
            "points": pts,
            "end_x": sx(last[0]), "end_y": sy(last[1]),
            "n_bets": s["n_bets"], "ruined": s["ruined"],
            "curve": s["curve"],
        })

    # 끝점 라벨을 겹치지 않게 **위로** 쌓는다. 파산한 계열은 모두 0원에서
    # 끝나 같은 자리에 모이는데, 아래로 밀면 가로축 눈금 글씨와 부딪힌다.
    floor = CHART_H - PAD_B - 4
    used: List[float] = []
    for ser in sorted(out, key=lambda r: (r["end_x"], r["end_y"])):
        y = min(ser["end_y"] + 4, floor)
        while any(abs(y - u) < 13 for u in used):
            y -= 13
        ser["label_y"] = max(y, PAD_T + 10)
        used.append(ser["label_y"])

    # 가로축 눈금 — 베팅 횟수
    xt = [0, max_x // 4, max_x // 2, (max_x * 3) // 4, max_x]
    yt = [0, max_y * 0.25, max_y * 0.5, max_y * 0.75, max_y]
    return {
        "w": CHART_W, "h": CHART_H, "series": out,
        "zero_y": sy(0), "start_y": sy(start_bankroll),
        "xticks": [{"x": sx(v), "label": f"{int(v):,}"} for v in xt],
        "yticks": [{"y": sy(v), "label": f"{int(v/10000):,}만"} for v in yt],
        "max_x": max_x, "max_y": max_y,
        "plot": {"l": PAD_L, "r": CHART_W - PAD_R, "t": PAD_T, "b": CHART_H - PAD_B},
    }


def load_metrics(path: Path = Path("models/metrics.json")) -> Dict:
    """학습·검증 규모를 화면에 그대로 노출한다.

    적중률만 크게 적고 표본 수와 검증 방법을 빼면 스스로를 속이게 된다.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    wf = raw.get("walk_forward") or {}
    out = {
        "trained_races": raw.get("trained_races"),
        "trained_rows": raw.get("trained_rows"),
        "n_features": raw.get("n_features"),
        "date_max": raw.get("date_max"),
        "verified_races": wf.get("n_races"),
        "hit_win": wf.get("top1_win"),
        "hit_place": wf.get("top1_top2"),
        "roi_win": wf.get("roi_win"),
        "lane1_win": wf.get("lane1_win"),
        "lane1_roi": wf.get("lane1_roi"),
        "form_win": wf.get("form_win"),
        "auc": wf.get("auc"),
        "logloss": wf.get("logloss"),
    }
    if out.get("hit_win") is not None and out.get("lane1_win") is not None:
        out["edge_win"] = out["hit_win"] - out["lane1_win"]
    return out


# ---------------------------------------------------------------------------
# 빌드
# ---------------------------------------------------------------------------

def asset_versions(static_dir: Path = STATIC_DIR) -> Dict[str, str]:
    """정적 파일의 내용 해시.

    ``style.css?v=ab12cd34`` 처럼 붙여 브라우저가 옛 파일을 계속 쓰지 못하게
    한다. 이게 없으면 CSS 를 고쳐도 화면이 안 바뀌고, 고치는 쪽은 코드가
    틀렸다고 착각하며 엉뚱한 곳을 파게 된다.
    """
    import hashlib

    out: Dict[str, str] = {}
    if not static_dir.exists():
        return out
    for f in static_dir.iterdir():
        if f.is_file():
            out[f.name] = hashlib.sha1(f.read_bytes()).hexdigest()[:8]
    return out


def build(db: str, out: Path, cfg: Dict) -> None:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True, lstrip_blocks=True,
    )
    env.filters["pct"] = lambda v, d=1: "—" if v is None else f"{v * 100:.{d}f}%"
    env.filters["num"] = lambda v, d=1: "—" if v is None else f"{v:,.{d}f}"
    env.filters["date_ko"] = fmt_date

    out.mkdir(parents=True, exist_ok=True)
    if STATIC_DIR.exists():
        shutil.copytree(STATIC_DIR, out / "static", dirs_exist_ok=True)
    # GitHub Pages 는 기본으로 Jekyll 을 돌린다. 이 파일이 없으면 밑줄로
    # 시작하는 이름이 조용히 무시된다 — 지금은 해당 없지만 나중에 하나만
    # 생겨도 원인을 찾기 어려운 실종이 된다.
    (out / ".nojekyll").write_text("", encoding="utf-8")
    # 커스텀 도메인은 **배포 산출물 안에** CNAME 이 있어야 유지된다. Actions 로
    # 배포하면 저장소 설정만으로는 매 배포마다 도메인이 풀릴 수 있다.
    host = urlparse(cfg.get("site", {}).get("url") or "").hostname
    if host:
        (out / "CNAME").write_text(host + "\n", encoding="utf-8")

    bcfg = cfg.get("build", {})
    # 배포 기준 경로. 환경변수가 설정을 이긴다 — 같은 소스로 로컬(루트)과
    # 배포(/저장소명)를 모두 구울 수 있어야 한다.
    base = (os.environ.get("BOATAI_BASE")
            or cfg.get("site", {}).get("base_path") or "").rstrip("/")
    ctx_base = {
        "site": cfg.get("site", {}),
        "built_at": now_kst().strftime("%Y-%m-%d %H:%M"),
        "mark_meaning": MARK_MEANING,
        "star_threshold": MARK_THRESHOLDS["star"],
        "version_label": VERSION_LABEL,
        "bet_order": BET_ORDER,
        "assets": asset_versions(),
        "base": base,
    }

    with session(db) as conn:
        metrics = load_metrics()
        live = load_races(conn, LIVE_VERSION)
        oos = load_races(conn, OOS_VERSION)

        today = today_kst().strftime("%Y%m%d")
        # 다가올 경주는 **가까운 날 · 이른 경주** 순이다. 곧 발주할 것이 위에 와야
        # 쓸모가 있다. (결과 목록은 최신 날짜가 위, 그 안에서는 1R 부터.)
        upcoming = sorted([r for r in live if not r["has_result"]],
                          key=lambda r: (r["race_ymd"] or "", r["race_no"] or 0)
                          )[: bcfg.get("upcoming_limit", 60)]
        # 결과가 나온 실전 경주가 먼저, 모자라면 모의 기록으로 채운다. 둘은
        # 화면에서 배지로 구분되므로 섞여도 오해가 없다.
        finished = [r for r in live if r["has_result"]]
        finished += [r for r in oos if r["has_result"]]
        finished = finished[: bcfg.get("results_limit", 300)]

        # ── 경주 상세 ────────────────────────────────────────────
        detail_targets = upcoming + finished[: bcfg.get("past_races", 400)]
        seen = set()
        race_pages: List[Dict] = []
        for r in detail_targets:
            if r["race_key"] in seen:
                continue
            seen.add(r["race_key"])
            runners = load_runners(conn, r["race_key"], r["version"])
            if not runners:
                continue
            page = dict(r)
            page.update({
                "runners": runners,
                "combos": betting_combos(runners),
                "payoffs": load_payoffs(conn, r["race_key"]),
                "picks": race_picks(runners),
                "outcome": race_outcome(runners),
                "sim": load_simulation(conn, r["race_key"], runners),
                "focus": focus_points(runners),
                "form": recent_form(runners),
            })
            race_pages.append(page)
            _write(out / "race" / r["race_key"] / "index.html",
                   env.get_template("race.html").render(
                       race=page, page_url=f"/race/{r['race_key']}/", **ctx_base))

        by_key = {p["race_key"]: p for p in race_pages}
        for lst in (upcoming, finished):
            for r in lst:
                p = by_key.get(r["race_key"])
                r["picks"] = p.get("picks") if p else []
                r["outcome"] = p.get("outcome") if p else {}
                r["has_page"] = p is not None

        # ── 경주일 ───────────────────────────────────────────────
        days: Dict[str, List[Dict]] = {}
        for r in upcoming + finished:
            if r["race_ymd"]:
                days.setdefault(r["race_ymd"], []).append(r)
        for ymd, rows in days.items():
            rows = sorted(rows, key=lambda x: x["race_no"] or 0)
            _write(out / "day" / ymd / "index.html",
                   env.get_template("day.html").render(
                       ymd=ymd, date_label=fmt_date(ymd), races=rows,
                       page_url=f"/day/{ymd}/", **ctx_base))

        # ── 검증 ─────────────────────────────────────────────────
        reports = {v: build_report(conn, v) for v in (LIVE_VERSION, OOS_VERSION)}

        # 고배당 카드에서 상세 페이지로 갈 수 있는 것만 링크한다. 오래된
        # 경주는 상세를 굽지 않으므로(build.past_races) 링크가 깨진다.
        have_page = {p["race_key"] for p in race_pages}
        for rep in reports.values():
            for group in (rep.get("highlights") or {}).values():
                for h in group:
                    h["has_page"] = h["race_key"] in have_page

        # ── 베팅 전략 ────────────────────────────────────────────
        strat = strategy_report(conn, OOS_VERSION)
        if not strat.get("empty"):
            for stage in strat["stages"]:
                stage["chart"] = build_curve_chart(stage, strat["start_bankroll"])

    _write(out / "strategy" / "index.html", env.get_template("strategy.html").render(
        s=strat, page_url="/strategy/", **ctx_base))

    _write(out / "index.html", env.get_template("index.html").render(
        upcoming=upcoming, finished=finished[:20], metrics=metrics,
        today=today, day_list=sorted(days, reverse=True)[:12],
        highlights=(reports.get(OOS_VERSION) or {}).get("highlights"),
        reports=reports, page_url="/", **ctx_base))

    _write(out / "results" / "index.html", env.get_template("results.html").render(
        races=finished, page_url="/results/", **ctx_base))

    _write(out / "accuracy" / "index.html", env.get_template("accuracy.html").render(
        reports=reports, metrics=metrics, min_sample=bcfg.get("min_sample", 30),
        page_url="/accuracy/", **ctx_base))

    log.info("경주 상세 %d개 · 경주일 %d개 · 다가올 %d경주 · 결과 %d경주",
             len(race_pages), len(days), len(upcoming), len(finished))
    log.info("빌드 완료 → %s", out)


def _write(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="정적 사이트 생성")
    ap.add_argument("--db", default="data/boatai.sqlite")
    ap.add_argument("--out", default="dist")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    except OSError:
        cfg = {}
    build(args.db, Path(args.out), cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
