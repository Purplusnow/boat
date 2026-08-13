"""실전 베팅 전략 시뮬레이션 — 베팅금 조절과 선택 규칙.

"동일 금액으로는 답이 없다면, 실패 뒤에 베팅금을 올리면 되지 않나?" 라는 질문에
숫자로 답하기 위한 모듈이다. 결론을 먼저 적어 둔다. **베팅금 조절은 기대값을
바꾸지 못한다.**

    E[손익] = Σ (베팅금ᵢ × (배당ᵢ × 적중ᵢ − 1))

베팅금은 각 항에 곱해지는 양수일 뿐이라, 개별 베팅의 기대값이 음수면 어떤
가중치를 줘도 합은 음수다. 마틴게일이 이겨 보이는 이유는 **작은 이익을 자주,
큰 손실을 드물게** 만들어 손실을 시야 밖으로 밀어내기 때문이지 기대값이
좋아져서가 아니다.

기대값을 바꿀 수 있는 것은 **금액이 아니라 선택**이다 — 어떤 경주에, 어떤
승식에 거느냐. 그래서 이 모듈은 둘을 나눠 다룬다.

  * ``STAKING`` — 베팅금 규칙. 기대값은 그대로, 분포만 바뀐다는 것을 보인다.
  * ``SELECTIONS`` — 선택 규칙. 실제로 회수율을 올린다(70% → 96%). 다만
    실측 결과 **어떤 규칙도 100% 를 넘지 못했다.**

경정에는 실전에서 더 큰 장벽이 하나 더 있다. **발주 전 배당률이 공개되지
않는다.** 회수 베팅의 금액은 원래 "이번에 맞히면 지금까지의 손실을 덮는" 크기여야
하는데, 배당을 모르면 그 크기를 계산할 수가 없다. 여기서는 과거 평균 배당으로
추정해 시뮬레이션하지만, 실전에서는 그 추정조차 사후에만 검증된다.

    python -m boatai.strategy --db data/boatai.sqlite
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .verify import load_payoffs, load_verified, race_level

log = logging.getLogger(__name__)

# 6정 편성에서 각 승식의 전체 조합 수. '모두 사면 반드시 맞는다' 는 성질로
# 승식별 실제 환급률을 구하는 데 쓴다.
POOL_COMBOS = {"단승": 6, "연승": 3, "쌍승": 30, "복승": 15, "삼복승": 20}

START_BANKROLL = 1_000_000.0   # 시드 100만원
BASE_STAKE = 10_000.0          # 기본 베팅 1만원

# **1회 구매 한도 10만원.** 경정은 한 경주·한 승식에 이 금액을 넘겨 살 수 없다.
# 이 한 줄이 회수 베팅의 결론을 바꾼다 — 마틴게일은 불리한 것이 아니라
# **애초에 성립하지 않는다.** 기본 1만원에서 두 배씩 올리면 네 번째 베팅이
# 8만원, 다섯 번째가 16만원이라 한도를 넘는다. 즉 4연패 뒤에는 회수에 필요한
# 금액을 걸 수 없고, 그때까지의 손실은 그대로 확정된다.
MAX_STAKE = 100_000.0


# ---------------------------------------------------------------------------
# 베팅 대상 만들기
# ---------------------------------------------------------------------------

def load_bets(conn: sqlite3.Connection, version: str = "v1-oos") -> pd.DataFrame:
    """경주 순서대로 정렬된 베팅 단위 표.

    한 행이 '한 경주에 한 승식을 한 통 산 것'이다. 배당 자료가 없는 경주는
    verify 단계에서 이미 빠져 있다 — 자료가 없는 것을 불발로 세면 성적이
    자료 결손만큼 나빠진다.
    """
    df = load_verified(conn, version)
    rl = race_level(df, load_payoffs(conn))
    if rl.empty:
        return pd.DataFrame()

    p1 = df[df.pred_rank == 1][["race_key", "p_win", "p_top2", "lane", "racer_grd"]]
    p2 = df[df.pred_rank == 2][["race_key", "p_win"]].rename(columns={"p_win": "p2_win"})
    rl = rl.merge(p1, on="race_key").merge(p2, on="race_key")
    rl["gap"] = rl["p_win"] - rl["p2_win"]

    rows: List[Dict] = []
    for r in rl.itertuples():
        for pool, b in (r.bets or {}).items():
            if pool not in POOL_COMBOS:
                continue
            rows.append({
                "race_key": r.race_key, "date": r.race_date, "race_no": r.race_no,
                "pool": pool, "tickets": b["cost"], "payout": b["payout"],
                "hit": bool(b["hit"]),
                "p_win": r.p_win, "p_top2": r.p_top2, "gap": r.gap,
                "lane": r.lane, "grd": r.racer_grd,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # 시간순이어야 한다. 연속 손실·회수 베팅은 순서에 의존하므로, 정렬이
    # 틀리면 시뮬레이션 결과가 통째로 무의미해진다.
    return out.sort_values(["date", "race_no"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 선택 규칙 — 기대값을 바꿀 수 있는 유일한 손잡이
# ---------------------------------------------------------------------------

SELECTIONS: Dict[str, Dict] = {
    "단승-전체": {"pool": "단승", "desc": "매 경주 1순위 단승"},
    "연승-전체": {"pool": "연승", "desc": "매 경주 1순위 연승"},
    "연승-고신뢰": {"pool": "연승", "p_min": 0.55,
                "desc": "1순위 승률 55% 이상일 때만 연승"},
    "연승-고신뢰-1코스": {"pool": "연승", "p_min": 0.55, "lanes": (1,),
                    "desc": "1순위가 1코스이고 승률 55% 이상일 때만 연승"},
    "단승-고신뢰": {"pool": "단승", "p_min": 0.55,
                "desc": "1순위 승률 55% 이상일 때만 단승"},
    "삼복승-전체": {"pool": "삼복승", "desc": "매 경주 상위 3정 삼복승"},
}


def select(bets: pd.DataFrame, rule: Dict) -> pd.DataFrame:
    m = bets["pool"] == rule["pool"]
    if rule.get("p_min"):
        m &= bets["p_win"] >= rule["p_min"]
    if rule.get("lanes"):
        m &= bets["lane"].isin(rule["lanes"])
    return bets[m].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 베팅금 규칙
# ---------------------------------------------------------------------------
#
# 각 함수는 (연속손실수, 직전적중여부, 현재자금, 누적손실액, 평균배당) 을 받아
# 이번 베팅 금액을 돌려준다. 기대값을 바꾸지 못한다는 것을 보이는 게 목적이므로
# 실제로 쓰이는 방식들을 그대로 구현한다.

def _flat(streak, last_hit, bankroll, drawdown, avg_payout):
    return BASE_STAKE


def _martingale_x2(streak, last_hit, bankroll, drawdown, avg_payout):
    """실패할 때마다 두 배.

    배당이 2배를 넘어야 성립하는 방식인데, 우리 1순위 단승의 평균 적중 배당은
    1.8배다. 두 배씩 올려도 한 번의 적중이 그때까지의 손실을 못 덮는다.
    """
    return BASE_STAKE * (2 ** min(streak, 30))


def _martingale_recover(streak, last_hit, bankroll, drawdown, avg_payout):
    """이번에 맞히면 지금까지의 손실 + 기본이익을 덮는 크기.

    회수 베팅의 정석이다. 필요한 금액은 ``(누적손실 + 목표이익) / (배당 − 1)``
    인데, **경정은 발주 전 배당이 공개되지 않으므로 이 값을 실전에서 계산할 수
    없다.** 여기서는 과거 평균 배당으로 추정한다 — 즉 실전보다 유리한 조건이다.
    """
    if streak == 0:
        return BASE_STAKE
    need = (drawdown + BASE_STAKE) / max(avg_payout - 1.0, 0.05)
    return min(need, bankroll)


def _dalembert(streak, last_hit, bankroll, drawdown, avg_payout):
    """실패하면 한 단위 더, 성공하면 한 단위 덜. 마틴게일의 완만한 판."""
    return BASE_STAKE * (1 + min(streak, 50))


def _anti_martingale(streak, last_hit, bankroll, drawdown, avg_payout):
    """성공했을 때 올린다(흐름 타기). 손실은 작게, 이익은 크게 — 라는 주장."""
    return BASE_STAKE if not last_hit else BASE_STAKE * 2


def _proportional(streak, last_hit, bankroll, drawdown, avg_payout):
    """자금의 일정 비율(2%). 파산하지 않는 대신 회복도 느리다."""
    return max(bankroll * 0.02, 1000.0)


STAKING: Dict[str, Callable] = {
    "정액": _flat,
    "마틴게일(2배)": _martingale_x2,
    "손실회수형": _martingale_recover,
    "달랑베르": _dalembert,
    "역마틴게일": _anti_martingale,
    "정률(2%)": _proportional,
}


# ---------------------------------------------------------------------------
# 시뮬레이션
# ---------------------------------------------------------------------------

def simulate(bets: pd.DataFrame, plan: Callable, *,
             bankroll: float = START_BANKROLL) -> Dict:
    """자금 곡선을 그린다.

    **파산하면 거기서 멈춘다.** 자금이 0 이 된 뒤의 베팅을 계속 세면 마틴게일이
    실제보다 좋아 보인다 — 현실에서는 그 시점에 게임이 끝난다.
    """
    if bets.empty:
        return {}
    # 적중이 하나도 없으면 평균 배당이 NaN 이다. ``NaN or 2.0`` 은 NaN 이 참이라
    # 그대로 통과해 자금 계산 전체를 NaN 으로 오염시킨다 — 값이 틀리는 게 아니라
    # 라운딩에서 터진다. 명시적으로 확인한다.
    hit_payouts = bets.loc[bets["hit"], "payout"]
    avg_payout = float(hit_payouts.mean()) if len(hit_payouts) else 2.0
    if not np.isfinite(avg_payout):
        avg_payout = 2.0

    cash = bankroll
    peak = bankroll
    streak = 0            # 연속 실패 수
    last_hit = False
    drawdown = 0.0        # 마지막 적중 이후 누적 손실액
    max_dd = 0.0
    max_streak = 0
    max_stake = 0.0
    n_bet = 0
    ruined_at = None
    curve = []

    for i, r in enumerate(bets.itertuples()):
        stake = plan(streak, last_hit, cash, drawdown, avg_payout)
        # 규정 한도와 보유 자금, 둘 다 넘길 수 없다.
        stake = min(stake, MAX_STAKE, cash)
        if stake < 1000:                  # 최소 베팅 미만이면 게임 종료
            ruined_at = i
            break

        cash -= stake
        n_bet += 1
        max_stake = max(max_stake, stake)
        if r.hit:
            cash += stake * r.payout
            streak = 0
            drawdown = 0.0
            last_hit = True
        else:
            streak += 1
            max_streak = max(max_streak, streak)
            drawdown += stake
            last_hit = False

        peak = max(peak, cash)
        max_dd = max(max_dd, (peak - cash) / peak if peak else 0.0)
        curve.append((i + 1, round(cash)))
        if cash < 1000:
            ruined_at = i
            break

    staked_total = float(bets.iloc[:n_bet]["tickets"].sum()) if n_bet else 0
    return {
        "n_bets": n_bet,
        "n_available": len(bets),
        "final": round(cash),
        "profit": round(cash - bankroll),
        "return_pct": cash / bankroll,
        "max_drawdown": max_dd,
        "max_losing_streak": max_streak,
        "max_stake": round(max_stake),
        "ruined": ruined_at is not None,
        "ruined_at": ruined_at,
        "hit_rate": float(bets.iloc[:n_bet]["hit"].mean()) if n_bet else None,
        "curve": _thin(curve),
        "_staked": staked_total,
    }


def _thin(curve: List[tuple], limit: int = 240) -> List[tuple]:
    """자금 곡선을 그릴 만한 점 수로 줄인다.

    **마지막 점은 반드시 남긴다** — 파산 지점이 곡선의 요점인데, 균등 추출로
    끝을 잘라 버리면 그래프가 '아직 돈이 남은 채' 끝나 보인다.
    """
    if len(curve) <= limit:
        return curve
    step = len(curve) / limit
    out = [curve[int(i * step)] for i in range(limit)]
    if out[-1] != curve[-1]:
        out.append(curve[-1])
    return out


def losing_streaks(bets: pd.DataFrame) -> Dict:
    """연속 실패의 실제 분포. 마틴게일에 필요한 자금을 가늠하는 근거다."""
    if bets.empty:
        return {}
    streaks, cur = [], 0
    for hit in bets["hit"]:
        if hit:
            if cur:
                streaks.append(cur)
            cur = 0
        else:
            cur += 1
    if cur:
        streaks.append(cur)
    if not streaks:
        return {}
    s = np.array(streaks)
    longest = int(s.max())
    # 규정 한도 안에서 두 배씩 올릴 수 있는 횟수. 1만 → 2 → 4 → 8만까지가
    # 한계이고 그다음(16만)은 살 수 없다.
    doublings = 0
    while BASE_STAKE * 2 ** (doublings + 1) <= MAX_STAKE:
        doublings += 1
    return {
        "longest": longest,
        "p90": int(np.percentile(s, 90)),
        "n_streaks": len(s),
        # 한도 안에서 버틸 수 있는 연패 수와, 그때까지 쌓이는 손실
        "max_doublings": doublings,
        "limit_streak": doublings + 1,
        "loss_at_limit": round(BASE_STAKE * (2 ** (doublings + 1) - 1)),
        "over_limit_races": int((s > doublings + 1).sum()),
    }


def payback_rates(conn: sqlite3.Connection) -> List[Dict]:
    """승식별 실제 환급률 — '모든 조합을 다 산 경우'의 회수율.

    이 값이 곧 넘어야 할 벽이다. 우리 회수율이 이 값보다 높다면 모델이 실력을
    보탠 것이고, 그래도 100% 에 못 미친다면 그 차이가 공제율이다.
    """
    out = []
    for pool, n in POOL_COMBOS.items():
        if pool == "연승":
            # 연승은 1착·2착 두 통이 맞는다. 6통 사서 2통 적중.
            r1 = conn.execute("SELECT AVG(payout) FROM payoffs WHERE pool='연승1'").fetchone()[0]
            r2 = conn.execute("SELECT AVG(payout) FROM payoffs WHERE pool='연승2'").fetchone()[0]
            if r1 and r2:
                out.append({"pool": "연승", "combos": 6, "avg_payout": (r1 + r2) / 2,
                            "payback": (r1 + r2) / 6})
            continue
        row = conn.execute(
            "SELECT AVG(payout) a, COUNT(*) n FROM payoffs WHERE pool=? AND payout IS NOT NULL",
            (pool,)).fetchone()
        if row and row[0]:
            out.append({"pool": pool, "combos": n, "avg_payout": row[0],
                        "payback": row[0] / n})
    return out


def selection_table(bets: pd.DataFrame, split_year: int = 2023) -> List[Dict]:
    """선택 규칙별 회수율을 **탐색 기간과 검증 기간으로 나눠** 낸다.

    한 기간에서 규칙을 여럿 시험하면 우연히 좋아 보이는 것이 반드시 나온다.
    나눠서 보지 않으면 그 우연을 발견으로 착각한다.
    """
    out = []
    for name, rule in SELECTIONS.items():
        sub = select(bets, rule)
        if sub.empty:
            continue
        tr = sub[sub["date"].dt.year < split_year]
        ho = sub[sub["date"].dt.year >= split_year]

        def _roi(g):
            return float(g["payout"].sum() / g["tickets"].sum()) if len(g) and g["tickets"].sum() else None

        out.append({
            "name": name, "desc": rule["desc"], "pool": rule["pool"],
            "n": len(sub), "roi": _roi(sub), "hit_rate": float(sub["hit"].mean()),
            "n_train": len(tr), "roi_train": _roi(tr),
            "n_test": len(ho), "roi_test": _roi(ho),
        })
    return sorted(out, key=lambda r: -(r["roi"] or 0))


def build_report(conn: sqlite3.Connection, version: str = "v1-oos") -> Dict:
    bets = load_bets(conn, version)
    if bets.empty:
        return {"empty": True}

    selections = selection_table(bets)
    # 베팅금 규칙은 **가장 좋은 선택 규칙 위에서** 비교한다. 나쁜 선택 위에
    # 얹으면 '금액 규칙이 문제였나' 하는 여지가 남는다.
    best = max(selections, key=lambda r: r["roi"] or 0)
    base = select(bets, SELECTIONS[best["name"]])

    # 베팅금 규칙을 **두 무대**에서 돌린다.
    #
    #   ① 가장 좋은 선택 규칙 — 적중률이 높아 연패가 짧다. 회수 베팅이 가장
    #      유리한 조건이며, 그런데도 지는지 본다.
    #   ② 매 경주 단승 — 적중률 47.8% 라 연패가 길다. 질문자가 떠올리는
    #      "실패하면 올린다" 는 바로 이 무대다. 여기서 무슨 일이 벌어지는지가
    #      마틴게일의 실제 모습이다.
    stages = []
    for label, sub in (("최적 선택 (%s)" % best["name"], base),
                       ("매 경주 단승", select(bets, SELECTIONS["단승-전체"]))):
        runs = []
        for name, plan in STAKING.items():
            s = simulate(sub, plan)
            s["name"] = name
            s["doc"] = (plan.__doc__ or "").strip().split("\n")[0]
            runs.append(s)
        stages.append({
            "label": label,
            "n_available": len(sub),
            "hit_rate": float(sub["hit"].mean()),
            "flat_roi": float(sub["payout"].sum() / sub["tickets"].sum()),
            "runs": runs,
            "streaks": losing_streaks(sub),
        })

    return {
        "version": version,
        "period": [str(bets["date"].min())[:10], str(bets["date"].max())[:10]],
        "n_races": int(bets["race_key"].nunique()),
        "payback": payback_rates(conn),
        "selections": selections,
        "best_selection": best,
        "stages": stages,
        "staking": stages[0]["runs"],
        "streaks": stages[0]["streaks"],
        "start_bankroll": START_BANKROLL,
        "base_stake": BASE_STAKE,
        "split_year": 2023,
    }


def report_text(rep: Dict) -> str:
    if rep.get("empty"):
        return "베팅 판정이 가능한 예측이 없습니다."
    L = [f"실전 베팅 전략 검토  {rep['period'][0]} ~ {rep['period'][1]}  "
         f"{rep['n_races']:,}경주", "=" * 66, "", "■ 승식별 실제 환급률 (모든 조합을 다 산 경우)"]
    for p in rep["payback"]:
        L.append(f"    {p['pool']:<5} 평균배당 {p['avg_payout']:>7.2f}배 / {p['combos']:>2}통"
                 f"  →  {p['payback']:>6.1%}")
    L += ["", "■ 선택 규칙별 회수율 (탐색 ~2022 / 검증 2023~)"]
    L.append(f"    {'규칙':<18}{'베팅':>7}{'적중률':>8}{'전체':>8}{'탐색':>8}{'검증':>8}")
    for s in rep["selections"]:
        f = lambda v: "   n/a" if v is None else f"{v:6.1%}"  # noqa: E731
        L.append(f"    {s['name']:<18}{s['n']:>7,}{s['hit_rate']:>8.1%}"
                 f"{f(s['roi']):>8}{f(s['roi_train']):>8}{f(s['roi_test']):>8}")

    L += ["", f"■ 베팅금 규칙  (시드 {rep['start_bankroll']:,.0f}원 · "
              f"기본 베팅 {rep['base_stake']:,.0f}원)"]
    for stage in rep["stages"]:
        st = stage["streaks"]
        L += ["", f"  ▸ {stage['label']} — {stage['n_available']:,}회 베팅 · "
                  f"적중률 {stage['hit_rate']:.1%} · 정액 회수율 {stage['flat_roi']:.1%}",
              f"    {'규칙':<16}{'베팅수':>8}{'최종자금':>14}{'수익률':>9}{'최대베팅':>14}{'결과':>8}"]
        for s in stage["runs"]:
            L.append(f"    {s['name']:<16}{s['n_bets']:>8,}{s['final']:>14,}"
                     f"{s['return_pct']:>9.1%}{s['max_stake']:>14,}"
                     f"{'파산' if s['ruined'] else '생존':>8}")
        if st:
            L.append(f"    최장 {st['longest']}연패 · 1회 한도 {MAX_STAKE:,.0f}원 안에서 "
                     f"2배 증액은 {st['limit_streak']}연패까지만 가능 "
                     f"(그 구간 손실 {st['loss_at_limit']:,}원) · "
                     f"한도를 넘는 연패 {st['over_limit_races']}회")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="베팅 전략 시뮬레이션")
    ap.add_argument("--db", default="data/boatai.sqlite")
    ap.add_argument("--version", default="v1-oos")
    ap.add_argument("--out", default="data/strategy.json")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        rep = build_report(conn, args.version)
    finally:
        conn.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str),
                              encoding="utf-8")
    print(report_text(rep))
    print(f"\n리포트 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
