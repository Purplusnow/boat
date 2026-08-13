"""전개 시뮬레이션의 물리 상수를 실제 결과로 보정한다.

시뮬레이션에는 실측 자료를 못 구한 상수가 넷 있다 — 선회 간격(LANE_W),
탈출 속도가 직선에서 얼마나 시간으로 바뀌는지(EXIT_GAIN), 스타트 편차,
기량 편차. 이 값들을 손으로 정하면 전개는 '그럴듯한 이야기'일 뿐이다.

그래서 **코스별 1착 비율**을 기준으로 맞춘다. 실제 경정에서 1코스는 36%,
6코스는 4% 를 이긴다. 시뮬레이션이 같은 경주들을 돌렸을 때 이 분포를
재현하지 못하면 물리 상수가 틀린 것이다.

    python tools/calibrate_sim.py --races 400
"""

from __future__ import annotations

import argparse
import itertools
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boatai import simulate as sm  # noqa: E402
from boatai.site import load_runners  # noqa: E402

# 2020년 이후 실측 (54,542행)
EMPIRICAL = {1: 0.361, 2: 0.232, 3: 0.162, 4: 0.125, 5: 0.078, 6: 0.041}


def sample_races(conn, n: int, version: str = "v1-oos"):
    rows = conn.execute(
        "SELECT DISTINCT race_key FROM predictions WHERE model_version=? "
        "ORDER BY race_key DESC LIMIT ?", (version, n)).fetchall()
    return [r[0] for r in rows]


def lane_win_rates(conn, keys, version, n_sims=300):
    tally = {i: 0.0 for i in range(1, 7)}
    total = 0
    for key in keys:
        runners = load_runners(conn, key, version)
        if len(runners) < 6:
            continue
        sim = sm.simulate(runners, n_sims=n_sims)
        if not sim:
            continue
        for b in sim["per_boat"]:
            tally[b["lane"]] = tally.get(b["lane"], 0.0) + b["sim_win"]
        total += 1
    return {k: v / total for k, v in tally.items()}, total


def error(rates) -> float:
    """실측과의 절대 오차 합. 코스별 비율이라 백분위 차이를 그대로 더한다."""
    return sum(abs(rates.get(k, 0) - v) for k, v in EMPIRICAL.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/boatai.sqlite")
    ap.add_argument("--races", type=int, default=300)
    ap.add_argument("--version", default="v1-oos")
    ap.add_argument("--sims", type=int, default=250)
    ap.add_argument("--grid", action="store_true", help="상수 격자 탐색")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    keys = sample_races(conn, args.races, args.version)
    print(f"표본 {len(keys)}경주 · 판당 {args.sims}회 반복\n")

    if not args.grid:
        rates, n = lane_win_rates(conn, keys, args.version, args.sims)
        print(f"{'코스':>4}{'시뮬':>9}{'실측':>9}{'차이':>9}")
        for k in range(1, 7):
            print(f"{k:>4}{rates[k]:>9.1%}{EMPIRICAL[k]:>9.1%}"
                  f"{rates[k]-EMPIRICAL[k]:>+9.1%}")
        print(f"\n절대오차 합 {error(rates):.3f}  (경주 {n}개)")
        return 0

    # ── 격자 탐색 ────────────────────────────────────────────────
    best = None
    grid = itertools.product(
        (1.6, 2.2, 3.0),        # LANE_W
        (6.0, 10.0, 16.0),      # EXIT_GAIN
        (0.055, 0.075, 0.095),  # ST_SIGMA
        (0.045, 0.065),         # PACE_SIGMA
    )
    print(f"{'LANE_W':>8}{'EXIT':>7}{'ST_σ':>8}{'PACE_σ':>8}{'오차':>8}   코스별")
    for lw, eg, sts, ps in grid:
        sm.LANE_W, sm.EXIT_GAIN = lw, eg
        sm.ST_SIGMA, sm.PACE_SIGMA = sts, ps
        rates, _ = lane_win_rates(conn, keys, args.version, args.sims)
        e = error(rates)
        mark = ""
        if best is None or e < best[0]:
            best = (e, lw, eg, sts, ps, rates)
            mark = "  ←"
        print(f"{lw:>8.1f}{eg:>7.1f}{sts:>8.3f}{ps:>8.3f}{e:>8.3f}   "
              + " ".join(f"{rates[k]:.0%}" for k in range(1, 7)) + mark)

    e, lw, eg, sts, ps, rates = best
    print(f"\n최적: LANE_W={lw} EXIT_GAIN={eg} ST_SIGMA={sts} PACE_SIGMA={ps}  오차 {e:.3f}")
    print(f"{'코스':>4}{'시뮬':>9}{'실측':>9}")
    for k in range(1, 7):
        print(f"{k:>4}{rates[k]:>9.1%}{EMPIRICAL[k]:>9.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
