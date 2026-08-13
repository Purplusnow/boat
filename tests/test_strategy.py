"""베팅 시뮬레이터 회귀 테스트.

시뮬레이터가 틀리면 결론이 통째로 뒤집힌다. 특히 **파산 뒤에도 베팅을 계속
세는 버그**는 마틴게일을 실제보다 좋아 보이게 만들고, 그 오류는 표에서
'생존'으로 조용히 나타난다. 그래서 손으로 계산할 수 있는 짧은 수열로 고정한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boatai import strategy as st  # noqa: E402


def _bets(seq, payout=2.0):
    """seq 는 적중 여부 리스트. 배당은 고정."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01"] * len(seq)),
        "race_no": range(len(seq)),
        "pool": ["단승"] * len(seq),
        "tickets": [1] * len(seq),
        "payout": [payout if h else 0.0 for h in seq],
        "hit": seq,
    })


def test_flat_bankroll_is_exact():
    # 3연패 뒤 1적중, 배당 2배, 기본 1만원.
    # 100만 − 4만(4회 베팅) + 2만(적중 환급) = 98만
    out = st.simulate(_bets([False, False, False, True]), st._flat)
    assert out["n_bets"] == 4
    assert out["final"] == 980_000


def test_simulation_stops_at_ruin():
    # 전부 실패. 100만 / 1만 = 100회에서 자금이 마르고 거기서 멈춰야 한다.
    out = st.simulate(_bets([False] * 500), st._flat)
    assert out["ruined"] is True
    assert out["n_bets"] == 100
    assert out["n_bets"] < out["n_available"]
    assert out["final"] == 0


def test_martingale_is_capped_by_the_purchase_limit():
    """1회 구매 한도 10만원을 넘겨 걸 수 없다.

    이 한도가 마틴게일의 결론을 바꾼다 — 불리한 정도가 아니라 **회수 자체가
    불가능**해진다. 1만원에서 두 배씩 가면 8만원이 한계이고 그다음은 못 산다.
    """
    out = st.simulate(_bets([False] * 6 + [True]), st._martingale_x2)
    assert out["max_stake"] == st.MAX_STAKE
    assert st.BASE_STAKE * 2 ** 3 <= st.MAX_STAKE < st.BASE_STAKE * 2 ** 4


def test_no_plan_ever_exceeds_the_purchase_limit():
    """어떤 베팅금 규칙도 규정 한도를 넘지 않아야 한다."""
    seq = [False] * 30 + [True] + [False] * 30
    for name, plan in st.STAKING.items():
        out = st.simulate(_bets(seq), plan)
        assert out["max_stake"] <= st.MAX_STAKE, f"{name} 이 한도를 넘겼다"


def test_martingale_ruins_faster_than_flat():
    """같은 수열에서 마틴게일이 정액보다 먼저 파산해야 한다.

    이 페이지의 핵심 주장이다. 뒤집히면 결론을 다시 써야 한다.
    """
    # 정액은 100회(=시드/기본베팅)를 버티므로 수열이 그보다 길어야 비교가 된다.
    seq = [False] * 200
    flat = st.simulate(_bets(seq), st._flat)
    mart = st.simulate(_bets(seq), st._martingale_x2)
    assert mart["ruined"] and flat["ruined"]
    assert mart["n_bets"] < flat["n_bets"]


def test_losing_streaks_counts_runs():
    b = _bets([False, False, True, False, False, False, True, False])
    s = st.losing_streaks(b)
    assert s["longest"] == 3
    assert s["n_streaks"] == 3
    # 1만원에서 두 배씩 → 2·4·8만원까지가 한도 안. 즉 4연패까지 버틴다.
    assert s["max_doublings"] == 3
    assert s["limit_streak"] == 4
    # 그 구간 누적 손실 = 1+2+4+8 = 15만원
    assert s["loss_at_limit"] == st.BASE_STAKE * 15
    assert s["over_limit_races"] == 0


def test_stake_never_exceeds_bankroll():
    # 회수형은 필요 금액이 자금을 넘을 수 있다. 없는 돈을 걸어선 안 된다.
    out = st.simulate(_bets([False] * 40), st._martingale_recover)
    assert out["max_stake"] <= st.START_BANKROLL
    assert out["final"] >= 0
