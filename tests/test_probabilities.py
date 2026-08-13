"""확률 산출 회귀 테스트.

세 확률(1착 / 2착 이내 / 3착 이내)은 **포개진 사건**이다. 독립 모델 셋이
따로 내놓은 값을 경주 안에서 각각 정규화하면 이 순서가 뒤집힐 수 있고,
실제로 화면에 '2착이내 84.0% · 3착이내 81.4%' 가 나갔다. 숫자를 확률로
읽으려면 지켜져야 하는 관계이므로 기계가 지키게 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boatai import model as md  # noqa: E402


class _Stub:
    """지정한 확률을 그대로 돌려주는 가짜 분류기."""

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.probs, self.probs])


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "race_key": ["R1"] * 6,
        "lane": list(range(1, 7)),
        "age": [30] * 6,
    })


def test_probabilities_stay_ordered():
    df = _frame()
    # 일부러 뒤집어 준다: 1번 정은 top3 원값이 top2 보다 작다.
    models = {
        "win": _Stub([0.60, 0.10, 0.10, 0.08, 0.07, 0.05]),
        "top2": _Stub([0.80, 0.30, 0.30, 0.25, 0.20, 0.15]),
        "top3": _Stub([0.20, 0.60, 0.60, 0.55, 0.50, 0.45]),
    }
    out = md.predict_frame(models, df, ["age"])
    assert (out["p_win_norm"] <= out["p_top2_norm"] + 1e-9).all()
    assert (out["p_top2_norm"] <= out["p_top3_norm"] + 1e-9).all()


def test_win_probabilities_sum_to_one_within_race():
    df = _frame()
    models = {"win": _Stub([0.6, 0.1, 0.1, 0.08, 0.07, 0.05]),
              "top2": _Stub([0.8] * 6), "top3": _Stub([0.9] * 6)}
    out = md.predict_frame(models, df, ["age"])
    assert abs(out["p_win_norm"].sum() - 1.0) < 1e-9


def test_pred_rank_follows_win_probability():
    df = _frame()
    models = {"win": _Stub([0.05, 0.07, 0.08, 0.10, 0.10, 0.60]),
              "top2": _Stub([0.3] * 6), "top3": _Stub([0.5] * 6)}
    out = md.predict_frame(models, df, ["age"]).sort_values("pred_rank")
    # 가장 확률이 높은 6번 정이 1순위여야 한다.
    assert int(out.iloc[0]["lane"]) == 6
    assert list(out["pred_rank"]) == [1, 2, 3, 4, 5, 6]
