"""예상 기호.

기호 규칙은 화면(site)과 검증(verify) 양쪽이 똑같이 써야 한다. 한쪽에만 두면
언젠가 갈리고, 그때 '기호별 실적'은 화면에서 벌어지는 일과 무관한 숫자가 된다.
그래서 규칙은 여기 한 곳에만 둔다.

경정은 여섯 척이 고정이라 경마보다 자리가 적다. 상위 네 자리에만 기호를 주고
나머지 둘은 비운다 — 여섯 중 다섯에 기호가 붙으면 기호가 아무것도 가리지 않는다.
"""

from __future__ import annotations

from typing import Dict, List

# 기본        ◎ ○ ▲ △
# 우세 뚜렷    ★ ○ ▲ △
#
# 자리를 고정하면 기호는 '절대적 약속'이 아니라 **경주 안에서의 상대 순위**가
# 된다. 절대 신호는 ★ 하나가 맡는다.
MARK_SEQUENCE = ["◎", "○", "▲", "△"]
MARK_SEQUENCE_STAR = ["★", "○", "▲", "△"]
MARK_LIMIT = len(MARK_SEQUENCE)

# ★ 기준은 손으로 정하지 않고 과거 기록에서 보정했다.
# (v1-oos 워크포워드 15,958경주 · 무작위는 1착 16.7% / 2착이내 33.3%,
#  1순위 전체 평균은 1착 47.8% / 2착이내 70.6%)
#
#   기준   ★출현   ★1착   ★2착이내
#   0.62    38%   60.3%    81.2%
#   0.65    28%   63.3%    83.3%
#   0.68    20%   66.8%    85.5%
#   0.70    15%   68.7%    86.3%   ← 채택
#   0.72    11%   71.3%    87.2%
#   0.75     7%   73.5%    88.3%
#
# 0.70 이 곡선의 무릎이다. 더 올리면 1착이 2.6%p 오르는 대신 출현이 15%→11% 로
# 줄어 하루 17경주에서 2~3개가 1~2개가 되고, 내리면 흔해져 '우세가 뚜렷'이
# 무색해진다. 15% 면 경주일마다 두세 번 나온다 — 드물지도 흔하지도 않다.
MARK_THRESHOLDS = {"star": 0.70}

# 기호 자체가 표기이므로 화면에 이름은 붙이지 않는다. 다만 처음 보는 사람을 위해
# '무엇을 뜻하는가'는 범례로 남긴다 — 이름이 아니라 뜻이다.
MARK_MEANING = {
    "★": "우세가 뚜렷한 축",
    "◎": "축",
    "○": "상대",
    "▲": "복병",
    "△": "참고",
}


def assign_marks(runners: List[Dict]) -> None:
    """예상 기호를 붙인다 (제자리 수정).

    자리 수가 고정이므로 예측 순위대로 배분한다. 1순위가 2착 이내에 들 확률이
    충분히 높으면 ◎ 대신 ★ 를 준다 — 접전 경주와 한 척이 압도하는 경주를 같은
    기호로 적으면, 읽는 쪽은 둘을 구분할 방법이 없다.
    """
    ordered = sorted(runners, key=lambda r: r.get("pred_rank") or 99)
    top = ordered[0] if ordered else None
    star = bool(top and (top.get("p_top2") or 0.0) >= MARK_THRESHOLDS["star"])
    seq = MARK_SEQUENCE_STAR if star else MARK_SEQUENCE

    for i, r in enumerate(ordered):
        mark = seq[i] if i < MARK_LIMIT else ""
        r["mark"] = mark
        r["mark_meaning"] = MARK_MEANING.get(mark, "")
