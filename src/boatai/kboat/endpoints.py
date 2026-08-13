"""경정 오픈API 엔드포인트 레지스트리.

경로·파라미터는 data.go.kr 데이터셋 페이지에 임베드된 Swagger 명세에서 그대로
받아 ``config/openapi/<데이터셋번호>.json`` 에 보관했다. 추측한 값이 하나도
없으므로 ``NO_OPENAPI_SERVICE_ERROR`` 는 여기 적힌 경로를 쓰는 한 나지 않는다.

명세를 다시 받고 싶으면::

    python -m boatai.kboat.spec refresh

승인 여부와 실제 응답 필드는 ``probe`` 로 확인한다 — 명세에 적힌 필드와 실제
응답이 다른 경우가 있고(경정 API 는 특히 회차/일차 라벨이 어긋나 있다),
파서를 필드명 추측 없이 쓰려면 실물을 봐야 한다.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

RESOLVED_PATH = Path(os.environ.get("BOATAI_ENDPOINTS", "config/endpoints.resolved.json"))

# 경정은 미사리 한 곳에서만 시행한다. 경마의 '경마장' 축이 아예 없다는 뜻이라
# 지역별 편차를 걱정할 필요가 없는 대신, 표본을 나눌 축도 그만큼 적다.
VENUE = "미사리"

# 정번(艇番)은 1~6 로 고정이고 곧 코스이기도 하다. 경마의 게이트와 달리
# **1코스의 우위가 압도적**이라(인코스 선회), 코스는 가장 강한 단일 피처다.
LANES = (1, 2, 3, 4, 5, 6)


@dataclass
class Endpoint:
    """하나의 오픈API 오퍼레이션."""

    key: str                    # 내부 식별자
    title: str                  # 한글 명칭
    service: str                # GW 서비스 세그먼트 (SRVC_...)
    operation: str              # 오퍼레이션 세그먼트 (TODZ_...)
    dataset_pk: str             # data.go.kr 데이터셋 번호
    required: List[str] = field(default_factory=list)   # serviceKey 외 필수 파라미터
    optional: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def path(self) -> str:
        """BASE 이후의 전체 경로. GW 는 서비스 세그먼트를 반드시 요구한다."""
        return f"{self.service}/{self.operation}"


# ---------------------------------------------------------------------------
# 레지스트리
#
# 예측 파이프라인에 실제로 쓰이는 것부터 적는다. 뒤쪽 콘텐츠용 API 는 승인만
# 받아 두고 수집은 하지 않아도 된다 — 일일 호출량은 유한하다.
# ---------------------------------------------------------------------------

RACE_CARD = Endpoint(
    key="race_card",
    title="출주표",
    service="SRVC_OD_API_VWEB_MBR_RACE_INFO",
    operation="TODZ_API_VWEB_MBR_RACE_I",
    dataset_pk="15107808",
    optional=["stnd_yr", "week_tcnt", "day_tcnt", "race_no"],
    note=(
        "**예측 피처의 근간.** 43개 필드로 경주 전에 확정되는 정보를 거의 다 담는다. "
        "선수(등급·연령·체중·평균착순·연대율·평균ST·FL횟수·평균사고점), "
        "모터(2연대율·3연대율·평균착순점), 보트(연대율), 그리고 결정적으로 "
        "**코스별 6개월 연대율**(mm_6_1~6_race_high_rank_ratio)까지 있다. "
        "경정에서 코스는 곧 정번이고 1코스 우위가 압도적이라, 이 값이 사실상 "
        "'이 선수가 이 자리에서 얼마나 하는가'를 직접 알려준다."
    ),
)

RACE_CARD_WEB = Endpoint(
    key="race_card_web",
    title="홈페이지 출주표 정보",
    service="SRVC_OD_API_VWEB_MBR_RACE_DOC",
    operation="TODZ_API_VWEB_MBR_RACE_DOC_I",
    dataset_pk="15107797",
    required=["stnd_yr"],
    optional=["week_tcnt", "day_tcnt", "organ_stat_cd", "race_no",
              "race_reg_no", "race_ymd"],
    note=(
        "출주표의 32개 필드 판. race_ymd(경주일자)와 racer_perio_no(기수)가 있어 "
        "**회차·일차를 실제 날짜로 잇는 다리**로 쓴다. 다른 API 들은 날짜 없이 "
        "연도/회차/일차만 주므로 이것이 없으면 달력에 붙일 수가 없다."
    ),
)

RACE_RESULT = Endpoint(
    key="race_result",
    title="경주결과",
    service="SRVC_OD_API_MBR_RACE_RESULT",
    operation="TODZ_API_MBR_RACE_RESULT_I",
    dataset_pk="15107847",
    optional=["stnd_yr", "day_tcnt", "week_tcnt", "race_no"],
    note=(
        "1~3위 선수와 승식별 배당. **6명 전원의 착순은 없고 3위까지만** 준다 — "
        "학습 레이블로는 1·2·3착 여부까지만 만들 수 있다는 뜻이다. "
        "전 착순이 필요하면 race_rank(경주결과순위)를 함께 쓴다."
    ),
)

RACE_RANK = Endpoint(
    key="race_rank",
    title="경주결과순위",
    service="SRVC_MRA_RACE_RANK",
    operation="TODZ_MRA_RACE_RANK",
    dataset_pk="15143984",
    optional=["stnd_year", "tms", "day_ord", "race_no", "race_day",
              "racer_no", "racer_nm"],
    note=(
        "선수 단위 착순. 파라미터 이름이 다른 API 와 **어긋난다** "
        "(stnd_year·tms·day_ord ↔ stnd_yr·week_tcnt·day_tcnt). 같은 기관 API 인데 "
        "규칙이 갈리므로 호출부에서 반드시 변환한다. 6정 전원 착순이 여기 있으면 "
        "레이블 품질이 올라가므로 프로브로 실제 필드를 먼저 확인한다."
    ),
)

PAYOFF = Endpoint(
    key="payoff",
    title="배당률",
    service="SRVC_OD_API_MBR_PAYOFF",
    operation="TODZ_API_MBR_PAYOFF_I",
    dataset_pk="15107811",
    required=["stnd_yr", "week_tcnt", "day_tcnt", "race_no"],
    note=(
        "승식별 확정배당(단승·연승1·연승2·쌍승·복승·삼복승). **경주 하나씩만** "
        "조회된다(4개 파라미터가 전부 필수) — 하루 12~16경주면 회차당 수십 회 "
        "호출이라 일일 한도 계산에 반드시 넣어야 한다. "
        "우리 추천 조합이 실제로 얼마를 돌려줬는지 검증하는 유일한 근거다."
    ),
)

RACER_TMS = Endpoint(
    key="racer_tms",
    title="선수 회차별 성적",
    service="SRVC_OD_API_VWEB_MBR_RACER_TMS_INFO",
    operation="TODZ_API_VWEB_RACER_TMS_I",
    dataset_pk="15107804",
    optional=["stnd_yr", "week_tcnt", "racer_no", "racer_nm"],
    note=(
        "선수의 회차 단위 누적 성적(1~6위 횟수, 평균착순, 평균ST, 승률·연대율). "
        "**회차별 스냅샷이라 시점을 지켜 쓸 수 있다** — 해당 회차 이전 값만 쓰면 "
        "누수 없이 '그때까지의 폼'을 만들 수 있다."
    ),
)

MOTOR_INFO = Endpoint(
    key="motor_info",
    title="모터정보",
    service="SRVC_OD_API_VWEB_MBR_MOTOR_INFO",
    operation="todz_api_vweb_motor_i",
    dataset_pk="15107812",
    required=["stnd_yr", "motor_no"],
    note=(
        "모터 단위 통산 성적. 경정은 모터를 추첨으로 배정하고 회차 내내 쓰므로 "
        "**모터 성능이 선수 기량만큼 결과를 가른다**. 다만 모터번호 하나씩만 "
        "조회되므로(필수 파라미터), 출주표에 이미 실린 mot_high_rank_ratio 로 "
        "대부분 갈음하고 이 API 는 보강용으로만 쓴다."
    ),
)

BOAT_INFO = Endpoint(
    key="boat_info",
    title="보트정보",
    service="SRVC_OD_API_VWEB_MBR_BOAT_INFO",
    operation="todz_api_vweb_mbr_boat_i",
    dataset_pk="15107810",
    required=["stnd_yr", "boat_no"],
    note="보트 단위 통산 성적. 모터보다 영향이 작다. 출주표 값으로 대개 충분하다.",
)

RACER_INFO = Endpoint(
    key="racer_info",
    title="선수정보",
    service="SRVC_VWEB_MBR_RACER_INFO",
    operation="TODZ_VWEB_MBR_RACER_INFO",
    dataset_pk="15107809",
    optional=["stnd_yr", "racer_nm", "racer_perio_no", "pageNo", "numOfRows"],
    note="선수 마스터(기수·신상). 선수 페이지 표시용이며 예측 기여는 작다.",
)

FL_INFO = Endpoint(
    key="fl_info",
    title="회차별 출발위반 현황",
    service="SRVC_MRA_FL_INFO",
    operation="TODZ_MRA_FL_INFO",
    dataset_pk="15128711",
    optional=["stnd_year", "tms", "day_ord", "race_no", "startrace_day",
              "endrace_day", "racer_nm"],
    note=(
        "플라잉(F)·출발지연(L) 이력. 경정에서 사전출발 위반은 **실격**이라 "
        "결과를 직접 뒤집는다. 위반이 잦은 선수는 스타트를 늦춰 잡는 경향이 있어 "
        "승률 자체에도 영향이 있다. 파라미터 규칙이 race_rank 계열과 같다."
    ),
)

RACER_START = Endpoint(
    key="racer_start",
    title="선수별 정상출발 현황",
    service="SRVC_MRA_RACER_STRT",
    operation="TODZ_MRA_RACER_STRT",
    dataset_pk="15139305",
    optional=["stnd_yr", "racer_no", "racer_nm"],
    note="선수별 정상출발 비율. FL 이력의 반대편 지표로 함께 본다.",
)

COURSE_WIN = Endpoint(
    key="course_win",
    title="코스별 우승전법",
    service="SRVC_MRA_COURSE_WIN",
    operation="TODZ_MRA_COURSE_WIN",
    dataset_pk="15125101",
    optional=["stnd_yr", "entry_course"],
    note=(
        "코스별 우승 전법(인빠지기·휘감기 등) 분포. 개별 경주 예측보다 "
        "**해설·사전 확률의 근거**로 쓸모가 있다. 코스 우위의 크기를 연도별로 "
        "확인할 수 있어 모델이 배운 것이 상식과 맞는지 대조하기도 좋다."
    ),
)

RACER_RECORD = Endpoint(
    key="racer_record",
    title="선수 상대전적",
    service="SRVC_MRA_RACER_RECORD",
    operation="TODZ_MRA_RACER_RECORD",
    dataset_pk="15125109",
    optional=["stnd_yr", "racer_nm1"],
    note="선수 간 상대전적. 표본이 얕아 피처보다 콘텐츠에 어울린다.",
)

RACER_WIN_RANK = Endpoint(
    key="racer_win_rank",
    title="선수 다승순위",
    service="SRVC_TODZ_API_MRA_RACER_WIN_RANK_I",
    operation="TODZ_API_MRA_RACER_WIN_RANK_I",
    dataset_pk="15107851",
    optional=["racer_rank", "stnd_yr", "racer_nm"],
    note="연도별 다승 순위. 신인·복귀 선수의 사전(prior) 값으로 쓸 수 있다.",
)

OPERATION = Endpoint(
    key="tilt",
    title="틸트각 정보 (운영정보)",
    service="SRVC_OD_API_MRA_SUPP_CD",
    operation="TODZ_API_MRA_RACER_TILT_I",
    dataset_pk="15107867",
    optional=["stnd_yr", "week_tcnt", "day_tcnt", "racer_no", "race_no"],
    note=(
        "경주별 틸트각·자켓중량·부과량. 틸트를 세우면 직진이 좋아지고 선회가 "
        "불안해진다 — **선수가 그 경주를 어떻게 타려는지 드러내는 설정값**이라 "
        "공개 데이터 중 작전에 가장 가깝다. 경주 전에 공개되면 강력한 피처다. "
        "프로브에서 '언제부터 값이 실리는지'를 반드시 확인하라(발주 후 공개면 "
        "학습에는 쓸 수 있어도 예측에는 못 쓴다)."
    ),
)

INTERVIEW = Endpoint(
    key="interview",
    title="출주선수 면담 (운영정보)",
    service="SRVC_OD_API_MRA_SUPP_CD",
    operation="TODZ_API_MAR_RACER_INTERVIEW_I",
    dataset_pk="15107867",
    optional=["stnd_yr", "week_tcnt", "racer_no"],
    note="선수 건강·훈련 상태 코멘트. 텍스트라 피처화는 나중 일이지만 해설에 좋다.",
)

PRIZE_STANDARD = Endpoint(
    key="prize_standard",
    title="상금지급기준",
    service="SRVC_TODZ_VW_API_MRA_PRIZ_STND_I",
    operation="TODZ_API_MRA_PRIZ_STND_I",
    dataset_pk="15107852",
    optional=["stnd_yr", "race_kind_nm"],
    note="경주 등급별 상금. 경주 격을 나타내는 보조 축.",
)

GRAND_PRIX = Endpoint(
    key="grand_prix",
    title="대상경정 연도별 순위",
    service="SRVC_OD_API_WEB_BOAT_GRND_PRIZE",
    operation="todz_api_web_grnd_prize_i",
    dataset_pk="15107786",
    optional=["stnd_yr"],
    note="대상경주 역대 순위. 콘텐츠용.",
)

RACE_VIDEO = Endpoint(
    key="race_video",
    title="경주영상",
    service="SRVC_OD_API_WEB_BOAT_RACE_VIDEO",
    operation="TODZ_API_MRA_BOAT_RACE_I",
    dataset_pk="15107792",
    optional=["stnd_yr", "week_tcnt", "day_tcnt"],
    note="경주 다시보기 URL. race_ymd 가 있어 회차↔날짜 대조에도 쓸 수 있다.",
)

REGISTRY: Dict[str, Endpoint] = {
    e.key: e
    for e in (
        # 예측 파이프라인 핵심
        RACE_CARD, RACE_CARD_WEB, RACE_RESULT, RACE_RANK, PAYOFF,
        # 보강
        RACER_TMS, MOTOR_INFO, BOAT_INFO, RACER_INFO,
        FL_INFO, RACER_START, OPERATION, INTERVIEW,
        # 콘텐츠·참고
        COURSE_WIN, RACER_RECORD, RACER_WIN_RANK, PRIZE_STANDARD,
        GRAND_PRIX, RACE_VIDEO,
    )
}

# 이것들이 없으면 파이프라인이 성립하지 않는다.
#   출주표 = 피처, 경주결과 = 레이블, 배당률 = 검증
REQUIRED_KEYS = ["race_card", "race_result", "payoff"]

# 파라미터 이름 규칙이 두 갈래다. 한쪽 이름으로 통일해 쓰고 호출 직전에 바꾼다.
# (이걸 호출부마다 손으로 맞추면 언젠가 한 곳을 빠뜨리고, 그 API 만 조용히
#  0건을 돌려준다 — 파라미터가 전부 '옵션'이라 오류조차 나지 않는다.)
ALT_PARAM_NAMES = {
    "stnd_yr": "stnd_year",
    "week_tcnt": "tms",
    "day_tcnt": "day_ord",
    "race_ymd": "race_day",
}
ALT_PARAM_KEYS = {"race_rank", "fl_info"}

# 2026-08-13 실측으로 확인한 승인 완료 API. 미승인 API 는 오류 코드가 아니라
# **HTTP 403** 으로 돌아온다. 승인 목록을 코드에 적어 두는 이유는, 미승인 API 를
# 수집 대상에 넣어 두면 매 실행마다 403 을 맞으면서도 파이프라인은 성공으로
# 끝나 '자료가 원래 없는 것'처럼 보이기 때문이다.
APPROVED = {
    "race_card",       # 출주표 — 피처
    "race_result",     # 경주결과 — 1~3착 + 승식 배당
    "race_rank",       # 경주결과순위 — 전 착순 + 실제 경주일자 (2002~)
    "payoff",          # 배당률 — 경주 단위 확정배당
    "racer_tms",       # 선수 회차별 성적
    "racer_win_rank",  # 선수 다승순위
}

# 경주번호는 **2자리 0채움 문자열**이어야 한다. "1" 을 주면 오류 없이 0건이
# 돌아온다 — 가장 찾기 어려운 종류의 실패라, 파라미터를 만드는 이 한 곳에서
# 강제한다. (실측: race_no="01" → 1건, race_no="1"/1 → 0건)
ZERO_PAD_2 = ("race_no",)


def to_api_params(key: str, params: Dict[str, object]) -> Dict[str, object]:
    """내부 표준 파라미터명·표기를 해당 API 가 받는 형태로 바꾼다."""
    out = dict(params)
    for name in ZERO_PAD_2:
        v = out.get(name)
        if v not in (None, ""):
            out[name] = f"{int(v):02d}" if str(v).strip().isdigit() else v
    if key in ALT_PARAM_KEYS:
        out = {ALT_PARAM_NAMES.get(k, k): v for k, v in out.items()}
    return out


# ---------------------------------------------------------------------------
# 프로브 결과 캐시
# ---------------------------------------------------------------------------

def load_resolved(path: Optional[Path] = None) -> Dict[str, str]:
    p = path or RESOLVED_PATH
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("paths", {})
        except (ValueError, OSError) as e:
            log.warning("확정 경로 캐시를 읽지 못했습니다 (%s): %s", p, e)
    return {}


def save_resolved(paths: Dict[str, str], meta: Optional[dict] = None,
                  path: Optional[Path] = None) -> Path:
    p = path or RESOLVED_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"paths": paths, "meta": meta or {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


def resolve(key: str) -> str:
    """호출 경로. 명세에서 그대로 왔으므로 후보를 두지 않는다."""
    return REGISTRY[key].path
