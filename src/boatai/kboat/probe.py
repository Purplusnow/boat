"""엔드포인트 프로브 — 승인 여부 확인과 실제 응답 필드 덤프.

경로는 명세에서 그대로 가져왔으므로 추측할 것이 없다. 대신 확인해야 할 것이
둘 있다.

1. **어느 API 가 실제로 승인됐나.** 활용신청은 API 단위이고 승인 반영에는
   시차가 있다. 미승인은 코드 20 으로 돌아오므로, 코드 12(경로 오류)와
   구분해 보여준다 — 둘을 뭉뚱그리면 '키가 안 된다'는 잘못된 결론으로 샌다.
2. **실제 응답 필드와 값의 생김새.** 경정 API 는 명세와 실물이 어긋난 곳이
   있다. 특히 ``week_tcnt``(회차)와 ``day_tcnt``(일차)는 같은 데이터셋
   안에서도 파라미터 설명과 응답 필드 설명이 서로 반대로 적혀 있다. 파서를
   쓰기 전에 실물로 확정해야 한다.

    python -m boatai.kboat.probe              # 전체 확인
    python -m boatai.kboat.probe --only race_card race_result
    python -m boatai.kboat.probe --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..clock import now_kst, today_kst
from .client import KboatApiError, KboatClient, _as_list, redact
from .endpoints import REGISTRY, REQUIRED_KEYS, save_resolved, to_api_params

log = logging.getLogger(__name__)

FIELD_DUMP = Path("config/api_fields.json")

# 회차 좌표를 아직 모를 때 쓰는 탐색 순서. 연도만 주면 대개 응답이 나오고,
# 거기서 실제 회차·일차를 읽어 나머지 API 에 넘긴다.
DISCOVERY_KEYS = ["race_result", "race_card", "race_card_web", "race_rank"]


def _year_candidates() -> List[str]:
    """올해부터 뒤로. 연초에는 올해 자료가 아직 없을 수 있다."""
    y = today_kst().year
    return [str(y), str(y - 1), str(y - 2)]


def _param_sets(key: str, ctx: Dict[str, object]) -> List[Dict[str, object]]:
    """이 엔드포인트에 시도해 볼 파라미터 조합.

    ``ctx`` 는 앞선 프로브에서 알아낸 실제 회차 좌표
    (stnd_yr / week_tcnt / day_tcnt / race_no / motor_no / boat_no / racer_no).
    """
    ep = REGISTRY[key]
    req, opt = set(ep.required), set(ep.optional)
    sets: List[Dict[str, object]] = []

    def add(d: Dict[str, object]) -> None:
        d = {k: v for k, v in d.items() if v not in (None, "")}
        if d not in sets:
            sets.append(d)

    years = [ctx.get("stnd_yr")] if ctx.get("stnd_yr") else _year_candidates()

    # 경주 하나를 특정해야 하는 API (배당률 등) — 좌표를 알아야만 부를 수 있다.
    if {"week_tcnt", "day_tcnt", "race_no"} <= req:
        for y in years:
            if ctx.get("week_tcnt"):
                add({"stnd_yr": y, "week_tcnt": ctx["week_tcnt"],
                     "day_tcnt": ctx["day_tcnt"], "race_no": ctx.get("race_no") or 1})
        return sets

    # 장비 번호가 필수인 API
    if "motor_no" in req or "boat_no" in req:
        num = ctx.get("motor_no") if "motor_no" in req else ctx.get("boat_no")
        for y in years:
            add({"stnd_yr": y, "motor_no" if "motor_no" in req else "boat_no": num or 1})
        return sets

    # 그 밖 — 좁은 조건부터 넓은 조건으로. 넓은 쪽이 먼저 성공하면 totalCount 가
    # 커서 페이지 순회 비용만 늘고, 좁은 쪽이 성공하면 회차 좌표를 바로 얻는다.
    for y in years:
        if ctx.get("week_tcnt") and {"week_tcnt", "day_tcnt"} & opt:
            add({"stnd_yr": y, "week_tcnt": ctx["week_tcnt"], "day_tcnt": ctx["day_tcnt"]})
        if "stnd_yr" in req or "stnd_yr" in opt:
            add({"stnd_yr": y})
    if not req:
        add({})
    return sets


def _learn(ctx: Dict[str, object], records: List[dict]) -> None:
    """응답에서 회차 좌표와 장비 번호를 배운다.

    가장 큰 회차를 고른다 — 최근 회차라야 다른 API 에도 자료가 있을 가능성이
    높다. 정렬 순서를 믿지 않고 값으로 고르는 것은, API 마다 정렬 기준이
    다르고 명시돼 있지도 않기 때문이다.
    """
    def _int(v):
        try:
            return int(str(v).strip())
        except (TypeError, ValueError):
            return None

    best = None
    for r in records:
        yr = _int(r.get("stnd_yr") or r.get("stnd_year"))
        wk = _int(r.get("week_tcnt") or r.get("tms"))
        dy = _int(r.get("day_tcnt") or r.get("day_ord"))
        if yr and wk and dy:
            cand = (yr, wk, dy, _int(r.get("race_no")) or 1)
            if best is None or cand[:3] > best[:3]:
                best = cand
    if best and not ctx.get("week_tcnt"):
        ctx["stnd_yr"], ctx["week_tcnt"], ctx["day_tcnt"], ctx["race_no"] = (
            str(best[0]), best[1], best[2], best[3])

    for r in records:
        for src, dst in (("motor_no", "motor_no"), ("boat_no", "boat_no"),
                         ("racer_no", "racer_no"), ("race_reg_no", "racer_no")):
            if not ctx.get(dst) and _int(r.get(src)):
                ctx[dst] = _int(r.get(src))
        # 회차↔날짜를 잇는 실제 경주일자. 달력에 붙이려면 이게 있어야 한다.
        if not ctx.get("race_ymd") and r.get("race_ymd"):
            ctx["race_ymd"] = str(r["race_ymd"]).strip()


def probe_one(client: KboatClient, key: str, ctx: Dict[str, object]) -> dict:
    ep = REGISTRY[key]
    attempts: List[dict] = []

    for params in _param_sets(key, ctx):
        try:
            body = client.raw(ep.path, to_api_params(key, params))
        except KboatApiError as e:
            attempts.append({"params": params, "error": f"[{e.code}] {e.msg}"})
            if e.fatal:
                # 권한·경로 문제는 파라미터를 바꿔도 똑같다. 더 두드리면
                # 일일 호출량만 태운다.
                return {"key": key, "ok": False, "fatal": True, "code": e.code,
                        "reason": str(e), "attempts": attempts}
            continue
        except Exception as e:  # noqa: BLE001
            attempts.append({"params": params, "error": redact(repr(e))})
            continue

        records = _as_list(body.get("items"))
        if records:
            _learn(ctx, records)
            return {
                "key": key, "ok": True, "path": ep.path, "sample_params": params,
                "total_count": body.get("totalCount"),
                "fields": sorted(records[0].keys()),
                "sample_record": records[0],
                "attempts": attempts,
            }
        attempts.append({"params": params,
                         "error": f"빈 응답 (totalCount={body.get('totalCount')})"})

    return {"key": key, "ok": False, "fatal": False, "code": "",
            "reason": "승인은 됐으나 레코드가 나오지 않음 (조회 조건 또는 자료 부재)",
            "attempts": attempts}


def probe_all(client: KboatClient, keys: Optional[List[str]] = None) -> dict:
    keys = keys or list(REGISTRY)
    # 회차 좌표를 먼저 알아내야 배당률처럼 좌표가 필수인 API 를 부를 수 있다.
    ordered = [k for k in DISCOVERY_KEYS if k in keys] + \
              [k for k in keys if k not in DISCOVERY_KEYS]
    ctx: Dict[str, object] = {}
    results = {}
    for key in ordered:
        log.info("프로브: %-16s %s", key, REGISTRY[key].title)
        results[key] = probe_one(client, key, ctx)
    results["_context"] = ctx
    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="경정 오픈API 엔드포인트 프로브")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 으로 출력")
    ap.add_argument("--only", nargs="*", help="특정 엔드포인트 키만 검사")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        client = KboatClient.from_env()
    except ValueError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2

    results = probe_all(client, args.only)
    ctx = results.pop("_context", {})

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        ok = [k for k, r in results.items() if r["ok"]]
        denied = [k for k, r in results.items() if r.get("code") in ("20", "401", "403")]
        badpath = [k for k, r in results.items() if r.get("code") == "12"]
        other = [k for k, r in results.items()
                 if not r["ok"] and k not in denied and k not in badpath]

        print(f"\n조회 좌표: {ctx or '(확보 실패)'}\n")
        for key, r in results.items():
            ep = REGISTRY[key]
            if r["ok"]:
                print(f"  ✓ {ep.title:<22} 필드 {len(r['fields']):>2}개  "
                      f"총 {r.get('total_count')}건  {r['sample_params']}")
            else:
                mark = "✗✗" if r.get("fatal") else "· "
                print(f"  {mark} {ep.title:<22} {r['reason']}")

        print(f"\n승인·응답 정상 {len(ok)}개")
        if denied:
            print(f"미승인(HTTP 403) {len(denied)}개: "
                  f"{', '.join(REGISTRY[k].title for k in denied)}")
            print("   → 활용신청을 하지 않았거나 승인 전입니다. "
                  "게이트웨이는 이 경우 오류 코드가 아니라 403 을 돌려줍니다.")
        if badpath:
            print(f"경로 오류(코드 12) {len(badpath)}개: "
                  f"{', '.join(REGISTRY[k].title for k in badpath)}")
        if other:
            print(f"응답 없음 {len(other)}개: "
                  f"{', '.join(REGISTRY[k].title for k in other)}")

    resolved = {k: r["path"] for k, r in results.items() if r["ok"]}
    if resolved:
        p = save_resolved(resolved, meta={
            "probed_at": now_kst().isoformat(timespec="seconds"),
            "context": ctx,
        })
        FIELD_DUMP.parent.mkdir(parents=True, exist_ok=True)
        FIELD_DUMP.write_text(json.dumps(
            {k: {"fields": r["fields"], "sample": r["sample_record"],
                 "params": r["sample_params"], "total": r.get("total_count")}
             for k, r in results.items() if r["ok"]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n확정 경로 → {p}\n응답 필드 덤프 → {FIELD_DUMP}")

    missing = [k for k in REQUIRED_KEYS if k not in resolved]
    if missing:
        print(f"\n필수 엔드포인트 미확보: "
              f"{', '.join(REGISTRY[k].title for k in missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
