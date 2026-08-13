"""예상 생성과 **확정 저장**.

적중률 숫자를 나중에 믿을 수 있으려면, 예측이 결과를 본 뒤에 바뀌지 않았다는
보장이 있어야 한다. 그래서 규칙은 하나다: **발주 시각이 지난 경주의 예측은
절대 다시 쓰지 않는다.** 모델을 새로 학습해도, 피처 파싱을 고쳐도 마찬가지다.

    python -m boatai.predict upcoming          # 아직 안 달린 경주
    python -m boatai.predict backfill          # 과거 경주 (워크포워드, 사후)

``upcoming`` 과 ``backfill`` 은 model_version 을 다르게 남긴다. 전자는 발주 전에
만든 **실전 기록**이고 후자는 사후에 만든 **모의 기록**이라, 둘을 한 숫자로
합치면 그 숫자는 아무 것도 뜻하지 않는다. verify 가 둘을 갈라서 집계한다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sqlite3
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

from .clock import now_kst
from .features import build_frame, feature_columns
from .kboat.store import session, upsert
from .model import MODEL_VERSION, fit, load, predict_frame

log = logging.getLogger(__name__)

# 사후에 만든 모의 예측임을 이름에 박아 둔다. 표에서 실전 기록과 나란히 놓였을
# 때 무엇인지 바로 보이지 않으면 언젠가 섞인다.
OOS_VERSION = f"{MODEL_VERSION}-oos"


def _post_datetime(row) -> Optional[dt.datetime]:
    """경주일자 + 발주시각 → datetime. 하나라도 없으면 None."""
    ymd, hhmm = row.get("race_ymd"), row.get("post_time")
    if not ymd or not hhmm:
        return None
    try:
        return dt.datetime.strptime(f"{str(ymd)[:8]} {str(hhmm)[:5]}", "%Y%m%d %H:%M")
    except ValueError:
        return None


def frozen_keys(conn: sqlite3.Connection, version: str) -> set:
    return {r[0] for r in conn.execute(
        "SELECT DISTINCT race_key FROM predictions WHERE model_version = ?", (version,))}


def _write(conn: sqlite3.Connection, pred: pd.DataFrame, version: str) -> int:
    rows = [{
        "race_key": r.race_key, "lane": int(r.lane), "racer_nm": r.racer_nm,
        "p_win": float(r.p_win_norm),
        "p_top2": None if pd.isna(r.p_top2_norm) else float(r.p_top2_norm),
        "p_top3": None if pd.isna(r.p_top3_norm) else float(r.p_top3_norm),
        "pred_rank": int(r.pred_rank), "model_version": version,
    } for r in pred.itertuples()]
    n = upsert(conn, "predictions", rows, ["race_key", "lane", "model_version"])
    conn.commit()
    return n


def predict_upcoming(conn: sqlite3.Connection, db_path: str) -> int:
    """아직 발주하지 않은 경주에 예측을 만든다.

    이미 예측이 있는 경주는 건드리지 않는다. 발주가 지난 경주도 만들지 않는다 —
    출주표는 나중에도 조회되므로, 막지 않으면 '결과를 아는 상태에서 만든 예측'이
    실전 기록에 섞여 들어간다.
    """
    bundle = load()
    models, cols = bundle["models"], bundle["features"]

    df = build_frame(conn, with_labels=False)
    if df.empty:
        log.info("출주표가 없습니다.")
        return 0

    now = now_kst()
    have = frozen_keys(conn, MODEL_VERSION)

    rows = df.to_dict("records")
    keep_keys = set()
    for r in rows:
        if r["race_key"] in have:
            continue
        if r.get("has_result"):
            continue
        pt = _post_datetime(r)
        # 발주 시각을 모르면 예측하지 않는다. 모른 채 만들면 그것이 발주 전에
        # 만들어졌다고 주장할 근거가 없다.
        if pt is None or pt <= now:
            continue
        keep_keys.add(r["race_key"])

    if not keep_keys:
        log.info("예측할 경주가 없습니다 (발주 전 미예측 경주 0개).")
        # 예측은 이미 있는데 전개 시뮬레이션만 없는 경우가 있다 — 기능이 나중에
        # 생겼기 때문이다. **아직 발주 전인 경주에 한해서만** 채운다. 발주가
        # 지난 경주에 지금 만들어 넣으면 그것은 사후 생성이다.
        pending = {r["race_key"] for r in rows
                   if r["race_key"] in have and not r.get("has_result")
                   and (_post_datetime(r) or now) > now}
        todo = [k for k in pending if not _has_simulation(conn, k)]
        if todo:
            sub = df[df["race_key"].isin(todo)].copy()
            sub["p_win_norm"] = sub.get("p_win_norm", np.nan)
            n = _freeze_simulations(conn, _attach_stored_probs(conn, sub))
            log.info("전개 시뮬레이션 %d경주 추가 저장", n)
        return 0

    target = df[df["race_key"].isin(keep_keys)].copy()
    pred = predict_frame(models, target, cols)
    n = _write(conn, pred, MODEL_VERSION)
    n_sim = _freeze_simulations(conn, pred)
    log.info("예상 %d행 / %d경주 확정 저장 (전개 시뮬레이션 %d경주)", n, len(keep_keys), n_sim)
    return n


def _has_simulation(conn: sqlite3.Connection, race_key: str) -> bool:
    return conn.execute("SELECT 1 FROM simulations WHERE race_key=?",
                        (race_key,)).fetchone() is not None


def _attach_stored_probs(conn: sqlite3.Connection, df: pd.DataFrame) -> pd.DataFrame:
    """**저장된** 예측 확률을 붙인다.

    시뮬레이션은 예측 확률에 닻을 내린다. 모델을 다시 돌려 확률을 새로 만들면
    이미 저장된 예상과 어긋난 전개가 나온다 — 화면의 순위와 전개가 다른 말을
    하게 된다. 그래서 DB 에 있는 그 값을 그대로 쓴다.
    """
    stored = pd.read_sql_query(
        "SELECT race_key, lane, p_win AS p_win_norm FROM predictions "
        "WHERE model_version = ?", conn, params=[MODEL_VERSION])
    return df.drop(columns=["p_win_norm"], errors="ignore").merge(
        stored, on=["race_key", "lane"], how="inner")


def _freeze_simulations(conn: sqlite3.Connection, pred: pd.DataFrame) -> int:
    """전개 시뮬레이션을 예측과 **같은 시점에** 고정한다.

    나중에 다시 돌리면 그때의 상수와 난수로 다른 전개가 나온다. 발주 전에
    화면에 있던 전개와 결과를 본 뒤의 전개가 다르면, 그 전개는 아무것도
    증명하지 못한다.
    """
    from .simulate import simulate as run_sim

    done = 0
    for key, g in pred.groupby("race_key"):
        runners = g.to_dict("records")
        for r in runners:
            # 시뮬레이션은 정규화된 확률을 닻으로 쓴다. 컬럼명을 맞춰 준다.
            r["p_win"] = r.get("p_win_norm", r.get("p_win"))
            lane = r.get("lane")
            r["own_course_rate"] = r.get(f"course{int(lane)}_high_rate") if lane else None
        sim = run_sim(runners)
        if not sim:
            continue
        conn.execute(
            "INSERT INTO simulations(race_key,payload,conf_label,conf_score,"
            "top_tactic,n_sims) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(race_key) DO NOTHING",
            (key, json.dumps(sim, ensure_ascii=False, default=float),
             sim["confidence"]["label"], sim["confidence"]["score"],
             sim["top_tactic"], sim["n_sims"]))
        done += 1
    conn.commit()
    return done


def predict_backfill(conn: sqlite3.Connection, n_folds: int = 5,
                     min_train_races: int = 2000) -> int:
    """과거 경주에 **워크포워드 표본외** 예측을 남긴다.

    각 구간의 예측은 그 구간 이전 자료만으로 학습한 모델이 만든다. 사후에
    만들었다는 점은 변하지 않지만, 적어도 자기 결과를 보고 만든 것은 아니다.
    승식별 회수율처럼 표본이 많아야 의미가 생기는 지표를 지금 당장 보기 위한
    용도이고, 실전 기록이 쌓이면 그쪽으로 대체된다.
    """
    from .features import build_training_frame

    df = build_training_frame(conn)
    if df.empty:
        log.info("학습 데이터가 없습니다.")
        return 0
    cols = feature_columns(df)
    df = df.sort_values("order_key").reset_index(drop=True)
    keys = df["order_key"].dropna().sort_values().unique()
    splits = np.linspace(len(keys) * 0.5, len(keys), n_folds + 1).astype(int)

    total = 0
    for i in range(n_folds):
        cut = keys[splits[i] - 1]
        end = keys[min(splits[i + 1] - 1, len(keys) - 1)]
        train = df[df["order_key"] <= cut]
        test = df[(df["order_key"] > cut) & (df["order_key"] <= end)]
        if train["race_key"].nunique() < min_train_races or test.empty:
            continue
        models = fit(train, cols)
        if "win" not in models:
            continue
        pred = predict_frame(models, test, cols)
        total += _write(conn, pred, OOS_VERSION)
        log.info("fold %d → %d경주 예측 기록", i + 1, test["race_key"].nunique())
    return total


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="경정 예측 생성")
    ap.add_argument("command", choices=["upcoming", "backfill"])
    ap.add_argument("--db", default="data/boatai.sqlite")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with session(args.db) as conn:
        if args.command == "upcoming":
            try:
                predict_upcoming(conn, args.db)
            except FileNotFoundError as e:
                print(f"✗ {e}", file=sys.stderr)
                return 1
        else:
            predict_backfill(conn, n_folds=args.folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
