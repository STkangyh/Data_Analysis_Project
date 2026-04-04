"""
submit_predictions_today.py
===========================
오늘(2026-04-04) 경기 홈팀 승리 확률을 예측하고 Statiz API에 제출합니다.

사용법:
    python submit_predictions_today.py            # 예측 + API 제출
    python submit_predictions_today.py --dry-run  # 예측만 출력, 실제 제출 안함

동작 흐름:
    1. 4월 일정 API 수집 → 4월 4일 경기 필터
    2. 각 경기 라인업 수집 (접근 불가 경기는 스킵)
    3. game_results_2023_2025.csv 기반 rolling stats 계산 (2026 3/28~4/3 포함)
    4. 저장된 XGBoost / LightGBM 모델로 앙상블 승리 확률 예측
    5. POST /prediction/savePrediction 으로 제출

특이사항:
    - 한화:두산 (14:00) 경기는 접근 가능
    - 라인업 미수집 경기는 스킵하지 않고 리그평균 대체로 예측 시도
    - 제출 실패 시 오류 코드/메시지 출력 후 다음 경기 진행
    - percent: 홈팀 앙상블 승리확률 (소수점 2자리, 예: 57.32)
"""

import argparse
import os
import sys
import time
import json
import hmac
import hashlib
import urllib.request
import urllib.parse
import warnings
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── 프로젝트 공통 모듈
from fetch_game_results import (
    StatizAPIClient, parse_game_record, parse_lineup_records,
    TEAM_CODE_MAP, STADIUM_CODE_MAP, REQUEST_DELAY,
)
from build_raw_data import (
    build_pitcher_lookup, build_bullpen_stats, build_throw_lookup,
    build_hitter_lookup, build_team_rs9_lookup,
    get_pitcher_features, get_hitter_features,
    PARK_FACTORS, DEFAULT_PARK_FACTOR,
    build_pitcher_stadium_lookup, build_hitter_stadium_lookup,
    get_pitcher_stadium_features, get_team_stadium_ops,
    STADIUM_NAME_TO_CODE, _normalize_stadium,
)
from baseball_baseline import load_models, predict_win_probability
from predict_2026 import (
    FOREIGN_NAME_KO_MAP,
    apply_foreign_pitcher_overrides,
    build_feature_rows,
)

load_dotenv()

# ──────────────────────────────────────────────
# 로깅
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR    = Path("data")
TARGET_DATE = "2026-04-04"


# ──────────────────────────────────────────────
# POST 지원 API 클라이언트 (StatizAPIClient 확장)
# ──────────────────────────────────────────────
class StatizAPIClientPlus(StatizAPIClient):
    """POST /prediction/savePrediction 지원을 추가한 확장 클라이언트."""

    def _sign_post(self, path: str, params: dict) -> tuple[str, str, str]:
        """POST 전용 서명 — payload에 'POST' 메서드명 사용, 파라미터는 쿼리스트링."""
        timestamp  = str(int(time.time()))
        safe       = "-_.!~*'()"
        normalized = "&".join(
            f"{quote(str(k), safe=safe)}={quote(str(v), safe=safe)}"
            for k, v in sorted(params.items())
        )
        payload   = f"POST|{path}|{normalized}|{timestamp}"
        signature = hmac.new(
            self.secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        logger.debug("POST payload: %s", payload)
        return timestamp, signature, normalized

    def post(self, path: str, params: dict = None) -> dict:
        """POST 요청 (타임아웃 15초, 재시도 3회).
        - 서명 payload: POST|{path}|{normalized_query}|{timestamp}
        - 파라미터는 form-encoded body로 전달 (서버가 POST body에서 읽음)
        """
        params = params or {}
        timestamp, signature, query_string = self._sign_post(path, params)
        # URL 쿼리스트링 → 서버 서명 검증에 사용
        # POST body(form-encoded) → 서버 파라미터 파싱에 사용
        url  = f"{self.base_url}/baseballApi/{path}?{query_string}"
        body = query_string.encode("utf-8")  # form-encoded
        headers = {
            "X-API-KEY":    self.api_key,
            "X-TIMESTAMP":  timestamp,
            "X-SIGNATURE":  signature,
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent":   "Mozilla/5.0",
        }
        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(
                    url, data=body, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                # 응답 바디에 서버의 구체적 오류 메시지가 있으므로 반드시 출력
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    err_body = "(body unavailable)"
                logger.warning(
                    "HTTP %s — %s (시도 %d/3) | 서버 응답: %s",
                    e.code, url, attempt, err_body,
                )
                if e.code in (401, 403):
                    raise
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning("POST 요청 오류: %s (시도 %d/3)", e, attempt)
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)


# ──────────────────────────────────────────────
# 경기 수집 (4월 일정 포함)
# ──────────────────────────────────────────────
def fetch_today_games(client: StatizAPIClientPlus) -> tuple:
    """
    4월 일정 수집 → TARGET_DATE 경기 필터 → 라인업 수집.
    라인업을 가져올 수 없는 경기는 스킵 없이 리스트에 그대로 포함
    (get_hitter_features가 리그평균으로 자동 대체).
    반환: (df_games, df_lineups)
    """
    logger.info("[1/4] 4월 경기 일정 수집 중 (대상 날짜: %s)...", TARGET_DATE)
    games_raw = client.get_schedule(2026, 4)
    time.sleep(REQUEST_DELAY)
    logger.info("  → 4월 일정 %d경기 발견", len(games_raw))

    target_games = []
    for g in games_raw:
        # ① 완료 경기 파싱 시도
        rec = parse_game_record(g)
        if rec and rec["game_date"] == TARGET_DATE:
            target_games.append(rec)
            continue

        # ② 미완료(오늘 예정) 경기도 포함
        game_date_raw = g.get("gameDate")
        try:
            gdate = datetime.fromtimestamp(int(game_date_raw)).strftime("%Y-%m-%d")
        except Exception:
            gdate = str(game_date_raw)
        if gdate != TARGET_DATE:
            continue

        home_tc = g.get("homeTeam")
        away_tc = g.get("awayTeam")
        s_code  = g.get("s_code")

        target_games.append({
            "game_id":        g.get("s_no"),
            "game_date":      gdate,
            "game_time":      str(g.get("hm", ""))[:5],
            "home_team":      TEAM_CODE_MAP.get(home_tc, str(home_tc)),
            "away_team":      TEAM_CODE_MAP.get(away_tc, str(away_tc)),
            "home_team_code": home_tc,
            "away_team_code": away_tc,
            "stadium":        STADIUM_CODE_MAP.get(s_code, str(s_code)),
            "stadium_code":   s_code,
            "home_sp_code":   g.get("homeSP"),
            "away_sp_code":   g.get("awaySP"),
            "home_sp_name":   g.get("homeSPName"),
            "away_sp_name":   g.get("awaySPName"),
            "home_score":     g.get("homeScore"),
            "away_score":     g.get("awayScore"),
            "result":         None,
        })

    df_games = pd.DataFrame(target_games) if target_games else pd.DataFrame()
    if df_games.empty:
        logger.warning("  ⚠️  %s 경기를 찾을 수 없습니다.", TARGET_DATE)
        return df_games, pd.DataFrame()

    logger.info("  ✅ %d경기 발견", len(df_games))
    logger.info(
        df_games[["game_time", "home_team", "away_team",
                  "home_sp_name", "away_sp_name"]].to_string(index=False)
    )

    # 라인업 수집
    logger.info("\n[2/4] 라인업 수집 중...")
    lineup_records = []
    for _, row in df_games.iterrows():
        gid = int(row["game_id"])
        try:
            raw  = client.get_lineup(gid)
            recs = parse_lineup_records(gid, raw)
            if recs:
                lineup_records.extend(recs)
                logger.info(
                    "  ✅ %s vs %s (game_id=%d) 라인업 %d명",
                    row["home_team"], row["away_team"], gid, len(recs),
                )
            else:
                logger.info(
                    "  ⚠️  %s vs %s (game_id=%d) 라인업 미제공 → 리그평균 대체",
                    row["home_team"], row["away_team"], gid,
                )
        except Exception as e:
            logger.info(
                "  ⚠️  %s vs %s (game_id=%d) 라인업 수집 실패: %s → 리그평균 대체",
                row["home_team"], row["away_team"], gid, e,
            )
        time.sleep(REQUEST_DELAY)

    df_lineups = pd.DataFrame(lineup_records) if lineup_records else pd.DataFrame()
    return df_games, df_lineups


# ──────────────────────────────────────────────
# CSV 기반 rolling stats (API 불필요)
# ──────────────────────────────────────────────
def build_rolling_from_csv(window: int = 10) -> tuple:
    """
    game_results_2023_2025.csv 전체 데이터로 rolling stats 계산.
    이미 2026 3/28~4/3 데이터가 포함되어 있으므로 API 호출 없이 정확한 값 반환.

    반환:
        rolling_lookup : {(game_id, team_code) → {'rwr', 'rrd', 'streak'}}
        current_form   : {team_code → {'rwr', 'rrd', 'streak'}}  ← 오늘 시작 시점
    """
    gres = pd.read_csv(DATA_DIR / "game_results_2023_2025.csv")
    gres["game_date"] = pd.to_datetime(gres["game_date"])
    gres = gres[gres["result"].isin([0, 1])].sort_values("game_date").reset_index(drop=True)

    def deque_stats(dq: deque) -> dict:
        hist = list(dq)
        if len(hist) < 3:
            return {"rwr": float("nan"), "rrd": float("nan"), "streak": 0}
        rwr = float(np.mean([r["win"] for r in hist]))
        rrd = float(np.mean([r["run_diff"] for r in hist]))
        streak = 0
        if hist:
            target = hist[-1]["win"]
            for r in reversed(hist):
                if r["win"] == target:
                    streak += 1 if target == 1 else -1
                else:
                    break
        return {"rwr": rwr, "rrd": rrd, "streak": streak}

    team_deques:   dict = {}
    rolling_lookup: dict = {}

    for _, row in gres.iterrows():
        gid     = int(row["game_id"])
        home_tc = int(row["home_team_code"])
        away_tc = int(row["away_team_code"])
        h       = int(row.get("home_score", 0) or 0)
        a       = int(row.get("away_score", 0) or 0)
        res     = int(row["result"])

        for tc in [home_tc, away_tc]:
            if tc not in team_deques:
                team_deques[tc] = deque(maxlen=window)

        # 경기 직전 시점 기록 (shift 의미)
        rolling_lookup[(gid, home_tc)] = deque_stats(team_deques[home_tc])
        rolling_lookup[(gid, away_tc)] = deque_stats(team_deques[away_tc])

        team_deques[home_tc].append({"win": res,     "run_diff": h - a})
        team_deques[away_tc].append({"win": 1 - res, "run_diff": a - h})

    current_form = {tc: deque_stats(dq) for tc, dq in team_deques.items()}

    # 최신 팀 컨디션 요약
    logger.info("\n[오늘 시점 팀 컨디션 (최근 %d경기)]", window)
    for tc, stats in sorted(current_form.items()):
        name = TEAM_CODE_MAP.get(tc, str(tc))
        logger.info(
            "  %-6s rwr=%.2f  rrd=%+.1f  streak=%+d",
            name,
            stats["rwr"]    if not np.isnan(stats["rwr"]) else float("nan"),
            stats["rrd"]    if not np.isnan(stats["rrd"]) else float("nan"),
            stats["streak"],
        )

    return rolling_lookup, current_form


# ──────────────────────────────────────────────
# 피처 조립 (predict_2026.build_feature_rows 재사용)
# ──────────────────────────────────────────────
def assemble_features(df_games: pd.DataFrame,
                       df_lineups: pd.DataFrame,
                       rolling_data: tuple) -> pd.DataFrame:
    """predict_2026.py의 build_feature_rows를 그대로 호출."""
    return build_feature_rows(df_games, df_lineups, rolling_data)


# ──────────────────────────────────────────────
# 예측 (앙상블)
# ──────────────────────────────────────────────
def run_predictions(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost + LightGBM 앙상블 예측.
    반환 컬럼: game_id, game_time, home_team, away_team,
               stadium, XGB_Prob, LGB_Prob, Ensemble_Prob,
               home_sp_name, away_sp_name
    """
    fitted = load_models()
    if not fitted:
        raise RuntimeError("models/ 디렉터리에 저장된 모델이 없습니다. "
                           "baseball_baseline.py를 먼저 실행하세요.")

    df_pred = df_feat.copy()
    df_pred["result"] = df_pred["result"].fillna(0).astype(int)

    probs = {}
    for model_name in ["XGBoost", "LightGBM"]:
        if model_name not in fitted:
            logger.warning("  %s 모델 없음 - 스킵", model_name)
            continue
        try:
            res = predict_win_probability(df_pred, fitted, model_name)
            probs[model_name] = res["Win_Prob_Pct"].values
        except Exception as e:
            logger.warning("  %s 예측 실패: %s", model_name, e)

    if not probs:
        raise RuntimeError("사용 가능한 모델이 없습니다.")

    base_cols = ["game_id", "game_date", "game_time",
                 "home_team", "away_team", "stadium"]
    out = df_feat[base_cols].copy().reset_index(drop=True)

    # 선발 투수명 보강
    for col in ["home_sp_name", "away_sp_name"]:
        if col in df_feat.columns:
            out[col] = df_feat[col].values

    if "XGBoost" in probs:
        out["XGB_Prob"]  = np.round(probs["XGBoost"], 2)
    if "LightGBM" in probs:
        out["LGB_Prob"]  = np.round(probs["LightGBM"], 2)

    prob_arrays = list(probs.values())
    out["Ensemble_Prob"] = np.round(np.mean(prob_arrays, axis=0), 2)
    out["Pred_Winner"]   = out.apply(
        lambda r: r["home_team"] if r["Ensemble_Prob"] >= 50 else r["away_team"],
        axis=1,
    )

    return out


# ──────────────────────────────────────────────
# API 제출
# ──────────────────────────────────────────────
def submit_prediction(client: StatizAPIClientPlus,
                       s_no: int,
                       percent: float,
                       dry_run: bool = False) -> dict | None:
    """
    POST /prediction/savePrediction 호출.
    - s_no    : 경기번호
    - percent : 홈팀 승리확률 (소수점 2자리, 예: 57.32)
    - dry_run : True면 실제 전송 없이 파라미터만 출력
    반환: API 응답 dict (dry_run이면 None)
    """
    params = {
        "s_no":    s_no,
        "percent": f"{percent:.2f}",
    }
    if dry_run:
        logger.info("  [DRY-RUN] POST savePrediction params=%s", params)
        return None

    try:
        resp = client.post("prediction/savePrediction", params=params)
        return resp
    except Exception as e:
        logger.error("  제출 실패 (s_no=%d): %s", s_no, e)
        return {"error": str(e)}


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="오늘 경기 승리 확률 예측 및 제출")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="예측만 출력하고 실제 API 제출은 하지 않음",
    )
    parser.add_argument(
        "--only", nargs="+", default=None, metavar="팀명",
        help="제출할 팀명 목록 (해당 팀이 포함된 경기만 제출, 예: --only 두산 한화)",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("★ DRY-RUN 모드: 예측만 출력합니다 (API 제출 없음)")
    if args.only:
        logger.info("★ 필터: %s 포함 경기만 제출합니다", args.only)

    print("=" * 65)
    print(f"  KBO 승부 예측 제출 / {TARGET_DATE}")
    print(f"  모드: {'DRY-RUN (제출 없음)' if args.dry_run else 'LIVE (API 제출)'}")
    print("=" * 65)

    # ── API 클라이언트
    client = StatizAPIClientPlus(
        api_key  = os.getenv("STATIZ_API_KEY", ""),
        secret   = os.getenv("STATIZ_SECRET", ""),
        base_url = os.getenv("STATIZ_API_BASE_URL", "https://api.statiz.co.kr"),
    )

    # ── [1/4] 경기 + 라인업 수집
    df_games, df_lineups = fetch_today_games(client)
    if df_games.empty:
        print("\n예측할 경기가 없습니다. 종료.")
        return

    # ── [2/4] Rolling stats (CSV 기반)
    logger.info("\n[3/4] Rolling stats 계산 중 (game_results CSV 기반)...")
    rolling_data = build_rolling_from_csv(window=10)

    # ── [3/4] 피처 조립
    logger.info("\n[4/4] 피처 조립 및 예측 중...")
    df_feat = assemble_features(df_games, df_lineups, rolling_data)

    # ── 예측
    df_result = run_predictions(df_feat)

    # ── 결과 출력
    print("\n" + "=" * 65)
    print("  🎯 예측 결과")
    print("=" * 65)
    display_cols = [c for c in
                    ["game_time", "home_team", "away_team",
                     "home_sp_name", "away_sp_name",
                     "XGB_Prob", "LGB_Prob", "Ensemble_Prob", "Pred_Winner"]
                    if c in df_result.columns]
    print(df_result[display_cols].to_string(index=False))

    # ── 제출
    print("\n" + "=" * 65)
    print("  📤 API 제출")
    print("=" * 65)

    submit_results = []
    for _, row in df_result.iterrows():
        s_no    = int(row["game_id"])
        percent = float(row["Ensemble_Prob"])
        home    = row["home_team"]
        away    = row["away_team"]
        gtime   = row.get("game_time", "")
        hs      = row.get("home_sp_name", "")
        as_     = row.get("away_sp_name", "")

        # --only 필터: 지정 팀이 포함된 경기만 제출
        if args.only:
            if not any(t in (home, away) for t in args.only):
                logger.info("  ○ 제출 스킵: %s vs %s (--only 필터)", home, away)
                continue

        logger.info(
            "  → [%s] %s(%s) vs %s(%s)  홈팀 승리확률=%.2f%%",
            gtime, home, hs, away, as_, percent,
        )

        resp = submit_prediction(client, s_no, percent, dry_run=args.dry_run)

        status = "DRY-RUN"
        if resp is not None:
            code = resp.get("code") or resp.get("cdoe")  # 오타 방어
            msg  = resp.get("result_msg", str(resp))
            if "error" in resp:
                status = f"실패 ({resp['error']})"
            else:
                status = f"성공 (code={code}, msg={msg})"
        logger.info("     결과: %s", status)

        submit_results.append({
            "game_id":       s_no,
            "home_team":     home,
            "away_team":     away,
            "Ensemble_Prob": percent,
            "submit_status": status,
        })

        time.sleep(REQUEST_DELAY)

    # ── 최종 요약
    print("\n" + "=" * 65)
    print("  📋 제출 요약")
    print("=" * 65)
    for r in submit_results:
        print(
            f"  game_id={r['game_id']}  "
            f"{r['home_team']} vs {r['away_team']}  "
            f"홈팀={r['Ensemble_Prob']:.2f}%  "
            f"상태: {r['submit_status']}"
        )

    # ── CSV 저장
    out_path = DATA_DIR / f"predictions_{TARGET_DATE.replace('-', '')}.csv"
    df_result.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("\n💾 예측 결과 저장: %s", out_path)


if __name__ == "__main__":
    main()
