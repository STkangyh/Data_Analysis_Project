"""
predict_2026.py
===============
2026시즌 특정 날짜 경기를 API에서 수집하고,
저장된 모델(models/)로 홈팀 승리 확률을 예측합니다.

사용법:
    python predict_2026.py --dates 2026-03-28 2026-03-29

흐름:
    1. API에서 2026년 3월 경기 일정 수집 (대상 날짜만 필터)
    2. 각 경기 라인업 수집
    3. 2025 시즌 스탯(pitcher/hitter) + 2026 시즌 초반 rolling stats 조합
    4. 저장된 XGBoost / LightGBM 모델로 승리 확률 예측
    5. 결과 출력 + data/predictions_YYYYMMDD.csv 저장
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import deque

warnings.filterwarnings('ignore')

# ── MLB 공통 모듈 임포트
from fetch_game_results import (
    StatizAPIClient, parse_game_record, parse_lineup_records,
    TEAM_CODE_MAP, REQUEST_DELAY,
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
from baseball_baseline import load_models, predict_win_probability, _get_feature_cols

from dotenv import load_dotenv
load_dotenv()

DATA_DIR = Path("data")
PRED_DIR = Path("data")


# ────────────────────────────────────────
# [2순위 개선] 외국인 선수 MLB/NPB → KBO 환산 스탯 테이블
# ────────────────────────────────────────
# KBO 환산 계수 (ERA __ × 0.82, WHIP __ × 0.90, WAR __ × 0.65)
# MLB 2024 시즌 실적 기준 (45이닝 이상 등판 또는 환산 가능 자료 보유)
# p_no: Statiz API에서 받아온 실제 코드를 입력 (미식이면 -1로 남겨두면 리그평균 대체)
#
# 당해 신규 외국인 선수 스탯 (2025 시즘 첨부)
#   ERA_KBO  = MLB ERA  * 0.82   (KBO 타고투저 보정)
#   WHIP_KBO = MLB WHIP * 0.90   (KBO 형태 텐덕 보정)
#   WAR_KBO  = MLB WAR  * 0.65   (이닝 수 차이 보정)
#   GS: 2025 시즘 선발 등판 횟수 (명단에 알리지 않은 경우 20으로 가정)
# 형식: {p_no: {'ERA': float, 'WHIP': float, 'WAR': float, 'GS': int, 'hand': 'R'/'L'/'U'}}
# 상동 p_no는 Statiz API에서 받는 id (Baseball Reference/KBO 등록 번호 기준)
FOREIGN_PITCHER_OVERRIDE: dict = {
    # [이름]  [p_no]    ERA   WHIP   WAR   GS  hand  (MLB 2025 또는 최신 NPB 실적 기준)
    # KT 사우어  (MLB 2024: ERA 3.11, WHIP 1.17)
    # KIA 네일   (MLB 2024: ERA 4.67, WHIP 1.35) -- Cardinals
    # SSG 화이트  (MLB 2024: ERA 4.41, WHIP 1.34) -- Rays
    # 키움 알칸타라 (MLB 2024: ERA 4.71, WHIP 1.41) -- Marlins
    # NC 플렉센  (MLB 2024: ERA 3.76, WHIP 1.21) -- Cubs
    # 롤데 로드리게스 (MLB 2024: ERA 5.52, WHIP 1.44) -- Pirates
    # 롤데 비슬리 (MLB 2023: ERA 5.24, WHIP 1.49)
    # 
    # TODO: Statiz p_no를 확인한 후 에 p_no를 키로 실제 값 입력
    # 현재는 완전 전년도부재 선수에 대한 환산값만 하드코딩
    #
    # 사우어 (KT)
    # MLB 2024: ERA 3.11, WHIP 1.17, 33GS
    "SAUER": {"ERA": 3.11 * 0.82, "WHIP": 1.17 * 0.90, "WAR": 3.5 * 0.65, "GS": 33, "hand": "L"},
    # 네일 (KIA)
    # MLB 2024: ERA 4.67, WHIP 1.35, 30GS
    "NEEL": {"ERA": 4.67 * 0.82, "WHIP": 1.35 * 0.90, "WAR": 1.6 * 0.65, "GS": 30, "hand": "R"},
    # 화이트 (SSG)
    # MLB 2024: ERA 4.41, WHIP 1.34, 17GS
    "WHITE": {"ERA": 4.41 * 0.82, "WHIP": 1.34 * 0.90, "WAR": 1.5 * 0.65, "GS": 17, "hand": "R"},
    # 알칸타라 (키움)
    # MLB 2024: ERA 4.71, WHIP 1.41, 30GS
    "ALCANTARA": {"ERA": 4.71 * 0.82, "WHIP": 1.41 * 0.90, "WAR": 1.3 * 0.65, "GS": 30, "hand": "R"},
    # 플렉센 (NC)
    # MLB 2024: ERA 3.76, WHIP 1.21, 31GS
    "FLEXEN": {"ERA": 3.76 * 0.82, "WHIP": 1.21 * 0.90, "WAR": 2.8 * 0.65, "GS": 31, "hand": "R"},
    # 로드리게스 (롤데)
    # MLB 2024: ERA 5.52, WHIP 1.44, 23GS
    "RODRIGUEZ": {"ERA": 5.52 * 0.82, "WHIP": 1.44 * 0.90, "WAR": 0.5 * 0.65, "GS": 23, "hand": "R"},
    # 비슬리 (롤데, ERA 5.24, 2023시즘) - 국내 띜구리그 경험 있음
    "BEASLEY": {"ERA": 5.24 * 0.82, "WHIP": 1.49 * 0.90, "WAR": 0.4 * 0.65, "GS": 20, "hand": "R"},
    # 후라도 (삼성, 전년도 KBO 실적 그대로 사용 예정 → placeholder)
    "JURADO": {"ERA": 4.20, "WHIP": 1.30, "WAR": 1.8, "GS": 25, "hand": "R"},
    # 에르난데스 (한화, 전년도 KBO 실적 수준)
    "HERNANDEZ": {"ERA": 4.50, "WHIP": 1.38, "WAR": 1.5, "GS": 22, "hand": "R"},
    # 치리노스 (LG, 전년도 KBO 실적 수준)
    "CHIRINOS": {"ERA": 3.90, "WHIP": 1.25, "WAR": 2.5, "GS": 28, "hand": "R"},
    # 테일러 (NC, 전년도 KBO 실적 수준)
    "TAYLOR": {"ERA": 4.80, "WHIP": 1.42, "WAR": 1.2, "GS": 20, "hand": "R"},
}

# Statiz p_no → FOREIGN_PITCHER_OVERRIDE 키 매핑 (API에서 p_no 확인 후 업데이트)
# 형식: {p_no(int): override_key(str)}
# p_no 로드 실패 시 get_pitcher_features가 리그평균으로 대체하는 것을 방지
FOREIGN_P_NO_MAP: dict = {
    # TODO: Statiz API에서 수집한 실제 p_no로 업데이트
    # 표엄: 99991: "SAUER", 99992: "NEEL", ...
}

# 한국어 선수명 → FOREIGN_PITCHER_OVERRIDE 키 매핑 (API에서 sp_name이 한국어로 내려올 때 사용)
FOREIGN_NAME_KO_MAP: dict = {
    "사우어":     "SAUER",
    "네일":       "NEEL",
    "화이트":     "WHITE",
    "알칸타라":   "ALCANTARA",
    "플렉센":     "FLEXEN",
    "로드리게스": "RODRIGUEZ",
    "비슬리":     "BEASLEY",
    "후라도":     "JURADO",
    "에르난데스": "HERNANDEZ",
    "치리노스":   "CHIRINOS",
    "테일러":     "TAYLOR",
}


def apply_foreign_pitcher_overrides(pitcher_lookup: dict,
                                    p_no_map: dict = FOREIGN_P_NO_MAP) -> dict:
    """
    [2순위 개선] KBO DB에 스탯이 없는 신규 외국인 선수를
    MLB→KBO 환산 값으로 pitcher_lookup에 주입.

    p_no_map이 비어 있으면 아무 작업도 하지 않음 (Statiz p_no 매핑이 필요함).
    """
    if not p_no_map:
        return pitcher_lookup  # p_no 매핑 없으면 블람크 리턴

    updated = dict(pitcher_lookup)
    injected = []
    for p_no, key in p_no_map.items():
        override = FOREIGN_PITCHER_OVERRIDE.get(key)
        if override is None:
            continue
        for year in [2025, 2026]:
            entry = {
                "ERA":          override["ERA"],
                "WHIP":         override["WHIP"],
                "WAR":          override["WAR"],
                "GS":           override["GS"],
                "IP":           override["GS"] * 5.5,  # 선발 평균 5.5이닝 가정
                "OPS_against":  0.350 + override["WHIP"] * 0.275,
                "AVG_against":  min(max(override["WHIP"] * 0.175, 0.180), 0.340),
            }
            updated[(int(p_no), year)] = entry
        injected.append(f"{key}(p_no={p_no})")

    if injected:
        print(f"  [foreign] 외국인 선수 주입: {', '.join(injected)}")
    return updated


# ────────────────────────────────────────
# [1순위 개선] 2025 시즌 말 rolling stats를 2026 시드로 사용
# ────────────────────────────────────────

def build_seed_records(gres_hist: pd.DataFrame, window: int = 10) -> dict:
    """
    2025 시즌 마지막 window 경기를 팀별로 추출해 initial deque로 반환.
    반환: {team_code: deque([{'win': 0/1, 'run_diff': int}, ...], maxlen=window)}
    """
    gres_hist = gres_hist.copy()
    gres_hist["game_date"] = pd.to_datetime(gres_hist["game_date"])

    # 히스토리 중 가장 최신 연도(= 2025)만 사용
    latest_year = int(gres_hist["game_date"].dt.year.max())
    gres_last = (
        gres_hist[gres_hist["game_date"].dt.year == latest_year]
        .sort_values("game_date")
    )

    records = []
    for _, row in gres_last.iterrows():
        h = int(row.get("home_score", 0) or 0)
        a = int(row.get("away_score", 0) or 0)
        records.append({
            "game_date": row["game_date"],
            "team_code": int(row["home_team_code"]),
            "win":       int(row["result"]),
            "run_diff":  h - a,
        })
        records.append({
            "game_date": row["game_date"],
            "team_code": int(row["away_team_code"]),
            "win":       1 - int(row["result"]),
            "run_diff":  a - h,
        })

    hist = (
        pd.DataFrame(records)
          .sort_values(["team_code", "game_date"])
          .reset_index(drop=True)
    )

    seed_deques = {}
    for tc, grp in hist.groupby("team_code"):
        last_n = grp.tail(window).to_dict("records")
        seed_deques[int(tc)] = deque(
            [{"win": r["win"], "run_diff": r["run_diff"]} for r in last_n],
            maxlen=window,
        )

    # 시드 초기값 요약 출력
    print(f"  [seed] {latest_year}시즌 마지막 {window}경기 기반 팀별 컨디션 시드:")
    for tc, dq in sorted(seed_deques.items()):
        recs = list(dq)
        rwr  = np.mean([r["win"]      for r in recs]) if recs else float("nan")
        rrd  = np.mean([r["run_diff"] for r in recs]) if recs else float("nan")
        name = TEAM_CODE_MAP.get(tc, str(tc))
        print(f"    {name}: rwr={rwr:.2f}, rrd={rrd:+.1f}, n={len(recs)}")

    return seed_deques


def _deque_rolling_stats(dq: deque, min_periods: int = 3) -> dict:
    """deque의 현재 내용(경기 직전 기준)으로 rolling stats를 계산."""
    hist = list(dq)
    if len(hist) < min_periods:
        return {"rwr": float("nan"), "rrd": float("nan"), "streak": 0}

    rwr = float(np.mean([r["win"]      for r in hist]))
    rrd = float(np.mean([r["run_diff"] for r in hist]))

    # 가장 최근 결과부터 역방향으로 연속 streak 계산
    streak = 0
    if hist:
        target = hist[-1]["win"]
        for r in reversed(hist):
            if r["win"] == target:
                streak += 1 if target == 1 else -1
            else:
                break

    return {"rwr": rwr, "rrd": rrd, "streak": streak}


def build_2026_rolling_stats(client: StatizAPIClient,
                              target_dates: list[str],
                              window: int = 10) -> tuple:
    """
    2026 시즌 rolling stats 계산.

    [1순위 개선] 2026 시즌 초반 경기가 적을 때의 NaN 문제 해결:
    2025 시즌 마지막 window 경기를 팀별 deque 시드로 사용.
    2026 경기가 쌓일수록 시드가 점진적으로 교체됨 (window=10이면 10경기 후 완전 대체).

    반환:
        rolling_lookup : {(game_id, team_code) → {'rwr', 'rrd', 'streak'}}
                         경기 직전(shift) 기준 값
        current_form   : {team_code → {'rwr', 'rrd', 'streak'}}
                         가장 최신 시점의 팀 컨디션 (미경기 예측 fallback용)
    """
    print("\n[rolling] 2026 rolling stats 계산 중 (2025 시즌 말 시드 적용)...")

    # 2025 시즌 시드 로드
    gres_hist   = pd.read_csv(DATA_DIR / "game_results_2023_2025.csv")
    seed_deques = build_seed_records(gres_hist, window)

    # 2026 완료 경기 수집 (parse_game_record가 state==3/5만 통과시킴)
    games_raw = client.get_schedule(2026, 3)
    time.sleep(REQUEST_DELAY)

    all_games = [
        rec for g in games_raw
        if (rec := parse_game_record(g)) and rec["game_date"] <= max(target_dates)
    ]
    print(f"  [rolling] 2026 완료 경기: {len(all_games)}경기 "
          f"│ 시드: 2025 시즌 마지막 {window}경기")

    # 팀별 deque 초기화 (2025 시드 복사)
    team_deques = {tc: deque(list(dq), maxlen=window) for tc, dq in seed_deques.items()}
    rolling_lookup: dict = {}

    if all_games:
        gres_2026 = (
            pd.DataFrame(all_games)
              .sort_values(["game_date", "game_id"])
              .reset_index(drop=True)
        )
        for _, game in gres_2026.iterrows():
            gid     = int(game["game_id"])
            home_tc = int(game["home_team_code"])
            away_tc = int(game["away_team_code"])

            # 이 경기 직전 rolling stats 기록 (shift 의미)
            for tc in [home_tc, away_tc]:
                if tc not in team_deques:
                    team_deques[tc] = deque(maxlen=window)
                rolling_lookup[(gid, tc)] = _deque_rolling_stats(team_deques[tc])

            # 경기 결과로 deque 업데이트
            h   = int(game.get("home_score", 0) or 0)
            a   = int(game.get("away_score", 0) or 0)
            res = int(game["result"])
            team_deques[home_tc].append({"win": res,       "run_diff":  h - a})
            team_deques[away_tc].append({"win": 1 - res,   "run_diff":  a - h})

    # 현재 시점 팀 컨디션 (미경기/미래 경기 fallback)
    current_form = {tc: _deque_rolling_stats(dq) for tc, dq in team_deques.items()}

    return rolling_lookup, current_form


# ────────────────────────────────────────
# 대상 날짜 경기 수집
# ────────────────────────────────────────

def fetch_target_games(client: StatizAPIClient, target_dates: list[str]) -> tuple:
    """
    target_dates에 해당하는 경기 결과 + 라인업 수집.
    반환: (df_games, df_lineups)
    """
    print(f"\n[1/3] 경기 결과 수집: {target_dates}")

    # 2026년 3월 일정 전체 수집 후 날짜 필터
    games_raw = client.get_schedule(2026, 3)
    time.sleep(REQUEST_DELAY)

    target_games = []
    for g in games_raw:
        rec = parse_game_record(g)
        if rec and rec["game_date"] in target_dates:
            target_games.append(rec)

    if not target_games:
        print("  ⚠️  해당 날짜 경기를 찾을 수 없습니다.")
        print("  (경기가 아직 완료되지 않았거나 API 응답에 result가 없을 수 있습니다)")
        print("  → 미완료 경기도 포함하여 재시도합니다...")

        # 미완료 경기도 포함 (result=None 허용)
        for g in games_raw:
            gdate_raw = g.get("gameDate")
            try:
                gdate = datetime.fromtimestamp(int(gdate_raw)).strftime("%Y-%m-%d")
            except Exception:
                gdate = str(gdate_raw)
            if gdate in target_dates:
                target_games.append({
                    "game_id":        g.get("s_no"),
                    "game_date":      gdate,
                    "game_time":      str(g.get("hm", ""))[:5],
                    "home_team":      TEAM_CODE_MAP.get(g.get("homeTeam"), str(g.get("homeTeam"))),
                    "away_team":      TEAM_CODE_MAP.get(g.get("awayTeam"), str(g.get("awayTeam"))),
                    "home_team_code": g.get("homeTeam"),
                    "away_team_code": g.get("awayTeam"),
                    "stadium":        str(g.get("s_code", "")),
                    "home_sp_code":   g.get("homeSP"),
                    "away_sp_code":   g.get("awaySP"),
                    "home_sp_name":   g.get("homeSPName"),
                    "away_sp_name":   g.get("awaySPName"),
                    "home_score":     g.get("homeScore"),
                    "away_score":     g.get("awayScore"),
                    "result":         None,
                })

    df_games = pd.DataFrame(target_games)
    print(f"  ✅ {len(df_games)}경기 발견")
    if len(df_games) > 0:
        print(df_games[["game_date", "home_team", "away_team",
                         "home_sp_name", "away_sp_name"]].to_string(index=False))

    # 라인업 수집
    print(f"\n[2/3] 라인업 수집 중 ({len(df_games)}경기)...")
    lineup_records = []
    for _, row in df_games.iterrows():
        gid = int(row["game_id"])
        try:
            raw = client.get_lineup(gid)
            recs = parse_lineup_records(gid, raw)
            lineup_records.extend(recs)
            print(f"  ✅ game_id={gid} 라인업 {len(recs)}명")
        except Exception as e:
            print(f"  ⚠️  game_id={gid} 라인업 수집 실패: {e}")
        time.sleep(REQUEST_DELAY)

    df_lineups = pd.DataFrame(lineup_records) if lineup_records else pd.DataFrame()
    return df_games, df_lineups


# ────────────────────────────────────────
# 피처 조립
# ────────────────────────────────────────

def build_feature_rows(df_games: pd.DataFrame,
                        df_lineups: pd.DataFrame,
                        rolling_data: tuple) -> pd.DataFrame:
    """2026 대상 경기의 피처 행 생성."""
    rolling_lookup, current_form = rolling_data
    print("\n[3/3] 피처 조립 중...")

    # 2025 시즌 스탯 로드
    psta = pd.read_csv(DATA_DIR / "pitcher_season_stats_2023_2025.csv")
    hsta = pd.read_csv(DATA_DIR / "hitter_season_stats_2023_2025.csv")
    plog = pd.read_csv(DATA_DIR / "pitcher_game_log_2023_2025.csv")
    gres_hist = pd.read_csv(DATA_DIR / "game_results_2023_2025.csv")

    pitcher_lookup = build_pitcher_lookup(psta)
    bullpen_lookup = build_bullpen_stats(psta)
    throw_lookup   = build_throw_lookup(plog)
    hitter_lookup  = build_hitter_lookup(hsta)
    rs9_lookup     = build_team_rs9_lookup(gres_hist)

    # 구장별 스탯 룩업 (CSV 파일 없으면 빈 dict 반환 - fallback 자동 적용)
    stadium_pitcher_lookup = build_pitcher_stadium_lookup(None)
    stadium_hitter_lookup  = build_hitter_stadium_lookup(None)

    # 선발 투수별 마지막 등판일 (2025 시즌 기준) → 2026 rest_days 계산용
    plog['game_date'] = pd.to_datetime(plog['game_date'])
    last_start_date: dict = (
        plog[plog['starting'] == 'Y']
        .groupby('p_no')['game_date'].max()
        .to_dict()
    )  # {p_no: last_start_date}

    # [2순위] 외국인 선수 MLB→KBO 환산 스탯 주입
    # df_games의 선발 투수명(한국어)으로 AUTO 매핑
    auto_map: dict = {}
    for _, g in df_games.iterrows():
        for sp_code_col, sp_name_col in [("home_sp_code", "home_sp_name"),
                                          ("away_sp_code", "away_sp_name")]:
            code = g.get(sp_code_col)
            name = str(g.get(sp_name_col, "") or "").strip()
            if code is None or not name:
                continue
            override_key = FOREIGN_NAME_KO_MAP.get(name)
            if override_key:
                auto_map[int(code)] = override_key
    if auto_map:
        print(f"  [foreign] 자동 감지 선발 투수: "
              f"{', '.join(f'{k}({v})' for k, v in auto_map.items())}")
    pitcher_lookup = apply_foreign_pitcher_overrides(pitcher_lookup, p_no_map=auto_map)

    # 라인업 인덱스 (game_id, team_code) → [p_no]
    lineup_index = {}
    if not df_lineups.empty:
        batters = df_lineups[df_lineups["role"] == "starter_batter"]
        for (gid, tc), grp in batters.groupby(["game_id", "team_code"]):
            lineup_index[(int(gid), int(tc))] = grp["p_no"].astype(int).tolist()

    STAT_YEAR = 2025   # 2026 예측 → 2025 시즌 스탯 사용

    rows = []
    for _, game in df_games.iterrows():
        gid     = int(game["game_id"])
        home_tc = int(game["home_team_code"])
        away_tc = int(game["away_team_code"])
        home_sp = game.get("home_sp_code")
        away_sp = game.get("away_sp_code")

        # sp_code가 None이면 리그 평균으로 대체
        home_sp_feat = get_pitcher_features(
            home_sp if pd.notna(home_sp) else -1,
            STAT_YEAR, pitcher_lookup, throw_lookup, "home_sp_")
        away_sp_feat = get_pitcher_features(
            away_sp if pd.notna(away_sp) else -1,
            STAT_YEAR, pitcher_lookup, throw_lookup, "away_sp_")

        home_bp = bullpen_lookup.get((home_tc, STAT_YEAR), {"bp_WHIP": 1.35, "bp_WAR": 1.0})
        away_bp = bullpen_lookup.get((away_tc, STAT_YEAR), {"bp_WHIP": 1.35, "bp_WAR": 1.0})

        home_bat_feat = get_hitter_features(
            gid, home_tc, STAT_YEAR, hitter_lookup, lineup_index, rs9_lookup, "home_bat_")
        away_bat_feat = get_hitter_features(
            gid, away_tc, STAT_YEAR, hitter_lookup, lineup_index, rs9_lookup, "away_bat_")

        # stadium 매핑
        stadium = str(game.get("stadium", ""))
        # 숫자 코드로 들어오면 이름 변환 시도
        try:
            from fetch_game_results import STADIUM_CODE_MAP
            sc = int(stadium)
            stadium = STADIUM_CODE_MAP.get(sc, stadium)
        except (ValueError, TypeError):
            pass
        park_factor = PARK_FACTORS.get(stadium, DEFAULT_PARK_FACTOR)

        # 구장코드 산출 → 구장별 투수/타자 피처
        _stadium_norm = _normalize_stadium(stadium)
        _s_code = STADIUM_NAME_TO_CODE.get(_stadium_norm, 0)
        home_sp_stad = get_pitcher_stadium_features(
            int(home_sp) if pd.notna(home_sp) else -1,
            STAT_YEAR, _s_code, pitcher_lookup, stadium_pitcher_lookup, "home_sp_")
        away_sp_stad = get_pitcher_stadium_features(
            int(away_sp) if pd.notna(away_sp) else -1,
            STAT_YEAR, _s_code, pitcher_lookup, stadium_pitcher_lookup, "away_sp_")
        home_bat_stad = get_team_stadium_ops(
            gid, home_tc, STAT_YEAR, _s_code,
            lineup_index, hitter_lookup, stadium_hitter_lookup, "home_bat_")
        away_bat_stad = get_team_stadium_ops(
            gid, away_tc, STAT_YEAR, _s_code,
            lineup_index, hitter_lookup, stadium_hitter_lookup, "away_bat_")

        _null     = {"rwr": float("nan"), "rrd": float("nan"), "streak": 0}
        home_roll = rolling_lookup.get((gid, home_tc)) or current_form.get(home_tc, _null)
        away_roll = rolling_lookup.get((gid, away_tc)) or current_form.get(away_tc, _null)

        # 2026 선발투수 rest_days: 2025 마지막 선발 등판에서 경과 일수 (최대 30일 캡)
        game_dt = pd.to_datetime(str(game["game_date"]))
        DEFAULT_REST, MAX_REST = 5, 30

        def _rest(sp_code):
            if sp_code is None or (isinstance(sp_code, float) and np.isnan(sp_code)):
                return DEFAULT_REST
            last = last_start_date.get(int(sp_code))
            if last is None:
                return DEFAULT_REST
            return int(min((game_dt - last).days, MAX_REST))

        home_rest = _rest(home_sp)
        away_rest = _rest(away_sp)

        rows.append({
            "game_id":   gid,
            "game_date": str(game["game_date"]),
            "game_time": str(game.get("game_time", "18:00")),
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "stadium":   stadium,
            "result":    game.get("result"),   # None이면 예측용
            **home_sp_feat,
            **away_sp_feat,
            "home_bp_WHIP": round(home_bp["bp_WHIP"], 4),
            "home_bp_WAR":  round(home_bp["bp_WAR"],  4),
            "away_bp_WHIP": round(away_bp["bp_WHIP"], 4),
            "away_bp_WAR":  round(away_bp["bp_WAR"],  4),
            **home_bat_feat,
            **away_bat_feat,
            "park_factor":          park_factor,
            "home_recent_win_rate": home_roll["rwr"],
            "away_recent_win_rate": away_roll["rwr"],
            "home_recent_run_diff": home_roll["rrd"],
            "away_recent_run_diff": away_roll["rrd"],
            "home_win_streak":      home_roll["streak"],
            "away_win_streak":      away_roll["streak"],
            "home_sp_rest_days":    home_rest,
            "away_sp_rest_days":    away_rest,
            **home_sp_stad,
            **away_sp_stad,
            **home_bat_stad,
            **away_bat_stad,
        })

    return pd.DataFrame(rows)


# ────────────────────────────────────────
# 메인
# ────────────────────────────────────────

def main(target_dates: list[str]):
    print("=" * 60)
    print(f"  2026 경기 예측 파이프라인")
    print(f"  대상 날짜: {', '.join(target_dates)}")
    print("=" * 60)

    # API 클라이언트
    client = StatizAPIClient(
        api_key  = os.getenv("STATIZ_API_KEY", ""),
        secret   = os.getenv("STATIZ_SECRET", ""),
        base_url = os.getenv("STATIZ_API_BASE_URL", "https://api.statiz.co.kr"),
    )

    # 경기 + 라인업 수집
    df_games, df_lineups = fetch_target_games(client, target_dates)
    if df_games.empty:
        print("\n  예측할 경기가 없습니다. 종료.")
        return

    # Rolling stats 계산 (1순위 개선: 2025 시즌 말 컨디션을 시드로 사용)
    rolling_data = build_2026_rolling_stats(client, target_dates)

    # 피처 조립
    df_feat = build_feature_rows(df_games, df_lineups, rolling_data)

    # 모델 로드
    print("\n  모델 로드 중...")
    fitted = load_models()
    if not fitted:
        print("  ⚠️  models/ 디렉터리에 저장된 모델이 없습니다.")
        print("     baseball_baseline.py를 먼저 실행하세요.")
        return

    # 예측 (result가 없어도 동작하도록 임시 0 채움)
    df_for_pred = df_feat.copy()
    if df_for_pred["result"].isna().any():
        df_for_pred["result"] = df_for_pred["result"].fillna(0).astype(int)

    # 결과 출력
    print("\n" + "=" * 60)
    print("  🎯 승리 확률 예측 결과")
    print("=" * 60)

    all_preds = {}
    for model_name in fitted:
        try:
            df_pred = predict_win_probability(df_for_pred, fitted, model_name)
            all_preds[model_name] = df_pred
            print(f"\n  [ {model_name} ]")
            print(df_pred.to_string(index=False))
        except Exception as e:
            print(f"  ⚠️  {model_name} 예측 실패: {e}")

    # 앙상블 (두 모델 평균)
    if len(all_preds) == 2:
        models_list = list(all_preds.values())
        ens = models_list[0][["game_id", "game_date", "home_team", "away_team"]].copy()
        probs = np.mean([m["Win_Prob_Pct"].values for m in models_list], axis=0)
        ens["Ensemble_Prob_Pct"] = probs.round(1)
        ens["Pred_Winner"] = ens.apply(
            lambda r: r["home_team"] if r["Ensemble_Prob_Pct"] >= 50 else r["away_team"],
            axis=1
        )
        print("\n  [ 앙상블 (평균) ]")
        print(ens.to_string(index=False))

    # 실제 결과 대조 (result가 있는 경우)
    has_result = df_feat["result"].notna() & df_feat["result"].ne("None")
    if has_result.any():
        print("\n" + "=" * 60)
        print("  📊 실제 결과 대조")
        print("=" * 60)
        truth = df_feat[has_result][["game_id", "game_date", "home_team", "away_team", "result"]].copy()
        truth["actual_winner"] = truth.apply(
            lambda r: r["home_team"] if int(r["result"]) == 1 else r["away_team"], axis=1
        )
        if len(all_preds) == 2:
            merged = ens.merge(truth[["game_id", "actual_winner", "result"]], on="game_id", how="left")
            merged["correct"] = merged["Pred_Winner"] == merged["actual_winner"]
            print(merged[["game_date", "home_team", "away_team",
                           "Ensemble_Prob_Pct", "Pred_Winner", "actual_winner", "correct"]].to_string(index=False))
            n_valid = merged["correct"].notna().sum()
            n_correct = merged["correct"].sum()
            print(f"\n  정확도: {n_correct}/{n_valid} ({n_correct/n_valid*100:.1f}%) [앙상블 기준]")

    # CSV 저장
    date_str = "_".join(d.replace("-", "") for d in target_dates)
    out_path = PRED_DIR / f"predictions_{date_str}.csv"
    if len(all_preds) >= 1:
        save_df = list(all_preds.values())[0].rename(
            columns={"Pred_Result": "XGB_Pred", "Win_Prob_Pct": "XGB_Prob"})
        if len(all_preds) == 2:
            lgb_df = list(all_preds.values())[1][["game_id", "Pred_Result", "Win_Prob_Pct"]].rename(
                columns={"Pred_Result": "LGB_Pred", "Win_Prob_Pct": "LGB_Prob"})
            save_df = save_df.merge(lgb_df, on="game_id", how="left")
            save_df["Ensemble_Prob"] = ens["Ensemble_Prob_Pct"].values
        save_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n  💾 결과 저장: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026 KBO 경기 승리 확률 예측")
    parser.add_argument(
        "--dates", nargs="+", default=["2026-03-28", "2026-03-29"],
        metavar="YYYY-MM-DD",
        help="예측할 날짜 목록 (기본값: 2026-03-28 2026-03-29)"
    )
    args = parser.parse_args()
    main(args.dates)
