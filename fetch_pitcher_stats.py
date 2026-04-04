"""
KBO 투수 데이터 수집기 (Statiz Prediction API v4)
==================================================
2023~2025 투수 관련 데이터를 수집하여 3개 CSV로 저장합니다.

출력:
  [Phase 1] data/pitcher_game_log_2023_2025.csv      (API 없이 기존 CSV 조인)
    - 선발투수별 경기 등판 기록 (game_lineups + game_results 조인)
    - 1행 = 선발투수 1명 × 1경기

  [Phase 2] data/pitcher_roster_2023_2025.csv        (playerRoster API)
    - 연도별 팀별 1군 투수 명단 (불펜 포함 전체)
    - 1행 = 선수 1명 × 연도

  [Phase 3] data/pitcher_season_stats_2023_2025.csv  (playerSeason API)
    - 투수별 연도별 시즌 스탯 (ERA, WHIP, WAR, FIP 등)
    - 1행 = 투수 1명 × 연도

사용법:
  python fetch_pitcher_stats.py
"""

import os
import time
import hmac
import hashlib
import logging
import json
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# .env 로드
# ──────────────────────────────────────────────
load_dotenv()

API_KEY       = os.getenv("STATIZ_API_KEY", "")
SECRET        = os.getenv("STATIZ_SECRET", "")
BASE_URL      = os.getenv("STATIZ_API_BASE_URL", "https://api.statiz.co.kr")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.5"))

RESULTS_CSV  = os.getenv("OUTPUT_CSV_PATH",     "data/game_results_2023_2025.csv")
LINEUPS_CSV  = os.getenv("OUTPUT_LINEUP_PATH",  "data/game_lineups_2023_2025.csv")

OUT_GAMELOG  = "data/pitcher_game_log_2023_2025.csv"
OUT_ROSTER   = "data/pitcher_roster_2023_2025.csv"
OUT_STATS    = "data/pitcher_season_stats_2023_2025.csv"

TARGET_YEARS = [2023, 2024, 2025]

# ──────────────────────────────────────────────
# 코드 매핑
# ──────────────────────────────────────────────
TEAM_CODE_MAP = {
    1001:  "삼성",
    2002:  "KIA",
    3001:  "롯데",
    5002:  "LG",
    6002:  "두산",
    7002:  "한화",
    9002:  "SSG",
    10001: "키움",
    11001: "NC",
    12001: "KT",
}

# 연도별 시즌 말 기준 날짜 (로스터 조회용)
SEASON_END_DATES = {
    2023: "2023-10-06",
    2024: "2024-09-28",
    2025: "2025-09-27",  # 예상 종료일
}

HAND_MAP = {1: "우투", 2: "우언", 3: "좌투", 4: "좌언"}


# ──────────────────────────────────────────────
# API 클라이언트 (some.py 방식 HMAC-SHA256)
# ──────────────────────────────────────────────
class StatizAPIClient:
    def __init__(self, api_key: str, secret: str, base_url: str):
        if not api_key or not secret:
            raise ValueError(".env 파일에 STATIZ_API_KEY, STATIZ_SECRET를 설정하세요.")
        self.api_key  = api_key
        self.secret   = secret
        self.base_url = base_url.rstrip("/")

    def _call(self, path: str, params: dict) -> dict | None:
        """HMAC-SHA256 서명 후 GET 요청, 실패 시 None 반환"""
        method    = "GET"
        timestamp = str(int(time.time()))
        safe      = "-_.!~*'()"
        normalized = "&".join(
            f"{quote(str(k), safe=safe)}={quote(str(v), safe=safe)}"
            for k, v in sorted(params.items())
        )
        payload   = f"{method}|{path}|{normalized}|{timestamp}"
        signature = hmac.new(
            self.secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        url = f"{self.base_url}/baseballApi/{path}?{normalized}"
        headers = {
            "X-API-KEY":   self.api_key,
            "X-TIMESTAMP": timestamp,
            "X-SIGNATURE": signature,
            "User-Agent":  "Mozilla/5.0",
        }
        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                logger.warning("HTTP %s — %s (시도 %d/3)", e.code, path, attempt)
                if e.code in (401, 403):
                    return None
                if attempt == 3:
                    return None
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning("요청 오류: %s (시도 %d/3)", e, attempt)
                if attempt == 3:
                    return None
                time.sleep(2 ** attempt)

    def get_player_roster(self, date: str, t_code: int, code: int = 1) -> list[dict]:
        """날짜 기준 팀 1군 로스터 반환 (code=1: 1군)
        응답 구조: {"0": {player}, "1": {player}, ..., "result_cd": ...}
        """
        data = self._call(
            "prediction/playerRoster",
            {"date": date, "code": code, "t_code": t_code},
        )
        if not data:
            return []
        # 숫자 키만 추출 (result_cd, result_msg, update_time 제외)
        return [v for k, v in data.items() if k.isdigit()]

    def get_player_season(self, p_no: int) -> dict | None:
        """선수번호로 전체 시즌 스탯 반환 (투수/타자 공통)"""
        return self._call("prediction/playerSeason", {"p_no": p_no})


# ──────────────────────────────────────────────
# Phase 1: 선발투수 경기 등판 기록 (API 없음)
# ──────────────────────────────────────────────
def build_pitcher_game_log(results_csv: str, lineups_csv: str) -> pd.DataFrame:
    """
    기존 game_results + game_lineups CSV를 조인하여
    선발투수별 경기 등판 기록을 생성합니다.
    추가 피처:
      - is_home    : 해당 투수팀이 홈이면 1
      - opponent   : 상대팀명
      - team_result: 해당팀 승=1 / 패=0
    """
    results = pd.read_csv(results_csv)
    lineups = pd.read_csv(lineups_csv)

    # 선발투수만 필터링
    pitchers = lineups[lineups["role"] == "starter_pitcher"].copy()

    # game_results에서 필요한 컬럼만 추출
    game_info = results[[
        "game_id", "game_date", "game_time",
        "home_team", "away_team", "home_team_code", "away_team_code",
        "stadium", "home_score", "away_score", "result",
    ]].copy()

    # 조인
    df = pitchers.merge(game_info, on="game_id", how="left")

    # 연도 파생
    df["year"] = df["game_date"].str[:4].astype(int)

    # 홈/원정 여부 및 상대팀, 팀 승패
    df["is_home"] = (df["team_code"] == df["home_team_code"]).astype(int)
    df["opponent"] = df.apply(
        lambda r: r["away_team"] if r["is_home"] == 1 else r["home_team"], axis=1
    )
    df["team_result"] = df.apply(
        lambda r: r["result"] if r["is_home"] == 1 else 1 - r["result"], axis=1
    )

    cols = [
        "year", "game_id", "game_date", "game_time",
        "p_no", "p_name", "team_code", "team_name",
        "is_home", "opponent", "stadium",
        "home_score", "away_score", "team_result",
        "p_throw", "starting", "lineup_state",
    ]
    return df[[c for c in cols if c in df.columns]].sort_values(["year", "game_date"])


# ──────────────────────────────────────────────
# Phase 2: 팀별 연도별 1군 로스터 → 투수 명단
# ──────────────────────────────────────────────
def fetch_pitcher_roster(client: StatizAPIClient) -> pd.DataFrame:
    """
    각 연도 시즌 말 기준으로 모든 팀의 1군 로스터를 조회하고
    투수(p_throw 정보가 있는 선수 중 pitcher로 분류)만 추출합니다.
    playerRoster는 포지션 정보를 직접 제공하지 않으므로,
    game_lineups에서 이미 확인된 pitcher p_no 목록을 seed로 활용합니다.
    """
    records = []
    teams   = list(TEAM_CODE_MAP.keys())
    total   = len(TARGET_YEARS) * len(teams)
    done    = 0

    for year in TARGET_YEARS:
        date = SEASON_END_DATES[year]
        for t_code in teams:
            done += 1
            t_name = TEAM_CODE_MAP[t_code]
            logger.info("[로스터 %d/%d] %d년 %s (%s) 조회 중...",
                        done, total, year, t_name, date)
            try:
                players = client.get_player_roster(date=date, t_code=t_code, code=1)
            except Exception as e:
                logger.warning("  실패: %s", e)
                time.sleep(REQUEST_DELAY)
                continue

            for p in players:
                records.append({
                    "year":      year,
                    "t_code":    t_code,
                    "team_name": t_name,
                    "p_no":      p.get("p_no"),
                    "p_name":    p.get("name") or p.get("p_name"),
                    "roster_date": date,
                })
            logger.info("  → %d명 수집", len(players))
            time.sleep(REQUEST_DELAY)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# Phase 3: 투수별 시즌 스탯 (playerSeason API)
# ──────────────────────────────────────────────
def _parse_pitcher_season(p_no: int, p_name: str, t_code: int, data: dict) -> list[dict]:
    """playerSeason 응답에서 투수 스탯 파싱 (연도별 여러 레코드 가능)"""
    records = []
    basic_list = (data or {}).get("basic", {}).get("list", [])

    for b in basic_list:
        year_raw = b.get("year")
        year = int(year_raw) if year_raw is not None else None
        # 투수 판별: AVG 없고 ERA 있음
        if b.get("AVG") is not None:
            continue  # 타자 시즌은 스킵

        # deepen 데이터 (FIP, KBB 등) — 같은 연도 매칭 (year 문자열/정수 모두 대응)
        d = next(
            (item for item in (data.get("deepen", {}).get("list", []))
             if str(item.get("year", "")) == str(year_raw)),
            None,
        )

        def sf(val):
            try:
                return round(float(val), 4) if val not in (None, "", "-") else None
            except (TypeError, ValueError):
                return None

        tc_raw = b.get("t_code", t_code)
        tc_int = int(tc_raw) if tc_raw is not None else t_code

        records.append({
            "p_no":        p_no,
            "p_name":      p_name,
            "t_code":      tc_int,
            "team_name":   TEAM_CODE_MAP.get(tc_int, str(tc_int)),
            "year":        year,
            "league_type": b.get("leagueType"),
            "G":           b.get("G"),
            "GS":          b.get("GS"),
            "W":           b.get("W"),
            "L":           b.get("L"),
            "S":           b.get("S"),
            "HD":          b.get("HD"),
            "IP":          sf(b.get("IP")),
            "ER":          b.get("ER"),
            "ERA":         sf(b.get("ERA")),
            "WHIP":        sf(b.get("WHIP")),
            "WAR":         sf(b.get("WAR")),
            "FIP":         sf(b.get("FIP")),           # basic에 포함
            "KBB":         sf(d.get("KBB")) if d else None,
            "K9":          sf(d.get("K9"))  if d else None,
            "BB9":         sf(d.get("BB9")) if d else None,
            "HR9":         sf(d.get("HR9")) if d else None,
            "OPS_against": sf(d.get("OPS")) if d else None,  # 피OPS
            "AVG_against": sf(b.get("AVG")),
            "SO":          b.get("SO"),
            "BB":          b.get("BB"),
            "HR":          b.get("HR"),
            "TBF":         b.get("TBF"),
            "NP":          b.get("NP"),
            "RS9":         sf(b.get("RS9")),
        })
    return records


def fetch_pitcher_season_stats(
    client: StatizAPIClient,
    pitcher_list: pd.DataFrame,
) -> pd.DataFrame:
    """
    투수 명단(p_no 포함)으로 playerSeason API를 호출하여
    연도별 시즌 스탯을 수집합니다.
    """
    # 중복 p_no 제거
    unique = pitcher_list[["p_no", "p_name", "t_code"]].drop_duplicates("p_no")
    total  = len(unique)
    all_records = []

    for i, row in enumerate(unique.itertuples(), 1):
        if i % 50 == 0 or i == 1:
            logger.info("  [시즌스탯 %d/%d] p_no=%s %s", i, total, row.p_no, row.p_name)
        try:
            data = client.get_player_season(int(row.p_no))
            records = _parse_pitcher_season(row.p_no, row.p_name, row.t_code, data)
            # 수집 대상 연도만 필터 (2023~2025)
            records = [r for r in records if r["year"] in TARGET_YEARS]
            all_records.extend(records)
        except Exception as e:
            logger.warning("  실패 p_no=%s: %s", row.p_no, e)
        time.sleep(REQUEST_DELAY)

    return pd.DataFrame(all_records)


# ──────────────────────────────────────────────
# 실행 진입점
# ──────────────────────────────────────────────
def main():
    logger.info("=" * 55)
    logger.info("KBO 투수 데이터 수집 시작 (%s)", ", ".join(map(str, TARGET_YEARS)))
    logger.info("=" * 55)

    Path("data").mkdir(exist_ok=True)
    client = StatizAPIClient(API_KEY, SECRET, BASE_URL)

    # ── Phase 1: 경기 등판 기록 (API 불필요) ──────────────
    logger.info("[Phase 1] 선발투수 경기 등판 기록 생성 중... (기존 CSV 조인)")
    df_log = build_pitcher_game_log(RESULTS_CSV, LINEUPS_CSV)
    df_log.to_csv(OUT_GAMELOG, index=False, encoding="utf-8-sig")
    logger.info("[Phase 1 완료] %d행 저장 → %s", len(df_log), OUT_GAMELOG)
    print(f"\n[연도별 선발 등판 수]")
    print(df_log.groupby("year")["game_id"].count().to_string())

    # ── Phase 2: 로스터 조회 (팀 × 연도) ──────────────────
    logger.info("\n[Phase 2] 팀별 연도별 1군 로스터 조회 중...")
    df_roster = fetch_pitcher_roster(client)

    if df_roster.empty:
        logger.warning("로스터 수집 실패. Phase 3 스킵.")
        return

    # 투수 필터링: game_lineups에서 이미 알고 있는 pitcher p_no 목록 활용
    lineups = pd.read_csv(LINEUPS_CSV)
    known_pitcher_pnos = set(
        lineups[lineups["role"] == "starter_pitcher"]["p_no"].dropna().astype(int)
    )
    # roster에서 알려진 투수만 우선 표시 (is_pitcher 컬럼 추가)
    df_roster["is_known_pitcher"] = df_roster["p_no"].isin(known_pitcher_pnos)
    df_roster.to_csv(OUT_ROSTER, index=False, encoding="utf-8-sig")
    logger.info("[Phase 2 완료] %d행 저장 → %s", len(df_roster), OUT_ROSTER)

    print(f"\n[로스터 연도별 인원]")
    print(df_roster.groupby("year")["p_no"].count().to_string())

    # ── Phase 3: 시즌 스탯 (게임 로그의 투수 기준) ────────
    logger.info("\n[Phase 3] 투수 시즌 스탯 수집 중... (playerSeason API)")
    # 게임 등판 기록에 있는 투수 + 로스터에서 known pitcher 조합
    pitcher_from_log    = df_log[["p_no", "p_name", "team_code"]].rename(columns={"team_code": "t_code"}).drop_duplicates()
    pitcher_from_roster = df_roster[df_roster["is_known_pitcher"]][["p_no", "p_name", "t_code"]].drop_duplicates()
    all_pitchers = pd.concat([pitcher_from_log, pitcher_from_roster]).drop_duplicates("p_no")

    logger.info("  수집 대상 투수 수: %d명", len(all_pitchers))
    logger.info("  예상 소요 시간: 약 %.0f분", len(all_pitchers) * REQUEST_DELAY / 60)

    df_stats = fetch_pitcher_season_stats(client, all_pitchers)

    if df_stats.empty:
        logger.warning("시즌 스탯 수집 결과 없음.")
    else:
        df_stats = df_stats.sort_values(["year", "t_code", "p_no"]).reset_index(drop=True)
        df_stats.to_csv(OUT_STATS, index=False, encoding="utf-8-sig")
        logger.info("[Phase 3 완료] %d행 저장 → %s", len(df_stats), OUT_STATS)

        print(f"\n[연도별 투수 시즌 스탯 수]")
        print(df_stats.groupby("year")["p_no"].count().to_string())
        print(f"\n[스탯 샘플 (상위 5행)]")
        print(df_stats[["year", "p_name", "team_name", "G", "GS", "ERA", "WHIP", "WAR"]].head().to_string(index=False))

    logger.info("\n[완료] 저장된 파일:")
    logger.info("  %s", OUT_GAMELOG)
    logger.info("  %s", OUT_ROSTER)
    logger.info("  %s", OUT_STATS)


if __name__ == "__main__":
    main()
