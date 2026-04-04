"""
2026 KBO 경기 결과 + 라인업 수집 후 기존 CSV에 append
=======================================================
3월 28일 ~ 4월 3일 (오늘 4/4 이전까지) 완료 경기만 수집하여
  data/game_results_2023_2025.csv
  data/game_lineups_2023_2025.csv
에 중복 없이 추가합니다.

사용법:
  python append_2026_games.py
  python append_2026_games.py --cutoff 2026-04-03  # 기본값
"""

import os
import sys
import time
import hmac
import hashlib
import json
import urllib.request
import logging
import datetime
import argparse
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

DATA_DIR = Path("data")

RESULTS_CSV  = DATA_DIR / "game_results_2023_2025.csv"
LINEUPS_CSV  = DATA_DIR / "game_lineups_2023_2025.csv"

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

STADIUM_CODE_MAP = {
    1001: "잠실",   1003: "고척",   1004: "동대문",
    2001: "인천",   2002: "수원",   2003: "숭의",
    3001: "청주",
    4001: "한밭",   4003: "대전",
    5001: "군산",   5002: "전주",
    6001: "광주",   6002: "무등",
    7001: "시민",   7002: "포항",   7003: "대구",
    8001: "사직",   8002: "마산",   8003: "울산",   8004: "구덕",   8005: "창원",
    9001: "춘천",
    10001: "제주",
}

LEAGUE_REGULAR = 10100
STATE_FINISHED  = 3
STATE_RAIN_COLD = 5


# ──────────────────────────────────────────────
# API 클라이언트 (fetch_game_results.py와 동일)
# ──────────────────────────────────────────────
class StatizAPIClient:
    def __init__(self, api_key: str, secret: str, base_url: str):
        if not api_key:
            raise ValueError(".env 파일에 STATIZ_API_KEY가 없습니다.")
        if not secret:
            raise ValueError(".env 파일에 STATIZ_SECRET가 없습니다.")
        self.api_key  = api_key
        self.secret   = secret
        self.base_url = base_url.rstrip("/")

    def _sign(self, path: str, params: dict):
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
        return timestamp, signature, normalized

    def get(self, path: str, params: dict = None) -> dict:
        params = params or {}
        timestamp, signature, query_string = self._sign(path, params)
        url = f"{self.base_url}/baseballApi/{path}?{query_string}"
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
                logger.warning("HTTP %s — %s (시도 %d/3)", e.code, url, attempt)
                if e.code in (401, 403):
                    raise
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning("요청 오류: %s (시도 %d/3)", e, attempt)
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def get_schedule(self, year: int, month: int) -> list:
        data = self.get("prediction/gameSchedule", params={"year": year, "month": month})
        games = []
        for key, value in (data or {}).items():
            if isinstance(value, list):
                games.extend(value)
        return games

    def get_lineup(self, s_no: int) -> dict:
        return self.get("prediction/gameLineup", params={"s_no": s_no}) or {}


# ──────────────────────────────────────────────
# 파싱 함수 (fetch_game_results.py와 동일)
# ──────────────────────────────────────────────
def parse_game_record(game: dict):
    league_type = game.get("leagueType")
    state       = game.get("state")

    if league_type != LEAGUE_REGULAR:
        return None
    if state not in (STATE_FINISHED, STATE_RAIN_COLD):
        return None

    home_score = game.get("homeScore")
    away_score = game.get("awayScore")

    if home_score is None or away_score is None:
        return None
    if home_score == away_score:
        return None

    game_date_raw = game.get("gameDate")
    try:
        game_date = datetime.datetime.fromtimestamp(int(game_date_raw)).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        game_date = str(game_date_raw)

    hm_raw    = str(game.get("hm", ""))
    game_time = hm_raw[:5] if len(hm_raw) >= 5 else hm_raw

    home_team_code = game.get("homeTeam")
    away_team_code = game.get("awayTeam")
    stadium_code   = game.get("s_code")

    return {
        "game_id":        game.get("s_no"),
        "game_date":      game_date,
        "game_time":      game_time,
        "home_team":      TEAM_CODE_MAP.get(home_team_code, str(home_team_code)),
        "away_team":      TEAM_CODE_MAP.get(away_team_code, str(away_team_code)),
        "home_team_code": home_team_code,
        "away_team_code": away_team_code,
        "stadium":        STADIUM_CODE_MAP.get(stadium_code, str(stadium_code)),
        "stadium_code":   stadium_code,
        "home_score":     home_score,
        "away_score":     away_score,
        "result":         1 if home_score > away_score else 0,
        "home_sp_code":   game.get("homeSP"),
        "away_sp_code":   game.get("awaySP"),
        "home_sp_name":   game.get("homeSPName"),
        "away_sp_name":   game.get("awaySPName"),
        "game_type":      game.get("gameType", 1),
        "state":          state,
        "weather_code":   game.get("weather"),
    }


POSITION_NAME_MAP = {
    1: "투수", 2: "포수", 3: "1루수", 4: "2루수", 5: "3루수",
    6: "유격수", 7: "좌익수", 8: "중견수", 9: "우익수", 10: "지명타자",
    11: "대타", 12: "대주자", 13: "내야수", 14: "외야수",
}

# 기존 lineup CSV 컬럼 순서
LINEUP_COLUMNS = [
    "game_id", "team_code", "team_name", "p_no", "p_name",
    "position", "position_name", "role", "starting",
    "lineup_state", "batting_order", "p_bat", "p_throw",
]


def parse_lineup_records(game_id: int, lineup_data: dict) -> list:
    records = []
    for team_code_str, players in lineup_data.items():
        if not isinstance(players, list):
            continue
        try:
            team_code = int(team_code_str)
        except ValueError:
            continue
        team_name = TEAM_CODE_MAP.get(team_code, team_code_str)

        for p in players:
            if not isinstance(p, dict):
                continue
            position = p.get("position", 0)
            starting = p.get("starting", "N")
            if position == 1:
                role = "starter_pitcher" if starting == "Y" else "bullpen_pitcher"
            else:
                role = "starter_batter" if starting == "Y" else "sub_batter"
            records.append({
                "game_id":       game_id,
                "team_code":     team_code,
                "team_name":     team_name,
                "p_no":          p.get("p_no"),
                "p_name":        p.get("p_name"),
                "position":      position,
                "position_name": POSITION_NAME_MAP.get(position, str(position)),
                "role":          role,
                "starting":      starting,
                "lineup_state":  p.get("lineupState"),
                "batting_order": p.get("battingOrder"),
                "p_bat":         p.get("p_bat"),
                "p_throw":       p.get("p_throw"),
            })
    return records


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default="2026-04-03",
                        help="이 날짜 이하 경기만 수집 (YYYY-MM-DD, 기본: 2026-04-03)")
    parser.add_argument("--year", type=int, default=2026,
                        help="수집 연도 (기본: 2026)")
    parser.add_argument("--months", nargs="+", type=int, default=[3, 4],
                        help="수집 월 목록 (기본: 3 4)")
    args = parser.parse_args()

    cutoff_date = args.cutoff  # YYYY-MM-DD string for comparison
    logger.info("수집 대상: %d년 %s월 | cutoff: %s", args.year, args.months, cutoff_date)

    # ── 기존 CSV 로드 (중복 체크용)
    if RESULTS_CSV.exists():
        existing_results = pd.read_csv(RESULTS_CSV)
        existing_game_ids = set(existing_results["game_id"].astype(int).tolist())
        logger.info("기존 game_results: %d경기", len(existing_results))
    else:
        existing_results = pd.DataFrame()
        existing_game_ids = set()
        logger.warning("game_results CSV 없음 — 새로 생성합니다.")

    if LINEUPS_CSV.exists():
        existing_lineups = pd.read_csv(LINEUPS_CSV)
        existing_lineup_ids = set(existing_lineups["game_id"].astype(int).tolist())
        logger.info("기존 game_lineups: %d행", len(existing_lineups))
    else:
        existing_lineups = pd.DataFrame()
        existing_lineup_ids = set()

    # ── API 클라이언트
    client = StatizAPIClient(API_KEY, SECRET, BASE_URL)

    new_results  = []
    new_lineups  = []
    skipped_dup  = 0
    skipped_date = 0
    total_api_calls = 0

    # ── 월별 경기 수집
    for month in args.months:
        logger.info("  %d년 %d월 일정 조회 중...", args.year, month)
        try:
            games = client.get_schedule(args.year, month)
            total_api_calls += 1
        except Exception as e:
            logger.error("  일정 조회 실패 (%d년 %d월): %s", args.year, month, e)
            continue

        logger.info("  → %d경기 발견", len(games))
        time.sleep(REQUEST_DELAY)

        for game in games:
            record = parse_game_record(game)
            if record is None:
                continue

            # cutoff 날짜 필터
            if record["game_date"] > cutoff_date:
                skipped_date += 1
                continue

            game_id = int(record["game_id"])

            # 경기 결과 중복 체크 (라인업과 독립적으로 처리)
            if game_id not in existing_game_ids:
                new_results.append(record)
                existing_game_ids.add(game_id)
            else:
                skipped_dup += 1

            # 라인업 수집 (경기 결과와 별도 체크)
            if game_id not in existing_lineup_ids:
                try:
                    lineup_data = client.get_lineup(game_id)
                    total_api_calls += 1
                    lineup_recs = parse_lineup_records(game_id, lineup_data)
                    new_lineups.extend(lineup_recs)
                    existing_lineup_ids.add(game_id)
                    time.sleep(REQUEST_DELAY)
                except Exception as e:
                    logger.warning("    라인업 수집 실패 (game_id=%d): %s", game_id, e)

    logger.info("")
    logger.info("=== 수집 결과 ===")
    logger.info("  신규 경기:       %d경기", len(new_results))
    logger.info("  신규 라인업:     %d경기분", len(set(r["game_id"] for r in new_lineups)) if new_lineups else 0)
    logger.info("  라인업 총 행:    %d행", len(new_lineups))
    logger.info("  건너뜀(중복):    %d경기", skipped_dup)
    logger.info("  건너뜀(cutoff):  %d경기", skipped_date)
    logger.info("  총 API 호출:     %d회", total_api_calls)

    if not new_results and not new_lineups:
        logger.info("추가할 신규 데이터가 없습니다. CSV 변경 없음.")
        return

    # ── 새 데이터 출력 확인
    if new_results:
        new_df = pd.DataFrame(new_results)
        logger.info("\n신규 경기 목록:")
        for _, row in new_df.iterrows():
            logger.info("  %s  %s vs %s  홈%d-원%d  결과=%s",
                        row["game_date"], row["home_team"], row["away_team"],
                        row["home_score"], row["away_score"],
                        "홈승" if row["result"] == 1 else "원정승")
        # ── game_results CSV append
        if existing_results.empty:
            combined_results = new_df
        else:
            combined_results = pd.concat([existing_results, new_df], ignore_index=True)
        combined_results.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
        logger.info("\ngame_results 저장 완료: %d → %d행", len(existing_results), len(combined_results))
    else:
        logger.info("신규 경기 없음 — game_results CSV 미변경")

    # ── game_lineups CSV append
    if new_lineups:
        new_lin_df = pd.DataFrame(new_lineups)
        if existing_lineups.empty:
            combined_lineups = new_lin_df
        else:
            combined_lineups = pd.concat([existing_lineups, new_lin_df], ignore_index=True)
        combined_lineups.to_csv(LINEUPS_CSV, index=False, encoding="utf-8-sig")
        logger.info("game_lineups 저장 완료: %d → %d행", len(existing_lineups), len(combined_lineups))
    else:
        logger.info("신규 라인업 없음 — game_lineups CSV 미변경")

    logger.info("\n✅ 완료! 이제 build_raw_data.py 와 baseball_baseline.py 를 실행하세요.")


if __name__ == "__main__":
    main()
