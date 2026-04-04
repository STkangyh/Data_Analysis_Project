"""
KBO 경기 결과 + 라인업 수집기 (Statiz Prediction API v4)
=======================================================
2023~2025 정규시즌 경기 결과 및 전체 라인업을 API로 수집하여 CSV로 저장합니다.

사용법:
  1. .env.example 을 복사해 .env 파일 생성
     cp .env.example .env
  2. .env 파일에 STATIZ_API_KEY, STATIZ_SECRET 입력
  3. 스크립트 실행
     python fetch_game_results.py

출력 (2개 CSV):
  [Phase 1] data/game_results_2023_2025.csv
    - 1행 = 1경기 (홈팀 기준), result: 홈팀 승=1 / 패=0

  [Phase 2] data/game_lineups_2023_2025.csv
    - 1행 = 1경기 × 1선수
    - role: starter_pitcher / bullpen_pitcher / starter_batter / sub_batter
    - 불펜 투수: position==1, starting=="N"
    - 선발 타자: position!=1, starting=="Y"
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

API_KEY          = os.getenv("STATIZ_API_KEY", "")
SECRET           = os.getenv("STATIZ_SECRET", "")
BASE_URL         = os.getenv("STATIZ_API_BASE_URL", "https://api.statiz.co.kr")
START_YEAR       = int(os.getenv("COLLECT_START_YEAR", "2023"))
END_YEAR         = int(os.getenv("COLLECT_END_YEAR",   "2025"))
REQUEST_DELAY    = float(os.getenv("REQUEST_DELAY",    "0.5"))
OUTPUT_CSV_PATH  = os.getenv("OUTPUT_CSV_PATH",  "data/game_results_2023_2025.csv")
OUTPUT_LINEUP_PATH = os.getenv("OUTPUT_LINEUP_PATH", "data/game_lineups_2023_2025.csv")

# ──────────────────────────────────────────────
# 코드 매핑 (API 문서 index 참조)
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

# 정규시즌 리그타입 코드
LEAGUE_REGULAR = 10100

# 경기 상태 코드 (완료 간주)
STATE_FINISHED   = 3   # 경기 종료
STATE_RAIN_COLD  = 5   # 강우 콜드 (유효 경기)

# KBO 정규시즌 월 범위 (3월 ~ 10월)
SEASON_MONTHS = list(range(3, 11))


# ──────────────────────────────────────────────
# API 클라이언트
# ──────────────────────────────────────────────
class StatizAPIClient:
    """Statiz Prediction API v4 클라이언트 (HMAC-SHA256 서명 인증)"""

    def __init__(self, api_key: str, secret: str, base_url: str):
        if not api_key:
            raise ValueError(
                ".env 파일에 STATIZ_API_KEY가 설정되지 않았습니다.\n"
                ".env.example 을 참고하여 .env 파일을 생성하세요."
            )
        if not secret:
            raise ValueError(
                ".env 파일에 STATIZ_SECRET가 설정되지 않았습니다.\n"
                ".env.example 을 참고하여 .env 파일을 생성하세요."
            )
        self.api_key   = api_key
        self.secret    = secret
        self.base_url  = base_url.rstrip("/")

    def _sign(self, path: str, params: dict) -> tuple[str, str, str]:
        """HMAC-SHA256 서명 생성 → (timestamp, signature, query_string) 반환"""
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
        """
        GET 요청 공통 처리 (타임아웃 15초, 재시도 3회)
        path: 'prediction/gameSchedule' 형태 (앞 슬래시 없이)
        """
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
                req  = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                logger.warning("HTTP %s — %s (시도 %d/3)", e.code, url, attempt)
                if e.code in (401, 403):
                    raise  # 인증 오류는 재시도 무의미
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning("요청 오류: %s (시도 %d/3)", e, attempt)
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def get_schedule(self, year: int, month: int) -> list[dict]:
        """월별 경기 일정 조회 → 게임 리스트 반환
        응답 구조: {"MMDD": [game, ...], "MMDD": [game, ...], ...}
        """
        data = self.get("prediction/gameSchedule", params={"year": year, "month": month})
        games = []
        for key, value in (data or {}).items():
            # 날짜 키("0501" 등)의 값이 경기 목록 배열
            if isinstance(value, list):
                games.extend(value)
        return games

    def get_lineup(self, s_no: int) -> dict:
        """경기번호(s_no)로 전체 라인업 조회
        응답 구조: {"팀코드": [선수, ...], "팀코드": [선수, ...]}
        """
        return self.get("prediction/gameLineup", params={"s_no": s_no}) or {}


# ──────────────────────────────────────────────
# 경기 데이터 파싱
# ──────────────────────────────────────────────
POSITION_NAME_MAP = {
    1: "투수", 2: "포수", 3: "1루수", 4: "2루수", 5: "3루수",
    6: "유격수", 7: "좌익수", 8: "중견수", 9: "우익수", 10: "지명타자",
    11: "대타", 12: "대주자", 13: "내야수", 14: "외야수",
}
def parse_game_record(game: dict) -> dict | None:
    """
    gameSchedule 응답의 게임 Dict를 경기 결과 레코드로 변환.
    정규시즌 완료 경기만 처리하며, 조건 미충족 시 None 반환.
    """
    import datetime

    league_type = game.get("leagueType")
    state       = game.get("state")  # 실제 필드명은 'state' (s_state 아님)

    # 정규시즌 + 경기종료(3) 또는 강우콜드(5)만 수집
    if league_type != LEAGUE_REGULAR:
        return None
    if state not in (STATE_FINISHED, STATE_RAIN_COLD):
        return None

    home_score = game.get("homeScore")
    away_score = game.get("awayScore")

    # 점수 없으면 스킵 (취소 처리 누락 방어)
    if home_score is None or away_score is None:
        return None
    # 동점(무승부) 제외 — 혹시 있으면 스킵
    if home_score == away_score:
        return None

    # gameDate 필드: Unix timestamp(초) → YYYY-MM-DD
    game_date_raw = game.get("gameDate")
    try:
        game_date = datetime.datetime.fromtimestamp(int(game_date_raw)).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        game_date = str(game_date_raw)

    # hm 필드: "18:30:00" 형태 → "18:30"
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
        "result":         1 if home_score > away_score else 0,  # 홈팀 승=1, 패=0
        "home_sp_code":   game.get("homeSP"),
        "away_sp_code":   game.get("awaySP"),
        "home_sp_name":   game.get("homeSPName"),
        "away_sp_name":   game.get("awaySPName"),
        "game_type":      game.get("gameType", 1),
        "state":          state,
        "weather_code":   game.get("weather"),
    }

def parse_lineup_records(game_id: int, lineup_data: dict) -> list[dict]:
    """
    gameLineup 응답을 선수별 레코드 리스트로 변환.
    role 분류:
      starter_pitcher  : position==1, starting=="Y"  (선발 투수)
      bullpen_pitcher  : position==1, starting=="N"  (불펜 투수)
      starter_batter   : position!=1, starting=="Y"  (선발 타자)
      sub_batter       : position!=1, starting=="N"  (대타/대주자 등)
    """
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
            position = p.get("position")
            starting = p.get("starting", "N")
            is_pitcher = (position == 1)

            if is_pitcher and starting == "Y":
                role = "starter_pitcher"
            elif is_pitcher and starting == "N":
                role = "bullpen_pitcher"
            elif not is_pitcher and starting == "Y":
                role = "starter_batter"
            else:
                role = "sub_batter"

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


def collect_game_lineups(
    client: StatizAPIClient,
    game_ids: list[int],
) -> pd.DataFrame:
    """
    [Phase 2] game_id 목록을 받아 경기별 라인업을 수집하여 DataFrame 반환.
    불펜 투수(bullpen_pitcher) 집계와 선발 타자(starter_batter) 피처 엔지니어링에 사용.
    """
    all_records = []
    total = len(game_ids)

    for i, game_id in enumerate(game_ids, 1):
        if i % 50 == 0 or i == 1:
            logger.info("  [라인업 %d/%d] game_id=%d", i, total, game_id)
        try:
            lineup_data = client.get_lineup(game_id)
            records = parse_lineup_records(game_id, lineup_data)
            all_records.extend(records)
        except Exception as e:
            logger.warning("  라인업 수집 실패 game_id=%d: %s", game_id, e)
        time.sleep(REQUEST_DELAY)

    if not all_records:
        return pd.DataFrame()
    return pd.DataFrame(all_records)


def collect_game_results(
    client: StatizAPIClient,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    연도 범위의 정규시즌 경기 결과를 수집하여 DataFrame 반환.
    월별로 API를 호출하고 완료 경기만 필터링합니다.
    """
    records = []
    total_months = sum(len(SEASON_MONTHS) for _ in range(start_year, end_year + 1))
    processed = 0

    for year in range(start_year, end_year + 1):
        for month in SEASON_MONTHS:
            processed += 1
            logger.info("[%d/%d] %d년 %d월 수집 중...", processed, total_months, year, month)

            try:
                games = client.get_schedule(year, month)
            except Exception as e:
                logger.error("%d년 %d월 수집 실패: %s", year, month, e)
                time.sleep(REQUEST_DELAY)
                continue

            month_count = 0
            for game in games:
                record = parse_game_record(game)
                if record:
                    records.append(record)
                    month_count += 1

            logger.info("  → %d경기 수집 완료", month_count)
            time.sleep(REQUEST_DELAY)

    if not records:
        logger.warning("수집된 경기 데이터가 없습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # 날짜 오름차순 정렬
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values("game_date").reset_index(drop=True)
    df["game_date"] = df["game_date"].dt.strftime("%Y-%m-%d")

    return df


# ──────────────────────────────────────────────
# 실행 진입점
# ──────────────────────────────────────────────
def main():
    logger.info("=" * 55)
    logger.info("KBO 경기 결과 + 라인업 수집 시작 (%d ~ %d 정규시즌)", START_YEAR, END_YEAR)
    logger.info("Base URL       : %s", BASE_URL)
    logger.info("결과 출력 경로 : %s", OUTPUT_CSV_PATH)
    logger.info("라인업 출력    : %s", OUTPUT_LINEUP_PATH)
    logger.info("=" * 55)

    client = StatizAPIClient(api_key=API_KEY, secret=SECRET, base_url=BASE_URL)

    # ── Phase 1: 경기 결과 수집 ──────────────────────────
    logger.info("[Phase 1] 경기 결과(gameSchedule) 수집 중...")
    df_results = collect_game_results(client, START_YEAR, END_YEAR)

    if df_results.empty:
        logger.error("수집된 경기 데이터가 없습니다. API 키와 URL을 확인하세요.")
        return

    out_path = Path(OUTPUT_CSV_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("[Phase 1 완료] %d경기 저장 → %s", len(df_results), out_path)

    # ── Phase 2: 경기별 라인업 수집 ──────────────────────
    logger.info("[Phase 2] 라인업(gameLineup) 수집 중... (총 %d경기, 약 %.0f분 소요)",
                len(df_results), len(df_results) * REQUEST_DELAY / 60)
    game_ids = df_results["game_id"].dropna().astype(int).tolist()
    df_lineups = collect_game_lineups(client, game_ids)

    if df_lineups.empty:
        logger.warning("라인업 데이터 수집 실패.")
    else:
        lineup_path = Path(OUTPUT_LINEUP_PATH)
        lineup_path.parent.mkdir(parents=True, exist_ok=True)
        df_lineups.to_csv(lineup_path, index=False, encoding="utf-8-sig")
        logger.info("[Phase 2 완료] %d행 저장 → %s", len(df_lineups), lineup_path)

    # ── 요약 출력 ─────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("기간       : %s ~ %s", df_results["game_date"].min(), df_results["game_date"].max())
    logger.info("홈팀 승률  : %.1f%%", df_results["result"].mean() * 100)
    logger.info("=" * 55)

    print("\n[연도별 경기 수]")
    df_results["year"] = df_results["game_date"].str[:4]
    print(df_results.groupby("year")["game_id"].count().to_string())

    if not df_lineups.empty:
        print("\n[라인업 role 분포]")
        print(df_lineups["role"].value_counts().to_string())

    print("\n[결과 샘플 (상위 5행)]")
    print(df_results[["game_date", "home_team", "away_team", "home_score", "away_score", "result"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
