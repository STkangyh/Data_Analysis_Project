"""
KBO 타자 데이터 수집기 (Statiz Prediction API v4)
==================================================
2023~2025 타자 관련 데이터를 수집하여 2개 CSV로 저장합니다.

출력:
  [Phase 1] data/hitter_game_log_2023_2025.csv       (API 없이 기존 CSV 조인)
    - 선발타자별 경기 출장 기록 (game_lineups + game_results 조인)
    - 1행 = 타자 1명 × 1경기

  [Phase 2] data/hitter_season_stats_2023_2025.csv   (playerSeason API)
    - 타자별 연도별 시즌 스탯 (OPS, wRCplus, WAR, BABIP, wOBA 등)
    - 1행 = 타자 1명 × 연도

사용법:
  python fetch_hitter_stats.py
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

RESULTS_CSV  = os.getenv("OUTPUT_CSV_PATH",    "data/game_results_2023_2025.csv")
LINEUPS_CSV  = os.getenv("OUTPUT_LINEUP_PATH", "data/game_lineups_2023_2025.csv")

OUT_GAMELOG  = "data/hitter_game_log_2023_2025.csv"
OUT_STATS    = "data/hitter_season_stats_2023_2025.csv"

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

POSITION_MAP = {
    1: "투수", 2: "포수", 3: "1루수", 4: "2루수", 5: "3루수",
    6: "유격수", 7: "좌익수", 8: "중견수", 9: "우익수", 10: "지명타자",
    11: "대타", 12: "대주자", 13: "내야수", 14: "외야수",
}

BAT_HAND_MAP = {1: "우타", 2: "좌타", 3: "양타"}


# ──────────────────────────────────────────────
# API 클라이언트 (HMAC-SHA256)
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

    def get_player_season(self, p_no: int) -> dict | None:
        """선수번호로 전체 시즌 스탯 반환"""
        return self._call("prediction/playerSeason", {"p_no": p_no})


# ──────────────────────────────────────────────
# Phase 1: 선발타자 경기 출장 기록 (API 없음)
# ──────────────────────────────────────────────
def build_hitter_game_log(results_csv: str, lineups_csv: str) -> pd.DataFrame:
    """
    기존 game_results + game_lineups CSV를 조인하여
    선발타자별 경기 출장 기록을 생성합니다.
    추가 피처:
      - is_home    : 해당 타자팀이 홈이면 1
      - opponent   : 상대팀명
      - team_result: 해당팀 승=1 / 패=0
    """
    results = pd.read_csv(results_csv)
    lineups = pd.read_csv(lineups_csv)

    # 선발타자만 필터링
    batters = lineups[lineups["role"] == "starter_batter"].copy()

    # game_results에서 필요한 컬럼만 추출
    game_info = results[[
        "game_id", "game_date", "game_time",
        "home_team", "away_team", "home_team_code", "away_team_code",
        "stadium", "home_score", "away_score", "result",
    ]].copy()

    # 조인
    df = batters.merge(game_info, on="game_id", how="left")

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
        "position", "position_name", "starting",
    ]
    return df[[c for c in cols if c in df.columns]].sort_values(["year", "game_date"])


# ──────────────────────────────────────────────
# Phase 2: 타자별 시즌 스탯 (playerSeason API)
# ──────────────────────────────────────────────
def _parse_hitter_season(p_no: int, p_name: str, t_code: int, data: dict) -> list[dict]:
    """
    playerSeason 응답에서 타자 스탯 파싱 (연도별 여러 레코드 가능)
    타자 판별: basic.list[].AVG is not None
    deepen은 year 필드 없이 basic과 동일 인덱스 순서로 대응
    """
    records = []
    basic_list  = (data or {}).get("basic", {}).get("list", [])
    deepen_list = (data or {}).get("deepen", {}).get("list", [])

    for i, b in enumerate(basic_list):
        # 타자 판별: AVG 있음 (투수는 None)
        if b.get("AVG") is None:
            continue

        year_raw = b.get("year")
        year = int(year_raw) if year_raw is not None else None

        # deepen은 basic과 동일 인덱스로 매핑 (year 필드 없음)
        d = deepen_list[i] if i < len(deepen_list) else None

        tc_raw = b.get("t_code", t_code)
        tc_int = int(tc_raw) if tc_raw is not None else t_code

        def sf(val):
            try:
                return round(float(val), 4) if val not in (None, "", "-") else None
            except (TypeError, ValueError):
                return None

        records.append({
            "p_no":        p_no,
            "p_name":      p_name,
            "t_code":      tc_int,
            "team_name":   TEAM_CODE_MAP.get(tc_int, str(tc_int)),
            "year":        year,
            "league_type": b.get("leagueType"),
            "p_position":  b.get("p_position"),
            "G":           b.get("G"),
            "PA":          b.get("PA"),
            "AB":          b.get("AB"),
            "R":           b.get("R"),
            "H":           b.get("H"),
            "2B":          b.get("2B"),
            "3B":          b.get("3B"),
            "HR":          b.get("HR"),
            "RBI":         b.get("RBI"),
            "SB":          b.get("SB"),
            "CS":          b.get("CS"),
            "BB":          b.get("BB"),
            "SO":          b.get("SO"),
            "GDP":         b.get("GDP"),
            "SH":          b.get("SH"),
            "SF":          b.get("SF"),
            "AVG":         sf(b.get("AVG")),
            "OBP":         sf(b.get("OBP")),
            "SLG":         sf(b.get("SLG")),
            "OPS":         sf(b.get("OPS")),
            "wRCplus":     sf(b.get("wRCplus")),
            "WAR":         sf(b.get("WAR")),
            "WAROff":      sf(b.get("WAROff")),
            "WARDef":      sf(b.get("WARDef")),
            # deepen (index 기준 매핑)
            "BBK":         sf(d.get("BBK")) if d else None,
            "BABIP":       sf(d.get("BABIP")) if d else None,
            "wOBA":        sf(d.get("wOBA")) if d else None,
        })
    return records


def fetch_hitter_season_stats(
    client: StatizAPIClient,
    batter_list: pd.DataFrame,
) -> pd.DataFrame:
    """
    타자 명단(p_no 포함)으로 playerSeason API를 호출하여
    연도별 시즌 스탯을 수집합니다.
    """
    unique = batter_list[["p_no", "p_name", "t_code"]].drop_duplicates("p_no")
    total  = len(unique)
    all_records = []

    for i, row in enumerate(unique.itertuples(), 1):
        if i % 50 == 0 or i == 1:
            logger.info("  [타자스탯 %d/%d] p_no=%s %s", i, total, row.p_no, row.p_name)
        try:
            data = client.get_player_season(int(row.p_no))
            records = _parse_hitter_season(row.p_no, row.p_name, row.t_code, data)
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
    logger.info("KBO 타자 데이터 수집 시작 (%s)", ", ".join(map(str, TARGET_YEARS)))
    logger.info("=" * 55)

    Path("data").mkdir(exist_ok=True)
    client = StatizAPIClient(API_KEY, SECRET, BASE_URL)

    # ── Phase 1: 경기 출장 기록 (API 불필요) ──────────────
    logger.info("[Phase 1] 선발타자 경기 출장 기록 생성 중... (기존 CSV 조인)")
    df_log = build_hitter_game_log(RESULTS_CSV, LINEUPS_CSV)
    df_log.to_csv(OUT_GAMELOG, index=False, encoding="utf-8-sig")
    logger.info("[Phase 1 완료] %d행 저장 → %s", len(df_log), OUT_GAMELOG)

    print("\n[연도별 선발 출장 수]")
    print(df_log.groupby("year")["game_id"].count().to_string())
    print("\n[포지션별 출장 수 (2023)]")
    print(df_log[df_log["year"] == 2023]["position_name"].value_counts().head(10).to_string())

    # ── Phase 2: 타자 시즌 스탯 (playerSeason API) ────────
    logger.info("\n[Phase 2] 타자 시즌 스탯 수집 중... (playerSeason API)")

    # game_log에서 unique p_no 추출 (team_code도 함께)
    batter_ids = (
        df_log[["p_no", "p_name", "team_code"]]
        .rename(columns={"team_code": "t_code"})
        .drop_duplicates("p_no")
    )
    logger.info("  수집 대상 타자 수: %d명", len(batter_ids))
    logger.info("  예상 소요 시간: 약 %.0f분", len(batter_ids) * REQUEST_DELAY / 60)

    df_stats = fetch_hitter_season_stats(client, batter_ids)

    if df_stats.empty:
        logger.warning("타자 시즌 스탯 수집 결과 없음.")
    else:
        df_stats = df_stats.sort_values(["year", "t_code", "p_no"]).reset_index(drop=True)
        df_stats.to_csv(OUT_STATS, index=False, encoding="utf-8-sig")
        logger.info("[Phase 2 완료] %d행 저장 → %s", len(df_stats), OUT_STATS)

        print("\n[연도별 타자 시즌 스탯 수]")
        print(df_stats.groupby("year")["p_no"].count().to_string())
        print("\n[스탯 샘플 (OPS 순 상위 10명, 2023년, PA>=200)]")
        s2023 = df_stats[
            (df_stats["year"] == 2023) &
            (df_stats["PA"].astype(float) >= 200)
        ].sort_values("OPS", ascending=False)
        print(
            s2023[["year", "p_name", "team_name", "G", "PA", "AVG", "OBP", "SLG",
                   "OPS", "wRCplus", "WAR", "BABIP", "wOBA"]]
            .head(10).to_string(index=False)
        )

    logger.info("\n[완료] 저장된 파일:")
    logger.info("  %s", OUT_GAMELOG)
    logger.info("  %s", OUT_STATS)


if __name__ == "__main__":
    main()
