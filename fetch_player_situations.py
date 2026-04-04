"""
선수별 구장/상대팀 상황별 스탯 수집기 (Statiz Prediction API v4)
==============================================================
/prediction/playerSituation (si=2) 호출:
  - stadium.{stadiumCode}: 구장별 타자/투수 스탯
  - vsTeam.{teamCode}    : 상대팀별 타자/투수 스탯

출력 (2개 CSV):
  data/pitcher_situations_2023_2025.csv
    - (p_no, year, stadium_code) → ERA, WHIP, OPS, AVG, IP, G
  data/hitter_situations_2023_2025.csv
    - (p_no, year, stadium_code) → OPS, AVG, SLG, OBP, PA, G

사용법:
  python fetch_player_situations.py
"""

import os
import time
import hmac
import hashlib
import json
import logging
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from dotenv import load_dotenv

from fetch_game_results import StatizAPIClient

# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

API_KEY       = os.getenv("STATIZ_API_KEY", "")
SECRET        = os.getenv("STATIZ_SECRET", "")
BASE_URL      = os.getenv("STATIZ_API_BASE_URL", "https://api.statiz.co.kr")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.2"))  # 0.5 → 0.2s
DATA_DIR      = Path("data")
CHECKPOINT_EVERY = 50  # 매 50명마다 CSV 저장

# 구장코드 ↔ 구장명 매핑 (API_codes 문서 기준)
STADIUM_CODE_TO_NAME: dict[int, str] = {
    1001: "잠실", 1002: "목동", 1003: "고척", 1004: "동대문",
    2001: "인천", 2002: "수원", 2003: "숭의",
    3001: "청주",
    4001: "한밭", 4003: "대전",
    5001: "군산", 5002: "전주",
    6001: "광주", 6002: "무등",
    7001: "시민", 7002: "포항", 7003: "대구",
    8001: "사직", 8002: "마산", 8003: "울산", 8004: "구덕", 8005: "창원",
    9001: "춘천",
    10001: "제주",
}


def safe_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def fetch_player_situation(client: StatizAPIClient, p_no: int, year: int) -> dict:
    """playerSituation(si=2) 응답 반환. 실패 시 {}"""
    try:
        resp = client.get(
            "prediction/playerSituation",
            params={"p_no": p_no, "year": year, "si": 2},
        )
        return resp or {}
    except Exception as e:
        logger.warning("p_no=%d year=%d 요청 실패: %s", p_no, year, e)
        return {}


# ──────────────────────────────────────────────────────────────
# 투수 구장별 스탯 수집
# ──────────────────────────────────────────────────────────────
def collect_pitcher_situations(client: StatizAPIClient) -> pd.DataFrame:
    psta = pd.read_csv(DATA_DIR / "pitcher_season_stats_2023_2025.csv")
    all_players = psta[["p_no", "year"]].drop_duplicates().values.tolist()

    out_path = DATA_DIR / "pitcher_situations_2023_2025.csv"
    # ── 재개(Resume): 이미 수집된 (p_no, year) 조합 스킵
    done_set: set = set()
    existing_rows: list = []
    if out_path.exists():
        existing = pd.read_csv(out_path)
        done_set = set(zip(existing["p_no"].astype(int), existing["year"].astype(int)))
        existing_rows = existing.to_dict("records")
        logger.info("기존 투수 데이터 로드: %d개 (p_no,year) 이미 완료", len(done_set))

    players = [(p, y) for p, y in all_players if (int(p), int(y)) not in done_set]
    total_all = len(all_players)
    total_new = len(players)
    logger.info("투수: 전체 %d개 중 신규 %d개 수집 시작", total_all, total_new)

    rows = list(existing_rows)
    for idx, (p_no, year) in enumerate(players):
        if idx % 20 == 0:
            logger.info("[투수 %d/%d] p_no=%d year=%d", idx, total_new, int(p_no), int(year))

        data = fetch_player_situation(client, int(p_no), int(year))
        stadium_data = data.get("stadium", {})

        # 구장별 스탯 행 생성
        for s_code_str, stats in stadium_data.items():
            if not isinstance(stats, dict):
                continue
            rows.append({
                "p_no":         int(p_no),
                "year":         int(year),
                "stadium_code": int(s_code_str),
                "stadium_name": STADIUM_CODE_TO_NAME.get(int(s_code_str), ""),
                "G":            safe_float(stats.get("G"),    0),
                "GS":           safe_float(stats.get("GS"),   0),
                "IP":           safe_float(stats.get("IP"),   0),
                "ER":           safe_float(stats.get("ER"),   None),
                "ERA":          safe_float(stats.get("ERA"),  None),
                "WHIP":         safe_float(stats.get("WHIP"), None),
                "AVG":          safe_float(stats.get("AVG"),  None),
                "OPS":          safe_float(stats.get("OPS"),  None),
                "H":            safe_float(stats.get("H"),    None),
                "BB":           safe_float(stats.get("BB"),   None),
                "SO":           safe_float(stats.get("SO"),   None),
            })

        # 체크포인트: 매 CHECKPOINT_EVERY명마다 중간 저장
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
            logger.info("  체크포인트 저장: %d행", len(rows))

        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(rows)
    logger.info("투수 구장별 스탯: %d행", len(df))
    return df


# ──────────────────────────────────────────────────────────────
# 타자 구장별 스탯 수집
# ──────────────────────────────────────────────────────────────
def collect_hitter_situations(client: StatizAPIClient) -> pd.DataFrame:
    hsta = pd.read_csv(DATA_DIR / "hitter_season_stats_2023_2025.csv")
    all_players = hsta[["p_no", "year"]].drop_duplicates().values.tolist()

    out_path = DATA_DIR / "hitter_situations_2023_2025.csv"
    # ── 재개(Resume): 이미 수집된 (p_no, year) 조합 스킵
    done_set: set = set()
    existing_rows: list = []
    if out_path.exists():
        existing = pd.read_csv(out_path)
        done_set = set(zip(existing["p_no"].astype(int), existing["year"].astype(int)))
        existing_rows = existing.to_dict("records")
        logger.info("기존 타자 데이터 로드: %d개 (p_no,year) 이미 완료", len(done_set))

    players = [(p, y) for p, y in all_players if (int(p), int(y)) not in done_set]
    total_all = len(all_players)
    total_new = len(players)
    logger.info("타자: 전체 %d개 중 신규 %d개 수집 시작", total_all, total_new)

    rows = list(existing_rows)
    for idx, (p_no, year) in enumerate(players):
        if idx % 50 == 0:
            logger.info("[타자 %d/%d] p_no=%d year=%d", idx, total_new, int(p_no), int(year))

        data = fetch_player_situation(client, int(p_no), int(year))
        stadium_data = data.get("stadium", {})

        for s_code_str, stats in stadium_data.items():
            if not isinstance(stats, dict):
                continue
            rows.append({
                "p_no":         int(p_no),
                "year":         int(year),
                "stadium_code": int(s_code_str),
                "stadium_name": STADIUM_CODE_TO_NAME.get(int(s_code_str), ""),
                "G":            safe_float(stats.get("G"),   0),
                "PA":           safe_float(stats.get("PA"),  0),
                "AB":           safe_float(stats.get("AB"),  0),
                "H":            safe_float(stats.get("H"),   None),
                "HR":           safe_float(stats.get("HR"),  None),
                "RBI":          safe_float(stats.get("RBI"), None),
                "AVG":          safe_float(stats.get("AVG"), None),
                "OBP":          safe_float(stats.get("OBP"), None),
                "SLG":          safe_float(stats.get("SLG"), None),
                "OPS":          safe_float(stats.get("OPS"), None),
            })

        # 체크포인트: 매 CHECKPOINT_EVERY명마다 중간 저장
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
            logger.info("  체크포인트 저장: %d행", len(rows))

        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(rows)
    logger.info("타자 구장별 스탯: %d행", len(df))
    return df


# ──────────────────────────────────────────────────────────────
def main():
    client = StatizAPIClient(
        api_key  = API_KEY,
        secret   = SECRET,
        base_url = BASE_URL,
    )

    print("=" * 55)
    print("  선수별 구장 상황 스탯 수집 시작")
    print("=" * 55)

    # 투수
    print("\n[1/2] 투수 구장별 스탯 수집 중...")
    pitcher_df = collect_pitcher_situations(client)
    out1 = DATA_DIR / "pitcher_situations_2023_2025.csv"
    pitcher_df.to_csv(out1, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {out1}  ({len(pitcher_df)}행)")

    # 타자 (투수 완료 후 시작 – 중간에 Ctrl+C 해도 투수 데이터는 보존됨)
    print("\n[2/2] 타자 구장별 스탯 수집 중...")
    hitter_df = collect_hitter_situations(client)
    out2 = DATA_DIR / "hitter_situations_2023_2025.csv"
    hitter_df.to_csv(out2, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {out2}  ({len(hitter_df)}행)")

    print("\n완료!")


if __name__ == "__main__":
    main()
