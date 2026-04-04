"""
build_raw_data.py
=================
수집된 6개 CSV 데이터를 baseball_baseline.py 입력 형식(raw_data.csv)으로 변환.

피처 조립 전략
──────────────
1. 시즌 스탯 연도: 당해 경기 연도 - 1 (전년도) 우선.
   전년도 데이터 없으면 당년도 사용 (Baseline 허용 수준).
2. 최근 IP/ER: 게임별 투구 기록이 없으므로
   시즌 평균 IP/GS, ERA×IP/GS/9 로 근사.
3. vs 상대팀 피안타율/피OPS: 상대별 스플릿 없어
   전체 시즌 OPS_against, AVG_against를 대리 변수로 사용.
4. 불펜: pitcher_season_stats 중 GS < 5 투수를 팀/연도 단위로 집계.
5. 타자 평균 스탯: 당일 선발 라인업 9명의 OPS/AVG/RS9 평균.
6. 플래툰 보정값: 스플릿 데이터 미수집 → 0.0 고정.
7. 파크 팩터: KBO 구장별 추정값 하드코딩.

출력: data/raw_data.csv  (1경기 = 1행)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

DATA_DIR = Path('data')

# ── KBO 구장 파크 팩터 (2023~2025 평균 추정치)
# 출처: 각종 야구통계 사이트 & 실전 손조정
PARK_FACTORS = {
    '잠실': 1.06,   # LG·두산 홈 (개방형, 타자 친화)
    '고척': 0.93,   # 키움 홈 (돔 구장, 투수 친화)
    '광주': 1.02,   # KIA 홈
    '광주-기아 챔피언스 필드': 1.02,
    '대전': 1.00,   # 한화 홈
    '대전 한화생명이글스파크': 1.00,
    '창원': 0.98,   # NC 홈
    '창원NC파크': 0.98,
    '사직': 1.01,   # 롯데 홈
    '인천': 0.97,   # SSG 홈 (크고 바람 영향)
    '인천SSG랜더스필드': 0.97,
    '수원': 1.00,   # KT 홈
    '수원KT위즈파크': 1.00,
    '대구': 1.03,   # 삼성 홈 (타자 친화)
    '대구삼성라이온즈파크': 1.03,
    '포항': 1.00,
    '청주': 1.01,
    '울산': 1.00,
}
DEFAULT_PARK_FACTOR = 1.00

# 투구 방향 코드 변환
THROW_MAP = {1: 'R', 2: 'L', 3: 'U'}


# ────────────────────────────────────────
# 유틸리티
# ────────────────────────────────────────

def ip_to_decimal(ip_val) -> float:
    """야구 표기 IP(100.2=100⅔이닝)를 실수로 변환."""
    try:
        ip = float(ip_val)
        full = int(ip)
        tenths = round((ip - full) * 10)   # 소수점 = 0, 1, 2
        return full + tenths / 3.0
    except (TypeError, ValueError):
        return 0.0


def safe_float(val, default=np.nan):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ────────────────────────────────────────
# 데이터 로드
# ────────────────────────────────────────

def load_data():
    gres = pd.read_csv(DATA_DIR / 'game_results_2023_2025.csv')
    glin = pd.read_csv(DATA_DIR / 'game_lineups_2023_2025.csv')
    plog = pd.read_csv(DATA_DIR / 'pitcher_game_log_2023_2025.csv')
    psta = pd.read_csv(DATA_DIR / 'pitcher_season_stats_2023_2025.csv')
    hsta = pd.read_csv(DATA_DIR / 'hitter_season_stats_2023_2025.csv')
    return gres, glin, plog, psta, hsta


# ────────────────────────────────────────
# 룩업 테이블 구성
# ────────────────────────────────────────

def build_pitcher_lookup(psta: pd.DataFrame) -> dict:
    """(p_no: int, year: int) → Series 행"""
    lookup = {}
    for _, row in psta.iterrows():
        key = (int(row['p_no']), int(row['year']))
        lookup[key] = row
    return lookup


def build_bullpen_stats(psta: pd.DataFrame) -> dict:
    """팀/연도별 불펜 집계.
    불펜 기준: GS < 5 (선발 전업 투수 제외).
    반환: (t_code: int, year: int) → {'bp_WHIP': float, 'bp_WAR': float}
    """
    bpen = psta[psta['GS'] < 5].copy()
    agg = (
        bpen.groupby(['t_code', 'year'])
        .agg(bp_WHIP=('WHIP', 'mean'), bp_WAR=('WAR', 'sum'))
        .reset_index()
    )
    return {
        (int(r['t_code']), int(r['year'])): {'bp_WHIP': r['bp_WHIP'], 'bp_WAR': r['bp_WAR']}
        for _, r in agg.iterrows()
    }


def build_throw_lookup(plog: pd.DataFrame) -> dict:
    """p_no → 투구 방향 문자열 (R/L/U).  가장 최근 기록 사용."""
    lookup = {}
    for _, row in plog.sort_values('game_date').iterrows():
        raw = row.get('p_throw')
        if pd.notna(raw):
            lookup[int(row['p_no'])] = THROW_MAP.get(int(raw), 'R')
    return lookup


def build_hitter_lookup(hsta: pd.DataFrame) -> dict:
    """(p_no: int, year: int) → Series 행"""
    lookup = {}
    for _, row in hsta.iterrows():
        key = (int(row['p_no']), int(row['year']))
        lookup[key] = row
    return lookup


def build_lineup_index(glin: pd.DataFrame) -> dict:
    """(game_id, team_code) → list of p_no (선발 타자 한정)"""
    idx = {}
    batters = glin[glin['role'] == 'starter_batter']
    for (gid, tc), grp in batters.groupby(['game_id', 'team_code']):
        idx[(int(gid), int(tc))] = grp['p_no'].dropna().astype(int).tolist()
    return idx


def build_team_rs9_lookup(gres: pd.DataFrame) -> dict:
    """(team_code: int, year: int) → 경기당 평균 득점 (RS9 근사).
    홈/원정 평균을 합산 평균하여 편중 제거."""
    g = gres.copy()
    g['year'] = pd.to_datetime(g['game_date']).dt.year
    home = g.groupby(['home_team_code', 'year'])['home_score'].mean().reset_index()
    home.columns = ['team_code', 'year', 'avg_run']
    away = g.groupby(['away_team_code', 'year'])['away_score'].mean().reset_index()
    away.columns = ['team_code', 'year', 'avg_run']
    combined = pd.concat([home, away]).groupby(['team_code', 'year'])['avg_run'].mean()
    return {(int(tc), int(yr)): round(val, 3) for (tc, yr), val in combined.items()}


def build_team_rolling_stats(gres: pd.DataFrame, window: int = 10) -> dict:
    """각 경기 직전 window 경기 기준 팀별 최근 성적 지표 계산.

    반환: {(game_id: int, team_code: int) → {'rwr', 'rrd', 'streak'}}
      - rwr   : 최근 win rate (0.0~1.0)
      - rrd   : 평균 득실차 (scored - allowed)
      - streak: 연속 승(양수) / 연속 패(음수) 횟수
    """
    records = []
    for _, row in gres.iterrows():
        h = int(row.get('home_score', 0) or 0)
        a = int(row.get('away_score', 0) or 0)
        records.append({
            'game_id': int(row['game_id']), 'game_date': row['game_date'],
            'team_code': int(row['home_team_code']), 'win': int(row['result']),
            'run_diff': h - a,
        })
        records.append({
            'game_id': int(row['game_id']), 'game_date': row['game_date'],
            'team_code': int(row['away_team_code']), 'win': 1 - int(row['result']),
            'run_diff': a - h,
        })

    hist = (
        pd.DataFrame(records)
          .sort_values(['team_code', 'game_date', 'game_id'])
          .reset_index(drop=True)
    )

    grp = hist.groupby('team_code')
    hist['rwr'] = grp['win'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )
    hist['rrd'] = grp['run_diff'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )

    def apply_streak(series: pd.Series) -> pd.Series:
        """경기 직전 누적 연속 승(양수)/패(음수) 스트릭."""
        prev, streaks = 0, []
        for w in series:
            streaks.append(prev)
            prev = (prev + 1 if prev >= 0 else 1) if w == 1 \
                   else (prev - 1 if prev <= 0 else -1)
        return pd.Series(streaks, index=series.index)

    hist['streak'] = grp['win'].transform(apply_streak)

    return {
        (int(r['game_id']), int(r['team_code'])): {
            'rwr':    float(r['rwr'])    if pd.notna(r['rwr'])    else np.nan,
            'rrd':    float(r['rrd'])    if pd.notna(r['rrd'])    else np.nan,
            'streak': int(r['streak']),
        }
        for _, r in hist.iterrows()
    }


def _get_stat_year(game_year: int, available_years: set) -> int:
    """경기 연도에 대응하는 시즌 스탯 연도 반환.
    전년도 데이터 우선; 없으면 당년도 (Baseline 허용).
    """
    prev = game_year - 1
    return prev if prev in available_years else game_year


# ────────────────────────────────────────
# 피처 생성 함수
# ────────────────────────────────────────

# 리그 평균값 (데이터 없을 때 대체값)
LEAGUE_AVG = {
    'ERA': 4.50, 'WAR': 1.50, 'WHIP': 1.35,
    'avg_ip': 5.5, 'avg_er': 2.75,
    'ops_against': 0.720, 'avg_against': 0.262,
}


def get_pitcher_features(p_no, stat_year: int, pitcher_lookup: dict,
                          throw_lookup: dict, prefix: str) -> dict:
    """투수 1명 피처 딕셔너리 반환."""
    stats = pitcher_lookup.get((int(p_no), stat_year))
    hand = throw_lookup.get(int(p_no), 'R')

    if stats is not None:
        era = safe_float(stats.get('ERA'), LEAGUE_AVG['ERA'])
        war = safe_float(stats.get('WAR'), LEAGUE_AVG['WAR'])
        whip = safe_float(stats.get('WHIP'), LEAGUE_AVG['WHIP'])
        ops_ag = safe_float(stats.get('OPS_against'), np.nan)
        avg_ag = safe_float(stats.get('AVG_against'), np.nan)

        ip_dec = ip_to_decimal(stats.get('IP', 0))
        gs = max(int(stats.get('GS', 1)), 1)
        avg_ip = ip_dec / gs if ip_dec > 0 else LEAGUE_AVG['avg_ip']
        avg_er = era * avg_ip / 9.0

        # OPS_against/AVG_against 결측 처리
        if np.isnan(ops_ag):
            ops_ag = 0.350 + whip * 0.275  # WHIP 기반 근사
        if np.isnan(avg_ag):
            avg_ag = min(max(whip * 0.175, 0.180), 0.340)
    else:
        era, war, whip = LEAGUE_AVG['ERA'], LEAGUE_AVG['WAR'], LEAGUE_AVG['WHIP']
        avg_ip, avg_er = LEAGUE_AVG['avg_ip'], LEAGUE_AVG['avg_er']
        ops_ag, avg_ag = LEAGUE_AVG['ops_against'], LEAGUE_AVG['avg_against']

    return {
        f'{prefix}ERA': round(era, 4),
        f'{prefix}WAR': round(war, 4),
        f'{prefix}WHIP': round(whip, 4),
        f'{prefix}recent_IP': round(avg_ip, 2),
        f'{prefix}recent_ER': round(avg_er, 2),
        f'{prefix}vs_opp_BA': round(avg_ag, 4),
        f'{prefix}vs_opp_OPS': round(ops_ag, 4),
        f'{prefix}hand': hand,
    }


def get_hitter_features(game_id, team_code, stat_year: int,
                         hitter_lookup: dict, lineup_index: dict,
                         rs9_lookup: dict, prefix: str) -> dict:
    """선발 라인업 9명의 평균 타자 스탯 피처 반환."""
    p_nos = lineup_index.get((int(game_id), int(team_code)), [])

    ops_list, avg_list = [], []
    for p_no in p_nos:
        h = hitter_lookup.get((int(p_no), stat_year))
        if h is not None:
            ops = safe_float(h.get('OPS'))
            avg = safe_float(h.get('AVG'))
            if not np.isnan(ops):
                ops_list.append(ops)
            if not np.isnan(avg):
                avg_list.append(avg)

    team_avg_ops = np.mean(ops_list) if ops_list else 0.700
    team_avg_risp = np.mean(avg_list) if avg_list else 0.250  # AVG ≈ RISP proxy
    team_rs9 = rs9_lookup.get((int(team_code), int(stat_year)), 4.5)

    return {
        f'{prefix}avg_OPS': round(team_avg_ops, 4),
        f'{prefix}avg_RISP': round(team_avg_risp, 4),
        f'{prefix}avg_RS9': round(team_rs9, 3),
        f'{prefix}platoon_adj': 0.0,   # 스플릿 데이터 미수집 → 0 고정
    }


# ────────────────────────────────────────
# 구장별 투수/타자 스탯 룩업
# ────────────────────────────────────────

# 구장 이름 → API stadium_code 매핑
STADIUM_NAME_TO_CODE: dict[str, int] = {
    '잠실': 1001, '목동': 1002, '고척': 1003, '동대문': 1004,
    '인천': 2001, '수원': 2002, '숭의': 2003,
    '청주': 3001,
    '한밭': 4001, '대전': 4003,
    '군산': 5001, '전주': 5002,
    '광주': 6001, '무등': 6002,
    '시민': 7001, '포항': 7002, '대구': 7003,
    '사직': 8001, '마산': 8002, '울산': 8003, '구덕': 8004, '창원': 8005,
    '춘천': 9001,
    '제주': 10001,
}
# 공식 구장명 → 약식명 정규화 (game_results stadium 컬럼 대응)
STADIUM_ALIAS: dict[str, str] = {
    '광주-기아 챔피언스 필드': '광주',
    '대전 한화생명이글스파크': '대전',
    '창원NC파크': '창원',
    '인천SSG랜더스필드': '인천',
    '수원KT위즈파크': '수원',
    '대구삼성라이온즈파크': '대구',
}


def _normalize_stadium(name: str) -> str:
    """구장명 정규화 (약식명 통일)."""
    return STADIUM_ALIAS.get(str(name).strip(), str(name).strip())


def build_pitcher_stadium_lookup(psta_df: pd.DataFrame) -> dict:
    """
    투수별 구장 스탯 룩업.
    CSV(pitcher_situations_2023_2025.csv) 로드 후 구성.
    반환: {(p_no, stat_year, stadium_code): {'ERA': float, 'WHIP': float}}

    파일이 없으면 빈 dict 반환 (fetch_player_situations.py 실행 필요).
    """
    fpath = DATA_DIR / 'pitcher_situations_2023_2025.csv'
    if not fpath.exists():
        print("  [경고] pitcher_situations_2023_2025.csv 없음 → 구장별 투수 스탯 미반영")
        return {}

    df = pd.read_csv(fpath)
    lookup: dict = {}
    for _, row in df.iterrows():
        ip_val = safe_float(row.get('IP'), 0)
        # 이닝 부족 시 신뢰도 낮으므로 스킵 (3이닝 미만)
        if ip_val < 3:
            continue
        era  = safe_float(row.get('ERA'),  None)
        whip = safe_float(row.get('WHIP'), None)
        if era is None and whip is None:
            continue
        key = (int(row['p_no']), int(row['year']), int(row['stadium_code']))
        lookup[key] = {
            'ERA':  era,
            'WHIP': whip,
        }
    return lookup


def build_hitter_stadium_lookup(hsta_df: pd.DataFrame) -> dict:
    """
    타자별 구장 스탯 룩업.
    CSV(hitter_situations_2023_2025.csv) 로드 후 구성.
    반환: {(p_no, stat_year, stadium_code): {'OPS': float}}

    파일이 없으면 빈 dict 반환.
    """
    fpath = DATA_DIR / 'hitter_situations_2023_2025.csv'
    if not fpath.exists():
        print("  [경고] hitter_situations_2023_2025.csv 없음 → 구장별 타자 스탯 미반영")
        return {}

    df = pd.read_csv(fpath)
    lookup: dict = {}
    for _, row in df.iterrows():
        pa_val = safe_float(row.get('PA'), 0)
        # 타석 부족 시 스킵 (10타석 미만)
        if pa_val < 10:
            continue
        ops = safe_float(row.get('OPS'), None)
        if ops is None:
            continue
        key = (int(row['p_no']), int(row['year']), int(row['stadium_code']))
        lookup[key] = {'OPS': ops}
    return lookup


def get_pitcher_stadium_features(p_no: int, stat_year: int, stadium_code: int,
                                  pitcher_lookup: dict,
                                  stadium_pitcher_lookup: dict,
                                  prefix: str) -> dict:
    """선발 투수의 해당 구장 ERA/WHIP 반환.
    구장별 데이터 없으면 시즌 전체 스탯으로 fallback."""
    key = (p_no, stat_year, stadium_code)
    sdata = stadium_pitcher_lookup.get(key)
    if sdata:
        era  = sdata.get('ERA')
        whip = sdata.get('WHIP')
        # None이면 시즌 전체 스탯 fallback
        base = pitcher_lookup.get((p_no, stat_year))
        if era  is None: era  = safe_float(base.get('ERA'),  LEAGUE_AVG['ERA'])  if base is not None else LEAGUE_AVG['ERA']
        if whip is None: whip = safe_float(base.get('WHIP'), LEAGUE_AVG['WHIP']) if base is not None else LEAGUE_AVG['WHIP']
    else:
        base = pitcher_lookup.get((p_no, stat_year))
        era  = safe_float(base.get('ERA'),  LEAGUE_AVG['ERA'])  if base is not None else LEAGUE_AVG['ERA']
        whip = safe_float(base.get('WHIP'), LEAGUE_AVG['WHIP']) if base is not None else LEAGUE_AVG['WHIP']
    return {
        f'{prefix}stadium_ERA':  round(float(era),  4),
        f'{prefix}stadium_WHIP': round(float(whip), 4),
    }


def get_team_stadium_ops(game_id: int, team_code: int, stat_year: int,
                          stadium_code: int, lineup_index: dict,
                          hitter_lookup: dict,
                          stadium_hitter_lookup: dict,
                          prefix: str) -> dict:
    """선발 라인업 9명의 해당 구장 OPS 평균 반환.
    구장별 데이터 없는 선수는 시즌 전체 OPS로 fallback."""
    p_nos = lineup_index.get((int(game_id), int(team_code)), [])
    ops_list = []
    for p_no in p_nos:
        skey = (int(p_no), stat_year, stadium_code)
        sdata = stadium_hitter_lookup.get(skey)
        if sdata and sdata.get('OPS') is not None:
            ops_list.append(sdata['OPS'])
        else:
            hrow = hitter_lookup.get((int(p_no), stat_year))
            if hrow is not None:
                ops = safe_float(hrow.get('OPS'))
                if not np.isnan(ops):
                    ops_list.append(ops)
    team_stadium_ops = np.mean(ops_list) if ops_list else 0.700
    return {f'{prefix}stadium_OPS': round(team_stadium_ops, 4)}


# ────────────────────────────────────────
# 선발 투수 등판 간격 룩업
# ────────────────────────────────────────

def build_sp_rest_days_lookup(gres: pd.DataFrame, plog: pd.DataFrame) -> dict:
    """
    (p_no, game_id) → rest_days (직전 선발 등판으로부터 경과 일수)
    첫 등판 또는 off-season 복귀: DEFAULT_REST(5) 반환
    최대값 MAX_REST(30) 캡 적용
    """
    DEFAULT_REST = 5
    MAX_REST = 30

    starters = plog[plog['starting'] == 'Y'][['p_no', 'game_id', 'game_date']].copy()
    starters['game_date'] = pd.to_datetime(starters['game_date'])
    starters = starters.sort_values(['p_no', 'game_date']).reset_index(drop=True)

    lookup: dict = {}
    for p_no, grp in starters.groupby('p_no', sort=False):
        grp = grp.reset_index(drop=True)
        for i, row in grp.iterrows():
            if i == 0:
                rest = DEFAULT_REST
            else:
                delta = (row['game_date'] - grp.loc[i - 1, 'game_date']).days
                rest = int(min(delta, MAX_REST))
            lookup[(int(p_no), int(row['game_id']))] = rest

    return lookup


# ────────────────────────────────────────
# 메인 변환 파이프라인
# ────────────────────────────────────────

def build_raw_data() -> pd.DataFrame:
    print("=" * 55)
    print("  raw_data.csv 빌드 시작")
    print("=" * 55)

    # 1. 로드
    print("\n[1/5] CSV 로드 중...")
    gres, glin, plog, psta, hsta = load_data()

    gres['game_date'] = pd.to_datetime(gres['game_date'])
    plog['game_date'] = pd.to_datetime(plog['game_date'])
    gres = gres.sort_values('game_date').reset_index(drop=True)
    gres['year'] = gres['game_date'].dt.year

    # 무승부/미완료 경기 제외 (result가 0 또는 1인 행만 사용)
    gres = gres[gres['result'].isin([0, 1])].reset_index(drop=True)
    print(f"  경기 수: {len(gres)}  (무승부/미완료 제거 후)")

    available_years = set(psta['year'].unique())

    # 2. 룩업 테이블 구성
    print("\n[2/5] 룩업 테이블 구성 중...")
    pitcher_lookup = build_pitcher_lookup(psta)
    bullpen_lookup = build_bullpen_stats(psta)
    throw_lookup = build_throw_lookup(plog)
    hitter_lookup = build_hitter_lookup(hsta)
    lineup_index = build_lineup_index(glin)
    rs9_lookup = build_team_rs9_lookup(gres)
    rolling_lookup = build_team_rolling_stats(gres)
    sp_rest_lookup = build_sp_rest_days_lookup(gres, plog)
    stadium_pitcher_lookup = build_pitcher_stadium_lookup(psta)
    stadium_hitter_lookup  = build_hitter_stadium_lookup(hsta)
    print(f"  투수 스탯: {len(pitcher_lookup)}건  |  타자 스탯: {len(hitter_lookup)}건")
    print(f"  구장별 투수: {len(stadium_pitcher_lookup)}건  |  구장별 타자: {len(stadium_hitter_lookup)}건")
    print(f"  라인업 인덱스: {len(lineup_index)}건  |  투구방향 룩업: {len(throw_lookup)}건")
    print(f"  팀 컨디션 롤링 스탯: {len(rolling_lookup)}건")

    # 3. 경기별 피처 조립
    print(f"\n[3/5] 경기별 피처 조립 중... (총 {len(gres)}경기)")
    rows = []
    stats_miss_home, stats_miss_away = 0, 0

    for i, game in gres.iterrows():
        if i % 300 == 0:
            print(f"  [{i:4d}/{len(gres)}] 진행 중...")

        year = int(game['year'])
        stat_year = _get_stat_year(year, available_years)
        home_tc = int(game['home_team_code'])
        away_tc = int(game['away_team_code'])
        home_sp = int(game['home_sp_code'])
        away_sp = int(game['away_sp_code'])

        # 선발 투수 스탯
        home_pitcher_key = (home_sp, stat_year)
        away_pitcher_key = (away_sp, stat_year)
        if home_pitcher_key not in pitcher_lookup:
            stats_miss_home += 1
        if away_pitcher_key not in pitcher_lookup:
            stats_miss_away += 1

        home_sp_feat = get_pitcher_features(home_sp, stat_year, pitcher_lookup, throw_lookup, 'home_sp_')
        away_sp_feat = get_pitcher_features(away_sp, stat_year, pitcher_lookup, throw_lookup, 'away_sp_')

        # 불펜 스탯
        home_bp = bullpen_lookup.get((home_tc, stat_year), {'bp_WHIP': 1.35, 'bp_WAR': 1.0})
        away_bp = bullpen_lookup.get((away_tc, stat_year), {'bp_WHIP': 1.35, 'bp_WAR': 1.0})

        # 타자 스탯
        home_bat_feat = get_hitter_features(
            game['game_id'], home_tc, stat_year, hitter_lookup, lineup_index, rs9_lookup, 'home_bat_')
        away_bat_feat = get_hitter_features(
            game['game_id'], away_tc, stat_year, hitter_lookup, lineup_index, rs9_lookup, 'away_bat_')

        # 파크 팩터
        park_factor = PARK_FACTORS.get(str(game['stadium']), DEFAULT_PARK_FACTOR)

        # 최근 팀 컨디션 (rolling stats)
        gid = int(game['game_id'])
        home_roll = rolling_lookup.get((gid, home_tc), {'rwr': np.nan, 'rrd': np.nan, 'streak': 0})
        away_roll = rolling_lookup.get((gid, away_tc), {'rwr': np.nan, 'rrd': np.nan, 'streak': 0})

        # 구장 코드 조회
        stadium_name_norm = _normalize_stadium(str(game.get('stadium', '')))
        stadium_code = STADIUM_NAME_TO_CODE.get(stadium_name_norm, 0)

        # 구장별 선발투수 스탯
        home_sp_stad = get_pitcher_stadium_features(
            home_sp, stat_year, stadium_code, pitcher_lookup, stadium_pitcher_lookup, 'home_sp_')
        away_sp_stad = get_pitcher_stadium_features(
            away_sp, stat_year, stadium_code, pitcher_lookup, stadium_pitcher_lookup, 'away_sp_')

        # 구장별 팀 타자 OPS
        home_bat_stad = get_team_stadium_ops(
            gid, home_tc, stat_year, stadium_code,
            lineup_index, hitter_lookup, stadium_hitter_lookup, 'home_bat_')
        away_bat_stad = get_team_stadium_ops(
            gid, away_tc, stat_year, stadium_code,
            lineup_index, hitter_lookup, stadium_hitter_lookup, 'away_bat_')

        rows.append({
            'game_id': game['game_id'],
            'game_date': game['game_date'].strftime('%Y-%m-%d'),
            'game_time': game['game_time'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'stadium': game['stadium'],
            'result': int(game['result']),
            **home_sp_feat,
            **away_sp_feat,
            'home_bp_WHIP': round(home_bp['bp_WHIP'], 4),
            'home_bp_WAR': round(home_bp['bp_WAR'], 4),
            'away_bp_WHIP': round(away_bp['bp_WHIP'], 4),
            'away_bp_WAR': round(away_bp['bp_WAR'], 4),
            **home_bat_feat,
            **away_bat_feat,
            'park_factor': park_factor,
            'home_recent_win_rate': home_roll['rwr'],
            'away_recent_win_rate': away_roll['rwr'],
            'home_recent_run_diff': home_roll['rrd'],
            'away_recent_run_diff': away_roll['rrd'],
            'home_win_streak':      home_roll['streak'],
            'away_win_streak':      away_roll['streak'],
            'home_sp_rest_days':    sp_rest_lookup.get((home_sp, gid), 5),
            'away_sp_rest_days':    sp_rest_lookup.get((away_sp, gid), 5),
            **home_sp_stad,
            **away_sp_stad,
            **home_bat_stad,
            **away_bat_stad,
        })

    df = pd.DataFrame(rows)
    print(f"\n  투수 스탯 미매칭 - 홈: {stats_miss_home}건, 원정: {stats_miss_away}건 (리그평균 대체)")

    # 4. 저장
    print("\n[4/5] 파일 저장 중...")
    out_path = DATA_DIR / 'raw_data.csv'
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ 저장 완료: {out_path}")
    print(f"  → {len(df)}행 × {len(df.columns)}열")

    # 5. 요약
    print("\n[5/5] 데이터 요약")
    df['_year'] = pd.to_datetime(df['game_date']).dt.year
    print(df.groupby('_year').agg(
        경기수=('game_id', 'count'),
        홈승률=('result', 'mean'),
        홈SP_ERA평균=('home_sp_ERA', 'mean'),
        팀평균OPS=('home_bat_avg_OPS', 'mean'),
        홈최근승률평균=('home_recent_win_rate', 'mean'),
    ).round(3).to_string())
    df.drop(columns=['_year'], inplace=True)

    return df


if __name__ == '__main__':
    df = build_raw_data()
    print("\n[컬럼 목록]")
    print(df.columns.tolist())
    print("\n[샘플 (첫 2행)]")
    print(df.head(2).to_string())
