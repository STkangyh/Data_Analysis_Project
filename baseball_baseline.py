"""
야구 승부 예측 머신러닝 베이스라인 V1.0
========================================
지침 파일(claude.md) 기반으로 작성.
전처리(preprocess_data) → Feature Engineering → Time-Series Split 학습 → 승리 확률 예측
까지 한 번에 실행되는 End-to-End 파이프라인.

【원본 데이터(Raw Data) 필수 컬럼 정의】
────────────────────────────────────────
경기 단위(game-level) 1행 = 1경기(홈팀 기준)

[공통]
  - game_id          : 경기 고유 ID
  - game_date        : 경기 날짜 (YYYY-MM-DD)
  - game_time        : 경기 시작 시각 (HH:MM, 24h) → Day/Night 더미 생성
  - home_team        : 홈팀 코드
  - away_team        : 어웨이팀 코드
  - stadium          : 구장명
  - result           : 홈팀 승(1) / 패(0) ← 타깃 변수

[홈 선발 투수]
  - home_sp_ERA          : 홈 선발 ERA
  - home_sp_WAR          : 홈 선발 WAR
  - home_sp_WHIP         : 홈 선발 WHIP
  - home_sp_recent_IP    : 최근 5경기 평균 IP (이닝)
  - home_sp_recent_ER    : 최근 5경기 평균 ER (자책점)
  - home_sp_vs_opp_BA    : 상대 팀 대비 피안타율 (천적)
  - home_sp_vs_opp_OPS   : 상대 팀 대비 피OPS (천적)
  - home_sp_hand         : 투구 유형 (R/L/U) → 플래툰 보정에 사용

[어웨이 선발 투수]  (컬럼명 접두어: away_sp_)
  - away_sp_ERA, away_sp_WAR, away_sp_WHIP
  - away_sp_recent_IP, away_sp_recent_ER
  - away_sp_vs_opp_BA, away_sp_vs_opp_OPS
  - away_sp_hand

[홈 불펜]
  - home_bp_WHIP         : 홈 불펜 평균 WHIP
  - home_bp_WAR          : 홈 불펜 합산 WAR

[어웨이 불펜]
  - away_bp_WHIP, away_bp_WAR

[홈 타자 (선발 라인업 9명 평균값으로 미리 집계한 형태)]
  - home_bat_avg_OPS     : 팀 평균 OPS
  - home_bat_avg_RISP    : 팀 평균 득점권 타율
  - home_bat_avg_RS9     : 팀 평균 9이닝당 득점 지원
  - home_bat_platoon_adj : 상대 선발 투구 유형에 맞춘 플래툰 보정 OPS (사전 계산 또는 파생)

[어웨이 타자]  (컬럼명 접두어: away_bat_)
  - away_bat_avg_OPS, away_bat_avg_RISP, away_bat_avg_RS9
  - away_bat_platoon_adj

[구장 파크 팩터]
  - park_factor           : 해당 구장의 파크 팩터 (1.0 = 중립)
"""

# ──────────────────────────────────────────────
# 0. 라이브러리 임포트
# ──────────────────────────────────────────────
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')    # 비대화형 백엔드 (plt.show() 블로킹 방지)
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. 상수 정의
# ──────────────────────────────────────────────
# 낮/밤 경기 구분 기준 시각 (14시 = 오후 2시 이전이면 Day=1)
DAY_CUTOFF_HOUR = 14

# 불펜 WHIP 리그 평균 (시즌 시작 전 추정치; 실제 데이터로 대체 가능)
LEAGUE_AVG_BP_WHIP = 1.35

# 불펜 WHIP 페널티 스케일 팩터 (악영향을 '강하게' 반영하기 위해 2.0 배율)
BP_WHIP_PENALTY_SCALE = 2.0

# 최근 컨디션 ERA 계산 시 이닝이 0인 경우 대체할 최솟값
MIN_IP = 0.1

# Time-Series Split 폴드 수
N_SPLITS = 5

# ──────────────────────────────────────────────
# 2. 샘플 데이터 생성 (Raw Data가 없을 때 테스트용)
# ──────────────────────────────────────────────
def generate_sample_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    실제 원본 데이터가 없는 경우 파이프라인 동작 검증용 샘플 데이터를 생성합니다.
    실제 데이터 사용 시 이 함수는 불필요합니다.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2022-04-01", periods=n, freq="D")
    teams = ["KIA", "LG", "SSG", "두산", "한화", "NC", "롯데", "KT", "삼성", "키움"]
    stadiums = ["광주", "잠실", "인천", "잠실", "대전", "창원", "사직", "수원", "대구", "고척"]
    hand_types = ["R", "L", "U"]

    rows = []
    for i in range(n):
        home_idx = rng.integers(0, len(teams))
        away_idx = rng.integers(0, len(teams))
        while away_idx == home_idx:
            away_idx = rng.integers(0, len(teams))

        rows.append({
            "game_id": f"G{i+1:04d}",
            "game_date": dates[i].strftime("%Y-%m-%d"),
            "game_time": f"{rng.integers(13, 20):02d}:00",
            "home_team": teams[home_idx],
            "away_team": teams[away_idx],
            "stadium": stadiums[home_idx],
            "result": int(rng.random() > 0.48),   # 홈팀 약 52% 승률

            # 홈 선발 투수
            "home_sp_ERA":        round(rng.uniform(2.5, 6.5), 2),
            "home_sp_WAR":        round(rng.uniform(0.0, 5.0), 2),
            "home_sp_WHIP":       round(rng.uniform(0.9, 1.8), 2),
            "home_sp_recent_IP":  round(rng.uniform(4.0, 7.0), 1),
            "home_sp_recent_ER":  round(rng.uniform(0.5, 4.0), 1),
            "home_sp_vs_opp_BA":  round(rng.uniform(0.200, 0.320), 3),
            "home_sp_vs_opp_OPS": round(rng.uniform(0.550, 0.900), 3),
            "home_sp_hand":       hand_types[rng.integers(0, 3)],

            # 어웨이 선발 투수
            "away_sp_ERA":        round(rng.uniform(2.5, 6.5), 2),
            "away_sp_WAR":        round(rng.uniform(0.0, 5.0), 2),
            "away_sp_WHIP":       round(rng.uniform(0.9, 1.8), 2),
            "away_sp_recent_IP":  round(rng.uniform(4.0, 7.0), 1),
            "away_sp_recent_ER":  round(rng.uniform(0.5, 4.0), 1),
            "away_sp_vs_opp_BA":  round(rng.uniform(0.200, 0.320), 3),
            "away_sp_vs_opp_OPS": round(rng.uniform(0.550, 0.900), 3),
            "away_sp_hand":       hand_types[rng.integers(0, 3)],

            # 홈 불펜
            "home_bp_WHIP": round(rng.uniform(1.0, 1.8), 2),
            "home_bp_WAR":  round(rng.uniform(-0.5, 3.0), 2),

            # 어웨이 불펜
            "away_bp_WHIP": round(rng.uniform(1.0, 1.8), 2),
            "away_bp_WAR":  round(rng.uniform(-0.5, 3.0), 2),

            # 홈 타자
            "home_bat_avg_OPS":      round(rng.uniform(0.600, 0.900), 3),
            "home_bat_avg_RISP":     round(rng.uniform(0.200, 0.360), 3),
            "home_bat_avg_RS9":      round(rng.uniform(3.5, 6.5), 2),
            "home_bat_platoon_adj":  round(rng.uniform(-0.050, 0.050), 3),

            # 어웨이 타자
            "away_bat_avg_OPS":      round(rng.uniform(0.600, 0.900), 3),
            "away_bat_avg_RISP":     round(rng.uniform(0.200, 0.360), 3),
            "away_bat_avg_RS9":      round(rng.uniform(3.5, 6.5), 2),
            "away_bat_platoon_adj":  round(rng.uniform(-0.050, 0.050), 3),

            # 구장
            "park_factor": round(rng.uniform(0.85, 1.15), 3),
            # 최근 팀 컨디션
            "home_recent_win_rate": round(rng.uniform(0.3, 0.7), 3),
            "away_recent_win_rate": round(rng.uniform(0.3, 0.7), 3),
            "home_recent_run_diff": round(rng.uniform(-3.0, 3.0), 2),
            "away_recent_run_diff": round(rng.uniform(-3.0, 3.0), 2),
            "home_win_streak":      int(rng.integers(-5, 6)),
            "away_win_streak":      int(rng.integers(-5, 6)),
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 3. 전처리 및 Feature Engineering
# ──────────────────────────────────────────────
def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    원본 데이터(df_raw)를 입력받아 모델 학습에 사용할
    Feature DataFrame을 반환합니다.

    생성 피처 목록:
    ┌─ 투수 ─────────────────────────────────────────────────────────┐
    │ Home_SP_ERA_ParkAdj          : 홈 선발 ERA × 파크 팩터 보정      │
    │ Away_SP_ERA_ParkAdj          : 어웨이 선발 ERA × 파크 팩터 보정  │
    │ Home_SP_WAR                  : 홈 선발 WAR                      │
    │ Away_SP_WAR                  : 어웨이 선발 WAR                  │
    │ Home_SP_WHIP                 : 홈 선발 WHIP                     │
    │ Away_SP_WHIP                 : 어웨이 선발 WHIP                 │
    │ Home_SP_RecentERA            : 홈 선발 최근 평균자책점           │
    │ Away_SP_RecentERA            : 어웨이 선발 최근 평균자책점       │
    │ Home_SP_VsOpp_BA             : 홈 선발 상대팀 피안타율           │
    │ Away_SP_VsOpp_BA             : 어웨이 선발 상대팀 피안타율       │
    │ Home_SP_VsOpp_OPS            : 홈 선발 상대팀 피OPS             │
    │ Away_SP_VsOpp_OPS            : 어웨이 선발 상대팀 피OPS         │
    │ Home_Bullpen_WHIP_Penalty    : 홈 불펜 WHIP 페널티              │
    │ Away_Bullpen_WHIP_Penalty    : 어웨이 불펜 WHIP 페널티          │
    │ Home_BP_WAR                  : 홈 불펜 WAR                      │
    │ Away_BP_WAR                  : 어웨이 불펜 WAR                  │
    ├─ 타자 ─────────────────────────────────────────────────────────┤
    │ Home_Team_Avg_OPS            : 홈팀 평균 OPS                    │
    │ Away_Team_Avg_OPS            : 어웨이팀 평균 OPS                │
    │ Home_Team_Avg_RISP           : 홈팀 평균 득점권 타율            │
    │ Away_Team_Avg_RISP           : 어웨이팀 평균 득점권 타율        │
    │ Home_Team_Avg_RS9            : 홈팀 평균 RS9                    │
    │ Away_Team_Avg_RS9            : 어웨이팀 평균 RS9                │
    │ Home_Team_OPS_Platoon        : 홈팀 플래툰 보정 OPS             │
    │ Away_Team_OPS_Platoon        : 어웨이팀 플래툰 보정 OPS         │
    ├─ 대결 차분 (홈 - 어웨이) ──────────────────────────────────────┤
    │ Diff_SP_ERA_ParkAdj          : 선발 ERA 차분 (낮을수록 홈 유리) │
    │ Diff_SP_WAR                  : 선발 WAR 차분                    │
    │ Diff_SP_RecentERA            : 최근 ERA 차분                    │
    │ Diff_Bullpen_WHIP_Penalty    : 불펜 페널티 차분                 │
    │ Diff_Team_OPS_Platoon        : 타선 플래툰 보정 OPS 차분        │
    │ Diff_Team_Avg_RISP           : 득점권 타율 차분                 │
    ├─ 공통 ─────────────────────────────────────────────────────────┤
    │ Is_Day_Game                  : 낮 경기 더미 (1=Day, 0=Night)    │
    └────────────────────────────────────────────────────────────────┘

    Returns
    -------
    pd.DataFrame  : 피처 컬럼 + 'result' 타깃 컬럼 포함
    """
    df = df_raw.copy()

    # ── 날짜 정렬 (Time-Series Split을 위해 필수)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    # ─────────────────────────────────────────
    # A. 투수 피처
    # ─────────────────────────────────────────

    # A-1. 선발 ERA × 파크 팩터 보정
    #      파크 팩터가 높을수록(타자 친화 구장) 투수 ERA가 더 불리하게 보정됨
    df["Home_SP_ERA_ParkAdj"] = df["home_sp_ERA"] * df["park_factor"]
    df["Away_SP_ERA_ParkAdj"] = df["away_sp_ERA"] * df["park_factor"]

    # A-2. 선발 WAR
    df["Home_SP_WAR"] = df["home_sp_WAR"]
    df["Away_SP_WAR"] = df["away_sp_WAR"]

    # A-3. 선발 WHIP
    df["Home_SP_WHIP"] = df["home_sp_WHIP"]
    df["Away_SP_WHIP"] = df["away_sp_WHIP"]

    # A-4. 최근 평균자책점 = (최근 평균 ER × 9) / max(최근 평균 IP, MIN_IP)
    #      이닝이 0이면 분모 보정값(MIN_IP) 사용
    df["Home_SP_RecentERA"] = (df["home_sp_recent_ER"] * 9) / df["home_sp_recent_IP"].clip(lower=MIN_IP)
    df["Away_SP_RecentERA"] = (df["away_sp_recent_ER"] * 9) / df["away_sp_recent_IP"].clip(lower=MIN_IP)

    # A-5. 천적 지표 (당일 상대 팀 대비 피안타율 / 피OPS)
    df["Home_SP_VsOpp_BA"]  = df["home_sp_vs_opp_BA"]
    df["Away_SP_VsOpp_BA"]  = df["away_sp_vs_opp_BA"]
    df["Home_SP_VsOpp_OPS"] = df["home_sp_vs_opp_OPS"]
    df["Away_SP_VsOpp_OPS"] = df["away_sp_vs_opp_OPS"]

    # A-6. 불펜 WHIP 페널티 (핵심 파생 변수)
    #      계산식: max(0, (불펜 WHIP - 리그평균 WHIP)) × SCALE
    #      불펜 WHIP이 리그 평균 이하면 페널티 = 0 (좋은 불펜은 패널티 없음)
    df["Home_Bullpen_WHIP_Penalty"] = (
        (df["home_bp_WHIP"] - LEAGUE_AVG_BP_WHIP).clip(lower=0) * BP_WHIP_PENALTY_SCALE
    )
    df["Away_Bullpen_WHIP_Penalty"] = (
        (df["away_bp_WHIP"] - LEAGUE_AVG_BP_WHIP).clip(lower=0) * BP_WHIP_PENALTY_SCALE
    )

    # A-7. 불펜 WAR
    df["Home_BP_WAR"] = df["home_bp_WAR"]
    df["Away_BP_WAR"] = df["away_bp_WAR"]

    # ─────────────────────────────────────────
    # B. 타자 피처
    # ─────────────────────────────────────────

    # B-1. 기본 팀 평균 스탯 (라인업 9명 평균값)
    df["Home_Team_Avg_OPS"]  = df["home_bat_avg_OPS"]
    df["Away_Team_Avg_OPS"]  = df["away_bat_avg_OPS"]
    df["Home_Team_Avg_RISP"] = df["home_bat_avg_RISP"]
    df["Away_Team_Avg_RISP"] = df["away_bat_avg_RISP"]
    df["Home_Team_Avg_RS9"]  = df["home_bat_avg_RS9"]
    df["Away_Team_Avg_RS9"]  = df["away_bat_avg_RS9"]

    # B-2. 플래툰 보정 OPS = 팀 평균 OPS + 상대 선발 유형 보정값
    #      home 타자는 away 선발(away_sp_hand)을 상대, 반대 동일
    #      platoon_adj 컬럼이 이미 계산된 경우 바로 더함
    df["Home_Team_OPS_Platoon"] = df["home_bat_avg_OPS"] + df["home_bat_platoon_adj"]
    df["Away_Team_OPS_Platoon"] = df["away_bat_avg_OPS"] + df["away_bat_platoon_adj"]

    # ─────────────────────────────────────────
    # C. 공통 피처
    # ─────────────────────────────────────────

    # C-1. Day/Night 더미 변수 (14시 이전 = Day = 1)
    df["game_hour"] = pd.to_datetime(df["game_time"], format="%H:%M").dt.hour
    df["Is_Day_Game"] = (df["game_hour"] < DAY_CUTOFF_HOUR).astype(int)

    # ─────────────────────────────────────────
    # D. 대결 차분 피처 (홈 - 어웨이)
    #    모델이 상대적 강약을 직접 학습하도록 도움
    # ─────────────────────────────────────────
    df["Diff_SP_ERA_ParkAdj"]       = df["Home_SP_ERA_ParkAdj"]       - df["Away_SP_ERA_ParkAdj"]
    df["Diff_SP_WAR"]               = df["Home_SP_WAR"]               - df["Away_SP_WAR"]
    df["Diff_SP_RecentERA"]         = df["Home_SP_RecentERA"]         - df["Away_SP_RecentERA"]
    df["Diff_Bullpen_WHIP_Penalty"] = df["Home_Bullpen_WHIP_Penalty"] - df["Away_Bullpen_WHIP_Penalty"]
    df["Diff_Team_OPS_Platoon"]     = df["Home_Team_OPS_Platoon"]     - df["Away_Team_OPS_Platoon"]
    df["Diff_Team_Avg_RISP"]        = df["Home_Team_Avg_RISP"]        - df["Away_Team_Avg_RISP"]

    # ─────────────────────────────────────────
    # E. 최근 팀 컨디션 피처
    # ─────────────────────────────────────────
    # E-1. 최근 10경기 기반 개별 팀 지표
    df["Home_Recent_Win_Rate"] = df["home_recent_win_rate"]
    df["Away_Recent_Win_Rate"] = df["away_recent_win_rate"]
    df["Home_Recent_Run_Diff"] = df["home_recent_run_diff"]
    df["Away_Recent_Run_Diff"] = df["away_recent_run_diff"]
    df["Home_Win_Streak"]      = df["home_win_streak"]
    df["Away_Win_Streak"]      = df["away_win_streak"]

    # E-2. 최근 컨디션 차분 (홈 - 어웨이)
    df["Diff_Recent_Win_Rate"] = df["Home_Recent_Win_Rate"] - df["Away_Recent_Win_Rate"]
    df["Diff_Recent_Run_Diff"] = df["Home_Recent_Run_Diff"] - df["Away_Recent_Run_Diff"]
    df["Diff_Win_Streak"]      = df["Home_Win_Streak"]      - df["Away_Win_Streak"]

    # ─────────────────────────────────────────
    # F. 개막 시즌 더미
    # ─────────────────────────────────────────
    KBO_OPENING_DAYS = {
        2023: pd.Timestamp('2023-04-01'),
        2024: pd.Timestamp('2024-03-23'),
        2025: pd.Timestamp('2025-03-22'),
        2026: pd.Timestamp('2026-03-28'),
    }
    df['_season_open'] = df['game_date'].dt.year.map(KBO_OPENING_DAYS)
    df['_days_into_season'] = (df['game_date'] - df['_season_open']).dt.days
    df['Is_Opening_Series'] = (df['_days_into_season'] <= 14).astype(int)
    df.drop(columns=['_season_open', '_days_into_season'], inplace=True)

    # ─────────────────────────────────────────
    # G. 선발투수 등판 간격
    # ─────────────────────────────────────────
    df['Home_SP_Rest_Days'] = df['home_sp_rest_days'].clip(upper=30)
    df['Away_SP_Rest_Days'] = df['away_sp_rest_days'].clip(upper=30)
    df['Diff_SP_Rest_Days'] = df['Home_SP_Rest_Days'] - df['Away_SP_Rest_Days']

    # ─────────────────────────────────────────
    # I. 구장별 투수/타자 피처
    # ─────────────────────────────────────────
    # I-1. 선발투수의 해당 구장 ERA/WHIP (없으면 시즌 전체 스탯 fallback)
    if 'home_sp_stadium_ERA' in df.columns:
        df['Home_SP_Stadium_ERA']  = df['home_sp_stadium_ERA'].fillna(df['Home_SP_ERA_ParkAdj'])
        df['Away_SP_Stadium_ERA']  = df['away_sp_stadium_ERA'].fillna(df['Away_SP_ERA_ParkAdj'])
        df['Home_SP_Stadium_WHIP'] = df['home_sp_stadium_WHIP'].fillna(df['Home_SP_WHIP'])
        df['Away_SP_Stadium_WHIP'] = df['away_sp_stadium_WHIP'].fillna(df['Away_SP_WHIP'])
    else:
        df['Home_SP_Stadium_ERA']  = df['Home_SP_ERA_ParkAdj']
        df['Away_SP_Stadium_ERA']  = df['Away_SP_ERA_ParkAdj']
        df['Home_SP_Stadium_WHIP'] = df['Home_SP_WHIP']
        df['Away_SP_Stadium_WHIP'] = df['Away_SP_WHIP']
    df['Diff_SP_Stadium_ERA']  = df['Home_SP_Stadium_ERA']  - df['Away_SP_Stadium_ERA']

    # I-2. 타자뒤의 해당 구장 평균 OPS (없으면 시즌 전체 OPS fallback)
    if 'home_bat_stadium_OPS' in df.columns:
        df['Home_Bat_Stadium_OPS'] = df['home_bat_stadium_OPS'].fillna(df['Home_Team_Avg_OPS'])
        df['Away_Bat_Stadium_OPS'] = df['away_bat_stadium_OPS'].fillna(df['Away_Team_Avg_OPS'])
    else:
        df['Home_Bat_Stadium_OPS'] = df['Home_Team_Avg_OPS']
        df['Away_Bat_Stadium_OPS'] = df['Away_Team_Avg_OPS']
    df['Diff_Bat_Stadium_OPS'] = df['Home_Bat_Stadium_OPS'] - df['Away_Bat_Stadium_OPS']

    # ─────────────────────────────────────────
    # H. 결측치 처리 (중앙값으로 대체)
    # ─────────────────────────────────────────
    feature_cols = _get_feature_cols()
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def _get_feature_cols() -> list:
    """모델 학습에 사용할 피처 컬럼 목록을 반환합니다."""
    return [
        # 투수 - 홈
        "Home_SP_ERA_ParkAdj", "Home_SP_WAR", "Home_SP_WHIP",
        "Home_SP_RecentERA", "Home_SP_VsOpp_BA", "Home_SP_VsOpp_OPS",
        "Home_Bullpen_WHIP_Penalty", "Home_BP_WAR",
        # 투수 - 어웨이
        "Away_SP_ERA_ParkAdj", "Away_SP_WAR", "Away_SP_WHIP",
        "Away_SP_RecentERA", "Away_SP_VsOpp_BA", "Away_SP_VsOpp_OPS",
        "Away_Bullpen_WHIP_Penalty", "Away_BP_WAR",
        # 타자 - 홈
        "Home_Team_Avg_OPS", "Home_Team_Avg_RISP", "Home_Team_Avg_RS9",
        "Home_Team_OPS_Platoon",
        # 타자 - 어웨이
        "Away_Team_Avg_OPS", "Away_Team_Avg_RISP", "Away_Team_Avg_RS9",
        "Away_Team_OPS_Platoon",
        # 차분 피처
        "Diff_SP_ERA_ParkAdj", "Diff_SP_WAR", "Diff_SP_RecentERA",
        "Diff_Bullpen_WHIP_Penalty", "Diff_Team_OPS_Platoon", "Diff_Team_Avg_RISP",
        # 공통
        "Is_Day_Game",
        # 최근 팀 컨디션 (개별)
        "Home_Recent_Win_Rate", "Away_Recent_Win_Rate",
        "Home_Recent_Run_Diff", "Away_Recent_Run_Diff",
        "Home_Win_Streak", "Away_Win_Streak",
        # 최근 컨디션 차분
        "Diff_Recent_Win_Rate", "Diff_Recent_Run_Diff", "Diff_Win_Streak",
        # 개막 시즌
        "Is_Opening_Series",
        # 선발투수 등판 간격
        "Home_SP_Rest_Days", "Away_SP_Rest_Days", "Diff_SP_Rest_Days",
        # 구장별 선발투수 ERA/WHIP
        "Home_SP_Stadium_ERA", "Away_SP_Stadium_ERA", "Diff_SP_Stadium_ERA",
        "Home_SP_Stadium_WHIP", "Away_SP_Stadium_WHIP",
        # 구장별 타자 OPS
        "Home_Bat_Stadium_OPS", "Away_Bat_Stadium_OPS", "Diff_Bat_Stadium_OPS",
    ]


# ──────────────────────────────────────────────
# 4. 모델 정의
# ──────────────────────────────────────────────
def get_models() -> dict:
    """XGBoost, LightGBM 모델 딕셔너리를 반환합니다."""
    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }
    return models


# ──────────────────────────────────────────────
# 5. 학습 및 평가 (Time-Series Split)
# ──────────────────────────────────────────────
def train_and_evaluate(df_processed: pd.DataFrame) -> dict:
    """
    전처리 완료된 DataFrame을 받아 Time-Series Cross-Validation으로
    각 모델을 학습·평가하고, 마지막 Fold에서 학습한 모델 객체를 반환합니다.

    Returns
    -------
    dict : {"모델명": {"model": fitted_model, "metrics": {...}, "scaler": scaler}}
    """
    feature_cols = _get_feature_cols()
    X = df_processed[feature_cols].values
    y = df_processed["result"].values

    tscv    = TimeSeriesSplit(n_splits=N_SPLITS)
    models  = get_models()
    results = {}

    print("=" * 60)
    print(f"  Time-Series {N_SPLITS}-Fold Cross-Validation 시작")
    print(f"  전체 샘플 수: {len(X):,}  |  피처 수: {len(feature_cols)}")
    print("=" * 60)

    for model_name, model in models.items():
        fold_metrics = {"accuracy": [], "roc_auc": [], "log_loss": []}
        last_model  = None
        last_scaler = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 스케일링 (각 폴드마다 train 기준 fit)
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)

            # 학습
            model.fit(X_train, y_train)

            # 예측
            y_pred      = model.predict(X_val)
            y_pred_prob = model.predict_proba(X_val)[:, 1]

            # 평가 지표 수집
            acc  = accuracy_score(y_val, y_pred)
            auc  = roc_auc_score(y_val, y_pred_prob)
            loss = log_loss(y_val, y_pred_prob)

            fold_metrics["accuracy"].append(acc)
            fold_metrics["roc_auc"].append(auc)
            fold_metrics["log_loss"].append(loss)

            print(f"  [{model_name}] Fold {fold:2d} | "
                  f"Accuracy: {acc:.4f}  ROC-AUC: {auc:.4f}  LogLoss: {loss:.4f}")

            last_model  = model
            last_scaler = scaler

        # 폴드 평균
        avg = {k: np.mean(v) for k, v in fold_metrics.items()}
        print(f"\n  ▶ [{model_name}] 평균 | "
              f"Accuracy: {avg['accuracy']:.4f}  "
              f"ROC-AUC: {avg['roc_auc']:.4f}  "
              f"LogLoss: {avg['log_loss']:.4f}\n")

        results[model_name] = {
            "model":   last_model,
            "scaler":  last_scaler,
            "metrics": avg,
        }

    return results


# ──────────────────────────────────────────────
# 6. 승리 확률 예측
# ──────────────────────────────────────────────
def predict_win_probability(
    df_new: pd.DataFrame,
    fitted_results: dict,
    model_name: str = "LightGBM",
) -> pd.DataFrame:
    """
    새 경기 데이터(df_new)에 대해 지정 모델의 홈팀 승리 확률(%)을 예측합니다.

    Parameters
    ----------
    df_new        : 원본 형태의 새 경기 데이터 (preprocess 전 Raw 형태)
    fitted_results: train_and_evaluate() 반환값
    model_name    : 사용할 모델명 ("XGBoost" 또는 "LightGBM")

    Returns
    -------
    pd.DataFrame : 원본 컬럼 + 예측 결과 컬럼(Pred_Result, Win_Prob_Pct) 추가
    """
    feature_cols = _get_feature_cols()
    df_proc      = preprocess_data(df_new)

    model      = fitted_results[model_name]["model"]
    scaler     = fitted_results[model_name]["scaler"]
    calibrator = fitted_results[model_name].get("calibrator")

    X_new = scaler.transform(df_proc[feature_cols].values)
    if calibrator is not None:
        pred_proba = calibrator.predict_proba(X_new)[:, 1]   # Platt 보정
    else:
        pred_proba = model.predict_proba(X_new)[:, 1]
    pred_label = (pred_proba >= 0.5).astype(int)

    df_out = df_new.copy()
    df_out["Pred_Result"]    = pred_label            # 1 = 홈팀 승, 0 = 홈팀 패
    df_out["Win_Prob_Pct"]   = (pred_proba * 100).round(1)  # 홈팀 승리 확률 (%)

    return df_out[["game_id", "game_date", "home_team", "away_team",
                   "Pred_Result", "Win_Prob_Pct"]]


# ──────────────────────────────────────────────
# 7. 피처 중요도 시각화
# ──────────────────────────────────────────────
def plot_feature_importance(fitted_results: dict, top_n: int = 20) -> None:
    """각 모델의 피처 중요도 상위 N개를 시각화합니다."""
    feature_cols = _get_feature_cols()

    fig, axes = plt.subplots(1, len(fitted_results), figsize=(14, 7), sharey=False)
    if len(fitted_results) == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, fitted_results.items()):
        model = res["model"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            print(f"[경고] {model_name} 은 feature_importances_ 미지원")
            continue

        idx   = np.argsort(importances)[::-1][:top_n]
        names = [feature_cols[i] for i in idx]
        vals  = importances[idx]

        ax.barh(names[::-1], vals[::-1], color="steelblue")
        ax.set_title(f"{model_name} 피처 중요도 Top {top_n}", fontsize=12)
        ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig("feature_importance.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✅ 피처 중요도 이미지 저장 완료: feature_importance.png")


# ──────────────────────────────────────────────
# 8. 모델 저장 / 불러오기
# ──────────────────────────────────────────────
MODEL_DIR = Path("models")


def save_models(fitted_results: dict) -> None:
    """
    학습된 모델과 스케일러를 models/ 디렉터리에 저장합니다.

    저장 파일:
        models/{ModelName}_model.pkl   - 학습된 분류 모델
        models/{ModelName}_scaler.pkl  - 학습에 사용한 StandardScaler
    """
    MODEL_DIR.mkdir(exist_ok=True)
    for model_name, res in fitted_results.items():
        model_path  = MODEL_DIR / f"{model_name}_model.pkl"
        scaler_path = MODEL_DIR / f"{model_name}_scaler.pkl"
        joblib.dump(res["model"],  model_path)
        joblib.dump(res["scaler"], scaler_path)
        print(f"  💾 [{model_name}] 저장 완료 → {model_path}, {scaler_path}")


def load_models(model_names: list = None) -> dict:
    """
    models/ 디렉터리에서 저장된 모델과 스케일러를 불러옵니다.

    Parameters
    ----------
    model_names : list or None
        불러올 모델명 리스트. None이면 디렉터리 내 전체 모델 자동 탐색.

    Returns
    -------
    dict : {"ModelName": {"model": ..., "scaler": ...}}
    """
    if model_names is None:
        model_names = [
            p.stem.replace("_model", "")
            for p in MODEL_DIR.glob("*_model.pkl")
        ]

    results = {}
    for model_name in model_names:
        model_path  = MODEL_DIR / f"{model_name}_model.pkl"
        scaler_path = MODEL_DIR / f"{model_name}_scaler.pkl"
        if not model_path.exists() or not scaler_path.exists():
            print(f"  ⚠️  [{model_name}] 저장 파일 없음 — 건너뜀")
            continue
        results[model_name] = {
            "model":  joblib.load(model_path),
            "scaler": joblib.load(scaler_path),
        }
        print(f"  📂 [{model_name}] 불러오기 완료 ← {model_path}")
        cal_path = MODEL_DIR / f"{model_name}_calibrator.pkl"
        if cal_path.exists():
            results[model_name]["calibrator"] = joblib.load(cal_path)
            print(f"  📐 [{model_name}] Platt 캘리브레이터 로드 완료 ← {cal_path}")
    return results


def calibrate_and_save(df_processed: pd.DataFrame, fitted_results: dict) -> dict:
    """
    [4순위 개선] Platt Scaling을 이용해 확률 캘리브레이션 적용.
    홀드아웃 마지막 20%에 cv='prefit' 시그모이드 피팅.
    저장: models/{ModelName}_calibrator.pkl
    """
    feature_cols = _get_feature_cols()
    X = df_processed[feature_cols].values
    y = df_processed["result"].values
    split_idx = int(len(X) * 0.8)
    X_cal, y_cal = X[split_idx:], y[split_idx:]
    print(f"  쾔리브레이션 피팅 세트: {len(X_cal)}경기")

    MODEL_DIR.mkdir(exist_ok=True)
    for model_name, res in fitted_results.items():
        X_scaled = res["scaler"].transform(X_cal)
        calibrator = CalibratedClassifierCV(
            estimator=res["model"], cv="prefit", method="sigmoid"
        )
        calibrator.fit(X_scaled, y_cal)
        cal_path = MODEL_DIR / f"{model_name}_calibrator.pkl"
        joblib.dump(calibrator, cal_path)
        fitted_results[model_name]["calibrator"] = calibrator
        print(f"  📐 [{model_name}] Platt 캘리브레이터 저장 → {cal_path}")
    return fitted_results


# ──────────────────────────────────────────────
# 9. 마지막 Fold 상세 평가 리포트
# ──────────────────────────────────────────────
def print_final_report(df_processed: pd.DataFrame, fitted_results: dict) -> None:
    """
    전체 데이터 중 마지막 20%를 홀드아웃 테스트셋으로 삼아
    각 모델의 분류 리포트와 혼동 행렬을 출력합니다.
    """
    feature_cols = _get_feature_cols()
    X = df_processed[feature_cols].values
    y = df_processed["result"].values

    split_idx = int(len(X) * 0.8)
    X_test, y_test = X[split_idx:], y[split_idx:]

    print("\n" + "=" * 60)
    print("  최종 홀드아웃 테스트셋 평가 리포트 (마지막 20% 데이터)")
    print("=" * 60)

    for model_name, res in fitted_results.items():
        model  = res["model"]
        scaler = res["scaler"]

        X_test_scaled = scaler.transform(X_test)
        y_pred        = model.predict(X_test_scaled)
        y_prob        = model.predict_proba(X_test_scaled)[:, 1]

        print(f"\n[ {model_name} ]")
        print(classification_report(y_test, y_pred,
                                    target_names=["홈팀 패(0)", "홈팀 승(1)"]))
        print("혼동 행렬:")
        print(confusion_matrix(y_test, y_pred))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")


# ──────────────────────────────────────────────
# 10. 메인 실행 파이프라인
# ──────────────────────────────────────────────
def main(csv_path: str = None) -> None:
    """
    End-to-End 파이프라인 실행 함수.

    Parameters
    ----------
    csv_path : str or None
        실제 원본 데이터 CSV 경로.
        None이면 샘플 데이터를 자동 생성하여 실행합니다.

    실제 데이터 사용 예시
    ---------------------
        main(csv_path="/Users/younghoon-kang/Data_Analysis_Project/raw_data.csv")
    """
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass
    # ── 데이터 로드
    if csv_path:
        print(f"  📂 실제 데이터 로드: {csv_path}")
        df_raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    else:
        print("  ℹ️  원본 데이터 없음 → 샘플 데이터(500경기) 자동 생성")
        df_raw = generate_sample_data(n=500)

    print(f"  ✅ 데이터 로드 완료: {df_raw.shape[0]}행 × {df_raw.shape[1]}열\n")

    # ── 전처리 & Feature Engineering
    print("  🔧 전처리 및 Feature Engineering 시작...")
    df_processed = preprocess_data(df_raw)
    print(f"  ✅ 전처리 완료: 피처 컬럼 {len(_get_feature_cols())}개 생성\n")

    # ── 모델 학습 및 Time-Series CV 평가
    fitted_results = train_and_evaluate(df_processed)

    # ── 홀드아웃 최종 리포트
    print_final_report(df_processed, fitted_results)

    # ── 피처 중요도 시각화
    print("\n  📊 피처 중요도 시각화 중...")
    plot_feature_importance(fitted_results, top_n=20)

    # ── 모델 저장
    print("\n  💾 모델 저장 중...")
    save_models(fitted_results)

    # ── Platt Scaling 캘리브레이션 (4순위 개선)
    print("\n  📐 Platt Scaling 캘리브레이션 적용 중...")
    calibrate_and_save(df_processed, fitted_results)

    # ── 샘플 승리 확률 예측 (마지막 5경기)
    print("\n" + "=" * 60)
    print("  🎯 승리 확률 예측 샘플 (마지막 5경기)")
    print("=" * 60)
    df_sample = df_raw.tail(5).copy()
    for model_name in fitted_results:
        print(f"\n  [ {model_name} 예측 결과 ]")
        df_pred = predict_win_probability(df_sample, fitted_results, model_name)
        print(df_pred.to_string(index=False))

    print("\n  🏁 파이프라인 전체 실행 완료!")


# ──────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────
    # ▼ build_raw_data.py로 생성한 실제 데이터를 사용합니다.
    # ─────────────────────────────────────────────────────────────────
    RAW_DATA_PATH = "data/raw_data.csv"

    main(csv_path=RAW_DATA_PATH)
