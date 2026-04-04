# ⚾ KBO 승부 예측 머신러닝 프로젝트

세이버메트릭스 기반 KBO 경기 데이터를 활용하여 **홈팀의 승/패 및 승리 확률**을 예측하고, Statiz API로 결과를 제출하는 종단간(End-to-End) 머신러닝 파이프라인입니다.

> **학습 데이터:** 2023~2026 KBO 정규시즌 경기 (2,145경기)  
> **현재 모델 성능:** XGBoost Holdout 56.0% / LightGBM Holdout 58.0%  
> **마지막 데이터 업데이트:** 2026-04-03

---

## 📁 프로젝트 구조

```
Data_Analysis_Project/
│
├── 📄 스크립트 (실행 순서대로)
│   ├── fetch_game_results.py        # [1] Statiz API → 경기결과·라인업 수집
│   ├── fetch_pitcher_stats.py       # [2] 투수 시즌/게임로그/시추에이션 수집
│   ├── fetch_hitter_stats.py        # [3] 타자 시즌/게임로그/시추에이션 수집
│   ├── fetch_player_situations.py   # [4] 선수 상황별(좌우/RISP) 스탯 수집
│   ├── append_2026_games.py         # [5] 신규 경기 결과 CSV에 추가
│   ├── build_raw_data.py            # [6] 수집 데이터 → 피처 엔지니어링 → raw_data.csv
│   ├── baseball_baseline.py         # [7] 모델 학습·평가·저장 (XGBoost, LightGBM)
│   ├── predict_2026.py              # [8] 특정 날짜 예측 (오프라인 배치)
│   └── submit_predictions_today.py  # [9] 오늘 경기 예측 + API 자동 제출
│
├── 📦 데이터 (git 제외)
│   └── data/
│       ├── game_results_2023_2025.csv       # 경기 결과 원본
│       ├── game_lineups_2023_2025.csv       # 라인업 원본
│       ├── pitcher_*/hitter_* CSV들         # 투타 스탯 원본
│       ├── raw_data.csv                     # 피처 엔지니어링 완료본
│       └── predictions_YYYYMMDD.csv         # 일별 예측 결과
│
├── 📦 모델 (git 제외)
│   └── models/
│       ├── XGBoost_model.pkl / _scaler.pkl / _calibrator.pkl
│       └── LightGBM_model.pkl / _scaler.pkl / _calibrator.pkl
│
├── 📚 문서
│   ├── README.md                    # 프로젝트 개요 (현재 파일)
│   ├── PIPELINE_GUIDE.md            # 전체 파이프라인 상세 가이드
│   ├── DATA_PREPROCESSING_GUIDE.md  # 원본 데이터 컬럼 명세
│   └── claude.md                    # 모델링 설계 지침 (Baseline V1.0)
│
├── 🔐 인증 (git 제외)
│   ├── .env                         # API Key / Secret (절대 커밋 금지)
│   └── .env.example                 # 환경변수 템플릿
│
└── docs/                            # Statiz API 명세 CSV (git 제외)
```

---

## 🔄 전체 파이프라인 흐름

```
[Statiz API]
     │
     ▼
fetch_*.py              ← 경기결과 / 라인업 / 투타 스탯 / 시추에이션 수집
     │
     ▼
append_2026_games.py    ← 신규 경기 결과를 기존 CSV에 누적 append
     │
     ▼
build_raw_data.py       ← 투수ERA·WHIP·WAR, 불펜페널티, 타자OPS·RISP·플래툰
                           구장 파크팩터, rolling 컨디션 → raw_data.csv (50열)
     │
     ▼
baseball_baseline.py    ← Time-Series 5-Fold CV → XGBoost·LightGBM 학습
                           Platt Scaling 캘리브레이션 → models/ 저장
     │
     ▼
submit_predictions_today.py  ← 오늘 라인업 수집 → 앙상블 예측
                                → POST /prediction/savePrediction 제출
```

---

## 🤖 모델

| 모델       | Holdout Accuracy | ROC-AUC | 비고                          |
| ---------- | ---------------- | ------- | ----------------------------- |
| XGBoost    | 56.0%            | 0.5898  | Gradient Boosting             |
| LightGBM   | 58.0%            | 0.5805  | Gradient Boosting (빠른 학습) |
| **앙상블** | —                | —       | XGB·LGB 확률 평균             |

- 검증: **Time-Series Split 5-Fold** (미래 데이터 누수 방지)
- 캘리브레이션: **Platt Scaling** (`predict_proba` 확률 보정)
- 학습 데이터: 2023-01 ~ 2026-04-03 (2,145경기)

---

## 🔧 주요 피처 (Feature Engineering)

### 투수 지표

| 피처                             | 설명                                    |
| -------------------------------- | --------------------------------------- |
| `Home/Away_SP_ERA_ParkAdj`       | 선발 ERA × 구장 파크 팩터 보정          |
| `Home/Away_SP_RecentERA`         | 최근 5경기 평균자책점 (`ER×9 / IP`)     |
| `Home/Away_SP_VsOpp_BA/OPS`      | 당일 상대팀 대비 천적 지표              |
| `Home/Away_Bullpen_WHIP_Penalty` | 불펜 WHIP 리그평균(1.35) 초과 시 페널티 |

### 타자 지표

| 피처                         | 설명                                     |
| ---------------------------- | ---------------------------------------- |
| `Home/Away_Team_Avg_OPS`     | 선발 라인업 9명 평균 OPS                 |
| `Home/Away_Team_Avg_RISP`    | 선발 라인업 9명 평균 득점권 타율         |
| `Home/Away_Team_OPS_Platoon` | 상대 선발 투수 유형(좌/우/언더) 보정 OPS |

### 팀 컨디션 (Rolling)

| 피처                      | 설명                    |
| ------------------------- | ----------------------- |
| `Home/Away_RecentWinRate` | 최근 10경기 승률        |
| `Home/Away_RecentRunDiff` | 최근 10경기 평균 득실차 |
| `Home/Away_Streak`        | 현재 연승/연패 수       |

### 대결 차분 피처

홈 − 어웨이 값을 차감 → 상대적 강약을 모델이 직접 학습
(`Diff_SP_ERA_ParkAdj`, `Diff_SP_WAR`, `Diff_Team_OPS_Platoon` 등)

### 공통

- `Is_Day_Game`: 14시 이전 경기 = 1 (Day), 이후 = 0 (Night)

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install xgboost lightgbm scikit-learn pandas numpy matplotlib python-dotenv
```

### 2. API 키 설정

```bash
cp .env.example .env
# .env 파일을 열어 STATIZ_API_KEY, STATIZ_SECRET 입력
```

### 3. 데이터 수집 (최초 1회)

```bash
python fetch_game_results.py       # 경기결과 + 라인업
python fetch_pitcher_stats.py      # 투수 스탯
python fetch_hitter_stats.py       # 타자 스탯
python fetch_player_situations.py  # 상황별 스탯
```

### 4. 피처 빌드 + 모델 학습

```bash
python build_raw_data.py           # → data/raw_data.csv 생성
python baseball_baseline.py        # → models/*.pkl 저장
```

### 5. 오늘 경기 예측 + 제출

```bash
python submit_predictions_today.py            # 전체 제출
python submit_predictions_today.py --dry-run  # 예측만 출력 (제출 없음)
python submit_predictions_today.py --only 두산 한화  # 특정 팀만 제출
```

> ⚠️ 제출 마감: **경기 시작 15분 전**까지 (17:00 경기는 16:45 마감)

---

## 📅 매일 반복 워크플로

```bash
# 1. 전날 경기 결과 추가 (--cutoff를 오늘 날짜로)
python append_2026_games.py --cutoff 2026-04-04

# 2. 피처 재빌드
python build_raw_data.py

# 3. 모델 재학습
python baseball_baseline.py

# 4. 오늘 경기 제출 (라인업 공개 후, 경기 시작 1시간 전 권장)
python submit_predictions_today.py
```

---

## 📊 출력 결과

```
game_time  home_team  away_team  XGB_Prob  LGB_Prob  Ensemble_Prob  Pred_Winner
    14:00        두산        한화      43.5      41.5          42.50          한화
    17:00        롯데       SSG      53.3      56.4          54.85          롯데
    17:00       KIA        NC      56.9      54.5          55.70         KIA
```

- `Ensemble_Prob`: 홈팀 승리 확률 (XGB + LGB 평균)
- `Pred_Winner`: 50% 기준 승자 예측
- 결과는 `data/predictions_YYYYMMDD.csv`로 자동 저장

---

## 📋 관련 문서

| 문서                                                         | 내용                                  |
| ------------------------------------------------------------ | ------------------------------------- |
| [PIPELINE_GUIDE.md](./PIPELINE_GUIDE.md)                     | 전체 파이프라인 상세 설명, 트러블슈팅 |
| [DATA_PREPROCESSING_GUIDE.md](./DATA_PREPROCESSING_GUIDE.md) | raw_data.csv 컬럼 명세 (50열)         |
| [claude.md](./claude.md)                                     | 모델링 설계 지침 Baseline V1.0        |

---

## ⚙️ 주요 상수

| 상수                    | 기본값         | 파일                          | 설명                       |
| ----------------------- | -------------- | ----------------------------- | -------------------------- |
| `DAY_CUTOFF_HOUR`       | `14`           | `baseball_baseline.py`        | 낮 경기 기준 시각          |
| `LEAGUE_AVG_BP_WHIP`    | `1.35`         | `build_raw_data.py`           | 불펜 WHIP 리그 평균        |
| `BP_WHIP_PENALTY_SCALE` | `2.0`          | `build_raw_data.py`           | 불펜 WHIP 페널티 배율      |
| `N_SPLITS`              | `5`            | `baseball_baseline.py`        | Time-Series CV 폴드 수     |
| `TARGET_DATE`           | `"2026-04-04"` | `submit_predictions_today.py` | 제출 대상 날짜 (매일 수정) |
