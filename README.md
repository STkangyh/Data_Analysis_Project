# ⚾ 야구 승부 예측 머신러닝 프로젝트

세이버메트릭스 기반 야구 경기 데이터를 활용하여 **홈팀의 승/패 및 승리 확률**을 예측하는 머신러닝 파이프라인입니다.

---

## 📁 프로젝트 구조

```
Data_Analysis_Project/
│
├── baseball_baseline.py          # 메인 파이프라인 (전처리 → 학습 → 예측)
├── DATA_PREPROCESSING_GUIDE.md   # 데이터 담당자용 원본 데이터 명세서
├── claude.md                     # 모델링 설계 지침 (Baseline V1.0)
├── feature_importance.png        # 피처 중요도 시각화 결과
└── README.md                     # 프로젝트 개요 (현재 파일)
```

---

## 🔄 파이프라인 흐름

```
Raw Data (CSV)
      │
      ▼
preprocess_data()       ← 투수/타자/불펜/구장 피처 엔지니어링, 결측치 처리
      │
      ▼
train_and_evaluate()    ← Time-Series 5-Fold Cross-Validation (XGBoost, LightGBM)
      │
      ▼
predict_win_probability()  ← predict_proba() 기반 승리 확률(%) 출력
```

---

## 🤖 모델

| 모델 | 라이브러리 | 비고 |
|---|---|---|
| XGBoost | `xgboost` | Gradient Boosting |
| LightGBM | `lightgbm` | Gradient Boosting (빠른 학습) |

- 검증 방식: **Time-Series Split** (미래 데이터 누수 방지)
- 평가 지표: Accuracy, ROC-AUC, Log Loss
- 최종 평가: 마지막 20% 홀드아웃 테스트셋

---

## 🔧 주요 피처 (Feature Engineering)

### 투수 지표
| 피처 | 설명 |
|---|---|
| `Home/Away_SP_ERA_ParkAdj` | 선발 ERA × 구장 파크 팩터 보정 |
| `Home/Away_SP_RecentERA` | 최근 5경기 평균자책점 (`ER×9 / IP`) |
| `Home/Away_SP_VsOpp_BA/OPS` | 당일 상대팀 대비 천적 지표 |
| `Home/Away_Bullpen_WHIP_Penalty` | 불펜 WHIP이 리그 평균 초과 시 페널티 (`max(0, WHIP-1.35) × 2.0`) |

### 타자 지표
| 피처 | 설명 |
|---|---|
| `Home/Away_Team_Avg_OPS` | 선발 라인업 9명 평균 OPS |
| `Home/Away_Team_Avg_RISP` | 선발 라인업 9명 평균 득점권 타율 |
| `Home/Away_Team_OPS_Platoon` | 상대 선발 유형(좌/우/언더) 보정 OPS |

### 대결 차분 피처
홈 - 어웨이 값을 직접 차감하여 **상대적 강약**을 모델이 학습하도록 합니다.
(`Diff_SP_ERA_ParkAdj`, `Diff_SP_WAR`, `Diff_Team_OPS_Platoon` 등 6개)

### 공통
- `Is_Day_Game`: 14시 이전 경기 = 1 (Day), 이후 = 0 (Night)

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 의존 패키지 설치
pip install xgboost lightgbm scikit-learn pandas numpy matplotlib

# macOS: XGBoost 실행에 필요한 OpenMP 설치
brew install libomp
```

### 2. 샘플 데이터로 동작 확인 (데이터 없어도 즉시 실행 가능)

```bash
python baseball_baseline.py
```

### 3. 실제 데이터로 실행

`baseball_baseline.py` 하단의 경로를 수정합니다.

```python
# baseball_baseline.py 하단
RAW_DATA_PATH = "/Users/younghoon-kang/Data_Analysis_Project/raw_data.csv"
```

그 후 동일하게 실행:

```bash
python baseball_baseline.py
```

---

## 📊 출력 결과

실행 시 아래 결과가 순서대로 출력됩니다.

1. **Time-Series CV 폴드별 성능** (Accuracy / ROC-AUC / Log Loss)
2. **홀드아웃 테스트셋 분류 리포트** (Precision / Recall / F1)
3. **피처 중요도 시각화** → `feature_importance.png` 저장
4. **승리 확률 예측 샘플** (마지막 5경기, 확률 % 포함)

예시 출력:
```
game_id  game_date  home_team  away_team  Pred_Result  Win_Prob_Pct
  G0496 2023-08-09        KIA       삼성            1          95.5
  G0497 2023-08-10        SSG       롯데            0          26.5
```

---

## 📋 데이터 담당자 안내

원본 데이터(Raw Data) 구성 방법은 아래 문서를 참고하세요.

👉 **[DATA_PREPROCESSING_GUIDE.md](./DATA_PREPROCESSING_GUIDE.md)**

- 필수 컬럼 36개 전체 명세 (컬럼명, 타입, 정상 범위, 예시 값)
- CSV 포맷 규칙 및 자주 하는 실수 정리
- 샘플 CSV 2행 예시 포함

---

## ⚙️ 주요 상수 (baseball_baseline.py 내 조정 가능)

| 상수 | 기본값 | 설명 |
|---|---|---|
| `DAY_CUTOFF_HOUR` | `14` | 낮 경기 기준 시각 (시) |
| `LEAGUE_AVG_BP_WHIP` | `1.35` | 불펜 WHIP 리그 평균 기준값 |
| `BP_WHIP_PENALTY_SCALE` | `2.0` | 불펜 WHIP 페널티 배율 |
| `N_SPLITS` | `5` | Time-Series CV 폴드 수 |