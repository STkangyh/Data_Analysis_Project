# 📘 KBO 승부 예측 파이프라인 상세 가이드

> 이 문서는 프로젝트의 전체 파이프라인을 스크립트 단위로 설명합니다.  
> 신규 팀원 온보딩, 디버깅, 정기 운영에 참고하세요.

---

## 목차

1. [아키텍처 개요](#1-아키텍처-개요)
2. [환경 설정](#2-환경-설정)
3. [데이터 수집 스크립트](#3-데이터-수집-스크립트)
4. [피처 엔지니어링 (build_raw_data.py)](#4-피처-엔지니어링)
5. [모델 학습 (baseball_baseline.py)](#5-모델-학습)
6. [당일 예측 및 제출 (submit_predictions_today.py)](#6-당일-예측-및-제출)
7. [API 인증 상세](#7-api-인증-상세)
8. [매일 운영 워크플로](#8-매일-운영-워크플로)
9. [데이터 파일 명세](#9-데이터-파일-명세)
10. [모델 성능 이력](#10-모델-성능-이력)
11. [트러블슈팅](#11-트러블슈팅)

---

## 1. 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                      Statiz API v4                          │
│  /prediction/gameSchedule  /prediction/gameLineup           │
│  /player/pitcher/season    /player/hitter/season  ...       │
└──────────────────────┬──────────────────────────────────────┘
                       │ HMAC-SHA256 인증
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   데이터 수집 레이어                          │
│  fetch_game_results.py  fetch_pitcher_stats.py              │
│  fetch_hitter_stats.py  fetch_player_situations.py          │
│  append_2026_games.py                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ CSV 누적 저장
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  피처 엔지니어링 레이어                       │
│  build_raw_data.py                                          │
│  → 투수ERA/WAR/WHIP·불펜페널티·타자OPS/RISP/플래툰          │
│  → 구장 파크팩터·롤링 컨디션·차분 피처                       │
│  → data/raw_data.csv (2,145행 × 50열)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    모델 학습 레이어                           │
│  baseball_baseline.py                                       │
│  → Time-Series 5-Fold CV                                    │
│  → XGBoost + LightGBM + Platt Scaling 캘리브레이션           │
│  → models/*.pkl 저장                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   예측 및 제출 레이어                         │
│  submit_predictions_today.py                                │
│  → 당일 라인업 수집 → 앙상블 예측                            │
│  → POST /prediction/savePrediction                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 환경 설정

### 2-1. 패키지 설치

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install xgboost lightgbm scikit-learn pandas numpy matplotlib python-dotenv
```

### 2-2. API 키 설정

`.env.example`을 복사해 `.env` 파일 생성:

```bash
cp .env.example .env
```

`.env` 내용 (실제 값 입력):

```
STATIZ_API_KEY=발급받은_API_키
STATIZ_SECRET=발급받은_시크릿
STATIZ_API_BASE_URL=https://api.statiz.co.kr
```

> ⚠️ `.env`는 `.gitignore`에 등록되어 있어 절대 git에 올라가지 않습니다.

---

## 3. 데이터 수집 스크립트

### 3-1. fetch_game_results.py — 경기결과 + 라인업

**역할:** 연도별 경기 결과(승패·점수·선발 투수)와 라인업(타순·타자 p_no) 수집

**출력:**

- `data/game_results_2023_2025.csv` — 경기별 1행 (game_id, date, home/away, 선발투수, 점수, 승패)
- `data/game_lineups_2023_2025.csv` — 선수별 1행 (game_id, team, 타순, p_no, 포지션)

**실행:**

```bash
python fetch_game_results.py
```

**특이사항:**

- `StatizAPIClient` 클래스: HMAC-SHA256 서명 인증 담당
- 라인업의 `p_no`가 null인 선수 행은 `build_raw_data.py`에서 자동 제외됨

---

### 3-2. fetch_pitcher_stats.py — 투수 스탯

**역할:** 투수별 시즌 성적, 경기 로그, 시추에이션 스탯 수집

**출력:**

- `data/pitcher_season_stats_2023_2025.csv`
- `data/pitcher_game_log_2023_2025.csv`
- `data/pitcher_situations_2023_2025.csv`
- `data/pitcher_roster_2023_2025.csv`

---

### 3-3. fetch_hitter_stats.py — 타자 스탯

**역할:** 타자별 시즌 성적, 경기 로그, 시추에이션 스탯 수집

**출력:**

- `data/hitter_season_stats_2023_2025.csv`
- `data/hitter_game_log_2023_2025.csv`
- `data/hitter_situations_2023_2025.csv`

---

### 3-4. append_2026_games.py — 신규 경기 추가

**역할:** 새 시즌(2026~) 경기 결과를 기존 CSV에 누적 append

**실행:**

```bash
# --cutoff: 이 날짜까지 수집 (기본: 오늘 -1일)
python append_2026_games.py --cutoff 2026-04-04
```

**동작:**

1. `game_results_2023_2025.csv`에서 마지막 game_id 확인
2. 해당 game_id 이후 경기만 API 수집
3. 결과를 기존 CSV 하단에 append (중복 없음)

---

## 4. 피처 엔지니어링

### build_raw_data.py

**입력:** `data/` 하위 모든 수집 CSV
**출력:** `data/raw_data.csv` (2,145행 × 50열)

### 주요 파생 변수

| 변수                   | 로직                                               |
| ---------------------- | -------------------------------------------------- |
| `SP_ERA_ParkAdj`       | `ERA × park_factor` (구장별 파크팩터 적용)         |
| `SP_RecentERA`         | 최근 5경기 `ER × 9 / IP` 가중평균                  |
| `SP_VsOpp_BA/OPS`      | 현재 시즌 해당 상대팀 피안타율·피OPS               |
| `Bullpen_WHIP_Penalty` | `max(0, bullpen_WHIP − 1.35) × 2.0`                |
| `Team_Avg_OPS`         | 선발 라인업 9명 OPS 평균                           |
| `Team_OPS_Platoon`     | 상대 선발 투수 유형에 맞춘 좌타/우타/양타 OPS 보정 |
| `RecentWinRate`        | 최근 10경기 승률 (rolling)                         |
| `RecentRunDiff`        | 최근 10경기 평균 득실차 (rolling)                  |
| `Streak`               | 현재 연승(양수) / 연패(음수)                       |
| `Diff_*`               | 홈 − 어웨이 차분 피처 (ERA, WAR, OPS 등)           |
| `Is_Day_Game`          | 14시 이전=1, 이후=0                                |

### 외국인 선수 처리

외국인 타자/투수의 한국어 이름은 `FOREIGN_NAME_KO_MAP` 딕셔너리로 매핑:

```python
# predict_2026.py
FOREIGN_NAME_KO_MAP = {
    "BEASLEY": 16433,
    "TAYLOR":  16450,
    ...
}
```

라인업에서 외국인 선수 이름이 탐지되면 `p_no`를 자동 주입합니다.

---

## 5. 모델 학습

### baseball_baseline.py

**입력:** `data/raw_data.csv`
**출력:** `models/` 하위 6개 파일

```
models/
├── XGBoost_model.pkl
├── XGBoost_scaler.pkl
├── XGBoost_calibrator.pkl    ← Platt Scaling
├── LightGBM_model.pkl
├── LightGBM_scaler.pkl
└── LightGBM_calibrator.pkl
```

### 학습 절차

```
raw_data.csv
    │
    ├─ 시간순 정렬
    ├─ 마지막 20% → holdout 테스트셋
    └─ 나머지 80% → Time-Series 5-Fold CV
                         │
                         ├─ XGBoost 학습·평가
                         ├─ LightGBM 학습·평가
                         └─ Platt Scaling 캘리브레이션
                                    │
                                    └─ models/*.pkl 저장
```

### 현재 성능 (2026-04-03 기준, 2,145경기)

| 모델     | Holdout Acc | ROC-AUC | Log Loss |
| -------- | ----------- | ------- | -------- |
| XGBoost  | 56.0%       | 0.5898  | —        |
| LightGBM | 58.0%       | 0.5805  | —        |

> 야구 승부 예측의 이론적 한계(정보 비대칭, 랜덤성)를 감안하면 58%는 실용적 수준입니다.

---

## 6. 당일 예측 및 제출

### submit_predictions_today.py

**역할:** 당일 경기 라인업 수집 → 앙상블 예측 → API 제출

**실행 옵션:**

```bash
# 전체 경기 제출
python submit_predictions_today.py

# 특정 팀이 포함된 경기만 제출
python submit_predictions_today.py --only 두산 한화

# 예측만 출력 (제출 없음)
python submit_predictions_today.py --dry-run
```

**내부 동작 순서:**

```
[1/4] 4월 일정 API 수집 → TARGET_DATE 경기 필터
[2/4] 각 경기 라인업 수집 (선발 라인업 공개 전이면 투수 2명만)
[3/4] game_results CSV 기반 rolling stats 계산 (최근 10경기)
[4/4] 피처 조립 → XGBoost·LightGBM 앙상블 예측
      → POST /prediction/savePrediction 제출
```

**중요 주의사항:**

| 항목             | 내용                                                |
| ---------------- | --------------------------------------------------- |
| 제출 마감        | 경기 시작 **15분 전**까지 (17:00 경기 → 16:45 마감) |
| 라인업 권장 시점 | 경기 시작 **1시간 전** 이후 (16:00~) 재실행         |
| `TARGET_DATE`    | 파일 상단에 하드코딩 — 매일 날짜 업데이트 필요      |

**출력 결과 CSV:** `data/predictions_YYYYMMDD.csv`

---

## 7. API 인증 상세

### Statiz API v4 HMAC-SHA256 인증

모든 API 요청에 아래 3개 헤더 포함:

| 헤더          | 값                            |
| ------------- | ----------------------------- |
| `X-API-KEY`   | 발급받은 API 키               |
| `X-TIMESTAMP` | Unix epoch seconds (10자리)   |
| `X-SIGNATURE` | HMAC-SHA256 서명 (hex string) |

### 서명 payload 구성

```
METHOD|PATH|NORMALIZED_QUERY|TIMESTAMP
```

- **`METHOD`**: HTTP 메서드 그대로 (`GET` 또는 `POST`)
- **`PATH`**: `/baseballApi/` 제외한 상대경로 (예: `prediction/savePrediction`)
- **`NORMALIZED_QUERY`**: 파라미터를 key 오름차순 정렬 후 URL 인코딩
- **`TIMESTAMP`**: epoch seconds 문자열

### Python 서명 코드

```python
import hmac, hashlib, time
from urllib.parse import quote

def make_signature(secret, method, path, params):
    safe = "-_.!~*'()"
    normalized = "&".join(
        f"{quote(str(k), safe=safe)}={quote(str(v), safe=safe)}"
        for k, v in sorted(params.items())
    )
    timestamp = str(int(time.time()))
    payload   = f"{method}|{path}|{normalized}|{timestamp}"
    signature = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return timestamp, signature, normalized
```

### POST savePrediction 요청 방식

서버가 서명 검증과 파라미터 파싱을 분리해서 처리함:

| 항목                 | 위치                         |
| -------------------- | ---------------------------- |
| 서명 검증용 파라미터 | **URL 쿼리스트링**           |
| 파라미터 파싱        | **POST body** (form-encoded) |

따라서 파라미터를 **URL + body 양쪽에 모두** 포함해야 합니다:

```python
url  = f"{base_url}/baseballApi/prediction/savePrediction?{normalized_query}"
body = normalized_query.encode("utf-8")
headers["Content-Type"] = "application/x-www-form-urlencoded"
req = urllib.request.Request(url, data=body, headers=headers, method="POST")
```

### 에러 코드 정리

| 코드 | 메시지                           | 원인 및 해결                        |
| ---- | -------------------------------- | ----------------------------------- |
| 401  | X-API-KEY 없음                   | 헤더 이름 확인                      |
| 401  | X-TIMESTAMP 형식 오류            | epoch seconds (10자리) 확인         |
| 401  | SIGNATURE 검증 실패              | payload 구성·PATH·QUERY 정규화 확인 |
| 403  | 허용되지 않은 IP                 | 등록된 IP 여부 확인                 |
| 403  | 입력 마감: 경기 시작 15분 전까지 | **시간 초과** — 다음 경기 제출 가능 |
| 400  | 필수 값이 누락                   | POST body 파라미터 확인             |

---

## 8. 매일 운영 워크플로

### 경기 당일 루틴

```bash
# ① 전날 경기 결과 수집 (당일 아침)
python append_2026_games.py --cutoff $(date +%Y-%m-%d)

# ② raw_data 재빌드
python build_raw_data.py

# ③ 모델 재학습
python baseball_baseline.py

# ④ submit_predictions_today.py 상단 TARGET_DATE 업데이트
#    TARGET_DATE = "2026-04-05"  ← 오늘 날짜로 변경

# ⑤ 라인업 공개 후 (경기 시작 1시간 전) 예측·제출
python submit_predictions_today.py
```

### 주간 점검 항목

- [ ] `data/game_results_2023_2025.csv` 행 수 증가 확인
- [ ] 모델 holdout accuracy 추이 모니터링
- [ ] 외국인 선수 교체 시 `FOREIGN_NAME_KO_MAP` 업데이트 (`predict_2026.py`)
- [ ] `TARGET_DATE` 매일 업데이트 (TODO: CLI 인수로 개선 예정)

---

## 9. 데이터 파일 명세

### data/game_results_2023_2025.csv

| 컬럼           | 타입 | 설명                          |
| -------------- | ---- | ----------------------------- |
| `game_id`      | int  | 경기 고유 번호 (예: 20260031) |
| `game_date`    | str  | `YYYY-MM-DD`                  |
| `home_team`    | str  | 홈팀 이름                     |
| `away_team`    | str  | 원정팀 이름                   |
| `home_score`   | int  | 홈팀 득점                     |
| `away_score`   | int  | 원정팀 득점                   |
| `home_win`     | int  | 홈팀 승=1, 패=0               |
| `home_sp_p_no` | int  | 홈팀 선발 투수 p_no           |
| `away_sp_p_no` | int  | 원정팀 선발 투수 p_no         |
| `stadium`      | str  | 구장 이름                     |

### data/raw_data.csv (50열)

피처 엔지니어링 최종 결과물. 전체 컬럼 명세는 [DATA_PREPROCESSING_GUIDE.md](./DATA_PREPROCESSING_GUIDE.md) 참고.

### data/predictions_YYYYMMDD.csv

당일 제출 결과 저장:

| 컬럼                      | 설명                   |
| ------------------------- | ---------------------- |
| `game_id`                 | 경기 번호              |
| `home_team` / `away_team` | 팀명                   |
| `XGB_Prob`                | XGBoost 홈팀 승리확률  |
| `LGB_Prob`                | LightGBM 홈팀 승리확률 |
| `Ensemble_Prob`           | 앙상블 확률 (제출값)   |
| `Pred_Winner`             | 예측 승자              |

---

## 10. 모델 성능 이력

| 날짜       | 학습 데이터               | XGB Acc | LGB Acc | XGB AUC | LGB AUC |
| ---------- | ------------------------- | ------- | ------- | ------- | ------- |
| 2026-04-03 | 2023~2026/4/3 (2,145경기) | 56.0%   | 58.0%   | 0.5898  | 0.5805  |

---

## 11. 트러블슈팅

### Q. 라인업 수집 시 2명만 나온다

경기 시작 약 1시간 전에 선발 라인업이 공개됩니다. 2명=선발 투수 양쪽만 등록된 상태입니다. 타자 라인업은 공개 후 재실행하면 반영됩니다.

### Q. 401 SIGNATURE 검증 실패

서버 오류 메시지에 서버가 직접 계산한 서명 해시가 포함됩니다.  
payload를 `INFO` 로그로 출력해 비교하세요:

```python
logger.info("payload: %s", payload)
```

흔한 원인: METHOD 불일치(`GET`/`POST`), PATH 앞 슬래시 유무, QUERY 정규화 방식 차이.

### Q. 403 입력 마감

경기 시작 15분 전 이후 제출 시 발생합니다. 해당 경기는 제출 불가이며 다음 경기는 정상 제출됩니다.

### Q. build_raw_data.py 실행 시 NaN 오류

`game_lineups_2023_2025.csv`에 `p_no`가 null인 행이 존재할 수 있습니다(API 미제공).  
`build_lineup_index()` 내 `dropna()` 처리로 자동 제외됩니다.

### Q. 외국인 선수 피처가 리그평균으로 치환됨

`predict_2026.py`의 `FOREIGN_NAME_KO_MAP`에 해당 선수의 한국 이름 → `p_no` 매핑을 추가하세요.

### Q. 모델 성능이 급격히 떨어짐

외국인 선수 교체, 트레이드, 부상 등으로 롤링 스탯이 오염됐을 가능성이 있습니다.  
`append_2026_games.py` → `build_raw_data.py` → `baseball_baseline.py` 순서로 재실행해 최신 데이터를 반영하세요.
