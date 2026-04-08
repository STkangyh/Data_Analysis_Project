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
6. [자동화 파이프라인 (daily_pipeline.py)](#6-자동화-파이프라인)
7. [당일 예측 및 제출 (submit_predictions_today.py)](#7-당일-예측-및-제출)
8. [API 인증 상세](#8-api-인증-상세)
9. [매일 운영 워크플로](#9-매일-운영-워크플로)
10. [데이터 파일 명세](#10-데이터-파일-명세)
11. [모델 성능 이력](#11-모델-성능-이력)
12. [트러블슈팅](#12-트러블슈팅)

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
│  → data/raw_data.csv (2,160행 × 50열)                      │
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
│                     예측 레이어                              │
│  predict_2026.py                                            │
│  → 당일 라인업 수집 → 앙상블 예측                            │
│  → data/predictions_YYYYMMDD.csv 저장                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     제출 레이어                              │
│  submit_predictions_today.py                                │
│  → POST /prediction/savePrediction                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              자동화 오케스트레이션                            │
│  daily_pipeline.py  — Step 1~5 순차 실행                    │
│  Task Scheduler: 13:00 / 16:00 / 17:30  (WakeToRun 활성)   │
│  최대 절전(Hibernate)에서 자동 기상 후 실행                   │
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

## 6. 자동화 파이프라인 (daily_pipeline.py)

### 역할

Step 1~5를 순서대로 실행하는 오케스트레이터입니다. Windows Task Scheduler에 등록되어 **매일 자동 실행**됩니다.

### 실행 단계

| 단계   | 스크립트                      | 설명                                 | 하루 1회 가드        |
| ------ | ----------------------------- | ------------------------------------ | -------------------- |
| Step 1 | `append_2026_games.py`        | 전날 경기 결과 수집 & CSV append     | ✅ 마커로 중복 방지  |
| Step 2 | `build_raw_data.py`           | raw_data.csv 재빌드                  | ✅ 마커로 중복 방지  |
| Step 3 | `baseball_baseline.py`        | 모델 재학습 (`--skip-train` 시 생략) | -                    |
| Step 4 | `predict_2026.py`             | 오늘 경기 예측 → CSV 저장            | ✅ 파일 존재 시 생략 |
| Step 5 | `submit_predictions_today.py` | API 제출                             | ✅ 마커로 중복 방지  |

### 마커 파일 (하루 1회 가드)

`logs/` 폴더에 마커 파일이 생성되어 당일 중복 실행을 방지합니다.

| 파일                                | 생성 시점      | 효과                 |
| ----------------------------------- | -------------- | -------------------- |
| `logs/data_updated_YYYYMMDD.marker` | Step 2 완료 후 | 당일 Step 1·2 건너뜀 |
| `logs/submitted_YYYYMMDD.marker`    | Step 5 완료 후 | 당일 Step 5 건너뜀   |

날짜가 바뀌면 마커 이름도 달라지므로 새로운 날에는 자동으로 초기화됩니다.

### Task Scheduler 등록 현황

| 태스크 이름           | 실행 시각 | 옵션                   | 동작                                |
| --------------------- | --------- | ---------------------- | ----------------------------------- |
| `KBO_Predict_Day`     | 13:00     | `--skip-train`         | 데이터 수집·예측·제출 (재학습 없이) |
| `KBO_Predict_Evening` | 16:00     | `--skip-train --force` | 라인업 반영 재예측·재제출           |
| `KBO_Predict_Night`   | 17:30     | (없음)                 | 재학습 포함 전체 파이프라인         |

모든 태스크에 `WakeToRun = True` 설정 완료.

> **절전 모드 주의:** 이 PC는 S0(Connected Standby) 방식입니다.  
> 일반 절전에서는 WakeToRun이 작동하지 않으므로 **최대 절전(Hibernate)** 또는 PC가 켜진 상태를 유지해야 합니다.  
> PC 끄기: `시작 → 전원 → 최대 절전` 또는 `shutdown /h`

### CLI 옵션

```bash
# 기본 실행 (재학습 없이, 당일 최초 실행 시)
python daily_pipeline.py --skip-train

# 라인업 반영 후 재예측·재제출 (마커 무시)
python daily_pipeline.py --skip-train --force

# 재학습 포함 전체 실행
python daily_pipeline.py

# 데이터 수집·재빌드도 강제 재실행
python daily_pipeline.py --force-data

# 날짜 직접 지정
python daily_pipeline.py --skip-train --date 2026-04-07
```

### 로그 파일

실행 결과는 `logs/pipeline_YYYYMMDD.log`에 기록됩니다.  
Step 별 성공/실패 여부와 최종 예측 결과를 확인할 수 있습니다.

---

## 7. 당일 예측 및 제출

### submit_predictions_today.py

**역할:** 당일 경기 라인업 수집 → 앙상블 예측 → API 제출

> 일반적으로 `daily_pipeline.py`가 자동 호출합니다. 수동 재제출이 필요할 때만 직접 실행합니다.

**실행 옵션:**

```bash
# 전체 경기 제출
python submit_predictions_today.py

# 특정 팀이 포함된 경기만 제출
python submit_predictions_today.py --only 두산 한화

# 예측만 출력 (제출 없음)
python submit_predictions_today.py --dry-run

# 날짜 직접 지정
python submit_predictions_today.py --date 2026-04-07
```

**내부 동작 순서:**

```
[1/4] 4월 일정 API 수집 → 대상 날짜 경기 필터
[2/4] 각 경기 라인업 수집 (선발 라인업 공개 전이면 투수 2명만)
[3/4] game_results CSV 기반 rolling stats 계산 (최근 10경기)
[4/4] 피처 조립 → XGBoost·LightGBM 앙상블 예측
      → POST /prediction/savePrediction 제출
```

**중요 주의사항:**

| 항목             | 내용                                                |
| ---------------- | --------------------------------------------------- |
| 제출 마감        | 경기 시작 **15분 전**까지 (17:00 경기 → 16:45 마감) |
| 라인업 권장 시점 | 경기 시작 **1시간 전** 이후 (16:00~) 재실행 권장    |
| `TARGET_DATE`    | `datetime.today()` 기본값 — 별도 업데이트 불필요    |

**출력 결과 CSV:** `data/predictions_YYYYMMDD.csv`

---

## 8. API 인증 상세

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

## 9. 매일 운영 워크플로

### 9-1. 자동 실행 (권장)

**경기 당일 할 일: PC를 최대 절전(Hibernate)으로 두기만 하면 됩니다.**

Task Scheduler가 13:00 / 16:00 / 17:30에 자동으로 PC를 깨워 파이프라인을 실행합니다.

```
경기 당일 아침 → PC를 최대 절전으로 전환
                 ↓
   13:00  KBO_Predict_Day 자동 실행
          (데이터 수집 + 예측 + 제출)
                 ↓
   16:00  KBO_Predict_Evening 자동 실행
          (라인업 반영 재예측 + 재제출)
                 ↓
   17:30  KBO_Predict_Night 자동 실행
          (모델 재학습 + 최종 예측 + 재제출)
```

### 9-2. 수동 실행 (자동화 실패 시)

```bash
# 오전: 데이터 수집 + 예측 + 제출
python daily_pipeline.py --skip-train

# 경기 시작 1시간 전: 라인업 반영 재예측 + 재제출
python daily_pipeline.py --skip-train --force

# 또는 submit 단독 실행
python submit_predictions_today.py
```

### 9-3. 주간 점검 항목

- [ ] `logs/pipeline_YYYYMMDD.log`에서 Step 5 성공(code=100) 여부 확인
- [ ] `data/game_results_2023_2025.csv` 행 수 증가 확인 (매일 +5경기 내외)
- [ ] 모델 holdout accuracy 추이 모니터링 (`logs/` 확인)
- [ ] 외국인 선수 교체 시 `FOREIGN_NAME_KO_MAP` 업데이트 (`predict_2026.py`)

---

## 10. 데이터 파일 명세

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

## 11. 모델 성능 이력

| 날짜       | 학습 데이터               | XGB Acc | LGB Acc | XGB AUC | LGB AUC |
| ---------- | ------------------------- | ------- | ------- | ------- | ------- |
| 2026-04-03 | 2023~2026/4/3 (2,145경기) | 56.0%   | 58.0%   | 0.5898  | 0.5805  |
| 2026-04-07 | 2023~2026/4/7 (2,160경기) | 54.0%   | 54.0%   | 0.5747  | 0.5814  |

---

## 12. 트러블슈팅

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

### Q. 자동화 태스크가 실행됐는데 제출이 안 됐다

1. `logs/pipeline_YYYYMMDD.log`를 열어 Step 5 결과 확인
2. `logs/submitted_YYYYMMDD.marker` 파일이 있으면 이미 제출된 것 (code=100)
3. 마커가 없고 로그에 `[FAIL]`이 있으면 Step 5 실패 → 수동으로 `python submit_predictions_today.py` 실행

### Q. 라인업이 바뀐 후 재제출하고 싶다

```bash
python daily_pipeline.py --skip-train --force
# 또는
python submit_predictions_today.py
```

`--force` 옵션은 `submitted_YYYYMMDD.marker`를 무시하고 재제출합니다.

### Q. PC를 완전히 껐는데 자동으로 안 켜졌다

Wake Timer는 **최대 절전(Hibernate)**에서만 작동합니다. 완전 종료(Shutdown)에서는 작동하지 않습니다.  
경기 당일에는 `시작 → 전원 → 최대 절전` 또는 `shutdown /h`를 사용하세요.

### Q. daily_pipeline.py가 없다는 오류 (exit=2)

파일이 삭제된 경우입니다. 프로젝트 루트 디렉터리에 `daily_pipeline.py`가 있는지 확인하고, 없으면 git에서 복구하거나 재생성하세요.
