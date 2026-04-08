# 📋 데이터 전처리 담당자 가이드 (Raw Data Spec)

> **목적:** 머신러닝 파이프라인(`baseball_baseline.py`)에 데이터를 올바르게 공급하기 위해,  
> 원본 데이터(Raw Data)를 **어떤 형태**로, **어떤 값을 담아서** 전달해야 하는지 명세합니다.  
> 이 문서에 기재된 내용을 충족하면, 이후 전처리/모델링은 코드가 자동으로 처리합니다.

---

## 1. 파일 형식 및 기본 규칙

| 항목                  | 규격                                                                         |
| --------------------- | ---------------------------------------------------------------------------- |
| **파일 형식**         | CSV (UTF-8-BOM 인코딩, `utf-8-sig`)                                          |
| **파일 경로**         | `C:\Users\yhkan\Data_Analysis_Project\data\raw_data.csv`                     |
| **행(Row) 단위**      | **1행 = 1경기** (홈팀 기준)                                                  |
| **열(Column) 구분자** | 쉼표 `,`                                                                     |
| **날짜 형식**         | `YYYY-MM-DD` (예: `2024-05-03`)                                              |
| **시각 형식**         | `HH:MM`, 24시간제 (예: `18:30`)                                              |
| **결측치 표기**       | 빈 칸 또는 `NaN` (코드가 중앙값으로 자동 대체)                               |
| **행 정렬**           | 날짜 오름차순 정렬 권장 (코드 내부에서도 재정렬하지만, 원본도 맞춰주면 좋음) |

---

## 2. 필수 컬럼 명세

> ⚠️ **컬럼명은 아래 표의 이름과 정확히 일치해야 합니다** (대소문자, 언더스코어 포함).

---

### 2-1. 공통 정보 (7개 컬럼)

| 컬럼명      | 타입   | 설명                                    | 예시 값                   |
| ----------- | ------ | --------------------------------------- | ------------------------- |
| `game_id`   | string | 경기 고유 식별자                        | `G0001`, `2024-KIA-LG-01` |
| `game_date` | string | 경기 날짜 (`YYYY-MM-DD`)                | `2024-05-03`              |
| `game_time` | string | 경기 시작 시각 (`HH:MM`, 24h)           | `18:30`, `14:00`          |
| `home_team` | string | 홈팀 명칭 또는 코드                     | `KIA`, `LG`, `SSG`        |
| `away_team` | string | 어웨이팀 명칭 또는 코드                 | `두산`, `한화`            |
| `stadium`   | string | 경기 구장명                             | `광주`, `잠실`, `인천`    |
| `result`    | int    | **타깃 변수.** 홈팀 승=`1`, 홈팀 패=`0` | `1` 또는 `0`              |

> 🔑 `result` 컬럼은 **학습 데이터에만** 존재하면 됩니다. 예측용 신규 데이터에는 없어도 됩니다.

---

### 2-2. 홈팀 선발 투수 (8개 컬럼)

| 컬럼명               | 타입   | 설명                                  | 정상 범위       | 예시 값 |
| -------------------- | ------ | ------------------------------------- | --------------- | ------- |
| `home_sp_ERA`        | float  | 홈 선발투수 ERA (평균자책점)          | `0.00 ~ 10.00`  | `3.45`  |
| `home_sp_WAR`        | float  | 홈 선발투수 WAR                       | `-1.0 ~ 8.0`    | `2.31`  |
| `home_sp_WHIP`       | float  | 홈 선발투수 WHIP                      | `0.70 ~ 2.50`   | `1.22`  |
| `home_sp_recent_IP`  | float  | 최근 5경기 **평균 투구 이닝** (IP)    | `0.1 ~ 9.0`     | `5.8`   |
| `home_sp_recent_ER`  | float  | 최근 5경기 **평균 자책점** (ER)       | `0.0 ~ 10.0`    | `2.4`   |
| `home_sp_vs_opp_BA`  | float  | 이번 상대 팀에 대한 **통산 피안타율** | `0.100 ~ 0.400` | `0.265` |
| `home_sp_vs_opp_OPS` | float  | 이번 상대 팀에 대한 **통산 피OPS**    | `0.400 ~ 1.100` | `0.720` |
| `home_sp_hand`       | string | 투구 유형 (우투/좌투/언더)            | `R` / `L` / `U` | `R`     |

---

### 2-3. 어웨이팀 선발 투수 (8개 컬럼)

홈팀 선발 투수와 동일한 구조이며, 컬럼명 접두어만 `away_sp_`로 변경됩니다.

| 컬럼명               | 타입   | 설명                                    |
| -------------------- | ------ | --------------------------------------- |
| `away_sp_ERA`        | float  | 어웨이 선발투수 ERA                     |
| `away_sp_WAR`        | float  | 어웨이 선발투수 WAR                     |
| `away_sp_WHIP`       | float  | 어웨이 선발투수 WHIP                    |
| `away_sp_recent_IP`  | float  | 최근 5경기 평균 투구 이닝               |
| `away_sp_recent_ER`  | float  | 최근 5경기 평균 자책점                  |
| `away_sp_vs_opp_BA`  | float  | 이번 상대 팀(홈팀)에 대한 통산 피안타율 |
| `away_sp_vs_opp_OPS` | float  | 이번 상대 팀(홈팀)에 대한 통산 피OPS    |
| `away_sp_hand`       | string | 투구 유형 (`R` / `L` / `U`)             |

---

### 2-4. 홈팀 불펜 (2개 컬럼)

| 컬럼명         | 타입  | 설명                                | 정상 범위     | 예시 값 |
| -------------- | ----- | ----------------------------------- | ------------- | ------- |
| `home_bp_WHIP` | float | 홈팀 불펜 **평균** WHIP (시즌 누계) | `0.80 ~ 2.50` | `1.41`  |
| `home_bp_WAR`  | float | 홈팀 불펜 **합산** WAR (시즌 누계)  | `-2.0 ~ 5.0`  | `1.20`  |

---

### 2-5. 어웨이팀 불펜 (2개 컬럼)

| 컬럼명         | 타입  | 설명                    |
| -------------- | ----- | ----------------------- |
| `away_bp_WHIP` | float | 어웨이팀 불펜 평균 WHIP |
| `away_bp_WAR`  | float | 어웨이팀 불펜 합산 WAR  |

---

### 2-6. 홈팀 타자 — 선발 라인업 9명 평균값 (4개 컬럼)

> 📌 **집계 방법:** 당일 선발 라인업 9명의 각 스탯을 산술 평균(Mean)하여 **팀 단위 1개 값**으로 제공합니다.

| 컬럼명                 | 타입  | 설명                                                     | 정상 범위         | 예시 값  |
| ---------------------- | ----- | -------------------------------------------------------- | ----------------- | -------- |
| `home_bat_avg_OPS`     | float | 선발 라인업 9명 평균 OPS                                 | `0.500 ~ 1.000`   | `0.763`  |
| `home_bat_avg_RISP`    | float | 선발 라인업 9명 평균 **득점권 타율** (RISP)              | `0.150 ~ 0.420`   | `0.285`  |
| `home_bat_avg_RS9`     | float | 선발 라인업 9명 평균 **9이닝당 득점 지원** (RS9)         | `2.0 ~ 8.0`       | `4.85`   |
| `home_bat_platoon_adj` | float | **플래툰 보정값.** 상대 선발 유형(R/L/U) 대비 OPS 증감치 | `-0.100 ~ +0.100` | `+0.023` |

> **플래툰 보정값 계산 방법 (예시):**  
> 홈팀 타자들의 `vs_RHP_OPS` 평균과 `vs_LHP_OPS` 평균을 미리 구한 뒤,  
> 상대 선발이 우투(R)이면 `avg_OPS_vs_RHP - avg_OPS_전체`를 `home_bat_platoon_adj`에 입력.  
> 상대 선발이 좌투(L)이면 `avg_OPS_vs_LHP - avg_OPS_전체`를 입력.  
> 언더(U)이면 별도 수집 데이터 또는 `0`으로 입력.

---

### 2-7. 어웨이팀 타자 (4개 컬럼)

홈팀 타자와 동일한 구조이며, 컬럼명 접두어만 `away_bat_`으로 변경됩니다.

| 컬럼명                 | 타입  | 설명                             |
| ---------------------- | ----- | -------------------------------- |
| `away_bat_avg_OPS`     | float | 선발 라인업 9명 평균 OPS         |
| `away_bat_avg_RISP`    | float | 선발 라인업 9명 평균 득점권 타율 |
| `away_bat_avg_RS9`     | float | 선발 라인업 9명 평균 RS9         |
| `away_bat_platoon_adj` | float | 플래툰 보정값                    |

---

### 2-8. 구장 파크 팩터 (1개 컬럼)

| 컬럼명        | 타입  | 설명                                 | 정상 범위     | 예시 값 |
| ------------- | ----- | ------------------------------------ | ------------- | ------- |
| `park_factor` | float | 해당 구장의 파크 팩터 (`1.0` = 중립) | `0.80 ~ 1.20` | `1.05`  |

> **파크 팩터 기준:**
>
> - `1.0` 이상 → 타자 친화 구장 (득점이 리그 평균보다 많이 남) → 투수에게 불리
> - `1.0` 미만 → 투수 친화 구장 (득점이 리그 평균보다 적게 남) → 투수에게 유리

---

### 2-9. 팀 최근 컨디션 — 롤링 스탯 (6개 컬럼)

> `build_raw_data.py`가 `game_results CSV`로부터 자동 계산합니다. 수동 입력 불필요.

| 컬럼명                 | 타입  | 설명                           | 정상 범위       |
| ---------------------- | ----- | ------------------------------ | --------------- |
| `home_recent_win_rate` | float | 홈팀 최근 10경기 승률          | `0.0 ~ 1.0`     |
| `away_recent_win_rate` | float | 원정팀 최근 10경기 승률        | `0.0 ~ 1.0`     |
| `home_recent_run_diff` | float | 홈팀 최근 10경기 평균 득실차   | `-10.0 ~ +10.0` |
| `away_recent_run_diff` | float | 원정팀 최근 10경기 평균 득실차 | `-10.0 ~ +10.0` |
| `home_win_streak`      | int   | 홈팀 현재 연승(+) / 연패(-)    | `-10 ~ +10`     |
| `away_win_streak`      | int   | 원정팀 현재 연승(+) / 연패(-)  | `-10 ~ +10`     |

---

### 2-10. 투수 등판 간격 (2개 컬럼)

> `build_raw_data.py`가 자동 계산합니다. 수동 입력 불필요.

| 컬럼명              | 타입 | 설명                                   | 정상 범위 |
| ------------------- | ---- | -------------------------------------- | --------- |
| `home_sp_rest_days` | int  | 홈 선발 투수의 전 등판 이후 휴식일수   | `0 ~ 30`  |
| `away_sp_rest_days` | int  | 원정 선발 투수의 전 등판 이후 휴식일수 | `0 ~ 30`  |

---

### 2-11. 구장-투수·타자 스플릿 (6개 컬럼)

> `build_raw_data.py`가 해당 구장에서의 역대 성적을 집계합니다. 수동 입력 불필요.

| 컬럼명                 | 타입  | 설명                                    |
| ---------------------- | ----- | --------------------------------------- |
| `home_sp_stadium_ERA`  | float | 홈 선발 투수의 해당 구장 통산 ERA       |
| `home_sp_stadium_WHIP` | float | 홈 선발 투수의 해당 구장 통산 WHIP      |
| `away_sp_stadium_ERA`  | float | 원정 선발 투수의 해당 구장 통산 ERA     |
| `away_sp_stadium_WHIP` | float | 원정 선발 투수의 해당 구장 통산 WHIP    |
| `home_bat_stadium_OPS` | float | 홈팀 타자들의 해당 구장 통산 평균 OPS   |
| `away_bat_stadium_OPS` | float | 원정팀 타자들의 해당 구장 통산 평균 OPS |

---

## 3. 전체 컬럼 목록 요약 (총 50개)

> 2-9~2-11의 롤링/구장 스플릿 컬럼은 `build_raw_data.py`가 자동 생성합니다.

```
# 공통 정보 (7)
game_id, game_date, game_time, home_team, away_team, stadium, result

# 홈 선발 투수 (8)
home_sp_ERA, home_sp_WAR, home_sp_WHIP, home_sp_recent_IP, home_sp_recent_ER,
home_sp_vs_opp_BA, home_sp_vs_opp_OPS, home_sp_hand

# 원정 선발 투수 (8)
away_sp_ERA, away_sp_WAR, away_sp_WHIP, away_sp_recent_IP, away_sp_recent_ER,
away_sp_vs_opp_BA, away_sp_vs_opp_OPS, away_sp_hand

# 불펜 (4)
home_bp_WHIP, home_bp_WAR, away_bp_WHIP, away_bp_WAR

# 타자 (8)
home_bat_avg_OPS, home_bat_avg_RISP, home_bat_avg_RS9, home_bat_platoon_adj,
away_bat_avg_OPS, away_bat_avg_RISP, away_bat_avg_RS9, away_bat_platoon_adj

# 파크 팩터 (1)
park_factor

# 팀 롤링 컨디션 — 자동 생성 (6)
home_recent_win_rate, away_recent_win_rate,
home_recent_run_diff, away_recent_run_diff,
home_win_streak, away_win_streak

# 투수 등판 간격 — 자동 생성 (2)
home_sp_rest_days, away_sp_rest_days

# 구장-선발 투수 스플릿 — 자동 생성 (4)
home_sp_stadium_ERA, home_sp_stadium_WHIP,
away_sp_stadium_ERA, away_sp_stadium_WHIP

# 구장-타자 스플릿 — 자동 생성 (2)
home_bat_stadium_OPS, away_bat_stadium_OPS
```

---

## 4. 코드가 자동으로 처리하는 것 (담당자가 하지 않아도 되는 것)

| 처리 항목                | 처리 방식                       | 생성 컬럼                        |
| ------------------------ | ------------------------------- | -------------------------------- |
| ERA × 파크 팩터 보정     | `ERA × park_factor` 곱셈        | `Home/Away_SP_ERA_ParkAdj`       |
| 최근 평균자책점 계산     | `(ER × 9) / IP` 공식 적용       | `Home/Away_SP_RecentERA`         |
| 불펜 WHIP 페널티         | `max(0, WHIP - 리그평균) × 2.0` | `Home/Away_Bullpen_WHIP_Penalty` |
| 플래툰 보정 OPS 최종값   | `avg_OPS + platoon_adj` 합산    | `Home/Away_Team_OPS_Platoon`     |
| 홈-어웨이 차분 피처 생성 | 홈 값 - 어웨이 값               | `Diff_*` 6개 컬럼                |
| Day/Night 더미 변수      | `game_time` 기준 14시 이전 = 1  | `Is_Day_Game`                    |
| 결측치 처리              | 각 컬럼 **중앙값**으로 대체     | (in-place 처리)                  |
| 날짜 기준 정렬           | `game_date` 오름차순            | (in-place 처리)                  |

---

## 5. 자주 하는 실수 & 주의사항

| ❌ 잘못된 예                     | ✅ 올바른 예                      | 비고                            |
| -------------------------------- | --------------------------------- | ------------------------------- |
| `home_sp_hand = "우투"`          | `home_sp_hand = "R"`              | 반드시 `R` / `L` / `U` 문자 1개 |
| `game_date = "2024.05.03"`       | `game_date = "2024-05-03"`        | 구분자는 `-` (하이픈)           |
| `game_time = "오후 6시"`         | `game_time = "18:00"`             | 24시간제 `HH:MM` 형식           |
| `result = "승"`                  | `result = 1`                      | 정수 `0` 또는 `1`               |
| `park_factor = 105`              | `park_factor = 1.05`              | `1.0` 기준 소수                 |
| 타자 스탯을 개인별 9행으로 제공  | **9명 평균값 1행**으로 집계       | 파이프라인은 팀 단위 1값만 받음 |
| `home_bat_platoon_adj` 컬럼 누락 | 해당 컬럼 `0.000`으로 채워서 제공 | 없으면 에러 발생                |

---

## 6. CSV 샘플 (첫 2행 예시)

```csv
game_id,game_date,game_time,home_team,away_team,stadium,result,home_sp_ERA,home_sp_WAR,home_sp_WHIP,home_sp_recent_IP,home_sp_recent_ER,home_sp_vs_opp_BA,home_sp_vs_opp_OPS,home_sp_hand,away_sp_ERA,away_sp_WAR,away_sp_WHIP,away_sp_recent_IP,away_sp_recent_ER,away_sp_vs_opp_BA,away_sp_vs_opp_OPS,away_sp_hand,home_bp_WHIP,home_bp_WAR,away_bp_WHIP,away_bp_WAR,home_bat_avg_OPS,home_bat_avg_RISP,home_bat_avg_RS9,home_bat_platoon_adj,away_bat_avg_OPS,away_bat_avg_RISP,away_bat_avg_RS9,away_bat_platoon_adj,park_factor
G0001,2024-04-05,18:30,KIA,LG,광주,1,3.21,3.10,1.05,6.2,1.8,0.241,0.680,R,4.55,1.80,1.31,5.4,2.6,0.275,0.745,L,1.28,1.50,1.42,0.90,0.778,0.291,5.12,0.025,0.741,0.268,4.88,-0.018,1.03
G0002,2024-04-06,14:00,SSG,두산,인천,0,5.10,0.90,1.52,4.8,3.5,0.305,0.830,R,3.88,2.40,1.18,6.0,2.1,0.258,0.710,R,1.55,0.40,1.22,1.30,0.712,0.253,4.55,-0.010,0.769,0.280,5.21,0.012,0.97
```

---

## 7. 문의

원본 데이터 관련 문의사항은 모델링 담당자에게 전달해주세요.  
파이프라인 코드 파일: `baseball_baseline.py`  
파이프라인 실행 방법: `python baseball_baseline.py` (샘플 데이터로 테스트 가능)
