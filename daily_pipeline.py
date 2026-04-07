"""
daily_pipeline.py
=================
매일 자동 실행되는 KBO 예측 파이프라인.

실행 순서:
  Step 1. append_2026_games  -- 전날 경기 결과 수집 & CSV append
  Step 2. build_raw_data     -- raw_data.csv 재빌드
  Step 3. baseball_baseline  -- 모델 재학습 (--skip-train 이면 생략)
  Step 4. predict_2026       -- 오늘 경기 예측

사용법:
  python daily_pipeline.py                    # 전체 실행 (재학습 포함)
  python daily_pipeline.py --skip-train       # 재학습 생략
  python daily_pipeline.py --skip-train --force  # 이미 예측 파일 있어도 재실행
  python daily_pipeline.py --date 2026-04-07  # 날짜 직접 지정
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────
PROJECT = Path(__file__).parent.resolve()
PYTHON = sys.executable
LOG_DIR = PROJECT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── 인수 파싱 ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="KBO 일일 예측 파이프라인")
parser.add_argument("--skip-train", action="store_true", help="모델 재학습 생략")
parser.add_argument("--force", action="store_true", help="예측 파일 존재해도 재실행")
parser.add_argument("--date", default=None, metavar="YYYY-MM-DD", help="기준 날짜 (기본: 오늘)")
args = parser.parse_args()

# ── 날짜 계산 ──────────────────────────────────────────────
today = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.today()
yesterday = today - timedelta(days=1)
today_str = today.strftime("%Y-%m-%d")
yesterday_str = yesterday.strftime("%Y-%m-%d")

# ── 로깅 설정 ──────────────────────────────────────────────
log_file = LOG_DIR / f"pipeline_{today.strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def run_step(name: str, cmd: list[str]) -> bool:
    """서브프로세스 실행 후 성공 여부 반환."""
    log.info(f"▶ {name}")
    log.info(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT))
    if result.returncode != 0:
        log.error(f"  ❌ {name} 실패 (exit={result.returncode})")
        return False
    log.info(f"  ✅ {name} 완료")
    return True


# ── 메인 ───────────────────────────────────────────────────
log.info("=" * 55)
log.info("  KBO 일일 파이프라인 시작")
log.info(f"  기준 날짜  : {today_str}")
log.info(f"  전날 날짜  : {yesterday_str}")
log.info(f"  모델 재학습: {'생략' if args.skip_train else '실행'}")
log.info("=" * 55)
log.info("")

# Step 1: 전날 결과 수집
log.info("[Step 1] 전날 경기 결과 수집 & append")
ok = run_step(
    "append_2026_games",
    [PYTHON, str(PROJECT / "append_2026_games.py"), "--cutoff", yesterday_str],
)
if not ok:
    sys.exit(1)
log.info("")

# Step 2: raw_data 재빌드
log.info("[Step 2] raw_data.csv 재빌드")
ok = run_step("build_raw_data", [PYTHON, str(PROJECT / "build_raw_data.py")])
if not ok:
    sys.exit(1)
log.info("")

# Step 3: 모델 재학습 (선택)
if args.skip_train:
    log.info("[Step 3] 모델 재학습 — 생략 (--skip-train)")
else:
    log.info("[Step 3] 모델 재학습")
    ok = run_step("baseball_baseline", [PYTHON, str(PROJECT / "baseball_baseline.py")])
    if not ok:
        sys.exit(1)
log.info("")

# Step 4: 예측 실행
pred_file = PROJECT / "data" / f"predictions_{today.strftime('%Y%m%d')}.csv"
if pred_file.exists() and not args.force:
    log.info(f"[Step 4] 예측 파일 이미 존재 — 생략 ({pred_file.name})")
    log.info("  (재실행하려면 --force 옵션 사용)")
else:
    log.info("[Step 4] 오늘 경기 예측")
    ok = run_step(
        "predict_2026",
        [PYTHON, str(PROJECT / "predict_2026.py"), "--dates", today_str],
    )
    if not ok:
        sys.exit(1)
log.info("")

# ── 결과 출력 ──────────────────────────────────────────────
log.info("=" * 55)
log.info("  🎉 전체 파이프라인 완료!")

if pred_file.exists():
    log.info(f"  📄 예측 결과 ({today_str}):")
    try:
        import pandas as pd
        df = pd.read_csv(pred_file)
        prob_col = next(
            (c for c in ["Ensemble_Prob", "Ensemble_Prob_Pct", "LGB_Prob"] if c in df.columns),
            None,
        )
        for _, row in df.iterrows():
            home = row.get("home_team", "?")
            away = row.get("away_team", "?")
            if prob_col:
                prob = row[prob_col]
                if prob > 0.5:
                    winner, pct = home, prob * 100 if prob <= 1 else prob
                else:
                    winner, pct = away, (1 - prob) * 100 if prob <= 1 else (100 - prob)
                log.info(f"    {away} @ {home}  →  예측 승자: {winner}  ({pct:.1f}%)")
            else:
                log.info(f"    {away} @ {home}")
    except Exception as e:
        log.warning(f"  결과 파일 파싱 실패: {e}")

log.info("=" * 55)
