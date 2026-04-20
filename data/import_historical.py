"""
Одноразовий імпорт historical predictions в таблицю predictions.

Джерела:
  Monster  → BETA/kelly_oos_only.xlsx  (Match List)  → monster_v1_kelly
  Aqua     → той самий файл, фільтр за niches.py     → aquamarine_v1_kelly
  WS Gap   → experiments/results/backtest_full_bets.csv → ws_gap_kelly_v1

Запуск (локально, поза Docker):
  python -m data.import_historical

Або в контейнері:
  docker compose exec scheduler python -m data.import_historical
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from db.models import Prediction
from db.session import SessionLocal

ROOT = Path(__file__).resolve().parent.parent

# ── Aqua niches whitelist ─────────────────────────────────────────────────────

def _aqua_set() -> set[tuple[str, str]]:
    from model.aquamarine.niches import MODELS as AQUA_MODELS
    return {(league, niche) for league, niches in AQUA_MODELS.items() for niche in niches}


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_date(val) -> date | None:
    if pd.isna(val):
        return None
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.date()
    try:
        return datetime.strptime(str(val).strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _won_to_result(val: str) -> str:
    return "win" if str(val).strip() in {"✓", "WIN", "1"} else "loss"


# ── importers ─────────────────────────────────────────────────────────────────

def _rows_monster_aqua() -> tuple[list[dict], list[dict]]:
    path = ROOT / "BETA" / "kelly_oos_only.xlsx"
    df = pd.read_excel(path, sheet_name="Match List")
    aqua = _aqua_set()

    monster_rows, aqua_rows = [], []
    for _, r in df.iterrows():
        match_date = _parse_date(r["Date"])
        if match_date is None:
            continue

        # Parse "Home vs Away"
        parts = str(r["Match"]).split(" vs ", 1)
        home_name = parts[0].strip() if len(parts) == 2 else str(r["Match"])
        away_name = parts[1].strip() if len(parts) == 2 else ""

        league   = str(r["League"]).strip()
        outcome  = str(r["Side"]).strip().lower()   # 'home' | 'away'
        niche    = str(r["Model"]).strip()
        odds     = float(r["Odds"])
        stake    = float(r["Kelly Stake $"])
        pnl      = float(r["Kelly P&L $"])
        result   = _won_to_result(r["Won"])

        row = dict(
            match_id=None,
            market="1x2",
            outcome=outcome,
            probability=0.0,
            odds_used=odds,
            ev=0.0,
            kelly_fraction=0.25,
            stake=stake,
            result=result,
            pnl=pnl,
            is_active=False,
            match_date=match_date,
            home_name=home_name,
            away_name=away_name,
            league_name=league,
        )
        monster_rows.append({**row, "model_version": "monster_v1_kelly"})
        if (league, niche) in aqua:
            aqua_rows.append({**row, "model_version": "aquamarine_v1_kelly"})

    return monster_rows, aqua_rows


def _rows_ws_gap() -> list[dict]:
    path = ROOT / "experiments" / "results" / "backtest_full_bets.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")

    rows = []
    for _, r in df.iterrows():
        match_date = _parse_date(r["Дата"])
        if match_date is None:
            continue

        home_name = str(r["Ставка_на"]).strip()
        away_name = str(r["Проти"]).strip()
        outcome   = str(r["Сторона"]).strip().lower()   # 'away' | 'home'
        odds      = float(r["Коеф"])
        stake     = float(r["Стейк_$"])
        pnl       = float(r["PnL_$"])
        result    = _won_to_result(r["WIN/LOSS"])
        league    = str(r["Ліга"]).strip()

        rows.append(dict(
            match_id=None,
            market="1x2",
            outcome=outcome,
            probability=0.0,
            odds_used=odds,
            ev=0.0,
            kelly_fraction=0.25,
            stake=stake,
            result=result,
            pnl=pnl,
            is_active=False,
            model_version="ws_gap_kelly_v1",
            match_date=match_date,
            home_name=home_name,
            away_name=away_name,
            league_name=league,
        ))
    return rows


# ── dedup helper ──────────────────────────────────────────────────────────────

def _existing_keys(db) -> set[tuple]:
    """Returns set of (model_version, match_date, home_name, outcome) for already-imported rows."""
    rows = (
        db.query(
            Prediction.model_version,
            Prediction.match_date,
            Prediction.home_name,
            Prediction.outcome,
        )
        .filter(Prediction.match_id.is_(None))
        .all()
    )
    return {(r.model_version, r.match_date, r.home_name, r.outcome) for r in rows}


# ── main ──────────────────────────────────────────────────────────────────────

def run_import(dry_run: bool = False) -> None:
    logger.info("Loading historical data...")
    monster_rows, aqua_rows = _rows_monster_aqua()
    ws_gap_rows = _rows_ws_gap()

    logger.info(f"  Monster: {len(monster_rows)} rows")
    logger.info(f"  Aqua:    {len(aqua_rows)} rows")
    logger.info(f"  WS Gap:  {len(ws_gap_rows)} rows")

    all_rows = monster_rows + aqua_rows + ws_gap_rows

    if dry_run:
        logger.info("Dry run — not writing to DB")
        return

    db = SessionLocal()
    try:
        existing = _existing_keys(db)
        logger.info(f"Already imported: {len(existing)} rows — will skip duplicates")

        added = 0
        skipped = 0
        for row in all_rows:
            key = (row["model_version"], row["match_date"], row["home_name"], row["outcome"])
            if key in existing:
                skipped += 1
                continue
            db.add(Prediction(**row))
            existing.add(key)
            added += 1

        db.commit()
        logger.info(f"Done: added {added}, skipped {skipped} duplicates")
    finally:
        db.close()


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    run_import(dry_run=dry)
