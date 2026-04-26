"""
CLV (Closing Line Value) monitor.

Computes 30-day rolling average CLV per production model and sends Telegram
alerts when a model crosses the Alert threshold (or recovers from one).

Threshold rationale:
  Healthy : avg_clv >= 0       → model beats the closing market on average
  Warning : -0.02 <= avg_clv < 0 → slight calibration drift, acceptable noise
  Alert   : avg_clv < -0.02    → consistently overestimating own edge → likely
                                 model decay; worth investigating or pausing.

The 2% cushion below zero acknowledges that bookmaker overround alone biases
implied probabilities ~2-3% higher than fair, so a tiny negative CLV is
neutral, while crossing -2% is a strong signal.

State persistence is a JSON file (no DB migration). The job alerts only on
state transitions: alert→ok or ok→alert.

Run manually for inspection:
    python -m scheduler.tasks.clv_monitor          # live (sends Telegram)
    python -m scheduler.tasks.clv_monitor --dry-run  # logs only, no Telegram

Required env vars (alert delivery):
    TELEGRAM_BOT_TOKEN  — already used by bot
    ADMIN_CHAT_ID       — Telegram chat ID receiving alerts (numeric)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from config.settings import settings
from db.models import Match, Prediction
from db.session import SessionLocal


# ── configuration ────────────────────────────────────────────────────────────

# (model_version → display name) — drives both the alert text and the report.
MODELS: dict[str, str] = {
    "ws_gap_kelly_v1":     "WS Gap",
    "monster_v1_kelly":    "Monster",
    "aquamarine_v1_kelly": "Aqua",
    "pure_v1":             "Pure",
    "gem_v1":              "Gem",
}

ROLLING_DAYS = 30
MIN_PICKS_REQUIRED = 5

# Thresholds — see module docstring.
WARNING_THRESHOLD = -0.02   # avg_clv >= this is at most "warning"
ALERT_THRESHOLD   = -0.02   # avg_clv < this triggers alert

STATE_FILE = Path("model/artifacts/clv_alert_state.json")


# ── state persistence ───────────────────────────────────────────────────────

def _load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[CLV] failed to read state file ({e}), resetting")
        return {}


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


# ── Telegram delivery ───────────────────────────────────────────────────────

def _send_alert(text: str) -> bool:
    """
    POSTs a Markdown message to the admin chat. Returns True on HTTP 200.
    Silently skips (and logs warning) if token / chat_id is missing.
    """
    token = settings.telegram_bot_token
    chat_id = settings.admin_chat_id
    if not token:
        logger.warning("[CLV] TELEGRAM_BOT_TOKEN not set — skipping alert")
        return False
    if not chat_id:
        logger.warning("[CLV] ADMIN_CHAT_ID not set — skipping alert")
        return False

    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.error(f"[CLV] Telegram returned {resp.status_code}: {resp.text}")
            return False
        return True
    except httpx.RequestError as e:
        logger.error(f"[CLV] Telegram request failed: {e}")
        return False


# ── core ────────────────────────────────────────────────────────────────────

def _classify(avg_clv: float) -> str:
    """Returns one of: 'healthy', 'warning', 'alert'."""
    if avg_clv >= 0:
        return "healthy"
    if avg_clv >= WARNING_THRESHOLD:
        return "warning"
    return "alert"


def _compute_avg_clv(db, model_version: str, cutoff: datetime) -> dict[str, Any] | None:
    """
    Returns dict with avg_clv, n, pos_rate — or None if insufficient data.
    Considers only settled (CLV-computed) picks within the rolling window.
    Uses Match.date when available, else falls back to Prediction.match_date
    so historical / imported picks (match_id IS NULL) still count.
    """
    cutoff_date = cutoff.date()
    rows = (
        db.query(Prediction, Match)
        .outerjoin(Match, Prediction.match_id == Match.id)
        .filter(
            Prediction.model_version == model_version,
            Prediction.clv.isnot(None),
        )
        .all()
    )

    clvs: list[float] = []
    for pred, match in rows:
        match_dt = match.date if match else None
        if match_dt is None and pred.match_date is not None:
            match_dt = datetime(
                pred.match_date.year, pred.match_date.month, pred.match_date.day,
                tzinfo=timezone.utc,
            )
        if match_dt is None:
            continue
        if match_dt.tzinfo is None:
            match_dt = match_dt.replace(tzinfo=timezone.utc)
        if match_dt < cutoff:
            continue
        clvs.append(float(pred.clv))

    if len(clvs) < MIN_PICKS_REQUIRED:
        return None

    return {
        "avg_clv":  sum(clvs) / len(clvs),
        "n":        len(clvs),
        "pos_rate": sum(1 for v in clvs if v > 0) / len(clvs),
    }


def _format_alert(model_name: str, stats: dict[str, Any]) -> str:
    return (
        "🚨 *CLV ALERT*\n"
        f"Model: `{model_name}`\n"
        f"{ROLLING_DAYS}d avg CLV: *{stats['avg_clv']*100:+.2f}%*\n"
        f"N picks: {stats['n']}\n"
        f"Status: model is overestimating edge — investigate or pause picks"
    )


def _format_recovery(model_name: str, stats: dict[str, Any], prev_avg: float | None) -> str:
    prev_str = f" (was {prev_avg*100:+.2f}% last alert)" if prev_avg is not None else ""
    return (
        "✅ *CLV recovered*\n"
        f"Model: `{model_name}`\n"
        f"{ROLLING_DAYS}d avg CLV: *{stats['avg_clv']*100:+.2f}%*{prev_str}"
    )


def run_clv_monitor(dry_run: bool = False) -> dict[str, dict[str, Any]]:
    """
    Daily entry point. Returns a per-model report dict (handy for tests / API).
    """
    logger.info(f"[CLV] starting monitor (dry_run={dry_run})")
    cutoff = datetime.now(timezone.utc) - timedelta(days=ROLLING_DAYS)
    state = _load_state()
    report: dict[str, dict[str, Any]] = {}

    db = SessionLocal()
    try:
        for model_version, model_name in MODELS.items():
            stats = _compute_avg_clv(db, model_version, cutoff)
            if stats is None:
                logger.info(f"[CLV] {model_name} ({model_version}): insufficient data")
                report[model_version] = {"status": "insufficient_data"}
                continue

            status = _classify(stats["avg_clv"])
            report[model_version] = {
                "status":   status,
                "avg_clv":  round(stats["avg_clv"], 4),
                "n":        stats["n"],
                "pos_rate": round(stats["pos_rate"], 4),
            }

            logger.info(
                f"[CLV] {model_name} ({model_version}): "
                f"avg={stats['avg_clv']*100:+.2f}% "
                f"n={stats['n']} "
                f"pos_rate={stats['pos_rate']:.0%} "
                f"status={status}"
            )

            prev = state.get(model_version, {})
            prev_status = prev.get("last_state")
            prev_avg = prev.get("last_avg_clv")

            crossed_into_alert    = status == "alert"  and prev_status != "alert"
            recovered_from_alert  = status != "alert"  and prev_status == "alert"

            if not dry_run and (crossed_into_alert or recovered_from_alert):
                msg = (
                    _format_alert(model_name, stats) if crossed_into_alert
                    else _format_recovery(model_name, stats, prev_avg)
                )
                if _send_alert(msg):
                    logger.info(f"[CLV] alert sent for {model_name}: "
                                f"{prev_status} → {status}")
                else:
                    logger.warning(f"[CLV] alert delivery failed for {model_name}")

            state[model_version] = {
                "last_state":    status,
                "last_avg_clv":  round(stats["avg_clv"], 4),
                "last_n":        stats["n"],
                "last_check_at": datetime.now(timezone.utc).isoformat(),
            }

        if not dry_run:
            _save_state(state)
    finally:
        db.close()

    return report


# ── CLI entry ───────────────────────────────────────────────────────────────

def _cli() -> int:
    parser = argparse.ArgumentParser(description="CLV monitor (daily cron)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and log results, but skip Telegram and state writes."
    )
    args = parser.parse_args()
    run_clv_monitor(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
