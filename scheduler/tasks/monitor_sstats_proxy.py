"""
scheduler/tasks/monitor_sstats_proxy.py

Health-check для Mac SSH tunnel + socat proxy chain. Без цього ланцюга VPS
блокують datacenter IP при зверненні до SStats — ми отримуємо truncated 14762
байт замість повних 137KB+.

Логіка:
1. Hit a known top-league fixture endpoint via the proxy.
2. Verify response size > 50KB (full payload), not 14762 (truncated).
3. If truncated 3 times in a row → Telegram alert (tunnel down).
4. State persisted в `model/artifacts/sstats_proxy_state.json` щоб не спамити.

Run every 30 minutes via cron.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import httpx
from loguru import logger

from config.settings import settings

# Stable fixture for testing — UPL Shakhtar vs Polessya 2026-04-20.
# Past fixture, but SStats keeps odds in catalog. If tunnel works → full
# 137KB response. If broken → truncated/timeout.
TEST_FIXTURE_ID = 1391598
TRUNCATE_THRESHOLD = 50_000  # bytes — anything below this is suspect
CONSECUTIVE_FAILURES_TO_ALERT = 3

STATE_FILE = Path(__file__).resolve().parents[2] / "model" / "artifacts" / "sstats_proxy_state.json"


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"consecutive_failures": 0, "last_state": "ok", "last_check_at": None}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"consecutive_failures": 0, "last_state": "ok", "last_check_at": None}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_check_at"] = datetime.utcnow().isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _send_telegram(text: str) -> bool:
    token = settings.telegram_bot_token
    chat_id = getattr(settings, "admin_chat_id", None)
    if not token or not chat_id:
        logger.warning("[ProxyMon] Telegram creds missing — alert logged only")
        return False
    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        return resp.status_code == 200
    except httpx.RequestError as e:
        logger.error(f"[ProxyMon] Telegram failed: {e}")
        return False


def run_monitor_sstats_proxy() -> None:
    """Single health-check tick. Updates state, alerts on transitions."""
    state = _load_state()

    proxy_url = settings.sstats_api_host
    elapsed_ms = None
    response_size = 0
    error_msg = None

    try:
        with httpx.Client(headers={"X-API-KEY": settings.sstats_api_key}, timeout=15.0) as c:
            t0 = datetime.utcnow()
            resp = c.get(f"{proxy_url}/Odds/{TEST_FIXTURE_ID}")
            elapsed_ms = (datetime.utcnow() - t0).total_seconds() * 1000
            response_size = len(resp.content)
            if resp.status_code != 200:
                error_msg = f"HTTP {resp.status_code}"
    except httpx.RequestError as e:
        error_msg = f"{type(e).__name__}: {str(e)[:80]}"

    is_healthy = error_msg is None and response_size > TRUNCATE_THRESHOLD

    if is_healthy:
        logger.info(
            f"[ProxyMon] OK · {response_size/1024:.0f}KB · {elapsed_ms:.0f}ms via {proxy_url}"
        )
        if state["last_state"] == "alert":
            # Recovery
            _send_telegram(
                f"✅ *SStats proxy recovered*\n"
                f"Response: {response_size/1024:.0f}KB · {elapsed_ms:.0f}ms\n"
                f"Was down for ~{state['consecutive_failures']} checks"
            )
        state = {"consecutive_failures": 0, "last_state": "ok"}
    else:
        state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
        logger.warning(
            f"[ProxyMon] DEGRADED · size={response_size} err={error_msg} "
            f"(consecutive_failures={state['consecutive_failures']})"
        )
        if (
            state["consecutive_failures"] >= CONSECUTIVE_FAILURES_TO_ALERT
            and state.get("last_state") != "alert"
        ):
            _send_telegram(
                f"🚨 *SStats proxy DOWN*\n"
                f"Tunnel `{proxy_url}` failed {state['consecutive_failures']}× in a row.\n"
                f"Latest: size={response_size}B err={error_msg or 'truncated'}\n"
                f"Action: check Mac launchd `com.capper.sstats-tunnel` status"
            )
            state["last_state"] = "alert"

    _save_state(state)


if __name__ == "__main__":
    run_monitor_sstats_proxy()
