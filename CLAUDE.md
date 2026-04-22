## Wiki

Project knowledge base lives in `.claude/wiki/`. Claude maintains it; you read it.

**Structure:**
- `index.md` — catalog of all pages; read this first on every session start
- `log.md` — append-only log of operations
- `pages/` — wiki pages (concepts, architecture decisions, status, research, etc.)

**Ingest** — when something new is worth preserving:
1. Write or update relevant page(s) in `pages/`
2. Update `index.md` if new pages were added
3. Append to `log.md`: `## [YYYY-MM-DD] ingest | <title>`

**Query** — read `index.md` first, then drill into relevant pages. Good answers can be saved back as new pages.

**Lint** (on request) — check for stale facts, orphan pages, missing cross-links, contradictions.

# Capper

## Project Overview

Capper — ML/AI предиктор для ставок на футбол, встроенный в Telegram-бот. Модель анализирует широкую футбольную линию и выявляет наиболее вероятные и value-исходы на основе Expected Value (EV) и Closing Line Value (CLV). Целевая аудитория — профессиональные беттеры, цель — стабильная система заработка на беттинге.

## Tech Stack

- **Language:** Python
- **Telegram Bot:** aiogram (async)
- **ML Models:** XGBoost / LightGBM (gradient boosting для табличных данных)
- **Database:** PostgreSQL + SQLAlchemy
- **Scheduler:** APScheduler
- **Data Source:** API-Football
- **Deployment:** VPS (Hetzner)

## Architecture

```
[API-Football] → [Data Collection] → [Feature Engineering] → [ML Model]
                                                                    ↓
                                            [Probability Calibration (CalibratedClassifierCV)]
                                                                    ↓
                                            [EV Filter vs Bookmaker Odds + CLV Check]
                                                                    ↓
                                            [Kelly Criterion → Stake Size]
                                                                    ↓
                                            [Telegram Bot → Daily Broadcast → User]
```

## ML Pipeline

```
Data Collection → Feature Engineering → Training → Backtesting → Deploy → Retrain (scheduled)
```

**Key concepts:**
- `EV = (our_probability × odds) - 1` — only bets with positive EV are sent
- `CLV` — closing line value used as long-term model quality metric
- `Kelly Criterion` — manages stake size based on user's bankroll: `f = (p × b - q) / b`
- `Fractional Kelly` (25%) — reduces risk from model errors
- Calibrated probabilities required before EV calculation

## Data & Features

**Match-level features:**
- xG (Expected Goals) — attack/defense strength without noise of actual goals
- Elo ratings — dynamic team strength
- Home/away form, head-to-head history
- Injuries and suspensions
- Odds movement (line changes = sharp money signal)
- Rolling averages (last 5, 10 games)

**Leagues covered:** Top-5 (EPL, La Liga, Bundesliga, Serie A, Ligue 1) + Champions League

## Markets by Phase

| Phase | Markets |
|---|---|
| v1 | 1X2, Totals (over/under), BTTS, Handicap |
| v2 | Double Chance, Accumulators (2-3 legs, high-risk label) |
| v3 | Player goal / assist (separate model, player-level data) |

**Accumulators note:** mathematically higher variance for pro bettors, offered as separate high-risk product with clear labeling.

## Project Structure

```
/bot        — Telegram bot (aiogram), handlers, daily broadcast
/model      — ML models, training, calibration, backtesting
/data       — API collection, feature engineering, data storage
/db         — PostgreSQL schemas, migrations (SQLAlchemy)
/scheduler  — APScheduler tasks (data refresh, predictions, retraining)
/config     — configs, environment variables
```

## Development Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run bot
python -m bot.main

# Run scheduler
python -m scheduler.main

# Train model
python -m model.train

# Backtest
python -m model.backtest

# Tests
pytest
```

## Claude's Working Style

- **Always plan first, then code.** Before implementing anything non-trivial, present the approach and wait for confirmation.
- **Short answers** with clarifications when needed. No unnecessary verbosity.
- **Ask before changing architecture** — never restructure modules or change data flow without discussion.

## Rules

- **Never touch the production database** directly. All DB changes go through migrations.
- **Always write tests** for new functionality.
- **Never change architecture without discussion** — propose first, implement after approval.
- **Never delete files without an explicit request.**
- **Monetization is v2+** — do not design for it in v1, keep the architecture simple.
- **Stop-loss is not implemented** — Kelly Criterion manages all stake sizing.
