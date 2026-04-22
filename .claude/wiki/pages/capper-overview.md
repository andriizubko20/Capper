# Capper — Project Overview

**Category:** Tools & Tech
**Created:** 2026-04-19
**Updated:** 2026-04-19

ML/AI предиктор для ставок на футбол, вбудований у Telegram-бот. Аналізує широку лінію і виявляє value-ісходи на основі EV і CLV. Цільова аудиторія — професійні беттери.

## Key Facts

- Language: Python
- Bot: aiogram (async)
- ML: XGBoost / LightGBM
- DB: PostgreSQL + SQLAlchemy
- Scheduler: APScheduler
- Data: [SStats API](sstats-api.md) (`api.sstats.net`)
- Deploy: VPS Hetzner
- Три продакшн-моделі: [WS Gap](ws-gap-model.md), [Monster](monster-model.md), [Aquamarine](aquamarine-model.md)

## Architecture

```
[API-Football] → [Data Collection] → [Feature Engineering] → [ML Model]
                                                                    ↓
                                        [Probability Calibration]
                                                                    ↓
                                        [EV Filter + CLV Check]
                                                                    ↓
                                        [Kelly Criterion → Stake]
                                                                    ↓
                                        [Telegram Bot → Daily Broadcast]
```

## Pipeline

```
Data Collection → Feature Engineering → Training → Backtesting → Deploy
```

Немає scheduled retraining — моделі навчаються вручну і деплояться після валідації.

## Markets (v1)

1X2, Totals (over/under), BTTS, Handicap

## Related

- [WS Gap Model](ws-gap-model.md)
- [Monster Model](monster-model.md)
- [Aquamarine Model](aquamarine-model.md)
- [Expected Value (EV)](expected-value.md)
- [Kelly Criterion](kelly-criterion.md)
- [SStats API](sstats-api.md)

## Sources

- `CLAUDE.md` — project configuration and architecture overview
