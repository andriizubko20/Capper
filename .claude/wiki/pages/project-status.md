# Project Status

**Category:** Tools & Tech
**Created:** 2026-04-19
**Updated:** 2026-04-20

Поточний стан розробки Capper станом на квітень 2026.

## Key Facts

- Три продакшн-моделі: WS Gap, Monster, Aquamarine — всі активні
- Miniapp backend integration: **ЗАВЕРШЕНО** (FastAPI + DB + frontend api.ts)
- Docker compose: 5 сервісів (db, pgadmin, bot, scheduler, api)
- API: `http://localhost:8000` (prod: VPS Hetzner port 8000)

## Що зроблено (квітень 2026)

### Backend
- FastAPI 5 ендпоінтів: `/api/health`, `/picks`, `/stats`, `/compare`, `/bankroll`
- Scheduler: live_tracker (кожні 3 хв), confirm_picks (07:00 UTC), update_results (backup)
- DB міграції: result/pnl/timing/is_active, bankroll_snapshots, users.bankroll, historical fields
- Баг-фікс: case-insensitive result calculation (був завжди `loss` для away бетів)
- Historical import: 681 рядків (Monster 370 + Aqua 121 + WS Gap 190) → тільки для майбутнього backtesting tab, **не** в stats

### Frontend (capper-miniapp)
- `src/lib/api.ts` — HTTP-шар замість mockData
- Всі 3 screens: useEffect + loading states
- PickCard: статуси `live/pending/win/loss/finished`; timing badges РАННЯ/ФІНАЛЬНА
- Stats curve: bankroll від $1000 (не cumulative PnL від 0)
- `.env`: `VITE_API_URL=http://localhost:8000`

## Production stats (2026-04-20)

| Модель | Bets | WR | ROI | Balance |
|--------|------|----|-----|---------|
| **Monster** | 13 | 61.5% | +13.7% | $1131 |
| **WS Gap**  | 44 | 63.6% | +28.6% | $1952 |
| **Aqua**    | 5  | 100%  | +64.6% | $1296 |

## Bug fixes (2026-04-22)

Проведено глибокий аудит стеку — знайдено 33 проблеми. Виправлено критичні та пріоритетні High:

- **Race condition** live_tracker + update_results → double bankroll: `db.refresh()` + `newly_settled` list
- **N+1 queries** в generate_picks_monster: batch-load odds + existing preds перед циклом
- **Bot handler**: try/except навколо `message.answer()`
- **PickCard NaN**: null check на `stake` перед арифметикою
- **Stale fetch**: `cancelled` flag у useEffect picks + stats
- **League flags**: Eredivisie 🇳🇱, Jupiler 🇧🇪, УПЛ 🇺🇦 (з УПЛ/England name collision fix)

Деталі: [Bug Audit 2026-04-22](bug-audit-2026-04-22.md)

## Що залишилось

- [ ] Medium баги: null checks в API, missing DB index, CompareScreen useMemo (деталі в [Bug Audit](bug-audit-2026-04-22.md))
- [ ] `confirm_picks.py` — порогові значення EV по моделях (потребує ресерчу)
- [ ] Deploy на VPS (після тестування)
- [ ] openpyxl додати в requirements.txt (потрібен для import_historical)

## MODEL_VERSIONS (поточні)

```python
"WS Gap":  ["ws_gap_v1", "ws_gap_kelly_v1"],
"Monster": ["monster_v1_kelly"],
"Aqua":    ["aquamarine_v1_kelly"],
```

## Related

- [Capper Overview](capper-overview.md)
- [WS Gap Model](ws-gap-model.md)
- [Monster Model](monster-model.md)
- [Aquamarine Model](aquamarine-model.md)
