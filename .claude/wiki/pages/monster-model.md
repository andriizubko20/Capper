# Monster Model

**Category:** ML & Modeling
**Created:** 2026-04-19
**Updated:** 2026-04-22

Niche-based модель на базі in-sample win rate (p_is) по нішах. Широке покриття, всі ніші перевірені на дистанції.

## Key Facts

- **Ніші (niches)** — дискретні патерни (сила команд, форма, ринок, ліга) з відомим IS win rate
- Окремий `p_is` для кожної ніші в таблиці `monster_p_is` (оновлюється щотижня)
- Широке покриття — більше ставок ніж у [WS Gap](ws-gap-model.md)
- Одна з трьох продакшн-моделей Capper

## Логіка відбору

```
1. build_team_state() → rolling stats per team (Elo, pts_5, xg_ratio_5)
2. build_upcoming_features() → features для конкретного матчу
3. match_niche() → визначаємо ніш для матчу
4. Знаходимо p_is для ніші з DB (monster_p_is)
5. EV = p_is × odds - 1 > 0 → pick
6. Kelly sizing: f = (p×b - q)/b × FRACTIONAL, capped at KELLY_CAP
```

## Features задіяні

- `elo_diff` — різниця Elo між командами
- `home_pts_5`, `away_pts_5` — форма (очки останніх 5 матчів)
- `xg_ratio_home_5`, `xg_ratio_away_5` — xGF/xGA ratio за 5 матчів
- `mkt_home_prob`, `mkt_away_prob` — market-implied probability (margin-normalized)

## Версії (MODEL_VERSION)

| Версія | Опис |
|--------|------|
| `monster_v1_kelly` | Продакшн, Kelly 25% + 10% cap |

## Файли

- `scheduler/tasks/generate_picks_monster.py` — генерація пікфів
- `model/monster/features.py` — `build_team_state()`, `build_upcoming_features()`
- `model/monster/niches.py` — `match_niche()`
- `scheduler/tasks/update_monster_p_is.py` — щотижневе оновлення p_is

## Bug-фікси (2026-04-22)

- **XG false NaN**: `row.get('home_xg') or np.nan` → `x if x is not None else np.nan` у `features.py:103-109`
  - Команди з реальним xGF=0.0 більше не трактуються як missing data
- **xg_ratio NaN check**: `isnan(h_xgf5 or np.nan)` → `isnan(h_xgf5)` у `features.py:213-214`
- Той самий фікс у `update_monster_p_is.py:113-114`

## Related

- [Aquamarine Model](aquamarine-model.md)
- [WS Gap Model](ws-gap-model.md)
- [xG — Expected Goals](xg-expected-goals.md)
- [Kelly Criterion](kelly-criterion.md)
- [Capper Overview](capper-overview.md)
