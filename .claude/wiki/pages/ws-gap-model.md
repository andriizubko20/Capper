# WS Gap Model

**Category:** ML & Modeling
**Created:** 2026-04-19
**Updated:** 2026-04-22

Продакшн-модель без ML на основі Weighted Score. Агресивно відсіює ставки, пропускає тільки найбільш впевнені сигнали.

## Key Facts

- **Не ML** — чистий rule-based фільтр через Weighted Score
- Мало ставок на виході — висока селективність
- Висока якість сигналів що пройшли фільтр
- Одна з трьох продакшн-моделей Capper

## Логіка відбору

```
1. Рахуємо ws_home і ws_away через compute_weighted_score()
2. dominant = max(ws_home, ws_away)
3. ws_gap = ws_dom - ws_weak
4. Фільтри: ws_gap >= WS_GAP_MIN AND ODDS_MIN <= odds <= ODDS_MAX
5. Sizing: Elo-based Kelly × FRACTIONAL, capped at KELLY_CAP
```

## Константи (поточні)

| Константа | Значення | Файл |
|-----------|----------|------|
| `WS_GAP_MIN` | 70 | `generate_picks_ws_gap.py` |
| `ODDS_MIN` | 2.0 | `generate_picks_ws_gap.py` |
| `ODDS_MAX` | 4.0 | `generate_picks_ws_gap.py` |
| `KELLY_CAP` | 0.10 (10%) | `generate_picks_ws_gap.py` |
| `FRACTIONAL` | 0.25 (25%) | `generate_picks_ws_gap.py` |

## Версії (MODEL_VERSION)

| Версія | Опис |
|--------|------|
| `ws_gap_v1` | Фінальні піки, 10% Kelly cap |
| `ws_gap_kelly_v1` | Фінальні піки, pure Kelly 25% |
| `ws_gap_v1_early` | Ранні піки (за кілька днів до матчу), 10% cap |
| `ws_gap_kelly_v1_early` | Ранні піки, pure Kelly 25% |

## Features задіяні

- `elo_diff`, `elo_home_win_prob` — сила команд, probability для Kelly
- `home_elo_momentum`, `away_elo_momentum` — динаміка Elo (були 0 до фікса 2026-04-22)
- `xg_ratio_home_5`, `xg_ratio_away_5` — атака/захист по xG
- `home_pts_5`, `away_pts_5` — форма (очки останніх 5 матчів)
- Odds (home, draw, away) через `build_match_features`

## Файли

- `scheduler/tasks/generate_picks_ws_gap.py` — генерація пікфів
- `model/weighted_score.py` — `compute_weighted_score()`
- `model/features/builder.py` — `build_match_features()`
- `model/features/elo.py` — `build_elo_snapshots()`, `compute_elo_momentum()`

## Bug-фікси (2026-04-22)

- Early scan `stake_kelly` тепер обмежений `KELLY_CAP` (був необмежений)
- `ODDS_MAX = 4.0` — новий верхній поріг для odds
- `build_elo_snapshots` тепер передається в `build_match_features` → ELO momentum дійсно рахується

## Related

- [Monster Model](monster-model.md)
- [Aquamarine Model](aquamarine-model.md)
- [Elo Ratings](elo-ratings.md)
- [Kelly Criterion](kelly-criterion.md)
- [Capper Overview](capper-overview.md)
