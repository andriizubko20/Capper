# BETA — Model Research Memory

## Goal
x10 bankroll за 2 роки ($1000 → $10,000). Мінімальна ціль: $4-5k (~58% annual ROI).

## Data Available
- **8177 finished matches** | Aug 2023 – Apr 2026 | 10 leagues
- **match_stats**: xG, shots, shots_on_target, possession, corners, Glicko, passes, saves, cards
- **odds**: 7749 матчів з 1x2. opening_value НЕ заповнений (немає line movement)
- **injuries**: ~30k rows
- **Feature matrix**: 201 колонок, 180 features

## Leagues
EPL(39), La Liga(140), Bundesliga(78), Serie A(135), Ligue 1(61),
Primeira Liga(94), Serie B(136), Eredivisie(88), Jupiler Pro League(144), Champions League(2)

## Betting Philosophy
- Не шукати розриви з букмекером — шукати якісні патерни де WR стабільно > 55%
- EV = (our_prob × odds) - 1 > 0 — обов'язкова умова
- Kelly 25% fractional + cap — управління ризиком
- Walk-forward (train 12m → test 3m → step 3m) — єдиний чесний метод
- **НІКОЛИ** не фільтрувати нічиї до симуляції — це data leakage (~20% inflation WR)
- MARKET_COLS виключити з тренування, використовувати тільки для EV check

## Key Research Findings

### xG Ratio — найсильніший сигнал
- `xg_ratio_home_5 >= 1.5`: n=1398, WR=67.7%, odds=1.69, flat ROI=+3.1% (in-sample)
- `xg_ratio_away_5` — #1 feature importance у моделі (gain 551.9)

### Value odds range [1.70-2.50]
- `home[1.7-2.5] + xg>=1.5`: n=297, WR=57.6%, odds=1.99, flat ROI=+13.3%
- `home[1.7-2.2] + xg>=1.5`: WR=60.4% ← досягнуто 60% target!
- Away [1.7-2.5] без фільтрів: 48-51% WR (near breakeven)

### Away signal — умови для прибутковості
- `elo_diff <= -50 + home_pts_5 <= 1.4 + away_pts_5 >= 1.6`: WR ~54-55%
- Слабшає влітку + ранній сезон (форма нестабільна)
- Потенційне покращення: додати `xg_ratio_away_5 >= 1.2`

## Model History

### v1 — M1/M2/M3/M4 (run_backtest.py)
- M1 LightGBM (sklearn wrapper) → BROKEN (sklearn 1.8.0)
- M2 XGBoost → -99% ROI (market features в training = модель копіює ринок)
- M3 Dixon-Coles Poisson — теоретично правильна, практично слабка
- **Ключовий урок:** MARKET_COLS мають бути виключені з тренування

### v2 — Selective xG Model (HOME ONLY) ← БАЗОВА ЛІНІЯ
**File:** `BETA/model_v2.py`
**Config:** XG_THRESHOLD=1.5, CONFIDENCE_PERCENTILE=35, MIN_ODDS=1.70, MAX_ODDS=2.50, KELLY=25%, CAP=4%

| Metric | Value |
|--------|-------|
| Bets | 68 (39W/29L) |
| Win rate | 57.4% |
| Avg odds | 1.94 |
| Bankroll | $1000 → $1,360 (1.36x) |
| ROI | +36.0% |
| Max DD | 21.6% |
| Bets/year | ~45 |

❌ Занадто мало ставок для target x10

### v3 — Combined Home + Away ← ПОТОЧНА РОБОЧА МОДЕЛЬ ✅
**File:** `BETA/model_v3.py`
**Config:**
- HOME: `xg_ratio_home_5 >= 1.5`, odds [1.70, 2.50]
- AWAY: `elo_diff <= -50`, `home_pts_5 <= 1.4`, `away_pts_5 >= 1.6`, odds [1.70, 2.50]
- CONFIDENCE_PERCENTILE=40, KELLY=25%, CAP=6%

| Metric | Value |
|--------|-------|
| Bets | 201 (113W/88L) |
| Win rate | 56.2% (HOME: 59.0%, AWAY: 54.5%) |
| Avg odds | 1.98 |
| Bankroll | $1000 → $3,263 (3.26x) |
| ROI | +226.3% |
| Max DD | 41.0% |
| Bets/year | ~80 |
| 2-yr projection | $4,839 |

⚠️ MaxDD 41% — занадто високий
⚠️ Останні 2 периоди 50% WR — away сигнал слабшає

## Research Roadmap

### Етап 1 — Калібровка v3
- [ ] Додати `xg_ratio_away_5 >= 1.2` до away фільтру
- [ ] Sweep: xg_threshold, confidence_pct, kelly_cap
- [ ] Порівняти, вибрати оптимум

### Етап 2 — Кластери за odds range (окрема міні-модель на кожен)
- [ ] [1.30-1.55] явні фаворити (home/away окремо)
- [ ] [1.55-1.80] середні
- [ ] [1.80-2.50] value range ← поточний фокус
- [ ] [2.50-4.00] аутсайдери — чи є тут edge?

### Етап 3 — Кластери за домінуючим сигналом
- [ ] xG dominant (ratio >= 1.5)
- [ ] Elo dominant (diff >= 100)
- [ ] Form dominant (pts_5 >= 2.5)
- [ ] Market undervalue (our_prob >> market_prob)

### Етап 4 — Аналіз по лігах
- [ ] WR і ROI окремо по кожній лізі
- [ ] Виявити слабкі → або окрема модель, або виключити

## Технічні нотатки
- LightGBM: `lgb.train()` native API (не sklearn wrapper — зламано в sklearn 1.8.0)
- sklearn 1.8.0: видалено `multi_class` з LogisticRegression
- Docker: `./BETA:/app/BETA` volume mount — зміни одразу в контейнері без rebuild
- `libgomp1` потрібен для LightGBM в Docker slim image

## File Structure
```
BETA/
  MEMORY.md           — цей файл
  model_v2.py         — v2: home-only xG selective model (baseline)
  model_v3.py         — v3: combined home+away (поточна робоча)
  research.py         — empirical filter research (all single/combo filters)
  research_away.py    — away favorites + home value [1.7-2.5] deep dive
  run_backtest.py     — entry point M1-M4 ensemble models
  data/
    extract.py        — pull data from DB → DataFrames
    features.py       — build feature matrix (no leakage)
  models/
    m1_lgbm.py        — LightGBM native API + Platt calibration
    m2_xgb.py         — XGBoost native API + Isotonic calibration
    m3_poisson.py     — Dixon-Coles Poisson
    m4_ensemble.py    — Ensemble (avg M1+M2+M3)
  backtest/
    engine.py         — walk-forward backtest core
    kelly.py          — Kelly sizing + compound bankroll sim
```
