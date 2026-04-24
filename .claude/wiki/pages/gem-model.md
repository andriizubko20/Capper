# Gem Model (v1)

**Status:** Phase 3 smoke-test пройшов успішно на малих параметрах (2026-04-25).
Повний запуск на 300 trials × 12 folds заплановано на ніч.

## Мета

- WR ≥ 70%, 7–10 ставок/геймвік, flat 4% від поточного банкролу
- Не ставимо на нічиї. Нічия = loss.
- Філософія: шукаємо реальні gem-матчі (домінація), не ринкові неефективності.
  Другорядні ліги часто недооцінюють справжніх фаворитів.

## Архітектура

```
load_historical() → team_state + h2h → build_feature_matrix → X/y/info
                                                                ↓
                         Walk-forward CV (12 folds, 10-day gap, 2-month val)
                                                                ↓
                    ┌─── Optuna 300 trials per model (last 3 folds, n_jobs=2) ───┐
                    ↓                    ↓                    ↓
                  XGBoost             LightGBM              CatBoost
                    ↓                    ↓                    ↓
                 OOF 12-fold          OOF 12-fold          OOF 12-fold
                    └──────┬─────────────┴───────────┬─────────┘
                           ↓                         ↓
                  L2 Logistic Meta-learner (stacking)
                           ↓
               Isotonic time-aware calibration (last 15%)
                           ↓
                 Gem Filter: P(draw)<0.28, P(side)>0.72, 1.45≤odds≤3.0
                           ↓
                 Flat 4% of current bankroll → simulate ROI
```

## Аудит (2026-04-24) — критичні знахідки

### 🔴 Data leakage в odds
- **94.1%** з 27,328 одд-рядків записані ПІСЛЯ матчу (avg +442 днi)
- `opening_value` = NULL для **всіх** 27,328 рядків
- **Рішення:** прибрано 7 market-features з features.py:
  `market_home/draw/away_odds`, `market_home/draw/away_prob`, `glicko_minus_market_home`
- Odds використовуються ЛИШЕ для:
  - Gem-filter threshold на inference
  - ROI simulation (з caveat що результат оптимістичний ~2-3pp)

### ✅ Glicko pre-match — підтверджено
- Tracing Bristol City 12 матчів: rating змінюється між row N та N+1 згідно з результатом N

### ✅ home_win_prob без leakage
- Correlation з actual result = 0.348 (здоровий діапазон)
- Hit rate коли prob>0.6: 74.2% | коли prob<0.3: 22.4%

## Feature Set (67 total)

| Категорія | К-сть | Примітка |
|---|---|---|
| Raw strength (xG, Glicko, PPG + diffs) | 13 | |
| Home/away splits | 6 | xG diff at home/road, PPG splits |
| Short-term momentum | 12 | form_5, xG trend, streaks, Glicko momentum |
| Style (possession, SoT, pass_acc) | 6 | |
| H2H | 3 | home_wr, avg_goals, last_result |
| Physical (rest_days, injuries) | 5 | |
| Glicko probs (pre-match) | 3 | home/draw/away |
| League cluster (binary) | 2 | top5 / second_tier |
| League priors (target-encoded) | 3 | home_wr/draw/away_wr per-fold |
| League one-hot | 14 | |

## Файли модуля

| Файл | Опис |
|---|---|
| `model/gem/niches.py` | Константи: TARGET_LEAGUES, filter thresholds, FLAT_STAKE_FRAC |
| `model/gem/data.py` | `load_historical()` → matches/stats/odds/injuries |
| `model/gem/team_state.py` | Rolling snapshots + H2H |
| `model/gem/features.py` | `build_gem_features()` → 67-фіча dict |
| `model/gem/feature_matrix.py` | X/y/info builder + `inject_league_priors()` |
| `model/gem/cross_val.py` | Walk-forward CV з 10-day gap |
| `model/gem/preprocessing.py` | `LeagueTargetEncoder` + Bayesian smoothing |
| `model/gem/ensemble.py` | 3-model stacking + Optuna + meta-learner |
| `model/gem/calibration.py` | Isotonic time-aware (per-class OvR + renormalise) |
| `model/gem/evaluate.py` | Metrics, baselines, gem simulation, SHAP |
| `model/gem/train.py` | Main entry + JSON experiment log |

## Smoke test результати (3 trials × 3 folds, 2026-04-25)

Цифри на мінімальних параметрах — не фінальні, але пайплайн працює end-to-end.

| Metric | Value |
|---|---|
| log-loss ensemble (raw) | 1.0001 |
| log-loss ensemble (calibrated) | 0.9981 |
| log-loss baseline dummy | 1.1689 |
| log-loss baseline glicko-only | 1.0724 |
| log-loss train (full refit) | 0.6988 |
| **Overfit gap** | **-0.47** (DUŽE high — очікуємо менше при 300 trials) |

**Gem sim (61 bets):**
- WR: 60.7%
- ROI: -3.55%
- Max drawdown: -22.5%
- Sharpe (weekly): -0.57

### Top-10 features (SHAP, XGBoost)

1. `glicko_home_prob` (0.096) — central signal ✅
2. `glicko_away_prob` (0.092)
3. `home_pass_acc` (0.039)
4. `glicko_draw_prob` (0.038)
5. `away_pass_acc` (0.034)
6. `home_xg_diff_at_home` (0.031)
7. `league_prior_draw_rate` (0.030) ✅ target-encoded feature працює
8. `rest_gap` (0.029)
9. `away_xg_diff_on_road` (0.029)
10. `context_ppg_gap` (0.028)

## Баги виправлені під час Phase 3

1. `lightgbm==4.5` несумісний з `sklearn==1.8` → upgrade до `4.6.0`
2. CatBoost `subsample` потребує `bootstrap_type="Bernoulli"`
3. CatBoost `bagging_temperature` тільки для Bayesian bootstrap → замінено на `colsample_bylevel`
4. XGBoost `early_stopping_rounds` падає на final refit без eval_set → пересоздаємо модель чисто
5. XGBoost `save_model` втрачає `_estimator_type` → fresh init перед final fit

## Як запустити

```bash
# Повний тренінг (~5-6 год з 300 trials × 12 folds)
docker exec -e PYTHONPATH=/app capper_scheduler python -m model.gem.train --trials 300 --folds 12

# Швидкий smoke (~10 хв)
docker exec -e PYTHONPATH=/app capper_scheduler python -m model.gem.train --trials 3 --folds 3
```

Артефакти:
- `model/gem/artifacts/` — моделі, calibrator, encoder
- `model/gem/experiments/exp_YYYYMMDD_HHMMSS.json` — повний лог запуску

## Caveat ROI

Через те що 94% історичних odds = closing snapshot, симульований ROI оптимістичний.
У production на opening odds очікуй 2-3pp зниження ROI.
