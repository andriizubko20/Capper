# Gem A/B Test — kmeans-3 Calibrator (2026-04-27)

Per-league calibration оптимізація — пошук найкращого `tail_frac`/`min_per_league`/cluster strategy через 6 паралельних експериментів.

## Контекст

Після введення per-league calibration heads (V1 PROD: +46.88 u/yr) шукали ще профіт через варіації:
- Hyperparameter tuning calibration (грід-серч 390 конфігів)
- Альтернативні методи калібровки (Platt, beta, blend)
- Per-league sweep guards customization
- Per-league ensemble blending (re-fit XGB/LGB/CatBoost meta)
- EPL drop investigation

## Master таблиця експериментів

| # | Variant | Active | Picks | u/yr | Δ vs PROD | Verdict |
|---|---|---|---|---|---|---|
| 0 | Global cal only | 8 | 215 | +21.39 | -25.49 | ❌ baseline |
| **1** | iso league + global (PROD V1) | 10 | 535 | **+46.88** | 0 | ✅ baseline |
| 2 | iso league+tier+global | 10 | 544 | +47.45 | +0.57 | ⚠️ noise |
| 3 | tier-only | 7 | 208 | +20.67 | -26.21 | ❌ |
| 4 | iso league+kmeans-4+global (default) | 10 | 552 | +46.84 | -0.04 | ⚠️ noise |
| 5 | kmeans-only | 8 | 219 | +25.44 | -21.44 | ❌ |
| 6 | Platt scaling | 8 | 406 | +34.07 | -12.81 | ❌ |
| 7 | Beta calibration | 8 | 319 | +33.39 | -13.49 | ❌ |
| 8 | iso+platt blend | 9 | 486 | +43.07 | -3.81 | ❌ |
| 9 | per-league META re-fit | 9 | 453 | +36.10 | -10.78 | ❌ overfit |
| 10 | Grid #1: kmeans-5, mpl=50, tail=0.20 | 10 | 506 | +45.92 | -0.96 | ⚠️ |
| **11** | **Grid #2: kmeans-3, mpl=20, tail=0.10** | **11** | 608 | **+52.49** | **+3.33** | ✅ **A/B candidate** |
| 12 | Per-league guards V2 (TOP5 lo95≥0.30) | +1-2 leagues | +10-20 | +1-2 | additive | можна додати поверх |

## Що працює

🥇 **Per-league iso heads — головний driver value** (V0 → V1: +25.5 u/yr)

🥈 **K-means clustering з тонкою настройкою** (Grid #2): додає +3.3 u/yr за рахунок:
- Менший `min_per_league` (20 vs 30) — більше ліг отримують власні голови
- Менший `tail_frac` (0.10 vs 0.15) — recency-фокусована калібровка
- 3 data-driven кластери (за середнім pH/pD/pA per league) як fallback

🥉 **Tier-aware sweep guards** (TOP5 lo95≥0.30): може повернути La Liga / Bundesliga / Ligue 1 на +1-2 u/yr

## Що НЕ працює

❌ **Per-league META re-fit** — overfit на малих per-league sample, -9 u/yr.
   Insight зберігся: LGB домінує 8 ліг (La Liga, Serie B, EPL...), XGB — 6 (Süper Lig, Eredivisie...), CAT — 4. Корисно для майбутнього sample weighting.

❌ **Platt / Beta / Blend** — точніші за log_loss, але менше picks → менше u/yr.
   Isotonic (current) — кращий саме для sweep optimization.

❌ **Tier-only / KMeans-only (без per-league heads)** — втрачає per-league granularity, на 25 u/yr гірше.

## Поточне рішення: A/B test

- **gem_v1** (baseline PROD) — V1 calibrator: per-league iso + global, +46.88 u/yr
- **gem_v2_kmeans3** (новий) — Grid #2: kmeans-3 fallback + mpl=20 + tail=0.10, +52.49 backtest

Обидва використовують **спільний ensemble** (XGB+LGB+CatBoost+meta). Різниця тільки в calibrator.pkl + per_league_thresholds*.json.

Cron: gem_v1 о :33, gem_v2 о :37. Через 30 днів живих picks бачимо чий ROI вищий → лишаємо переможця.

## Архітектурні зміни

- `model/gem/calibration.py:GemCalibrator` — додано hierarchical `league > cluster > global` з `cluster_strategy ∈ {none, tier, kmeans-K}`. Default `none` (V1 backwards compat).
- `model/gem/train.py` — fits BOTH calibrators per train run, зберігає `calibrator.pkl` + `calibrator_v2_kmeans3.pkl` + `oof.npz` ключем `oof_ensemble_calibrated_v2`.
- `model/gem/per_league_sweep.py` — `--variant v1|v2` CLI flag, читає відповідний OOF, пише `per_league_thresholds.json` або `_v2.json`.
- `scheduler/tasks/generate_picks_gem_v2.py` — thin wrapper що завантажує V2 calibrator + V2 thresholds, зберігає picks з `model_version='gem_v2_kmeans3'`.
- `api/utils.py` — реєстрація "Gem v2" → mini-app Compare screen покаже 6 моделей.

## Невирішені

- **EPL drop** на per-league cal: модель з global cal ловить +4.61 u/yr на EPL (88% WR), per-league cal все відсіває. Гіпотеза: per-league head на 81 tail samples overfit-ить, global cal "наївно" вгадує. Не investigated повністю — script мав bug. Можливо повернеться через kmeans-3 cluster fallback.
- **La Liga, Bundesliga, Ligue 1** не пройшли sweep при будь-яких варіаціях. Real model effect, не калібровка.
- **Sample-weighted training** (відкладено): надати 1.5× weight на проблемні ліги при тренуванні. Потребує retrain з `sample_weight` параметром у XGBoost/LGB/CatBoost.

## Files
- `model/gem/calibration.py` — нова hierarchical calibration
- `model/gem/train.py` — fits 2 cals
- `model/gem/per_league_sweep.py` — variant flag
- `scheduler/tasks/generate_picks_gem_v2.py` — A/B picker
- `scheduler/main.py` — :37 cron
- `api/utils.py` — Gem v2 registered

## Related
- [Session 2026-04-26](session-2026-04-26.md) — попередня сесія: бекфіл odds, retrain ground
- [Gem Model](gem-model.md)
- [Kelly Criterion](kelly-criterion.md)
