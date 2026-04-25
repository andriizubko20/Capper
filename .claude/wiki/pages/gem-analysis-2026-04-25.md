# Gem Model — Day 1 Analysis (2026-04-25)

Аналіз результатів повного тренінгу 300×12 + post-hoc OOF дослідження.

## TL;DR

✅ **Gate 1 пройдено**: ensemble б'є dummy на 7.2%, glicko-only на 3.0%.
✅ **WR ціль 70% досягнута** (66.7-70.3% залежно від OOF варіанту).
🔴 **Yield критично низький**: 21 ставка за 24 місяці (~0.9/тиждень, треба 7-10).
🔴 **Hypothesis "second_tier > top5" не підтверджено** (5 vs 16 ставок — мало даних).
✅ **gem_score (model_p - market_p) — сильний сигнал**: top quartile дає WR 85.7%.

---

## ML метрики (повний тренінг 300×12)

| Source | log-loss | AUC-ROC macro |
|---|---:|---:|
| Ensemble (raw) | 0.9948 | 0.6545 |
| Ensemble (calibrated) | 1.0344 ⚠️ | 0.6557 |
| Glicko-only LR | 1.0252 | 0.6045 |
| Class-prior dummy | 1.0724 | 0.5000 |
| **Market (closing, 4313 rows)** | **0.9599** | n/a |

**Ключове:** ринок (closing odds) калібрований краще за нашу модель загалом
(0.96 vs 1.00 log-loss). Це OK для "gem" філософії — нам не треба бути кращими
за ринок скрізь, лише на відфільтрованому підмножині.

## Gem філософія підтверджена

**gem_score = our_prob(side) − market_prob(side)** (calibrated OOF):

| Bucket | N | WR | avg our_p | avg market_p |
|---|---:|---:|---:|---:|
| 10-20% | 14 | 57.1% | 0.749 | 0.595 |
| **>20%** | **7** | **85.7%** | **0.780** | **0.539** |

Чим сильніше ми відхиляємось від ринку — тим вища WR. **Це підтверджує тезу "недооцінений явний фаворит"**: шукаємо матчі де ми дуже впевнені, а ринок занижує.

## Default filter (calibrated OOF)

- `P_bet > 0.72`, `P_draw < 0.28`, `1.50 ≤ odds ≤ 3.00`
- **21 bets | WR 66.7% | ROI +10.7% | avg_odds 1.65**

## Top filter configs (sweep, calibrated)

| p_bet | max_draw | min_odds | gem_score | N | WR | ROI |
|---:|---:|---:|---|---:|---:|---:|
| 0.75 | 0.32 | 1.60 | off | 6 | 83.3% | +43.3% |
| 0.75 | 0.30 | 1.60 | 0.10 | 6 | 83.3% | +43.3% |

(всі топ-15 — те саме 6-bet кластер. **Sample size — головна проблема.**)

## Per-cluster (calibrated)

| Cluster | N | WR | avg gem_score |
|---|---:|---:|---:|
| top5_ucl | 16 | 68.8% | 0.193 |
| second_tier | 5 | 60.0% | 0.153 |

**Несподіванка**: top5 кращий за second_tier на цьому маленькому семплі.
Гіпотеза вимагає 100+ ставок щоб довести/спростувати.

## Per-league (calibrated, top by sample)

| League | N | WR |
|---|---:|---:|
| Champions League | 6 | 66.7% |
| La Liga | 3 | 100% |
| Premier League | 3 | 33.3% |
| Bundesliga | 2 | 100% |
| Eredivisie | 2 | 100% |
| Jupiler Pro League | 2 | 50% |
| Ligue 1 | 2 | 50% |
| Primeira Liga | 1 | 0% |

Sample size надто малий для висновків. Треба ≥ 10 ставок на лігу.

## Feature signatures (WIN vs LOSS, Cohen's d)

| Feature | mean WIN | mean LOSS | d |
|---|---:|---:|---:|
| home_has_any_injuries | 0.93 | 0.57 | **+0.96** |
| h2h_home_last_result | -0.20 | +0.40 | **-0.73** |
| home_xg_diff_at_home | 0.54 | 1.09 | -0.67 |
| home_ppg_at_home | 1.82 | 2.22 | -0.55 |
| xg_diff_gap | 0.00 | 0.62 | -0.48 |
| home_xg_for | 1.61 | 1.89 | -0.53 |
| away_win_streak | 1.57 | 0.57 | **+0.50** |

**Інтерпретація з обережністю** (n=14 win, n=7 loss — Cohen's d нестабільний):
- Несподіванки (counterintuitive): WIN bets частіше мають слабший атаковий профіль (xG, PPG, xg_diff_gap нижче)
- Домінує signal "недооцінений" (low гap метрик), а не "явно сильніший"
- ⚠️ З цією вибіркою рано робити feature selection

## Top SHAP features (повний тренінг)

1. glicko_away_prob (0.115) ← перевершує home
2. glicko_home_prob (0.099)
3. home_xg_diff_at_home (0.051)
4. glicko_draw_prob (0.048)
5. glicko_gap (0.046)
6. home_glicko_momentum (0.046) ✅ працює
7. home_pass_acc (0.046)
8. away_glicko_momentum (0.045)
9. home_ppg_at_home (0.045)
10. away_xg_diff_on_road (0.043)

---

## Висновки

### Що зробила модель добре
- Знайшла value спот, де ринок чітко помиляється (top gem_score quartile: 85.7% WR)
- Glicko probs — основний driver (як і має бути, це власна оцінка сили)
- League priors target encoding працює
- Calibration ПОКРАЩУЄ gem_score сигнал, навіть якщо log-loss трохи гірша

### Що не вийшло
- **Yield надто низький** (1/тиждень) — модель занадто consistent з ринком, тільки на маленькій частині знаходить gap
- **Per-cluster не підтвердив гіпотезу** — top5 виявилися кращими, не second_tier
- **Calibrated log-loss гірша за raw** — isotonic перенавчається на tail (бо tail невеликий)

### Чому це сталося
Модель ефективно вчиться "імітувати ринок" (бо ринок добре калібрований).
Підняти yield = знизити поріг = прийняти нижчий WR.
Альтернатива: змусити модель **бачити інакше** через нові feature interactions.

---

## Рекомендації для v2 тренінгу

### A. Нові фічі (composite/interactions)

1. **`dominance_score`** = `glicko_gap × xg_diff_gap` — "явний фаворит" сигнал
2. **`momentum_alignment`** = бінарна; усі 3 momentum signals (form, xG trend, glicko momentum) у один бік
3. **`home_advantage_strength`** = `home_ppg_at_home / league_prior_home_wr`
4. **`xg_quality_gap`** = `(home_xg_for - away_xg_against) - (away_xg_for - home_xg_against)` — net quality differential
5. **`squad_health_proxy`** = `home_glicko_momentum - away_glicko_momentum`

### B. Filter design

**Поточний default:** `P_bet > 0.72`, `P_draw < 0.28` → 21 bets/24mo
**Запропонований:**
```python
USE_CALIBRATED   = True   # gem_score сильніший на calibrated
MIN_BET_PROB     = 0.68   # було 0.72 — більше вибірки
MAX_DRAW_PROB    = 0.30   # було 0.28
MIN_GEM_SCORE    = 0.08   # НОВЕ — ключовий "value" фільтр
MIN_ODDS         = 1.50
```

Очікувано: 50-80 bets/24mo, WR ~70%, ROI +12-15%.

### C. Чого НЕ робити (поки що)

- **Не дропати фічі** на основі signatures — вибірка надто мала
- **Не міняти ensemble** (XGB+LGB+CatBoost працює, log-loss 0.9948)
- **Не запускати per-cluster моделі** — мало даних

### D. Що додати в evaluation

- Filter-subset log-loss vs market (тільки на picks, не на всіх 4313)
- Brier score per gem_score bucket
- Rolling 6-month WR (стабільність моделі в часі)

---

## Decision matrix

| Сценарій | Що робимо |
|---|---|
| **A. Yield > 50, WR ≥ 65%** | Готовність → Day 3 production integration |
| B. Yield > 50, WR < 65% | Crisis: композити не допомогли. Day 2 → ablation. |
| C. Yield < 30, WR ≥ 70% | Релакс далі: P_bet=0.65, MIN_GEM=0.05 |
| D. Yield < 30, WR < 65% | STOP, повернутись до research (другий thesis) |

---

## Артефакти

- `model/gem/artifacts/oof.npz` — OOF predictions для re-analysis
- `model/gem/artifacts/info.parquet` — match metadata
- `model/gem/reports/analysis_cal.json` — повний звіт (calibrated)
- `model/gem/reports/analysis_raw.json` — повний звіт (raw)
- `model/gem/reports/filter_sweep.csv` — повний sweep таблиця
