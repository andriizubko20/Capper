# Gem Model — Roadmap 2 Weeks

Starting: **2026-04-25**
Target: Gem model у production, daily picks в Telegram, WR ≥ 65% на живих матчах.

---

## Week 1 (25 квітня → 1 травня)

### Day 1 — 25.04 (сьогодні/завтра)
- [ ] **Ніч 25→26:** повний тренінг 300 trials × 12 folds (~5-6 год)
- [ ] Ранок: перегляд метрик, SHAP, симуляції
- [ ] Decision gate: WR на OOS ≥ 65%? → ідемо в production. Інакше → Day 2 ablation.

### Day 2 — 26.04
**Якщо WR < 65%:**
- [ ] Ablation study: вимкнути по черзі league_priors / H2H / injuries / style → що реально драйвить log-loss
- [ ] Sensitivity: rolling_window = 10 vs 15 vs 20 (вже tuned params, швидко)
- [ ] Per-cluster тест: окрема модель для top5_ucl і second_tier

**Якщо WR ≥ 65%:**
- [ ] Commit Phase 3 + artifacts в git
- [ ] Oновити `requirements.txt`: catboost, optuna, shap, lightgbm>=4.6

### Day 3 — 27.04: Production integration part 1
- [ ] `model/gem/predictor.py`: load ensemble+calibrator+encoder → predict_on_match(match_id)
- [ ] DB migration: додати `model_name='gem'` в `predictions` таблицю
- [ ] Feature extraction для live match (використати existing team_state pipeline)

### Day 4 — 28.04: Production integration part 2
- [ ] Інтеграція в scheduler: daily job `scheduler/gem_picks.py`
- [ ] Apply gem filter: P(draw)<0.28, P(side)>0.72, odds ∈ [1.45, 3.0]
- [ ] Save picks → `predictions` (with `kelly_stake = 4%`, `is_gem = true`)

### Day 5 — 29.04: Forward-testing setup
- [ ] Paper-trading: Gem picks генеруються але не йдуть в бот
- [ ] Dashboard endpoint `/api/gem/picks` — показує свіжі picks
- [ ] Track vs Monster/Aquamarine у StatsScreen

### Day 6 — 30.04: Live validation
- [ ] Перші Gem picks на вихідних (Premier League, Bundesliga etc.)
- [ ] Моніторинг: чи filter пропускає правильні матчі, чи калібровано P(bet)

### Day 7 — 01.05: Week 1 review
- [ ] Перші ~20 ставок пройшли (якщо yield 7-10/тиждень)
- [ ] WR first batch?
- [ ] Update wiki: `gem-model-status.md` з живими цифрами

---

## Week 2 (2 → 8 травня)

### Day 8-9 — 02-03.05: Odds quality fix (критично)
- [ ] Проаналізувати чому `opening_value = NULL` скрізь. SStats API returns `openingValue`?
- [ ] Fix `backfill_odds.py` щоб правильно зберігати opening
- [ ] Set up live odds tracker (5-min scan) щоб зберігати opening + closing properly
- [ ] Re-backfill за останні 3 місяці з правильним opening

### Day 10 — 04.05: Train з правильними odds
- [ ] Rerun training з clean odds
- [ ] Пере-аудит: можливо відновити деякі market features (якщо справді pre-match)
- [ ] Compare: чи SHAP рейтинг фіч змінився?

### Day 11 — 05.05: Ensemble diversity
**Умова:** якщо WR все ще < 70% після Day 10
- [ ] Додати LogisticRegression як 4-ту модель (з scaler+impute pipeline) для diversity
- [ ] Спробувати Dirichlet calibration замість per-class isotonic
- [ ] Experiment: draw threshold 0.25 vs 0.28 vs 0.30

### Day 12-13 — 06-07.05: Telegram integration
- [ ] `bot/handlers/gem.py`: daily broadcast о ~11:00
- [ ] Формат: матч, pick, odds, stake, reasoning (top-3 SHAP features)
- [ ] Subscribe/unsubscribe команди
- [ ] Tests: bot handler + e2e

### Day 14 — 08.05: Deploy + monitoring
- [ ] Deploy Gem до VPS production
- [ ] Prometheus/logs: track daily yield, WR rolling 30d, max drawdown
- [ ] Final wiki update: production status + перший livewr тиждень

---

## Risk checkpoints

| Risk | Detection | Mitigation |
|---|---|---|
| WR смокел-тесту = 60% (not 70%) | Day 1 metrics | Day 2 ablation + rolling window tune |
| Overfit gap = -0.47 на smoke | Day 1 full run | Більше regularization; більше trials; дропнути worst features |
| Odds quality blocks real ROI | Day 7 forward-testing | Week 2 fix — вже заплановане |
| Yield < 5 bets/week | Day 6 live check | Пом'якшити filter (P(side)>0.65 замість 0.72) |
| Model деградує між ліг | SHAP per-league + per-cluster sim | Per-cluster models в Week 2 |

## Decision gates

- **Gate 1 (Day 1):** Overall log-loss + baseline порівняння. Якщо ensemble б'є glicko-only на <5%, зупинити.
- **Gate 2 (Day 2):** Sim WR ≥ 65% + sample size ≥ 500 bets. Якщо ні — ablation.
- **Gate 3 (Day 7):** Live WR перших 20 bets ≥ 55% (noise acceptable). Якщо нижче — stop live, повернутися до research.
- **Gate 4 (Day 14):** 2-week live WR ≥ 60%, ROI > 0% (after closing-odds correction). → production stays.

## Out of scope (v2)

- Multi-market (totals, BTTS, handicap) — зараз тільки 1x2
- Accumulators
- Player-level features (player stats, lineups, injuries attribution)
- Weather / referee / pitch data
- Automated retraining (buy-and-hold модель 3-6 міс)
