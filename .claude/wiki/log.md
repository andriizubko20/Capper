# Log

Append-only record of wiki operations.

## [2026-04-20] init | Wiki migrated from global ~/.claude/wiki to project .claude/wiki
## [2026-04-22] ingest | Bug Audit — знайдено 33 проблеми, виправлено 8 критичних+high; 12 medium задокументовано
## [2026-04-22] ingest | Project Status — оновлено з bug fixes, посилання на audit page
## [2026-04-22] ingest | Bug Audit — medium баги виправлені (9 з 12); audit page оновлено зі статусами
## [2026-04-22] ingest | Timing fix — confirm_picks кожні 2год; API визначає timing реально замість DB lag
## [2026-04-22] ingest | Model bugs — 5 model-level багів знайдено і виправлено: Kelly cap early scan, ODDS_MAX, ELO momentum, XG false NaN ×2
## [2026-04-22] ingest | WS Gap Model page — повністю переписано: константи, версії, features, bug-фікси
## [2026-04-22] ingest | Monster Model page — переписано: ніші, features, bug-фікси
## [2026-04-22] ingest | Bug Audit page — додано секцію Model-Level Bugs (items 21-25)
## [2026-04-22] ingest | Deep Audit Pass 2 — знайдено і виправлено 19 багів (items 26-44): UTC dates, cache TTL, lazy load, isoDate, rollback, commit granularity, history limit, auth_date, ELO NaN тощо
## [2026-04-22] ingest | Project Status — оновлено з pass 2 фіксами; оновлено "Що залишилось"
## [2026-04-22] ingest | Deploy VPS — 165.227.164.220; 44 bug fixes; docker recreate; всі 4 контейнери up
## [2026-04-22] ingest | EV thresholds research — 57 production picks, нуль деактивацій, threshold 0.0 залишається; revisit @100+ bets per model
## [2026-04-22] ingest | VPS ↔ SStats connectivity — network path NYC↔UK обриває chunked response (4% отримується); workaround через Mac proxy + SSH reverse tunnel + socat
## [2026-04-22] ingest | DB restore — match_stats(0→18506), injury_reports(0→30085), monster_p_is(0→76), відсутні колонки додані; pg_dump з локальної БД + psql stream
## [2026-04-22] ingest | Pick regeneration — WS Gap 29→33 (+4), Monster 21→23 (+2, Telegram OK), Aqua 7→7; Total 57→63 active picks
## [2026-04-22] ingest | Leagues sync — DROP UNIQUE(api_id) + ADD UNIQUE(api_id, season); +56 ліг з локальної; orphaned matches 11485→0
## [2026-04-22] ingest | Match 5247 backfill — OH Leuven 0:2 Westerlo pick created+settled (WS Gap away@2.45 WIN +$288.43); bankroll $2597.69→$2886.12
## [2026-04-22] ingest | Model efficacy analysis — 47 bets, WR 68%, ROI +40%; red flags: EV inversion (10-25% EV має WR 76.9% vs 50%+ EV має 58.3%), final picks -2.5% ROI, CLV not tracked, league_name empty
## [2026-04-22] ingest | Cloudflare Worker rejected — `aqua.andrii-zubko20.workers.dev` отримує ті самі truncated 14751 байт за 77с; api.sstats.net обмежує datacenter IP в цілому
## [2026-04-22] ingest | Proxy permanent setup — launchd (Mac proxy + SSH tunnel) + systemd capper-proxy-forwarder (VPS socat) переживають crash/sleep/reboot; Mac увімкненість — єдиний SPOF
## [2026-04-23] ingest | CLV + league_name fix — 4 bugs: generate_picks_*.py не писали league_name/home_name/away_name/match_date; update_clv.py фільтр "FT" замість "Finished"; closing odds без recorded_at. Fixed + backfilled 64 predictions + deployed via docker cp. CLV тепер працює (3 settled picks avg +0.68), per-league breakdown доступний (EPL лідер 72.7% WR, La Liga 44%)

## [2026-04-24] ingest | Gem Model v1 — architecture + audit findings
- 94% of historical odds recorded post-match → removed 7 market features from training
- Glicko + win_prob sanity-checked (pre-match ✅)
- Phase 3 code written: 7 new files under model/gem/
- Bugs fixed during smoke: lightgbm→4.6, catboost bootstrap_type, xgb early_stopping

## [2026-04-25] ingest | Gem smoke passed + 2-week roadmap
- Smoke test (3 trials × 3 folds) прошел: log-loss 0.9981 (calibrated), 61 bets sim (WR 60.7%, ROI -3.55%)
- SHAP top: glicko_home_prob (0.096), glicko_away_prob (0.092), home_pass_acc (0.039), league_prior_draw_rate (0.030)
- Fixed bugs: xgb early_stopping, xgb save_model, catboost bootstrap_type/bagging_temp, lgb sklearn compat
- Added n_jobs=2 в Optuna для 30-40% прискорення повного тренінгу
- Roadmap: 2 тижні план до production Telegram + live WR
## [2026-04-26] ingest | Work plan: 6 blocks data infrastructure overhaul
## [2026-04-26] ingest | Infra refactor session — per-league cherry-pick devig (+21u/yr), Kelly compounding, 3 Quick Wins, Mac proxy fix, ProxyMon + CLVMon Telegram alerts
