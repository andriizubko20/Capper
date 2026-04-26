# Gem Retrain Runbook — +6 Leagues

Adds UPL, Eliteserien, Premiership, Ekstraklasa, Allsvenskan, Süper Lig to the
Gem ensemble. Pre-req: historical odds backfill (`capper-backfill` on VPS) is
finished and the SStats proxy chain is up.

Run end-to-end on the VPS unless noted. All steps assume cwd `/opt/capper` and
the project venv is active (`source venv/bin/activate`).

---

## 1. Pre-flight checks

Backfill finished:
```bash
ssh root@165.227.164.220 "systemctl is-active capper-backfill"   # expect: inactive | (unit not found)
ssh root@165.227.164.220 "journalctl -u capper-backfill --since '6h ago' --no-pager | tail -50"
```

Odds coverage on the 6 new leagues (acceptance: each league > 200 finished
matches with 1x2 odds):
```bash
docker compose exec -T db psql -U capper -d capper <<'SQL'
SELECT l.country, l.name,
       COUNT(DISTINCT m.id)        AS finished,
       COUNT(DISTINCT o.match_id)  AS w_odds
FROM matches m
JOIN leagues l ON l.id = m.league_id
LEFT JOIN odds o ON o.match_id = m.id AND o.market = '1x2'
WHERE l.name    IN ('Premier League','Eliteserien','Premiership','Ekstraklasa','Allsvenskan','Süper Lig')
  AND l.country IN ('Ukraine','Norway','Scotland','Poland','Sweden','Turkey')
  AND m.home_score IS NOT NULL
GROUP BY l.country, l.name
ORDER BY l.country;
SQL
```

SStats proxy is healthy (Mac → SSH tunnel → socat):
```bash
ssh root@165.227.164.220 "tail -30 /var/log/capper/sstats_proxy.log"
docker compose exec -T scheduler python -m scheduler.tasks.monitor_sstats_proxy --once
```

Code is clean and on HEAD:
```bash
cd /opt/capper && git status --short && git log -1 --oneline
```

Tests pass:
```bash
pytest tests/test_movement_filter.py tests/test_best_odds.py -q
```

---

## 2. Update TARGET_LEAGUES

Edit `model/gem/niches.py` — add 6 entries to `SECOND_TIER`. NB: 3 of the 6
(Süper Lig, Eliteserien, Allsvenskan) are already present at HEAD; only 3 are
truly new. Final block must be:

```python
SECOND_TIER: set[tuple[str, str]] = {
    ("Championship",        "England"),
    ("2. Bundesliga",       "Germany"),
    ("Eredivisie",          "Netherlands"),
    ("Jupiler Pro League",  "Belgium"),
    ("Primeira Liga",       "Portugal"),
    ("Süper Lig",           "Turkey"),
    ("Eliteserien",         "Norway"),
    ("Allsvenskan",         "Sweden"),
    ("Premier League",      "Ukraine"),
    ("Premiership",         "Scotland"),
    ("Ekstraklasa",         "Poland"),
}
```

`LEAGUE_NAMES_ORDERED` regenerates automatically (sorted on import) — no other
file edits required. Commit this change in step 8.

---

## 3 + 4 + 5 + 6. Train ensemble (one entry point does all four)

`model/gem/train.py` is the master pipeline: it loads data via
`load_historical()`, builds the feature matrix, runs Optuna + walk-forward CV,
fits the L2 stacking meta, fits the isotonic calibrator on the 15% tail, runs
the SHAP audit, and writes every artifact below.

Single command:
```bash
nohup python -m model.gem.train --trials 200 --folds 12 --tail 0.15 \
  > /var/log/capper/gem_train_$(date +%Y%m%d_%H%M).log 2>&1 &
tail -F /var/log/capper/gem_train_*.log | grep --line-buffered -E \
  "Loaded:|Feature matrix|fold|Optuna|Calibrating|EVALUATION|GEM BET|SAVING|Traceback|Error"
```

Outputs (in `model/gem/artifacts/`):
- `xgb_model.json`, `lgb_model.txt`, `cat_model.cbm` — base learners
- `meta_model.pkl` — L2 logistic stack
- `league_encoder.pkl` — final `LeagueTargetEncoder`
- `params.json`, `feature_names.json` — tuned hyperparams + schema
- `calibrator.pkl` — isotonic OvR calibrator
- `oof.npz` — `oof_xgb`, `oof_lgb`, `oof_cat`, `oof_ensemble_raw`,
  `oof_ensemble_calibrated`, `y`, `covered`
- `info.parquet` — per-row metadata used by the sweep
- `experiments/exp_<ts>.json` — full run log

Estimated time on the VPS: **TODO: verify on VPS** — prior 7-league run took
~45–60 min; +3 leagues + multi-bookmaker odds rows likely **75–90 min** with
default `--trials 200 --folds 12`. If wall-time is tight, drop to
`--trials 120 --folds 10` (≈40 min) for a first pass; this is the highest-risk
step (Optuna × 12 folds × 3 base models is CPU-bound; no GPU used).

Sanity checks after train completes — open the new
`experiments/exp_*.json` and confirm:
- `config.n_samples` jumped vs the previous run (proportional to new league
  matches with odds).
- `metrics.ensemble_calibrated.log_loss` ≤ previous run + 0.01.
- `metrics.simulation.roi` ≥ previous run − 1pp.
- `metrics.overfit_gap_logloss` < 0.05.

---

## 7. Per-league sweep

```bash
python -m model.gem.per_league_sweep
```

Reads `artifacts/oof.npz` + `artifacts/info.parquet`, sweeps the
(devig × odds_lo × odds_hi × max_draw × min_bet × min_gem) grid per league,
filters with `n_bets ≥ 12`, `WR ≥ 0.55`, `ROI ≥ +5%`, `lo95 ≥ 0.40`, picks the
combo maximising `annual_units`.

Output: `model/gem/artifacts/per_league_thresholds.json` (canonical
"Country: Name" keys) + `model/gem/reports/per_league_sweep.csv`.

Acceptance:
- The 6 new leagues appear in the JSON, OR fail with a logged
  `❌ <league>: no viable combo` (acceptable for now — those leagues are
  simply ineligible until more data accumulates).
- Console total `units/yr` ≥ previous run's total.

---

## 8. Deploy

```bash
# locally / in worktree:
cd /Users/andrii.zubko/Desktop/projects/Capper
git add model/gem/niches.py model/gem/artifacts/ model/gem/reports/per_league_sweep.csv
git commit -m "Retrain Gem with 6 new leagues (UPL/NOR/SCO/POL/SWE/TUR)"
git push origin main

# VPS:
ssh root@165.227.164.220 "cd /opt/capper && git pull && \
  docker compose build scheduler && docker compose up -d scheduler"
```

Wipe & rebuild Gem picks (canonical league names changed; old picks would
collide):
```bash
docker compose exec -T db psql -U capper -d capper \
  -c "DELETE FROM predictions WHERE model_version='gem_v1';"

docker compose exec -T scheduler python -m scheduler.tasks.backfill_picks_gem --from 2026-04-12
docker compose exec -T scheduler python -m scheduler.tasks.update_results
docker compose exec -T scheduler python -m scheduler.tasks.rebuild_stakes_chronological --model gem_v1
```

---

## 9. Verification

API:
```bash
curl -s 'http://165.227.164.220:8000/api/compare?period=ALL' \
  | jq '.models[] | select(.name=="Gem")'
```

DB — picks per league (expect non-zero rows for each new league that produced
a viable threshold combo):
```sql
SELECT model_version, league_name, COUNT(*) AS n
FROM predictions
WHERE model_version='gem_v1'
GROUP BY model_version, league_name
ORDER BY n DESC;
```

Spot-check that movement filter is still additive (count today's filtered
picks):
```bash
docker compose logs scheduler --since 24h | grep -c "movement_filter: skip"
```

---

## 10. Rollback

If aggregate `units/yr` in the new sweep < previous run's total, restore
artifacts and config from the prior commit:

```bash
ssh root@165.227.164.220 "cd /opt/capper && \
  git checkout HEAD~1 -- model/gem/artifacts/ model/gem/niches.py && \
  docker compose restart scheduler && \
  docker compose exec -T db psql -U capper -d capper \
    -c \"DELETE FROM predictions WHERE model_version='gem_v1';\" && \
  docker compose exec -T scheduler python -m scheduler.tasks.backfill_picks_gem --from 2026-04-12 && \
  docker compose exec -T scheduler python -m scheduler.tasks.update_results && \
  docker compose exec -T scheduler python -m scheduler.tasks.rebuild_stakes_chronological --model gem_v1"
```

If `git checkout HEAD~1` is the wrong target (e.g. you committed unrelated
changes after step 8), revert to a known-good SHA from
`git log --oneline -- model/gem/artifacts/` instead.
