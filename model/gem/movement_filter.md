# Gem Movement Filter — Design (Phase 1 of Sharp-Money Integration)

Goal: use line-movement snapshots already collected by
`scheduler/tasks/track_odds_movement.py` to filter Gem picks where the
betting market has clearly moved AGAINST our pick (a sharp-money signal we
should respect).

---

## 1. Investigation summary

### What the tracker actually collects

`scheduler/tasks/track_odds_movement.py`:

- Inserts a NEW row into `odds` every ~30 min (deduped via
  `MIN_SNAPSHOT_GAP_MINUTES = 25`).
- Window: matches starting in the next `LOOKAHEAD_HOURS = 36` hours that are
  still `Not Started`.
- Markets tracked: `("1x2",)` only (for now).
- All bookmakers returned by SStats are kept (one row per
  match × bookmaker × outcome × `recorded_at`).
- `opening_value` per (match, market, outcome) is preserved across snapshots
  (`MIN(opening_value)` of any prior row, or fallback to API
  `openingValue` / current `value` on the first snapshot).
- `is_closing = False` for every snapshot — closing flag is only set by a
  separate task at kickoff.

So the timeline per (match, bookmaker, outcome) is:

```
recorded_at  →  value         opening_value (fixed)
T0 (kickoff-36h)   1.95        1.95
T0 + 30m           1.97        1.95
T0 + 60m           2.00        1.95
…
T_close            2.05        1.95   ← can be flagged is_closing later
```

### Data availability

- **Historical training data (2023–2025):** odds is a single static row per
  match (or one per bookmaker, all `recorded_at` ≈ ingestion time). NO
  movement curve. So Gem's training matrix CANNOT include movement
  features without retraining on prospective data only.
- **From now on:** every match in the next 36 h gets up to ~72 snapshots
  (36 h × 2/h). In practice picks are generated 1.5–6 h before kickoff
  (`settings.picks_hours_before`), so we typically see 3–12 snapshots per
  match before pick time, across multiple bookmakers.

### Implication — why **Option C**

- Option A (auxiliary model): needs labeled outcome data, which we won't
  have at meaningful sample size for ≥ 6 months.
- Option B (full retrain): same problem, plus retraining the ensemble is
  outside scope.
- **Option C (additive filter):** uses movement signals as a hard skip
  rule on top of Gem's existing decision. No model change. Default to
  "allow pick" when movement data is sparse → graceful degradation. Easy
  A/B with a single config flag. **Chosen.**

---

## 2. Feature definitions

All features are computed at pick-generation time from rows in the `odds`
table for one `match_id`, market `1x2`. Inputs are pandas-friendly tuples:

For each bookmaker `b` and outcome `s ∈ {home, draw, away}`:
- `current(b, s)` = latest `value` (max `recorded_at`)
- `opening(b, s)` = first `opening_value` seen, fallback to earliest `value`
- `value_30m_ago(b, s)` = latest value where `recorded_at ≤ now − 30 min`

Aggregations (median across bookmakers — robust to outliers):

| Feature | Formula | Meaning |
| --- | --- | --- |
| `drift_<side>` | `median_b (current − opening) / opening` | total fractional shift of side's odds since first snapshot. Positive = drifted out (market less confident in `side`). |
| `velocity_<side>` | `median_b (current − value_30m_ago) / opening` | recent (≤ 30 min) shift, normalized to opening. Last-minute sharp money. |
| `dispersion_<side>` | `std_b(current) / mean_b(current)` | how much bookmakers disagree right now. High = uncertain market. |
| `n_snapshots` | `max_b count(rows)` | how many snapshots we have (data quality / freshness). |
| `n_bookmakers` | `unique bookmakers in latest snapshot` | breadth of market. |

All features are returned as `float | None`. `None` ⇒ insufficient data for
that feature (graceful: filter does not skip).

### Pick-relative composite signal

Given a Gem pick with `pick_side ∈ {home, away}` (we never bet draws), we
compute the composite **drift_against_us**:

> Odds drifting **up** for our side (positive `drift_pick`) means the
> market lowered its implied probability for that side — i.e. the market
> moved **away** from us. So:

```
drift_against_us = drift_<pick_side>            # already pick-relative
velocity_against_us = velocity_<pick_side>
dispersion_pick = dispersion_<pick_side>
```

A negative `drift_against_us` (odds shortened) is *toward* us = good. We
only filter on positive (against-us) drift.

---

## 3. Hypothesis-driven thresholds

Conservative defaults, configurable via `niches.py` constants.

| Constant | Default | Rationale |
| --- | --- | --- |
| `MOVEMENT_DRIFT_THRESHOLD` | 0.05 | Industry rule of thumb: > 5 % drift on a single side over hours = sharp action. Below 5 % is noise. |
| `MOVEMENT_VELOCITY_THRESHOLD` | 0.03 | A 3 % shift in the last 30 min is steep — kickoff is close, sharps moving size. |
| `MOVEMENT_DISPERSION_THRESHOLD` | 0.10 | std/mean > 10 % across books = books disagree, market unstable, edge unreliable. |
| `MOVEMENT_MIN_SNAPSHOTS` | 3 | Need ≥ 3 snapshots to compute drift meaningfully. Otherwise: skip filter (allow pick). |
| `MOVEMENT_MIN_BOOKMAKERS` | 2 | Need ≥ 2 books to compute dispersion. Otherwise: dispersion = None, only drift/velocity active. |
| `ENABLE_MOVEMENT_FILTER` | True | Master kill-switch for A/B. |

Decision logic in plain English (executed AFTER Gem picks a side):

1. If `ENABLE_MOVEMENT_FILTER = False` → allow.
2. If `n_snapshots < MOVEMENT_MIN_SNAPSHOTS` → allow (sparse data).
3. If `drift_against_us > MOVEMENT_DRIFT_THRESHOLD` → **skip**.
4. If `velocity_against_us > MOVEMENT_VELOCITY_THRESHOLD` → **skip**.
5. If `dispersion_pick is not None and dispersion_pick > MOVEMENT_DISPERSION_THRESHOLD`
   → **skip**.
6. Else → allow.

All skips are logged with the offending metric for later analysis.

---

## 4. Validation plan

### Retroactive (limited)

Most historical Gem picks have a single odds row → no drift, filter is a
no-op. We can still query the small subset where ≥ 3 snapshots happened
(matches where the tracker was active ≥ 1.5 h before pick time). For
these, we can:

1. Recompute `drift_against_us` for each settled Gem pick.
2. Bucket picks into [drift < 0], [0 < drift < 5%], [drift ≥ 5%] and
   compare WR/ROI per bucket. Hypothesis: WR drops monotonically with
   drift_against_us; ROI in the ≥ 5% bucket is materially negative.
3. If sample is too small (< 30 picks per bucket), defer validation to
   live A/B.

Query draft (for a future analysis script):

```sql
SELECT pred.id, pred.outcome,
       MIN(o.recorded_at) AS first_snap,
       MAX(o.recorded_at) AS last_snap,
       COUNT(DISTINCT o.recorded_at) AS n_snaps
FROM predictions pred
JOIN odds o
  ON o.match_id = pred.match_id AND o.market = '1x2' AND o.outcome = pred.outcome
WHERE pred.model_version = 'gem_v1'
GROUP BY pred.id, pred.outcome
HAVING COUNT(DISTINCT o.recorded_at) >= 3;
```

### Forward A/B

The cleanest test: run with `ENABLE_MOVEMENT_FILTER = False` for two
weeks, store all "would have skipped" tags, then flip True for two weeks.
Compare WR and ROI on the filtered-out subset to baseline. Need ~50 picks
per arm for any meaningful signal (variance is huge in small samples).

---

## 5. Risks for review

1. **False positives.** Lines move for many non-sharp reasons (injury
   news public, lineup leaks, public-money tailwind on the *other* side).
   Threshold of 5 % is intentionally conservative; lowering it bleeds
   picks fast.
2. **Survivorship bias on validation.** Picks that the filter would block
   are *also* the ones with the strongest market disagreement — they
   could be either Gem-edge (we beat the market) or Gem-error (market
   sees something Gem missed). Without labels we can't tell, hence A/B.
3. **Bookmaker mix drift.** SStats may add/remove books over time; raw
   `dispersion` could become noisier. Median across books mitigates
   this; we also filter on `MOVEMENT_MIN_BOOKMAKERS`.
4. **Sparse data on weekday matches.** If the tracker missed a window
   (scheduler downtime), we degrade to "allow". This means filter
   strength varies day-to-day. Acceptable for v1.
5. **Conflicting with gem_score.** Gem already filters on `gem_score`
   (our_p − market_p) using a *single snapshot* of market_p. The
   movement filter operates on a different time axis (drift over time)
   so conceptually they're orthogonal, but in practice they may
   correlate: high drift against us also widens the gem_score, so the
   filter could be partly redundant. Worth measuring overlap.

---

## 6. Out of scope (explicit non-goals for v1)

- No model retraining. No new features fed into the ensemble.
- No double-chance / totals movement (only 1x2 is tracked today).
- No closing-line value computation here — `update_clv.py` already does
  that post-match.
- No per-league threshold tuning; v1 uses globals.
