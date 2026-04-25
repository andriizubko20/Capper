# Pure / Gem Variable Inventory

Total variables: **69** across 16 categories.

## MARKET  (8 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `home_odds` | continuous | low_means_home_favored | yes | 1.67…2.2…3.1 | Decimal 1x2 odds for HOME. Lower = market sees home as favorite. CAVEAT: 94% rows are post-match closing (leakage for training). |
| `draw_odds` | continuous | n/a | no (used for de-vig) | — | Decimal odds for draw. Used to remove bookmaker margin via 1/o normalization. |
| `away_odds` | continuous | low_means_away_favored | yes | 2.25…3.25…5.0 | Decimal 1x2 odds for AWAY. |
| `home_market_prob` | continuous | high_means_home_favored | yes (max_market_prob filter) | 0.306…0.426…0.565 | Implied probability HOME wins, removed bookmaker margin. Used to find 'undervalued' (cap max → only low-implied = mispriced fav). |
| `draw_market_prob` | continuous | n/a | no | — | Implied draw probability. |
| `away_market_prob` | continuous | high_means_away_favored | yes | 0.189…0.288…0.414 | Implied probability AWAY wins. |
| `odds_ratio` | continuous | high_means_home_favored | no (derivable) | — | How much shorter the home odds are vs away. Quick intuition for 'how clear is the favorite'. |
| `vig` | continuous | n/a | no (derivable) | — | Bookmaker margin. Typically 4-7%. Higher = thinner edge. |

## STRENGTH  (6 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `home_glicko` | continuous | high_means_home_strong | yes (via glicko_gap) | — | SStats Glicko-2 rating for home team. Default ~1500. Higher = stronger. |
| `away_glicko` | continuous | high_means_away_strong | yes (via glicko_gap) | — | Glicko rating for away team. |
| `glicko_gap` | continuous | positive_for_home | YES (key) | -102.772…-1.037…100.993 | STRONGEST single discriminative feature in calibration. Top decile WR 78% if betting home. |
| `home_glicko_prob` | continuous | high_means_home_strong | YES (key) | 0.29…0.405…0.542 | Pre-match Glicko model's win probability for home. Best single calibrated factor (top decile 79% WR). |
| `away_glicko_prob` | continuous | high_means_away_strong | YES | 0.215…0.325…0.445 | Glicko prob for away. |
| `glicko_prob_diff` | continuous | positive_for_home | no (derivable) | — | Our Glicko model says home is N% likely; market says M%. Diff = the 'gem' edge. |

## XG_ATTACK  (2 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `home_xg_for_10` | continuous | high_means_home_strong | yes (via xg_diff_home) | — | Last 10 matches xG scored. Range 0.5-3.0. |
| `away_xg_for_10` | continuous | high_means_away_strong | yes | — | Same as home_xg_for_10 but for away. |

## XG_DEFENSE  (2 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `home_xg_against_10` | continuous | low_means_home_strong | yes (via xg_diff_home) | — | Defense quality: lower better. |
| `away_xg_against_10` | continuous | low_means_away_strong | yes | — | Away team's defensive quality. |

## XG_NET  (2 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `xg_diff_home` | continuous | positive_for_home | YES | -0.439…-0.17…0.288 | Home team's net xG (attack minus defense). Positive = good team. |
| `xg_diff_away` | continuous | positive_for_away | YES | -0.416…-0.151…0.317 | Away team's net xG. |

## XG_MATCHUP  (3 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `attack_vs_def_home` | continuous | positive_for_home | YES | -0.339…-0.087…0.212 | Home attack vs away defense matchup. Positive = home outscores expected. |
| `attack_vs_def_away` | continuous | positive_for_away | YES | -0.341…-0.088…0.216 | Away attack vs home defense. |
| `xg_quality_gap` | continuous | positive_for_home | YES (Gem v2) | — | Net matchup advantage in xG terms. Captures both attack and defense imbalances. |

## XG_HOME_SPLIT  (2 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `xg_for_home_10` | continuous | high_means_home_strong | no (Gem only) | — | Home team's xG when playing AT HOME (different from overall). Captures home advantage in xG. |
| `xg_against_home_10` | continuous | low_means_home_strong | no | — | Home team's defensive xG at home. |

## XG_AWAY_SPLIT  (2 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `xg_for_away_10` | continuous | high_means_away_strong | no | — | Away team's xG when playing on the road. |
| `xg_against_away_10` | continuous | low_means_away_strong | no | — | Away team's defensive xG on the road. |

## FORM  (6 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `home_ppg_10` | continuous | high_means_home_strong | yes (via form_advantage) | — | Home team's points-per-game over last 10. Range 0-3. |
| `away_ppg_10` | continuous | high_means_away_strong | yes | — | Away team's PPG. |
| `form_advantage` | continuous | positive_for_home | YES (forensic key) | -0.6…0.0…0.5 | 🔥 STRONGEST signal in user-curated forensic analysis (Cohen's d=+0.70). |
| `home_form_5` | continuous | high_means_home_strong | no (Gem) | — | Shorter window than ppg_10 — captures recent shape change. |
| `away_form_5` | continuous | high_means_away_strong | no | — | Same for away. |
| `form_advantage_5` | continuous | positive_for_home | no (derivable) | — | Short-term form gap (last 5 vs last 5). Possibly more predictive than 10-match. |

## FORM_SPLIT  (2 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `ppg_home_10` | continuous | high_means_home_strong | no | — | Home team's PPG at home — captures home advantage strength. |
| `ppg_away_10` | continuous | high_means_away_strong | no | — | Away team's PPG on the road. |

## MOMENTUM  (10 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `home_xg_trend` | continuous | positive_for_home | no | — | Positive = home team's attack is accelerating recently. |
| `away_xg_trend` | continuous | positive_for_away | no | — | Same for away. |
| `xg_trend_advantage` | continuous | positive_for_home | no (derivable) | — | Net momentum diff. |
| `home_glicko_momentum` | continuous | positive_for_home | no | — | Glicko rating drift over last 5 matches. Proxy for squad health/rotation effect. |
| `away_glicko_momentum` | continuous | positive_for_away | no | — |  |
| `glicko_momentum_diff` | continuous | positive_for_home | no (derivable) | — | Composite momentum gap. |
| `home_win_streak` | ordinal | high_means_home_strong | no | — | 0 if not on win streak. |
| `home_lose_streak` | ordinal | low_means_home_strong | no | — | 0 if not on lose streak. |
| `away_win_streak` | ordinal | high_means_away_strong | no | — |  |
| `away_lose_streak` | ordinal | low_means_away_strong | no | — | Could be 'opponent weakness' signal: hot favorite vs cold opponent. |

## STYLE  (6 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `home_possession_10` | continuous | ambiguous | no | — | 0-100. Higher = more possession-based; not necessarily better. |
| `away_possession_10` | continuous | ambiguous | no | — |  |
| `home_sot_10` | continuous | high_means_home_strong | no | — | Higher = more attacking output. |
| `away_sot_10` | continuous | high_means_away_strong | no | — |  |
| `home_pass_acc_10` | continuous | high_means_home_strong | no | — | Quality of build-up. Top-tier teams ~85-90%. |
| `away_pass_acc_10` | continuous | high_means_away_strong | no | — |  |

## PHYSICAL  (5 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `home_rest_days` | ordinal | high_means_home_strong | no | — | Days since last match. 7+ = good rest, <4 = midweek-tired. |
| `away_rest_days` | ordinal | high_means_away_strong | no | — |  |
| `rest_advantage` | continuous | positive_for_home | no (derivable) | — | Travel/fatigue advantage. Positive = home is fresher. |
| `home_has_injuries` | binary | true_means_home_weak | no | — | Crude flag — doesn't weight player importance. |
| `away_has_injuries` | binary | true_means_away_weak | no | — |  |

## H2H  (3 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `h2h_home_wr` | continuous | positive_for_home | no | — | Range 0-1. Niches with strong dynastic dominance. |
| `h2h_avg_goals` | continuous | ambiguous | no | — | Style indicator; doesn't directly predict winner. |
| `h2h_home_last_result` | ordinal | positive_for_home | no | — | Recency: was last meeting a win? Some weight. |

## CONTEXT  (6 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `league_name` | categorical | n/a | YES (one-hot + per-league models) | — | Per-league niche tuning is essential — different leagues have different home advantage and form impacts. |
| `league_cluster_top5` | binary | n/a | no (Gem only) | — | Top-tier vs second-tier. |
| `league_home_wr` | continuous | high_means_home_advantage | no | — | Baseline. Range ~0.40-0.48. La Liga ~0.46, Serie A ~0.40. |
| `league_prior_home_wr` | continuous | high_means_home_advantage | Gem only | — | Computed per fold to avoid leakage. |
| `league_prior_draw_rate` | continuous | n/a | Gem only | — | League-specific draw rate (varies 22-28%). |
| `league_prior_away_wr` | continuous | high_means_away_strong | Gem only | — |  |

## META  (4 variables)

| name | type | direction | in_pure | range (q25–med–q75) | notes |
|---|---|---|---|---|---|
| `matches_played_home` | ordinal | n/a | yes (min_matches_played) | 19.0…40.0…70.0 | Filter early-season noise. Typical: require >=5. |
| `matches_played_away` | ordinal | n/a | yes | 19.0…40.5…70.0 |  |
| `date` | ordinal | n/a | yes (chronological splits) | — | Used for walk-forward CV and stability checks. |
| `last_match_date_home` | ordinal | n/a | no (derivable for rest) | — | Used to compute rest_days. |

