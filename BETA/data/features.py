"""
BETA/data/features.py

Builds a feature matrix for all finished matches.
STRICT no-leakage: every row uses only data from BEFORE that match's date.

Feature groups:
  - Elo (dynamic, K=32)
  - Glicko (from match_stats, pre-match snapshot)
  - Rolling form: last 5 / last 10 games (overall + home/away split)
  - Rolling xG, shots, possession, corners
  - Head-to-head last 3
  - Market implied probabilities
  - Rest days
  - Injury counts
  - Table position (cumulative points)
"""
import numpy as np
import pandas as pd
from collections import defaultdict


# ── Elo ───────────────────────────────────────────────────────────────────────

ELO_DEFAULT = 1500.0
ELO_K       = 32


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def _build_elo_series(matches: pd.DataFrame) -> dict[int, list[tuple]]:
    """Returns {team_id: [(match_id, elo_before_match), ...]} sorted by date."""
    elos: dict[int, float] = {}
    history: dict[int, list] = defaultdict(list)

    for _, row in matches.sort_values('date').iterrows():
        h, a = int(row.home_team_id), int(row.away_team_id)
        elo_h = elos.get(h, ELO_DEFAULT)
        elo_a = elos.get(a, ELO_DEFAULT)

        history[h].append((int(row.match_id), elo_h))
        history[a].append((int(row.match_id), elo_a))

        exp_h = _elo_expected(elo_h, elo_a)
        if row.home_score > row.away_score:
            s_h, s_a = 1.0, 0.0
        elif row.home_score == row.away_score:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        elos[h] = elo_h + ELO_K * (s_h - exp_h)
        elos[a] = elo_a + ELO_K * (s_a - (1 - exp_h))

    return history


# ── Rolling team stats ────────────────────────────────────────────────────────

def _rolling_team_features(matches: pd.DataFrame, stats: pd.DataFrame,
                            windows: tuple = (5, 10)) -> pd.DataFrame:
    """
    For each match builds pre-match rolling stats for home and away team.
    Returns DataFrame indexed by match_id.
    """
    # Merge stats onto matches
    s = stats[['match_id','home_xg','away_xg',
                'home_shots_on_target','away_shots_on_target',
                'home_possession','away_possession',
                'home_corners','away_corners',
                'home_gk_saves','away_gk_saves']].copy()

    m = matches[['match_id','date','home_team_id','away_team_id',
                  'home_score','away_score','result','league_id']].copy()
    m = m.merge(s, on='match_id', how='left')

    # Build per-team event log (one row per team per match)
    home_log = m.rename(columns={
        'home_team_id': 'team_id', 'away_team_id': 'opp_id',
        'home_score': 'gf', 'away_score': 'ga',
        'home_xg': 'xgf', 'away_xg': 'xga',
        'home_shots_on_target': 'sot_f', 'away_shots_on_target': 'sot_a',
        'home_possession': 'poss', 'home_corners': 'corners',
        'home_gk_saves': 'saves',
    }).assign(is_home=1)
    home_log['pts'] = home_log['result'].map({'H': 3, 'D': 1, 'A': 0})
    home_log['win']  = (home_log['result'] == 'H').astype(int)
    home_log['draw'] = (home_log['result'] == 'D').astype(int)
    home_log['loss'] = (home_log['result'] == 'A').astype(int)

    away_log = m.rename(columns={
        'away_team_id': 'team_id', 'home_team_id': 'opp_id',
        'away_score': 'gf', 'home_score': 'ga',
        'away_xg': 'xgf', 'home_xg': 'xga',
        'away_shots_on_target': 'sot_f', 'home_shots_on_target': 'sot_a',
        'away_possession': 'poss', 'away_corners': 'corners',
        'away_gk_saves': 'saves',
    }).assign(is_home=0)
    away_log['pts'] = away_log['result'].map({'A': 3, 'D': 1, 'H': 0})
    away_log['win']  = (away_log['result'] == 'A').astype(int)
    away_log['draw'] = (away_log['result'] == 'D').astype(int)
    away_log['loss'] = (away_log['result'] == 'H').astype(int)

    cols_keep = ['match_id','date','team_id','opp_id','league_id','is_home',
                 'gf','ga','xgf','xga','sot_f','sot_a','poss','corners','saves',
                 'pts','win','draw','loss']
    log = pd.concat([home_log[cols_keep], away_log[cols_keep]], ignore_index=True)
    log = log.sort_values('date').reset_index(drop=True)

    stat_cols = ['gf','ga','xgf','xga','sot_f','sot_a','poss','corners','saves','pts','win','loss']

    # Build rolling lookup: {team_id: sorted list of (date, match_id, stats...)}
    team_history: dict[int, list] = defaultdict(list)
    for _, row in log.iterrows():
        team_history[int(row.team_id)].append(row)

    # For each match compute rolling features for home/away team
    records = []
    elo_history = _build_elo_series(matches)
    elo_lookup  = {mid: elo for team_rows in elo_history.values()
                   for mid, elo in team_rows}

    for _, match in m.sort_values('date').iterrows():
        mid  = int(match.match_id)
        h_id = int(match.home_team_id)
        a_id = int(match.away_team_id)
        dt   = match.date

        feat = {'match_id': mid}

        # Elo pre-match (from history built above)
        feat['elo_home'] = elo_lookup.get(mid, ELO_DEFAULT)  # home team elo before this match
        # For away we need to look up from elo_history keyed by team
        elo_h_val = ELO_DEFAULT
        elo_a_val = ELO_DEFAULT
        if h_id in elo_history:
            for m_id, elo_val in elo_history[h_id]:
                if m_id == mid:
                    elo_h_val = elo_val
                    break
        if a_id in elo_history:
            for m_id, elo_val in elo_history[a_id]:
                if m_id == mid:
                    elo_a_val = elo_val
                    break
        feat['elo_home'] = elo_h_val
        feat['elo_away'] = elo_a_val
        feat['elo_diff'] = elo_h_val - elo_a_val
        feat['elo_home_win_prob'] = _elo_expected(elo_h_val, elo_a_val)

        # Rolling stats per team
        for side, team_id in [('home', h_id), ('away', a_id)]:
            past = [r for r in team_history.get(team_id, []) if r['date'] < dt]
            past_home = [r for r in past if r['is_home'] == 1]
            past_away = [r for r in past if r['is_home'] == 0]

            for w in windows:
                last_w    = past[-w:]      if len(past) >= 1    else past
                last_w_h  = past_home[-w:] if len(past_home) >= 1 else past_home
                last_w_a  = past_away[-w:] if len(past_away) >= 1 else past_away

                for suffix, subset in [(f'_{w}', last_w),
                                        (f'_h{w}', last_w_h),
                                        (f'_a{w}', last_w_a)]:
                    n = len(subset)
                    if n == 0:
                        for c in stat_cols:
                            feat[f'{side}_{c}{suffix}'] = np.nan
                    else:
                        for c in stat_cols:
                            vals = [r[c] for r in subset if not pd.isna(r[c])]
                            feat[f'{side}_{c}{suffix}'] = np.mean(vals) if vals else np.nan

                feat[f'{side}_n_games{suffix[-2:] if "_" in suffix else suffix}'] = n

        # xG ratio (safe division)
        for w in windows:
            hxf = feat.get(f'home_xgf_{w}', np.nan)
            hxa = feat.get(f'home_xga_{w}', np.nan)
            axf = feat.get(f'away_xgf_{w}', np.nan)
            axa = feat.get(f'away_xga_{w}', np.nan)
            feat[f'xg_ratio_home_{w}'] = hxf / max(hxa, 0.1) if not np.isnan(hxf or 0) else np.nan
            feat[f'xg_ratio_away_{w}'] = axf / max(axa, 0.1) if not np.isnan(axf or 0) else np.nan
            feat[f'xg_diff_home_{w}']  = (hxf - hxa) if (hxf is not None and not np.isnan(hxf)) else np.nan
            feat[f'xg_diff_away_{w}']  = (axf - axa) if (axf is not None and not np.isnan(axf)) else np.nan

        records.append(feat)

    return pd.DataFrame(records)


# ── Head-to-head ──────────────────────────────────────────────────────────────

def _h2h_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Last 3 H2H meetings before each match."""
    m = matches.sort_values('date').reset_index(drop=True)
    records = []

    for _, row in m.iterrows():
        h, a, dt = int(row.home_team_id), int(row.away_team_id), row.date
        past = m[
            (m['date'] < dt) &
            (((m['home_team_id'] == h) & (m['away_team_id'] == a)) |
             ((m['home_team_id'] == a) & (m['away_team_id'] == h)))
        ].tail(3)

        n = len(past)
        h_wins = h_draws = h_losses = 0
        for _, p in past.iterrows():
            if p.home_team_id == h:
                if p.result == 'H': h_wins += 1
                elif p.result == 'D': h_draws += 1
                else: h_losses += 1
            else:
                if p.result == 'A': h_wins += 1
                elif p.result == 'D': h_draws += 1
                else: h_losses += 1

        records.append({
            'match_id':    int(row.match_id),
            'h2h_n':       n,
            'h2h_home_wins':   h_wins   / max(n, 1),
            'h2h_home_draws':  h_draws  / max(n, 1),
            'h2h_home_losses': h_losses / max(n, 1),
        })

    return pd.DataFrame(records)


# ── Table position ────────────────────────────────────────────────────────────

def _table_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Cumulative table position + points per team per league at match date."""
    m = matches.sort_values('date').copy()
    points: dict[tuple, int] = defaultdict(int)   # (league_id, team_id) → pts
    records = []

    for _, row in m.iterrows():
        lid = int(row.league_id)
        h, a = int(row.home_team_id), int(row.away_team_id)

        # Snapshot before this match
        league_pts = {tid: pts for (l, tid), pts in points.items() if l == lid}
        sorted_teams = sorted(league_pts.items(), key=lambda x: -x[1])
        pos_map = {tid: i+1 for i, (tid, _) in enumerate(sorted_teams)}

        h_pts = league_pts.get(h, 0)
        a_pts = league_pts.get(a, 0)
        h_pos = pos_map.get(h, len(pos_map) + 1)
        a_pos = pos_map.get(a, len(pos_map) + 1)

        records.append({
            'match_id':        int(row.match_id),
            'table_home_pts':  h_pts,
            'table_away_pts':  a_pts,
            'table_pts_diff':  h_pts - a_pts,
            'table_home_pos':  h_pos,
            'table_away_pos':  a_pos,
            'table_pos_diff':  h_pos - a_pos,   # negative = home higher
        })

        # Update after match
        if row.result == 'H':
            points[(lid, h)] += 3
        elif row.result == 'D':
            points[(lid, h)] += 1
            points[(lid, a)] += 1
        else:
            points[(lid, a)] += 3

    return pd.DataFrame(records)


# ── Rest days ─────────────────────────────────────────────────────────────────

def _rest_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Days since last match per team."""
    m = matches.sort_values('date').copy()
    last_match: dict[int, pd.Timestamp] = {}
    records = []

    for _, row in m.iterrows():
        h, a, dt = int(row.home_team_id), int(row.away_team_id), row.date
        h_rest = (dt - last_match[h]).days if h in last_match else 14
        a_rest = (dt - last_match[a]).days if a in last_match else 14
        records.append({
            'match_id':     int(row.match_id),
            'rest_home':    min(h_rest, 21),
            'rest_away':    min(a_rest, 21),
            'rest_diff':    min(h_rest, 21) - min(a_rest, 21),
        })
        last_match[h] = dt
        last_match[a] = dt

    return pd.DataFrame(records)


# ── Injuries ──────────────────────────────────────────────────────────────────

def _injury_features(matches: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Injury count per team per match."""
    inj = injuries.groupby(['match_id','team_id'])['injury_count'].sum().reset_index()
    m = matches[['match_id','home_team_id','away_team_id']].copy()
    m = m.merge(inj.rename(columns={'team_id':'home_team_id','injury_count':'inj_home'}),
                on=['match_id','home_team_id'], how='left')
    m = m.merge(inj.rename(columns={'team_id':'away_team_id','injury_count':'inj_away'}),
                on=['match_id','away_team_id'], how='left')
    m['inj_home'] = m['inj_home'].fillna(0)
    m['inj_away'] = m['inj_away'].fillna(0)
    m['inj_diff'] = m['inj_home'] - m['inj_away']
    return m[['match_id','inj_home','inj_away','inj_diff']]


# ── Market implied prob ───────────────────────────────────────────────────────

def _market_features(odds: pd.DataFrame) -> pd.DataFrame:
    """Convert decimal odds to implied probabilities (normalized)."""
    o = odds.copy()
    o['raw_h'] = 1.0 / o['home_odds'].replace(0, np.nan)
    o['raw_d'] = 1.0 / o['draw_odds'].replace(0, np.nan)
    o['raw_a'] = 1.0 / o['away_odds'].replace(0, np.nan)
    o['margin'] = o[['raw_h','raw_d','raw_a']].sum(axis=1)
    o['mkt_home_prob'] = o['raw_h'] / o['margin']
    o['mkt_draw_prob'] = o['raw_d'] / o['margin']
    o['mkt_away_prob'] = o['raw_a'] / o['margin']
    o['home_odds_val'] = o['home_odds']
    o['draw_odds_val'] = o['draw_odds']
    o['away_odds_val'] = o['away_odds']
    return o[['match_id','home_odds_val','draw_odds_val','away_odds_val',
               'mkt_home_prob','mkt_draw_prob','mkt_away_prob']]


# ── Master builder ────────────────────────────────────────────────────────────

FEATURE_COLS = None   # set after first build


def build_feature_matrix(matches, stats, odds, injuries,
                          windows=(5, 10)) -> pd.DataFrame:
    """
    Builds the complete feature matrix.
    Returns DataFrame with match_id, date, label columns + all features.
    No leakage — all features are pre-match.
    """
    print("Building rolling team features...")
    rolling = _rolling_team_features(matches, stats, windows)

    print("Building H2H features...")
    h2h = _h2h_features(matches)

    print("Building table features...")
    table = _table_features(matches)

    print("Building rest features...")
    rest = _rest_features(matches)

    print("Building injury features...")
    inj = _injury_features(matches, injuries)

    print("Building market features...")
    mkt = _market_features(odds)

    # Merge everything
    base = matches[['match_id','date','league_id','league_name',
                    'home_team_id','away_team_id',
                    'home_score','away_score','result']].copy()

    for df in [rolling, h2h, table, rest, inj, mkt]:
        base = base.merge(df, on='match_id', how='left')

    # League one-hot
    league_dummies = pd.get_dummies(base['league_name'], prefix='lg', drop_first=False)
    base = pd.concat([base, league_dummies], axis=1)

    print(f"Feature matrix: {len(base)} rows × {len(base.columns)} columns")
    return base


LABEL_COL    = 'result'    # H / D / A
LABEL_BIN_H  = 'home_win'  # 1 if H else 0
LABEL_BIN_A  = 'away_win'  # 1 if A else 0


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Returns list of numeric feature columns (excludes meta + label cols)."""
    exclude = {
        'match_id','date','league_id','league_name',
        'home_team_id','away_team_id',
        'home_score','away_score','result',
    }
    return [c for c in df.columns
            if c not in exclude
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, bool]
            and not c.startswith('home_win') and not c.startswith('away_win')]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from BETA.data.extract import load_all

    matches, stats, odds, injuries = load_all()
    df = build_feature_matrix(matches, stats, odds, injuries)
    feat_cols = get_feature_cols(df)
    print(f"\nFeature columns ({len(feat_cols)}):")
    for c in feat_cols[:20]:
        print(f"  {c}")
    print("  ...")
    print(f"\nNull rate avg: {df[feat_cols].isnull().mean().mean():.1%}")
    print(df[feat_cols].describe().T[['mean','std','min','max']].head(10))
