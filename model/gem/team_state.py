"""
model/gem/team_state.py

Pre-computes rolling state per (team, match) — strictly using matches BEFORE
the current one. No leakage.

Output: dict keyed by (match_id, side) → state dict with rolling stats.
"""
from collections import defaultdict, deque
import pandas as pd
from loguru import logger

from model.gem.niches import ROLLING_10, ROLLING_5


def _result_points(team_score: int, opp_score: int) -> int:
    if team_score > opp_score:
        return 3
    if team_score == opp_score:
        return 1
    return 0


def _empty_state() -> dict:
    # Defaults returned when team has no history yet.
    return {
        "xg_for_10": None, "xg_against_10": None,
        "xg_for_home_10": None, "xg_against_home_10": None,
        "xg_for_away_10": None, "xg_against_away_10": None,
        "ppg_10": None, "ppg_home_10": None, "ppg_away_10": None,
        "form_5": None,
        "xg_trend": None,
        "win_streak": 0, "lose_streak": 0,
        "glicko_now": None, "glicko_momentum": None,
        "possession_10": None, "sot_10": None, "pass_acc_10": None,
        "last_match_date": None,
        "matches_played": 0,
    }


def _safe_avg(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def build_team_state(
    matches: pd.DataFrame,
    stats: pd.DataFrame,
) -> dict[tuple[int, str], dict]:
    """
    Iterates matches in chronological order, maintains a per-team history buffer,
    and snapshots the PRE-match state into output[(match_id, side)].

    side ∈ {'home', 'away'}.
    """
    df = matches.merge(stats, on="match_id", how="left").sort_values("date").reset_index(drop=True)

    # Per-team history: deques sized for max rolling window we need
    history: dict[int, dict] = defaultdict(lambda: {
        "all": deque(maxlen=20),       # stores dicts of per-match entries
        "home_only": deque(maxlen=20),
        "away_only": deque(maxlen=20),
        "results_chrono": deque(maxlen=20),  # 'W' / 'D' / 'L'
        "glicko_history": deque(maxlen=20),  # glicko rating at match time
        "last_match_date": None,
        "count": 0,
    })

    out: dict[tuple[int, str], dict] = {}

    for row in df.itertuples(index=False):
        for side, team_id, opp_side in (
            ("home", row.home_team_id, "away"),
            ("away", row.away_team_id, "home"),
        ):
            h = history[team_id]
            out[(row.match_id, side)] = _snapshot(h, side_glicko_now=_get_side_glicko(row, side))

        # After snapshotting pre-match state for BOTH teams, update histories with this match's result
        _update_history(history, row)

    logger.info(f"Computed team state for {len(df)} matches → {len(out)} (match, side) snapshots")
    return out


def _get_side_glicko(row, side: str):
    return row.home_glicko if side == "home" else row.away_glicko


def _snapshot(h: dict, side_glicko_now) -> dict:
    """Snapshot the team's PRE-match state from history buffers."""
    s = _empty_state()
    s["matches_played"] = h["count"]
    s["last_match_date"] = h["last_match_date"]
    s["glicko_now"] = side_glicko_now

    if h["count"] == 0:
        return s

    all_recent = list(h["all"])[-ROLLING_10:]
    home_recent = list(h["home_only"])[-ROLLING_10:]
    away_recent = list(h["away_only"])[-ROLLING_10:]
    last5 = all_recent[-ROLLING_5:]

    # xG rolling
    s["xg_for_10"]       = _safe_avg([m["xg_for"]     for m in all_recent])
    s["xg_against_10"]   = _safe_avg([m["xg_against"] for m in all_recent])
    s["xg_for_home_10"]  = _safe_avg([m["xg_for"]     for m in home_recent])
    s["xg_against_home_10"] = _safe_avg([m["xg_against"] for m in home_recent])
    s["xg_for_away_10"]  = _safe_avg([m["xg_for"]     for m in away_recent])
    s["xg_against_away_10"] = _safe_avg([m["xg_against"] for m in away_recent])

    # PPG rolling
    s["ppg_10"]      = _safe_avg([m["pts"] for m in all_recent])
    s["ppg_home_10"] = _safe_avg([m["pts"] for m in home_recent])
    s["ppg_away_10"] = _safe_avg([m["pts"] for m in away_recent])
    s["form_5"]      = _safe_avg([m["pts"] for m in last5])

    # xG trend: last-5 avg minus last-10 avg (positive = accelerating)
    if s["xg_for_10"] is not None and last5:
        xg_for_5 = _safe_avg([m["xg_for"] for m in last5])
        if xg_for_5 is not None:
            s["xg_trend"] = xg_for_5 - s["xg_for_10"]

    # Streaks
    res_chrono = list(h["results_chrono"])
    w_streak = l_streak = 0
    for r in reversed(res_chrono):
        if r == "W" and l_streak == 0:
            w_streak += 1
            continue
        if r == "L" and w_streak == 0:
            l_streak += 1
            continue
        break
    s["win_streak"], s["lose_streak"] = w_streak, l_streak

    # Glicko momentum: current − value 5 matches ago (using pre-match rating stored at time)
    glicko_hist = list(h["glicko_history"])
    if len(glicko_hist) >= ROLLING_5 and side_glicko_now is not None:
        prior = glicko_hist[-ROLLING_5]
        if prior is not None:
            s["glicko_momentum"] = side_glicko_now - prior

    # Style rolling
    s["possession_10"] = _safe_avg([m["possession"]  for m in all_recent])
    s["sot_10"]        = _safe_avg([m["sot"]         for m in all_recent])
    s["pass_acc_10"]   = _safe_avg([m["pass_acc"]    for m in all_recent])

    return s


def _update_history(history: dict, row) -> None:
    """After processing a match, push its result into both teams' histories."""
    home_xg, away_xg = row.home_xg, row.away_xg
    home_poss, away_poss = row.home_possession, row.away_possession
    home_sot, away_sot = row.home_sot, row.away_sot

    # Pass accuracy
    home_acc = None
    if row.home_pass_total and row.home_pass_acc and row.home_pass_total > 0:
        home_acc = row.home_pass_acc / row.home_pass_total
    away_acc = None
    if row.away_pass_total and row.away_pass_acc and row.away_pass_total > 0:
        away_acc = row.away_pass_acc / row.away_pass_total

    home_pts = _result_points(row.home_score, row.away_score)
    away_pts = _result_points(row.away_score, row.home_score)
    home_res = "W" if home_pts == 3 else ("D" if home_pts == 1 else "L")
    away_res = "W" if away_pts == 3 else ("D" if away_pts == 1 else "L")

    home_entry = {
        "xg_for": home_xg, "xg_against": away_xg,
        "pts": home_pts, "result": home_res,
        "possession": home_poss, "sot": home_sot, "pass_acc": home_acc,
        "date": row.date,
    }
    away_entry = {
        "xg_for": away_xg, "xg_against": home_xg,
        "pts": away_pts, "result": away_res,
        "possession": away_poss, "sot": away_sot, "pass_acc": away_acc,
        "date": row.date,
    }

    h_home = history[row.home_team_id]
    h_home["all"].append(home_entry)
    h_home["home_only"].append(home_entry)
    h_home["results_chrono"].append(home_res)
    h_home["glicko_history"].append(row.home_glicko)
    h_home["last_match_date"] = row.date
    h_home["count"] += 1

    h_away = history[row.away_team_id]
    h_away["all"].append(away_entry)
    h_away["away_only"].append(away_entry)
    h_away["results_chrono"].append(away_res)
    h_away["glicko_history"].append(row.away_glicko)
    h_away["last_match_date"] = row.date
    h_away["count"] += 1


def build_h2h(matches: pd.DataFrame) -> dict[tuple[int, int], dict]:
    """
    Computes head-to-head history per (home_team_id, away_team_id) ordered pair,
    snapshotting PRE-match H2H stats.

    Returns {match_id: {'h2h_home_wr', 'h2h_avg_goals', 'h2h_home_last_result'}}.
    """
    df = matches.sort_values("date").reset_index(drop=True)
    history: dict[frozenset, list] = defaultdict(list)
    out: dict[int, dict] = {}

    for row in df.itertuples(index=False):
        key = frozenset({row.home_team_id, row.away_team_id})
        prior = history[key][-5:]

        if not prior:
            out[row.match_id] = {
                "h2h_home_wr": None,
                "h2h_avg_goals": None,
                "h2h_home_last_result": None,
            }
        else:
            home_wins = sum(
                1 for m in prior
                if (m["home_id"] == row.home_team_id and m["home_score"] > m["away_score"])
                or (m["away_id"] == row.home_team_id and m["away_score"] > m["home_score"])
            )
            wr = home_wins / len(prior)
            goals = sum(m["home_score"] + m["away_score"] for m in prior) / len(prior)

            last = prior[-1]
            current_home_perspective = (
                "W" if (last["home_id"] == row.home_team_id and last["home_score"] > last["away_score"])
                       or (last["away_id"] == row.home_team_id and last["away_score"] > last["home_score"])
                else ("L" if (last["home_id"] == row.home_team_id and last["home_score"] < last["away_score"])
                             or (last["away_id"] == row.home_team_id and last["away_score"] < last["home_score"])
                      else "D")
            )
            last_res_numeric = {"W": 1, "D": 0, "L": -1}[current_home_perspective]

            out[row.match_id] = {
                "h2h_home_wr": wr,
                "h2h_avg_goals": goals,
                "h2h_home_last_result": last_res_numeric,
            }

        history[key].append({
            "home_id": row.home_team_id, "away_id": row.away_team_id,
            "home_score": row.home_score, "away_score": row.away_score,
        })

    return out
