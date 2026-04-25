"""
model/pure/selected_niches.py

User-curated production niche list. 41 niches across 8 leagues.
Picked by hand from the mass-search Excel.

Each entry parses the human-readable string into the dict schema used by
predictor.py / matcher.py.
"""
import re

# Grammar tokens (compact label → (column, comparison))
LABEL_MAP = {
    "g":   ("min_glicko_gap", ">="),
    "p":   ("min_glicko_prob", ">="),
    "xd":  ("min_xg_diff", ">="),
    "xq":  ("min_xg_quality_gap", ">="),
    "ad":  ("min_attack_vs_def", ">="),
    "f":   ("min_form_advantage", ">="),
    "ppg": ("min_ppg", ">="),
    "xt":  ("min_xg_trend", ">="),
    "gm":  ("min_glicko_momentum", ">="),
    "ws":  ("min_win_streak", ">="),
    "ols": ("min_opp_lose_streak", ">="),
    "pos": ("min_possession_10", ">="),
    "sot": ("min_sot_10", ">="),
    "pa":  ("min_pass_acc_10", ">="),
    "ra":  ("min_rest_advantage", ">="),
    "h2h": ("min_h2h_wr", ">="),
    "m":   ("max_market_prob", "<="),
}


# User's hand-picked list (v2: post-recency filter, 2026-04-25)
# Down to 20 niches selected after re-evaluation on 2025-08-01+ data.
RAW_NICHES = {
    "Bundesliga": [
        "home[1.7,2.1) ws>=1 pos>=50",
        "home[1.7,2.1) ra>=0 h2h>=0.5",
    ],
    "Eredivisie": [
        "home[1.4,1.6) gm>=0",
    ],
    "Jupiler Pro League": [
        "home[1.55,1.85) ppg>=1.5 ra>=0 m<=0.60",
    ],
    "La Liga": [
        "away[1.85,2.4) pa>=80 h2h>=0.5",
        "away[1.85,2.4) ppg>=1.5 pa>=80",
    ],
    "Ligue 1": [
        "home[1.55,1.85) ppg>=1.5 ra>=0 h2h>=0.5",
        "home[1.85,2.4) xt>=0 pos>=50",
        "away[1.7,2.1) ppg>=1.8 h2h>=0.5",
    ],
    "Serie A": [
        "home[1.7,2.1) xd>=0.2 ppg>=1.8",
        "away[1.85,2.4) f>=0.5 xt>=0",
        "away[1.7,2.1) ws>=1 ra>=0",
        "away[1.7,2.1) ppg>=1.8 ws>=1 pa>=80",
        "away[1.55,1.85) sot>=4",
        "away[1.55,1.85) f>=0.5 ra>=0 m<=0.60",
        "home[1.85,2.4) xd>=0.2 ppg>=1.8",
        "home[1.7,2.1) xd>=0.2 ppg>=1.8 ws>=1",
        "away[2.1,3.0) xt>=0 pos>=50 pa>=80",
    ],
}


_NICHE_RE = re.compile(
    r"^(home|away)\[([\d.]+),([\d.]+)\)(?:\s+(.+))?$"
)


def parse_niche_str(s: str) -> dict:
    """Parse 'home[1.4,1.6) xd>=0.2 h2h>=0.5' into structured dict."""
    s = s.strip()
    m = _NICHE_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse niche: {s!r}")

    side, lo, hi, rest = m.group(1), float(m.group(2)), float(m.group(3)), m.group(4) or ""
    out = {
        "side": side,
        "odds_range": (lo, hi),
        "min_glicko_gap": None,
        "min_glicko_prob": None,
        "min_xg_diff": None,
        "min_xg_quality_gap": None,
        "min_attack_vs_def": None,
        "min_form_advantage": None,
        "min_ppg": None,
        "min_xg_trend": None,
        "min_glicko_momentum": None,
        "min_win_streak": None,
        "min_opp_lose_streak": None,
        "min_possession_10": None,
        "min_sot_10": None,
        "min_pass_acc_10": None,
        "min_rest_advantage": None,
        "min_h2h_wr": None,
        "max_market_prob": None,
    }
    for token in rest.split():
        # Match label>=val or label<=val
        mt = re.match(r"([a-z][a-z0-9]*)([<>]=)([\d.]+)", token)
        if not mt:
            raise ValueError(f"Cannot parse token: {token!r} in {s!r}")
        label, op, val = mt.group(1), mt.group(2), float(mt.group(3))
        # Special: 'pa', 'pos' use 0-1 scale, but user wrote 'pa>=80' meaning 80%
        # Actually our match_factors columns use 0-1 for pass_acc and 0-1 for possession.
        # But our threshold grid uses pa>=0.80 (0-1). User wrote pa>=80 — keep as 80, will normalise below.
        if label not in LABEL_MAP:
            raise ValueError(f"Unknown label: {label!r}")
        col, _expected_op = LABEL_MAP[label]
        if op != _expected_op:
            raise ValueError(f"Op mismatch for {label}: got {op}, expected {_expected_op}")
        # Normalise pa, pos: user uses 0-100, our DB uses 0-1
        if label in ("pa", "pos") and val > 1.5:
            val = val / 100.0
        out[col] = val
    out["niche_id"] = s
    return out


def parse_all() -> dict[str, list[dict]]:
    parsed: dict[str, list[dict]] = {}
    for league, niches in RAW_NICHES.items():
        parsed[league] = [parse_niche_str(s) for s in niches]
    return parsed


if __name__ == "__main__":
    out = parse_all()
    total = sum(len(v) for v in out.values())
    print(f"Parsed {total} niches across {len(out)} leagues:")
    for lg, niches in out.items():
        print(f"  {lg}: {len(niches)} niches")
        for n in niches:
            non_null = {k: v for k, v in n.items() if v is not None and k != "niche_id"}
            print(f"    {n['niche_id']}")
