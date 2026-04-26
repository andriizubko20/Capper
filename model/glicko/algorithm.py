"""
model/glicko/algorithm.py

Pure-Python Glicko-2 implementation.

Reference: Glickman, M. E. (2013). Example of the Glicko-2 system.
http://www.glicko.net/glicko/glicko2.pdf

Each team has 3 numbers: rating (μ, default 1500), RD (φ, default 350),
volatility (σ, default 0.06).

Key API:
    rating  = TeamRating(rating=1500, rd=350, volatility=0.06)
    update_rating(player, opponents) → new TeamRating

Football-specific tuning:
    - τ (tau) = 0.5 (system constant; smaller = ratings more responsive)
    - Win = 1.0, Draw = 0.5, Loss = 0.0 — same as standard
"""
import math
from dataclasses import dataclass, replace
from typing import Sequence

GLICKO2_SCALE = 173.7178  # conversion constant between Glicko and Glicko-2


@dataclass(frozen=True)
class TeamRating:
    """Mutable-replace team rating as of a given moment."""
    rating: float = 1500.0
    rd: float = 350.0
    volatility: float = 0.06


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _new_volatility(
    sigma: float, phi: float, v: float, delta: float, tau: float = 0.5
) -> float:
    """Iterative volatility update via Illinois algorithm (Glickman 5.1, step 5)."""
    a = math.log(sigma * sigma)
    eps = 1e-6

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return num / den - (x - a) / (tau * tau)

    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
            if k > 100:
                break
        B = a - k * tau

    fA, fB = f(A), f(B)
    while abs(B - A) > eps:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA = fA / 2.0
        B, fB = C, fC

    return math.exp(A / 2.0)


def update_rating(
    player: TeamRating,
    opponents: Sequence[tuple[TeamRating, float]],
    tau: float = 0.5,
) -> TeamRating:
    """
    Update player's rating after a rating period.

    Args:
        player: current TeamRating
        opponents: list of (opponent_rating, score) where score ∈ {0, 0.5, 1}
        tau: system volatility constraint (typical 0.3-1.2)

    Returns: new TeamRating
    """
    if not opponents:
        # No games — only RD increases by volatility
        new_rd = math.sqrt(_glicko_to_glicko2_phi(player.rd) ** 2 + player.volatility ** 2)
        return replace(player, rd=_glicko2_to_glicko_rd(new_rd))

    mu = _glicko_to_glicko2_mu(player.rating)
    phi = _glicko_to_glicko2_phi(player.rd)

    # v: estimated variance of rating from games (step 3)
    v_inv = 0.0
    delta_sum = 0.0
    for opp, score in opponents:
        mu_j = _glicko_to_glicko2_mu(opp.rating)
        phi_j = _glicko_to_glicko2_phi(opp.rd)
        g_j = _g(phi_j)
        E_ij = _E(mu, mu_j, phi_j)
        v_inv += g_j * g_j * E_ij * (1 - E_ij)
        delta_sum += g_j * (score - E_ij)
    v = 1.0 / v_inv

    # delta: estimated improvement (step 4)
    delta = v * delta_sum

    # New volatility (step 5)
    sigma_new = _new_volatility(player.volatility, phi, v, delta, tau=tau)

    # Update phi* (step 6)
    phi_star = math.sqrt(phi * phi + sigma_new * sigma_new)

    # New phi (step 7)
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)

    # New mu (step 7)
    mu_new = mu + phi_new * phi_new * delta_sum

    return TeamRating(
        rating=_glicko2_to_glicko_mu(mu_new),
        rd=_glicko2_to_glicko_rd(phi_new),
        volatility=sigma_new,
    )


def expected_score(player: TeamRating, opponent: TeamRating) -> float:
    """Probability that `player` wins/draws (E ∈ [0, 1])."""
    mu = _glicko_to_glicko2_mu(player.rating)
    mu_j = _glicko_to_glicko2_mu(opponent.rating)
    phi_j = _glicko_to_glicko2_phi(opponent.rd)
    return _E(mu, mu_j, phi_j)


# ── Glicko ↔ Glicko-2 conversions ────────────────────────────────────────────
def _glicko_to_glicko2_mu(rating: float) -> float:
    return (rating - 1500.0) / GLICKO2_SCALE


def _glicko2_to_glicko_mu(mu: float) -> float:
    return mu * GLICKO2_SCALE + 1500.0


def _glicko_to_glicko2_phi(rd: float) -> float:
    return rd / GLICKO2_SCALE


def _glicko2_to_glicko_rd(phi: float) -> float:
    return phi * GLICKO2_SCALE


# ── Football-specific helper: 3 outcomes → score ─────────────────────────────
def football_score(home_score: int, away_score: int, side: str) -> float:
    """Convert match result to Glicko score for the given side."""
    if home_score > away_score:
        return 1.0 if side == "home" else 0.0
    if home_score < away_score:
        return 0.0 if side == "home" else 1.0
    return 0.5
