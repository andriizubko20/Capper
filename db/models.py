from datetime import date, datetime
from sqlalchemy import (
    BigInteger, Boolean, Date, DateTime, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.session import Base


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_id: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(100))
    country: Mapped[str] = mapped_column(String(100))
    season: Mapped[int] = mapped_column(Integer)

    __table_args__ = (UniqueConstraint("api_id", "season"),)

    teams: Mapped[list["Team"]] = relationship(back_populates="league")
    matches: Mapped[list["Match"]] = relationship(back_populates="league")


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_id: Mapped[int] = mapped_column(Integer, unique=True)
    name: Mapped[str] = mapped_column(String(100))
    country: Mapped[str] = mapped_column(String(100))
    league_id: Mapped[int | None] = mapped_column(ForeignKey("leagues.id"), nullable=True)
    elo: Mapped[float] = mapped_column(Float, default=1500.0)

    league: Mapped["League | None"] = relationship(back_populates="teams")


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_id: Mapped[int] = mapped_column(Integer, unique=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"))
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    date: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(50))
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    elapsed: Mapped[int | None] = mapped_column(Integer, nullable=True)

    league: Mapped["League"] = relationship(back_populates="matches")
    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id])
    stats: Mapped["MatchStats | None"] = relationship(back_populates="match", uselist=False)
    odds: Mapped[list["Odds"]] = relationship(back_populates="match")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="match")
    injury_reports: Mapped[list["InjuryReport"]] = relationship(back_populates="match")


class MatchStats(Base):
    __tablename__ = "match_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), unique=True)

    home_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_xg: Mapped[float | None] = mapped_column(Float, nullable=True)

    home_glicko: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_glicko: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_win_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    draw_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_win_prob: Mapped[float | None] = mapped_column(Float, nullable=True)

    home_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_shots_inside_box: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots_inside_box: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_possession: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_possession: Mapped[float | None] = mapped_column(Float, nullable=True)

    home_corners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_corners: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_passes_accurate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_passes_accurate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_passes_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_passes_total: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_gk_saves: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_gk_saves: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_yellow_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_yellow_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_red_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_red_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_fouls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_fouls: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_offsides: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_offsides: Mapped[int | None] = mapped_column(Integer, nullable=True)

    match: Mapped["Match"] = relationship(back_populates="stats")


class InjuryReport(Base):
    __tablename__ = "injury_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    player_api_id: Mapped[int] = mapped_column(Integer)
    player_name: Mapped[str] = mapped_column(String(150))
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)

    __table_args__ = (
        UniqueConstraint("match_id", "team_id", "player_api_id"),
    )

    match: Mapped["Match"] = relationship(back_populates="injury_reports")
    team: Mapped["Team"] = relationship()


class Odds(Base):
    __tablename__ = "odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    market: Mapped[str] = mapped_column(String(50))
    bookmaker: Mapped[str] = mapped_column(String(100))
    outcome: Mapped[str] = mapped_column(Text)
    value: Mapped[float] = mapped_column(Float)
    opening_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_closing: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        UniqueConstraint("match_id", "market", "bookmaker", "outcome", "recorded_at"),
    )

    match: Mapped["Match"] = relationship(back_populates="odds")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int | None] = mapped_column(ForeignKey("matches.id"), nullable=True)
    market: Mapped[str] = mapped_column(String(50))
    outcome: Mapped[str] = mapped_column(Text)
    probability: Mapped[float] = mapped_column(Float)
    odds_used: Mapped[float] = mapped_column(Float)
    ev: Mapped[float] = mapped_column(Float)
    kelly_fraction: Mapped[float] = mapped_column(Float)
    stake: Mapped[float | None] = mapped_column(Float, nullable=True)
    weighted_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    clv: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # result tracking
    result: Mapped[str | None] = mapped_column(Text, nullable=True)       # 'win' | 'loss'
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)        # profit/loss in $
    timing: Mapped[str | None] = mapped_column(String(10), nullable=True)  # 'early' | 'final'
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # denormalized fields for historical / imported predictions (match_id=None)
    match_date:  Mapped[date | None] = mapped_column(Date,        nullable=True)
    home_name:   Mapped[str | None]  = mapped_column(String(100), nullable=True)
    away_name:   Mapped[str | None]  = mapped_column(String(100), nullable=True)
    league_name: Mapped[str | None]  = mapped_column(String(100), nullable=True)

    match: Mapped["Match | None"] = relationship(back_populates="predictions")


class MonsterPIs(Base):
    __tablename__ = "monster_p_is"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league: Mapped[str] = mapped_column(String(100))
    niche_str: Mapped[str] = mapped_column(String(255))
    p_is: Mapped[float] = mapped_column(Float)
    n_samples: Mapped[int] = mapped_column(Integer)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("league", "niche_str"),)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    username: Mapped[str | None] = mapped_column(String(100), nullable=True)
    bankroll: Mapped[float] = mapped_column(Float, default=1000.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    snapshots: Mapped[list["BankrollSnapshot"]] = relationship(
        back_populates="user", order_by="BankrollSnapshot.created_at"
    )


class BankrollSnapshot(Base):
    __tablename__ = "bankroll_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    balance: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="snapshots")
